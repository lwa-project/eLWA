#!/usr/bin/env python3

"""
Correlator for LWA and/or VLA data.
"""

import os
import re
import sys
import time
import ephem
import numpy as np
import getpass
import argparse
from datetime import datetime

from astropy.constants import c as vLight
vLight = vLight.to('m/s').value

from lsl import astro
from lsl.common import stations, metabundle
from lsl.statistics import robust
from lsl.correlator import uvutils
from lsl.correlator import fx as fxc
from lsl.writer import fitsidi
from lsl.correlator.uvutils import compute_uvw

from lsl.reader import drx, errors
from lsl.reader.buffer import DRXFrameBuffer
from lsl.reader.base import CI8

import jones
import multirate
from utils import *


def main(args):
    # Build up the station
    site = stations.lwa1
    ## Updated 2018/3/8 with solutions from the 2018 Feb 28 eLWA
    ## run.  See createConfigFile.py for details.
    site.lat = 34.068956328 * np.pi/180
    site.long = -107.628103026 * np.pi/180
    site.elev = 2132.96837346
    observer = site.get_observer()
    
    # Parse the correlator configuration
    config, refSrc, filenames, metanames, foffsets, readers, antennas = read_correlator_configuration(args.filename)
    try:
        args.fft_length = config['channels']
        args.dump_time = config['inttime']
        print(f"NOTE: Set FFT length to {args.fft_length} and dump time to {args.dump_time:.3f} s per user defined configuration")
    except (TypeError, KeyError):
        pass
    if args.duration == 0.0:
        args.duration = refSrc.duration
    args.duration = min([args.duration, refSrc.duration])
    
    # Length of the FFT
    LFFT = args.fft_length
    
    # Get the raw configuration
    with open(args.filename, 'r') as fh:
        rawConfig = fh.readlines()
        
    # Antenna report
    print("Antennas:")
    for ant in antennas:
        print(f"  Antenna {ant.id}: Stand {ant.stand.id}, Pol. {ant.pol} ({ant.cable.clock_offset*1e6:.3f} us offset)")
        
    # Open and align the files
    fh = []
    nFramesFile = []
    srate = []
    beams = []
    tunepols = []
    beampols = []
    tStart = []
    cFreqs = []
    bitDepths = []
    buffers = []
    grossOffsets = []
    for i,(filename,metaname,foffset) in enumerate(zip(filenames, metanames, foffsets)):
        fh.append( open(filename, "rb") )
        
        go = int(round(antennas[2*i].cable.clock_offset*196e6)) / 196e6
        antennas[2*i+0].cable.clock_offset -= go
        antennas[2*i+1].cable.clock_offset -= go
        grossOffsets.append( -go )
        if go != 0:
            print(f"Correcting time tags for gross offset of {grossOffsets[i]*1e6:.3f} us")
            print(f"  Antenna clock offsets are now at {antennas[2*i+0].cable.clock_offset*1e6:.3f} us, {antennas[2*i+1].cable.clock_offset*1e6:.3f} us")
            
        nFramesFile.append( os.path.getsize(filename) // readers[i].FRAME_SIZE )
        junkFrame = readers[i].read_frame(fh[i])
        while junkFrame.header.decimation == 0:
            junkFrame = readers[i].read_frame(fh[i])
        readers[i].DATA_LENGTH = junkFrame.payload.data.size
        beam, tune, pol = junkFrame.id
        fh[i].seek(-readers[i].FRAME_SIZE, 1)
        
        beams.append( beam )
        srate.append( junkFrame.sample_rate )
        
        beampols.append( max(readers[i].get_frames_per_obs(fh[i])) )
            
        skip = args.skip + foffset
        if skip != 0:
            print(f"Skipping forward {skip:.3f} s")
            print(f"-> {float(junkFrame.time):.6f} ({junkFrame.time.datetime})")
            
            offset = int(skip*srate[i] / readers[i].DATA_LENGTH)
            fh[i].seek(beampols[i]*readers[i].FRAME_SIZE*offset, 1)
            junkFrame = readers[i].read_frame(fh[i])
            fh[i].seek(-readers[i].FRAME_SIZE, 1)
            
            print(f"-> {float(junkFrame.time):.6f} ({junkFrame.time.datetime})")
            
        tStart.append( junkFrame.time + grossOffsets[i] )
        
        # Get the frequencies
        cFreq1 = 0.0
        cFreq2 = 0.0
        for j in range(64):
            junkFrame = readers[i].read_frame(fh[i])
            b,t,p = junkFrame.id
            if p == 0:
                if t == 1:
                    cFreq1 = junkFrame.central_freq
                else:
                    cFreq2 = junkFrame.central_freq
            else:
                pass
        fh[i].seek(-64*readers[i].FRAME_SIZE, 1)
        cFreqs.append( [cFreq1,cFreq2] )
        try:
            bitDepths.append( junkFrame.header.bits_per_sample )
        except AttributeError:
            bitDepths.append( 8 )
            
        # Setup the frame buffers
        buffers.append( DRXFrameBuffer(beams=[beam,], tunes=[1,2], pols=[0,1], nsegments=16) )
    for i in range(len(filenames)):
        # Align the files as close as possible by the time tags
        junkFrame = readers[i].read_frame(fh[i])
        fh[i].seek(-readers[i].FRAME_SIZE, 1)
            
        j = 0
        while junkFrame.time + grossOffsets[i] < max(tStart):
            for k in range(beampols[i]):
                junkFrame = readers[i].read_frame(fh[i])
            j += beampols[i]
            
        jTime = j*readers[i].DATA_LENGTH/srate[i]/beampols[i]
        print(f"Shifted beam {beams[i]} data by {j} frames ({jTime:.4f} s)")
        
    # Set integration time
    tRead = 1.0
    nFrames = int(round(tRead*srate[-1]/readers[-1].DATA_LENGTH))
    tRead = nFrames*readers[-1].DATA_LENGTH/srate[-1]
    
    nFramesD = nFrames
    
    # Read in some data
    tFile = nFramesFile[-1] / beampols[-1] * readers[-1].DATA_LENGTH / srate[-1]
    if args.duration > 0.0:
        duration = args.duration
        duration = tRead * int(round(duration / tRead))
        tFile = duration
        
    # Date
    beginMJDs = []
    beginDates = []
    for i in range(len(filenames)):
        junkFrame = readers[i].read_frame(fh[i])
        fh[i].seek(-readers[i].FRAME_SIZE, 1)
        
        beginMJDs.append( (junkFrame.time + grossOffsets[i]).mjd )
        beginDates.append( (junkFrame.time + grossOffsets[i]).datetime )
        
    # Set the output base filename
    if args.tag is None:
        outbase = os.path.basename(filenames[0])
        outbase = os.path.splitext(outbase)[0][:8]
    else:
        outbase = args.tag
        
    # Report
    for i in range(len(filenames)):
        print(f"Filename: {os.path.basename(filenames[i])}")
        print(f"  Type/Reader: {readers[i].__name__}")
        print(f"  Date of First Frame: {beginDates[i]}")
        print(f"  Sample Rate: {srate[i]} Hz")
        print(f"  Tuning 1: {cFreqs[i][0]:.3f} Hz")
        print(f"  Tuning 2: {cFreqs[i][1]:.3f} Hz")
        print(f"  Bit Depth: {bitDepths[i]}")
    print("  ===")
    print("  Phase Center:")
    print(f"    Name: {refSrc.name}")
    print(f"    RA: {str(refSrc._ra)}")
    print(f"    Dec: {str(refSrc._dec)}")
    print("  ===")
    print(f"  Data Read Time: {tRead:.3f} s")
    print(f"  Data Reads in File: {int(tFile/tRead)}")
    print(" ")
    
    nDRXInputs = sum([1 for reader in readers if reader is drx])
    print(f"Processing {nDRXInputs} DRX input streams")
    print(" ")
    
    nFramesB = nFrames
    framesPerSecondB = srate[-1] / readers[-1].DATA_LENGTH
    if nDRXInputs:
        print(f"DRX Frames/s: {framesPerSecondB:.6f}")
        print(f"DRX Frames/Integration: {nFramesB}")
    print(" ")
    
    drxLFFT = LFFT * srate[-1] / srate[0]
    while drxLFFT != int(drxLFFT):
        LFFT += 1
        drxLFFT = LFFT * srate[-1] / srate[0]
    drxLFFT = int(drxLFFT)
    print(f"DRX Transform Size: {drxLFFT}")
    print(" ")
    
    nChunks = int(tFile/tRead)
    tSub = args.subint_time
    tSub = tRead / int(round(tRead/tSub))
    tDump = args.dump_time
    tDump = tSub * int(round(tDump/tSub))
    nDump = int(tDump / tSub)
    tDump = nDump * tSub
    nInt = int((nChunks*tRead) / tDump)
    print(f"Sub-integration time is: {tSub:.3f} s")
    print(f"Integration (dump) time is: {tDump:.3f} s")
    print(" ")
    
    if args.gpu is not None:
        try:
            import xcupy
            xcupy.select_gpu(args.gpu)
            xcupy.set_memory_usage_limit(1.5*1024**3)
            multirate.xengine = xcupy.xengine
            multirate.xengine_full = xcupy.xengine_full
            print(f"Loaded GPU X-engine support on GPU #{args.gpu} with {xcupy.get_memory_usage_limit()/1024.0**3:.2f} GB of device memory")
        except ImportError as e:
            pass
            
    subIntTimes = []
    subIntCount = 0
    fileCount   = 0
    wallStart = time.time()
    done = False
    oldStartRel = [0 for i in range(nDRXInputs)]
    username = getpass.getuser()
    for i in range(nChunks):
        wallTime = time.time()
        
        tStart = []
        tStartB = []
        
        drxRef  = [0 for j in range(nDRXInputs*2) ]
        
        # Read in the data
        with InterProcessLock(f"/dev/shm/sc-reader-{username}") as lock:
            try:
                dataDL['re'][...] = 0        # pylint: disable=possibly-used-before-assignment,used-before-assignment
                dataDL['im'][...] = 0        # pylint: disable=possibly-used-before-assignment,used-before-assignment
                dataDH['re'][...] = 0        # pylint: disable=possibly-used-before-assignment,used-before-assignment
                dataDH['im'][...] = 0        # pylint: disable=possibly-used-before-assignment,used-before-assignment
            except NameError:
                dataDL = np.zeros((len(drxRef),  readers[-1].DATA_LENGTH*nFramesD), dtype=CI8)
                dataDH = np.zeros((len(drxRef),  readers[-1].DATA_LENGTH*nFramesD), dtype=CI8)
                dataDL_view = dataDL.view(np.int16)
                dataDH_view = dataDH.view(np.int16)
            for j,f in enumerate(fh):
                ## DRX
                k = 0
                while k < beampols[j]*nFramesD:
                    try:
                        cFrame = readers[j].read_frame_ci8(f)
                        buffers[j].append( cFrame )
                    except errors.SyncError:
                        print(f"Error - DRX @ {i}, {j}")
                        continue
                    except errors.EOFError:
                        done = True
                        break
                        
                    frames = buffers[j].get()
                    if frames is None:
                        continue
                        
                    for cFrame in frames:
                        beam,tune,pol = cFrame.id
                        bid = 2*j + pol
                        
                        cFrame.payload.timetag += int(grossOffsets[j]*196e6)
                        
                        if k == 0:
                            tStart.append( cFrame.time )
                            tStartB.append( get_better_time(cFrame) )
                            
                            for p in (0,1):
                                pbid = 2*j + p
                                drxRef[pbid] = cFrame.payload.timetag
                                
                        count = cFrame.payload.timetag
                        count -= drxRef[bid]
                        count //= (4096*int(196e6/srate[-1]))
                        ### Fix from some LWA-SV files that seem to cause the current LSL
                        ### ring buffer problems
                        if count < 0:
                            continue
                        try:
                            if tune == 1:
                                dataDL_view[bid, count*readers[j].DATA_LENGTH:(count+1)*readers[j].DATA_LENGTH] = cFrame.payload.data.view(np.int16)     # pylint: disable=possibly-used-before-assignment,used-before-assignment
                            else:
                                dataDH_view[bid, count*readers[j].DATA_LENGTH:(count+1)*readers[j].DATA_LENGTH] = cFrame.payload.data.view(np.int16)     # pylint: disable=possibly-used-before-assignment,used-before-assignment
                            k += beampols[j]//2
                        except ValueError:
                            k = beampols[j]*nFramesD
                            break
                            
        print(f"RR - Read finished in {time.time()-wallTime:.3f} s for {tRead:.3f} s of data")
        
        # Time tag alignment (sample based)
        ## Initial time tags for each stream and the relative start time for each stream
        if args.verbose:
            ### TT = time tag
            print('TT - Start', tStartB)
        tStartMin = min([sec for sec,frac in tStartB])
        tStartRel = [(sec-tStartMin)+frac for sec,frac in tStartB]
        
        ## Sample offsets between the streams
        offsets = []
        for j in range(nDRXInputs):
            offsets.append( int( round(nsround(max(tStartRel) - tStartRel[j])*srate[j]) ) )
        if args.verbose:
            print('TT - Offsets', offsets)
            
        ## Roll the data to apply the sample offsets and then trim the ends to get rid 
        ## of the rolled part
        for j,offset in enumerate(offsets):
            if offset != 0:
                idx0 = 2*j + 0
                idx1 = 2*j + 1
                tStart[j] += offset/(srate[j])
                tStartB[j][1] += offset/(srate[j])
                while tStartB[j][1] >= 1.0:
                    tStartB[j][0] += 1
                    tStartB[j][1] -= 1
                while tStartB[j][1] < 0.0:
                    tStartB[j][0] -= 1
                    tStartB[j][1] += 1
                dataDL[idx0,:] = np.roll(dataDL[idx0,:], -offset)
                dataDL[idx1,:] = np.roll(dataDL[idx1,:], -offset)
                dataDH[idx0,:] = np.roll(dataDH[idx0,:], -offset)
                dataDH[idx1,:] = np.roll(dataDH[idx1,:], -offset)
                
        drxOffsets = offsets
        
        ## Apply the corrections to the original time tags and report on the sub-sample
        ## residuals
        if args.verbose:
            print('TT - Adjusted', tStartB)
        tStartMinSec  = min([sec  for sec,frac in tStartB])
        tStartMinFrac = min([frac for sec,frac in tStartB])
        tStartRel = [(sec-tStartMinSec)+(frac-tStartMinFrac) for sec,frac in tStartB]
        if args.verbose:
            print('TT - Residual', ["%.1f ns" % (r*1e9,) for r in tStartRel])
        for k in range(len(tStartRel)):
            antennas[2*k+0].cable.clock_offset -= tStartRel[k] - oldStartRel[k]
            antennas[2*k+1].cable.clock_offset -= tStartRel[k] - oldStartRel[k]
        oldStartRel = tStartRel
        
        # Setup everything we need to loop through the sub-integrations
        nSub = int(tRead/tSub)
        nSampD = int(srate[-1]*tSub)
        
        #tV = i*tRead + np.arange(dataV.shape[1]-max(vdifOffsets), dtype=np.float64)/srate[ 0]
        if nDRXInputs > 0:
            tD = i*tRead + np.arange(dataDL.shape[1]-max(drxOffsets), dtype=np.float64)/srate[-1]
            
        # Loop over sub-integrations
        for j in range(nSub):
            ## Select the data to work with
            tSubInt = tStart[0] + (j+1)*nSampD/srate[-1] - nSampD//2/srate[-1]
            tDSub    = tD[j*nSampD:(j+1)*nSampD]
            dataDLSub = dataDL[:,j*nSampD:(j+1)*nSampD]
            dataDHSub = dataDH[:,j*nSampD:(j+1)*nSampD]
            if nDRXInputs > 0:
                if dataDLSub.shape[1] != tDSub.size:
                    dataDLSub = dataDLSub[:,:tDSub.size]
                    dataDHSub = dataDHSub[:,:tDSub.size]
                if tDSub.size == 0:
                    continue
                    
                try:
                    if dataDLSubF.shape[1] != dataDLSub.shape[1]:
                        del dataDLSubF
                        del dataDHSubF
                    dataDLSubF.real[...] = dataDLSub['re']
                    dataDLSubF.imag[...] = dataDLSub['im']
                    dataDHSubF.real[...] = dataDHSub['re']
                    dataDHSubF.imag[...] = dataDHSub['im']
                except NameError:
                    dataDLSubF = dataDLSub['re'] + 1j*dataDLSub['im']
                    dataDLSubF = dataDLSubF.astype(np.complex64)
                    dataDHSubF = dataDHSub['re'] + 1j*dataDHSub['im']
                    dataDHSubF = dataDHSubF.astype(np.complex64)
                    
            ## Update the observation
            observer.date = astro.unix_to_utcjd(tSubInt) - astro.DJD_OFFSET
            refSrc.compute(observer)
            
            ## Correct for the LWA dipole power pattern
            dipoleX, dipoleY = jones.get_lwa_antenna_gain(observer, refSrc, freq=cFreqs[-1][0])
            dataDLSubF[0::2,:] /= np.sqrt(dipoleX) * 7
            dataDLSubF[1::2,:] /= np.sqrt(dipoleY) * 7
            dipoleX, dipoleY = jones.get_lwa_antenna_gain(observer, refSrc, freq=cFreqs[-1][1])
            dataDHSubF[0::2,:] /= np.sqrt(dipoleX) * 7
            dataDHSubF[1::2,:] /= np.sqrt(dipoleY) * 7
            
            ## Correlate
            delayPadding = multirate.get_optimal_delay_padding(antennas, [],
                                                               LFFT=drxLFFT, sample_rate=srate[-1], 
                                                               central_freq=cFreqs[-1][0], 
                                                               pol='*', phase_center=refSrc)
            freqDL, feoDL, veoDL, deoDL = multirate.fengine(dataDLSubF, antennas, LFFT=drxLFFT,
                                                            sample_rate=srate[-1], central_freq=cFreqs[-1][0], 
                                                            pol='*', phase_center=refSrc, 
                                                            delayPadding=delayPadding)
            freqDH, feoDH, veoDH, deoDH = multirate.fengine(dataDHSubF, antennas, LFFT=drxLFFT,
                                                            sample_rate=srate[-1], central_freq=cFreqs[-1][1], 
                                                            pol='*', phase_center=refSrc, 
                                                            delayPadding=delayPadding)
            
            if feoDL.shape[2] == 0:
                continue
                
            ## Sort out what goes where (channels and antennas) if we don't already know
            try:
                freqDL = freqDL[goodDL]        # pylint: disable=possibly-used-before-assignment,used-before-assignment
                feoDL = np.roll(feoDL, -goodDL[0], axis=1)[:,:len(goodDL),:]
                freqDH = freqDH[goodDL]        # pylint: disable=possibly-used-before-assignment,used-before-assignment
                feoDH = np.roll(feoDH, -goodDH[0], axis=1)[:,:len(goodDH),:]
            except NameError:
                ### Frequency overlap
                fMin, fMax = -1e12, 1e12
                fMin, fMax = max([fMin, freqDL.min()]), min([fMax, freqDL.max()])
                    
                ### Channels and antennas (X vs. Y)
                goodDL = np.where( (freqDL >= fMin) & (freqDL <= fMax) )[0]
                aXD = [k for (k,a) in enumerate(antennas) if a.pol == 0]
                aYD = [k for (k,a) in enumerate(antennas) if a.pol == 1]
                
                ### Apply
                freqDL = freqDL[goodDL]
                feoDL = np.roll(feoDL, -goodDL[0], axis=1)[:,:len(goodDL),:]
                
                ### Frequency overlap
                fMin, fMax = -1e12, 1e12
                fMin, fMax = max([fMin, freqDH.min()]), min([fMax, freqDH.max()])
                    
                ### Channels and antennas (X vs. Y)
                goodDH = np.where( (freqDH >= fMin) & (freqDH <= fMax) )[0]
                aXD = [k for (k,a) in enumerate(antennas) if a.pol == 0]
                aYD = [k for (k,a) in enumerate(antennas) if a.pol == 1]
                
                ### Apply
                freqDH = freqDH[goodDH]
                feoDH = np.roll(feoDH, -goodDH[0], axis=1)[:,:len(goodDH),:]
            nchan = freqDL.size
            fdt = feoDL.dtype
            vdt = veoDL.dtype        # pylint: disable=possibly-used-before-assignment,used-before-assignment
            ## Setup the intermediate F-engine products and trim the data
            ### Figure out the minimum number of windows
            nWin = 1e12
            nWin = min([nWin, feoDL.shape[2]])
            nWin = min([nWin, np.argmax(np.cumsum(veoDL.sum(axis=0)))+1])
                
            ### Initialize the intermediate arrays
            try:
                assert(feoXL.shape[2] == nWin)       # pylint: disable=possibly-used-before-assignment,used-before-assignment
            except (NameError, AssertionError):
                feoXL = np.zeros((nDRXInputs, nchan, nWin), dtype=fdt)
                feoYL = np.zeros((nDRXInputs, nchan, nWin), dtype=fdt)
                veoXL = np.zeros((nDRXInputs, nWin), dtype=vdt)
                veoYL = np.zeros((nDRXInputs, nWin), dtype=vdt)
                feoXH = np.zeros((nDRXInputs, nchan, nWin), dtype=fdt)
                feoYH = np.zeros((nDRXInputs, nchan, nWin), dtype=fdt)
                veoXH = np.zeros((nDRXInputs, nWin), dtype=vdt)
                veoYH = np.zeros((nDRXInputs, nWin), dtype=vdt)
                
            ### Trim
            feoDL = feoDL[:,:,:nWin]
            veoDL = veoDL[:,:nWin]
            feoDH = feoDH[:,:,:nWin]
            veoDH = veoDH[:,:nWin]
                
            ## Sort it all out by polarization
            for k in range(nDRXInputs):
                feoXL[k,:,:] = feoDL[aXD[k],:,:]      # pylint: disable=possibly-used-before-assignment,used-before-assignment
                feoYL[k,:,:] = feoDL[aYD[k],:,:]      # pylint: disable=possibly-used-before-assignment,used-before-assignment
                veoXL[k,:] = veoDL[aXD[k],:]          # pylint: disable=possibly-used-before-assignment,used-before-assignment
                veoYL[k,:] = veoDL[aYD[k],:]          # pylint: disable=possibly-used-before-assignment,used-before-assignment
                feoXH[k,:,:] = feoDH[aXD[k],:,:]      # pylint: disable=possibly-used-before-assignment,used-before-assignment
                feoYH[k,:,:] = feoDH[aYD[k],:,:]      # pylint: disable=possibly-used-before-assignment,used-before-assignment
                veoXH[k,:] = veoDH[aXD[k],:]          # pylint: disable=possibly-used-before-assignment,used-before-assignment
                veoYH[k,:] = veoDH[aYD[k],:]          # pylint: disable=possibly-used-before-assignment,used-before-assignment
                
            ## Cross multiply
            sfreqXXL = freqDL
            sfreqYYL = freqDL
            svisXXL, svisXYL, svisYXL, svisYYL = multirate.xengine_full(feoXL, veoXL, feoYL, veoYL)
            sfreqXXH = freqDH
            sfreqYYH = freqDH
            svisXXH, svisXYH, svisYXH, svisYYH = multirate.xengine_full(feoXH, veoXH, feoYH, veoYH)
            
            ## Accumulate
            if subIntCount == 0:
                subIntTimes = [tSubInt,]
                freqXXL = sfreqXXL
                freqYYL = sfreqYYL
                visXXL  = svisXXL / nDump
                visXYL  = svisXYL / nDump
                visYXL  = svisYXL / nDump
                visYYL  = svisYYL / nDump
                freqXXH = sfreqXXH
                freqYYH = sfreqYYH
                visXXH  = svisXXH / nDump
                visXYH  = svisXYH / nDump
                visYXH  = svisYXH / nDump
                visYYH  = svisYYH / nDump
            else:
                subIntTimes.append( tSubInt )
                visXXL += svisXXL / nDump
                visXYL += svisXYL / nDump
                visYXL += svisYXL / nDump
                visYYL += svisYYL / nDump
                visXXH += svisXXH / nDump
                visXYH += svisXYH / nDump
                visYXH += svisYXH / nDump
                visYYH += svisYYH / nDump
            subIntCount += 1
            
            ## Save
            if subIntCount == nDump:
                subIntCount = 0
                fileCount += 1
                
                ### CD = correlator dump
                outfile = f"{outbase}L-vis2-{fileCount:05d}.npz"
                np.savez(outfile, config=rawConfig, srate=srate[0]/2.0, freq1=freqXXL,       # pylint: disable=possibly-used-before-assignment,used-before-assignment
                            vis1XX=visXXL, vis1XY=visXYL, vis1YX=visYXL, vis1YY=visYYL, 
                            tStart=np.mean(np.array(subIntTimes, dtype=np.float64)), tInt=tDump)
                outfile = f"{outbase}H-vis2-{fileCount:05d}.npz"
                np.savez(outfile, config=rawConfig, srate=srate[0]/2.0, freq1=freqXXH,       # pylint: disable=possibly-used-before-assignment,used-before-assignment
                            vis1XX=visXXH, vis1XY=visXYH, vis1YX=visYXH, vis1YY=visYYH, 
                            tStart=np.mean(np.array(subIntTimes, dtype=np.float64)), tInt=tDump)
                print("CD - writing integration %i to disk, timestamp is %.3f s" % (fileCount, np.mean(np.array(subIntTimes, dtype=np.float64))))
                if fileCount == 1:
                    print("CD - each integration is %.1f MB on disk" % (os.path.getsize(outfile)/1024.0**2,))
                if (fileCount-1) % 25 == 0:
                    print("CD - average processing time per integration is %.3f s" % ((time.time() - wallStart)/fileCount,))
                    etc = (nInt - fileCount) * (time.time() - wallStart)/fileCount
                    eth = int(etc/60.0) // 60
                    etm = int(etc/60.0) % 60
                    ets = etc % 60
                    print("CD - estimated time to completion is %i:%02i:%04.1f" % (eth, etm, ets))
                    
        if done:
            break
            
    # Cleanup
    etc = time.time() - wallStart
    eth = int(etc/60.0) // 60
    etm = int(etc/60.0) % 60
    ets = etc % 60
    print("Processing finished after %i:%02i:%04.1f" % (eth, etm, ets))
    print(f"Average time per integration was {etc/fileCount:.3f} s")
    for f in fh:
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='the next generation of correlator for LWA/VLA/eLWA data', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('filename', type=str, 
                        help='configuration file to process')
    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false', default=True, 
                        help='disable verbose time tag information')
    parser.add_argument('-l', '--fft-length', type=int, default=512, 
                        help='set FFT length')
    parser.add_argument('-s', '--skip', type=float, default=0.0, 
                        help='amount of time in seconds to skip into the files')
    parser.add_argument('-u', '--subint-time', type=float, default=0.010, 
                        help='sub-integration time in seconds for the data')
    parser.add_argument('-t', '--dump-time', type=float, default=1.0, 
                        help='correlator dump time in seconds for saving the visibilties')
    parser.add_argument('-d', '--duration', type=float, default=0.0, 
                        help='duration in seconds of the file to correlate; 0 = everything')
    parser.add_argument('-g', '--tag', type=str, 
                        help='tag to use for the output file')
    parser.add_argument('--gpu', type=int,
                        help='enable the experimental GPU X-engine')
    parser.add_argument('-w', '--which', type=int, default=0, 
                        help='for LWA-only observations, which tuning to use for correlation; 0 = auto-select')
    args = parser.parse_args()
    main(args)
    
