#!/usr/bin/env python

"""
Correlator for LWA and/or VLA data.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import os
import re
import sys
import math
import time
import ephem
import numpy
import argparse
from datetime import datetime

from astropy.constants import c as vLight
vLight = vLight.to('m/s').value

from lsl import astro
from lsl.common import stations, metabundle
from lsl.statistics import robust
from lsl.correlator import uvutil
from lsl.correlator import fx as fxc
from lsl.writer import fitsidi
from lsl.correlator.uvutil import compute_uvw

from lsl.reader import drx, vdif, errors
from lsl.reader.buffer import DRXFrameBuffer, VDIFFrameBuffer

import guppi
from lsl.misc.dedispersion import delay as dispDelay

import jones
from utils import *
from buffer import GUPPIFrameBuffer


def bestFreqUnits(freq):
    """Given a numpy array of frequencies in Hz, return a new array with the
    frequencies in the best units possible (kHz, MHz, etc.)."""
    
    # Figure out how large the data are
    try:
        scale = int(math.log10(max(freq)))
    except TypeError:
        scale = int(math.log10(freq))
    if scale >= 9:
        divis = 1e9
        units = 'GHz'
    elif scale >= 6:
        divis = 1e6
        units = 'MHz'
    elif scale >= 3:
        divis = 1e3
        units = 'kHz'
    elif scale >= 0:
        divis = 1
        units = 'Hz'
    elif scale >= -3:
        divis = 1e-3
        units = 'mHz'
    else:
        divis = 1e-6
        units = 'uHz'
        
    # Convert the frequency
    newFreq = freq / divis
    
    # Return units and freq
    return (newFreq, units)


def main(args):
    # Select the multirate module to use
    if args.jit:
        from jit import multirate
    else:
        import multirate
        
    # Build up the station
    site = stations.lwa1
    ## Updated 2018/3/8 with solutions from the 2018 Feb 28 eLWA
    ## run.  See createConfigFile.py for details.
    site.lat = 34.068956328 * numpy.pi/180
    site.long = -107.628103026 * numpy.pi/180
    site.elev = 2132.96837346
    observer = site.get_observer()
    
    # Parse the correlator configuration
    config, refSrc, filenames, metanames, foffsets, readers, antennas = read_correlator_configuration(args.filename)
    if config is not None:
        args.fft_length = config['channels']
        args.dump_time = config['inttime']
        print("NOTE: Set FFT length to %i and dump time to %.3f s per user defined configuration" % (args.fft_length, args.dump_time))
    if args.duration == 0.0:
        args.duration = refSrc.duration
    args.duration = min([args.duration, refSrc.duration])
    
    # Length of the FFT
    LFFT = args.fft_length
    
    # Get the raw configuration
    fh = open(args.filename, 'r')
    rawConfig = fh.readlines()
    fh.close()
    
    # Get the raw polycos
    fh = open(refSrc._polycos.filename, 'r')
    rawPolycos = fh.readlines()
    fh.close()
    
    # Antenna report
    print("Antennas:")
    for ant in antennas:
        print("  Antenna %i: Stand %i, Pol. %i (%.3f us offset)" % (ant.id, ant.stand.id, ant.pol, ant.cable.clock_offset*1e6))
        
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
    delaySteps = []
    buffers = []
    grossOffsets = []
    for i,(filename,metaname,foffset) in enumerate(zip(filenames, metanames, foffsets)):
        fh.append( open(filename, "rb") )
        
        go = numpy.int32(antennas[2*i].cable.clock_offset)
        antennas[2*i+0].cable.clock_offset -= go
        antennas[2*i+1].cable.clock_offset -= go
        grossOffsets.append( go )
        if go != 0:
            print("Correcting time tags for gross offset of %i s" % grossOffsets[i])
            print("  Antenna clock offsets are now at %.3f us, %.3f us" % (antennas[2*i+0].cable.clock_offset*1e6, antennas[2*i+1].cable.clock_offset*1e6))
        
        if readers[i] in (vdif, guppi):
            header = vdif.read_guppi_header(fh[i])
            readers[i].FRAME_SIZE = readers[i].get_frame_size(fh[i])
            
        nFramesFile.append( os.path.getsize(filename) / readers[i].FRAME_SIZE )
        if readers[i] is vdif:
            junkFrame = readers[i].read_frame(fh[i], central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
            readers[i].DATA_LENGTH = junkFrame.payload.data.size
            beam, pol = junkFrame.id
        elif readers[i] is guppi:
            junkFrame = readers[i].read_frame(fh[i])
            readers[i].DATA_LENGTH = junkFrame.payload.data.size
            beam, pol = junkFrame.id
        elif readers[i] is drx:
            junkFrame = readers[i].read_frame(fh[i])
            while junkFrame.header.decimation == 0:
                junkFrame = readers[i].read_frame(fh[i])
            readers[i].DATA_LENGTH = junkFrame.payload.data.size
            beam, tune, pol = junkFrame.id
        fh[i].seek(-readers[i].FRAME_SIZE, 1)
        
        beams.append( beam )
        srate.append( junkFrame.sample_rate )
        
        if readers[i] in (vdif, guppi):
            tunepols.append( readers[i].get_thread_count(fh[i]) )
            beampols.append( tunepols[i] )
        elif readers[i] is drx:
            beampols.append( max(readers[i].get_frames_per_obs(fh[i])) )
            
        skip = args.skip + foffset
        if skip != 0:
            print("Skipping forward %.3f s" % skip)
            print("-> %.6f (%s)" % (sum(junkFrame.time, 0.0), datetime.utcfromtimestamp(sum(junkFrame.time, 0.0))))
            
            offset = int(skip*srate[i] / readers[i].DATA_LENGTH)
            fh[i].seek(beampols[i]*readers[i].FRAME_SIZE*offset, 1)
            if readers[i] is vdif:
                junkFrame = readers[i].read_frame(fh[i], central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
            else:
                junkFrame = readers[i].read_frame(fh[i])
            fh[i].seek(-readers[i].FRAME_SIZE, 1)
            
            print("-> %.6f (%s)" % (sum(junkFrame.time, 0.0), datetime.utcfromtimestamp(sum(junkFrame.time, 0.0))))
            
        tStart.append( sum(junkFrame.time, 0.0) + grossOffsets[i] )
        
        # Get the frequencies
        cFreq1 = 0.0
        cFreq2 = 0.0
        for j in xrange(32):
            if readers[i] is vdif:
                junkFrame = readers[i].read_frame(fh[i], central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
                s,p = junkFrame.id
                if p == 0:
                    cFreq1 = junkFrame.central_freq
                else:
                    pass
            elif readers[i] is guppi:
                junkFrame = readers[i].read_frame(fh[i])
                s,p = junkFrame.id
                if p == 0:
                    cFreq1 = junkFrame.central_freq
                else:
                    pass
            elif readers[i] is drx:
                junkFrame = readers[i].read_frame(fh[i])
                b,t,p = junkFrame.id
                if p == 0:
                    if t == 1:
                        cFreq1 = junkFrame.central_freq
                    else:
                        cFreq2 = junkFrame.central_freq
                else:
                    pass
        fh[i].seek(-32*readers[i].FRAME_SIZE, 1)
        cFreqs.append( [cFreq1,cFreq2] )
        try:
            bitDepths.append( junkFrame.header.bits_per_sample )
        except AttributeError:
            bitDepths.append( 8 )
            
        # Parse the metadata to get the delay steps
        delayStep = None
        if readers[i] is drx and metaname is not None:
            delayStep = parse_lwa_metadata(metaname)
        delaySteps.append( delayStep )
        
        # Setup the frame buffers
        if readers[i] is vdif:
            buffers.append( VDIFFrameBuffer(threads=[0,1]) )
        elif readers[i] is guppi:
            buffers.append( GUPPIFrameBuffer(threads=[0,1]) )
        elif readers[i] is drx:
            buffers.append( DRXFrameBuffer(beams=[beam,], tunes=[1,2], pols=[0,1], nsegments=16) )
    for i in xrange(len(filenames)):
        # Align the files as close as possible by the time tags
        if readers[i] is vdif:
            timetags = []
            for k in xrange(16):
                junkFrame = readers[i].read_frame(fh[i])
                timetags.append(junkFrame.header.frame_in_second)
            fh[i].seek(-16*readers[i].FRAME_SIZE, 1)
            
            j = 0
            while (timetags[j+0] != timetags[j+1]):
                j += 1
                fh[i].seek(readers[i].FRAME_SIZE, 1)
            
            nFramesFile[i] -= j
            
        elif readers[i] is guppi:
            pass
            
        elif readers[i] is drx:
            pass
            
        # Align the files as close as possible by the time tags
        if readers[i] is vdif:
            junkFrame = readers[i].read_frame(fh[i], central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
        else:
            junkFrame = readers[i].read_frame(fh[i])
        fh[i].seek(-readers[i].FRAME_SIZE, 1)
            
        j = 0
        while sum(junkFrame.time, 0.0) + grossOffsets[i] < max(tStart):
            if readers[i] is vdif:
                for k in xrange(beampols[i]):
                    try:
                        junkFrame = readers[i].read_frame(fh[i], central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
                    except errors.SyncError:
                        print("Error - VDIF @ %i" % (i,))
                        fh[i].seek(vdif.FRAME_SIZE, 1)
                        continue
            else:
                for k in xrange(beampols[i]):
                    junkFrame = readers[i].read_frame(fh[i])
            j += beampols[i]
            
        jTime = j*readers[i].DATA_LENGTH/srate[i]/beampols[i]
        print("Shifted beam %i data by %i frames (%.4f s)" % (beams[i], j, jTime))
        
    # Set integration time
    tRead = 1.0
    nFrames = int(round(tRead*srate[-1]/readers[-1].DATA_LENGTH))
    tRead = nFrames*readers[-1].DATA_LENGTH/srate[-1]
    
    nFramesV = tRead*srate[0]/readers[0].DATA_LENGTH
    nFramesD = nFrames
    while nFramesV != int(nFramesV):
        nFrames += 1
        tRead = nFrames*readers[-1].DATA_LENGTH/srate[-1]
        
        nFramesV = tRead*srate[0]/readers[0].DATA_LENGTH
        nFramesD = nFrames
    nFramesV = int(nFramesV)
    
    # Read in some data
    tFileV = nFramesFile[ 0] / beampols[ 0] * readers[ 0].DATA_LENGTH / srate[ 0]
    tFileD = nFramesFile[-1] / beampols[-1] * readers[-1].DATA_LENGTH / srate[-1]
    tFile = min([tFileV, tFileD])
    if args.duration > 0.0:
        duration = args.duration
        duration = tRead * int(round(duration / tRead))
        tFile = duration
        
    # Date
    beginMJDs = []
    beginDates = []
    for i in xrange(len(filenames)):
        if readers[i] is vdif:
            junkFrame = readers[i].read_frame(fh[i], central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
        else:
            junkFrame = readers[i].read_frame(fh[i])
        fh[i].seek(-readers[i].FRAME_SIZE, 1)
        
        beginMJDs.append( astro.unix_to_utcjd(sum(junkFrame.time, 0.0)) - astro.MJD_OFFSET)
        beginDates.append( datetime.utcfromtimestamp(sum(junkFrame.time, 0.0)) )
        
    # Set the output base filename
    if args.tag is None:
        outbase = os.path.basename(filenames[0])
        outbase = os.path.splitext(outbase)[0][:8]
    else:
        outbase = args.tag
        
    # Report
    for i in xrange(len(filenames)):
        print("Filename: %s" % os.path.basename(filenames[i]))
        print("  Type/Reader: %s" % readers[i].__name__)
        print("  Delay Steps Avaliable: %s" % ('No' if delaySteps[i] is None else 'Yes',))
        print("  Date of First Frame: %s" % beginDates[i])
        print("  Sample Rate: %i Hz" % srate[i])
        print("  Tuning 1: %.3f Hz" % cFreqs[i][0])
        print("  Tuning 2: %.3f Hz" % cFreqs[i][1])
        print("  Bit Depth: %i" % bitDepths[i])
    print("  ===")
    print("  Phase Center:")
    print("    Name: %s" % refSrc.name)
    print("    RA: %s" % refSrc._ra)
    print("    Dec: %s" % refSrc._dec)
    print("  ===")
    print("  Data Read Time: %.3f s" % tRead)
    print("  Data Reads in File: %i" % int(tFile/tRead))
    print(" ")
    
    nVDIFInputs = sum([1 for reader in readers if reader is vdif]) + sum([1 for reader in readers if reader is guppi])
    nDRXInputs = sum([1 for reader in readers if reader is drx])
    print("Processing %i VDIF and %i DRX input streams" % (nVDIFInputs, nDRXInputs))
    print(" ")
    
    nFramesV = int(round(tRead*srate[0]/readers[0].DATA_LENGTH))
    framesPerSecondV = int(srate[0] / readers[0].DATA_LENGTH)
    nFramesB = nFrames
    framesPerSecondB = srate[-1] / readers[-1].DATA_LENGTH
    if nVDIFInputs:
        print("VDIF Frames/s: %.6f" % framesPerSecondV)
        print("VDIF Frames/Integration: %i" % nFramesV)
    if nDRXInputs:
        print("DRX Frames/s: %.6f" % framesPerSecondB)
        print("DRX Frames/Integration: %i" % nFramesB)
    if nVDIFInputs*nDRXInputs:
        print("Sample Count Ratio: %.6f" % (1.0*(nFramesV*readers[0].DATA_LENGTH)/(nFramesB*4096),))
        print("Sample Rate Ratio: %.6f" % (srate[0]/srate[-1],))
    print(" ")
    
    vdifLFFT = LFFT * (2 if nVDIFInputs else 1)	# Fix to deal with LWA-only correlations
    drxLFFT = vdifLFFT * srate[-1] / srate[0]
    while drxLFFT != int(drxLFFT):
        vdifLFFT += 1
        drxLFFT = vdifLFFT * srate[-1] / srate[0]
    vdifLFFT = vdifLFFT // (2 if nVDIFInputs else 1)	# Fix to deal with LWA-only correlations
    drxLFFT = int(drxLFFT)
    if nVDIFInputs:
        print("VDIF Transform Size: %i" % vdifLFFT)
    if nDRXInputs:
        print("DRX Transform Size: %i" % drxLFFT)
    print(" ")
    
    vdifPivot = 1
    if abs(cFreqs[0][0] - cFreqs[-1][1]) < abs(cFreqs[0][0] - cFreqs[-1][0]):
        vdifPivot = 2
    if nVDIFInputs == 0 and args.which is not None:
        vdifPivot = args.which
    if nVDIFInputs*nDRXInputs:
        print("VDIF appears to correspond to tuning #%i in DRX" % vdifPivot)
    elif nDRXInputs:
        print("Correlating DRX tuning #%i" % vdifPivot)
    print(" ")
    
    nChunks = int(tFile/tRead)
    tSub = args.subint_time
    tSub = tRead / int(round(tRead/tSub))
    tDump = args.dump_time
    tDump = tSub * int(round(tDump/tSub))
    nDump = int(tDump / tSub)
    tDump = nDump * tSub
    nInt = int((nChunks*tRead) / tDump)
    print("Sub-integration time is: %.3f s" % tSub)
    print("Integration (dump) time is: %.3f s" % tDump)
    print(" ")
    
    # Solve for the pulsar binning
    observer.date = beginMJDs[0] + astro.MJD_OFFSET - astro.DJD_OFFSET
    refSrc.compute(observer)
    pulsarPeriod = refSrc.period
    nProfileBins = args.profile_bins
    if nProfileBins <= 0:
        nProfileBins = int(pulsarPeriod / tSub)
        nProfileBins = min([nProfileBins, 64])
    profileBins = numpy.linspace(0, 1+1.0/nProfileBins, nProfileBins+2)
    profileBins -= (profileBins[1]-profileBins[0])/2.0
    print("Pulsar frequency: %.6f Hz" % refSrc.frequency)
    print("Pulsar period: %.6s seconds" % pulsarPeriod)
    print("Number of profile bins:  %i" % nProfileBins)
    print("Phase coverage per bin: %.3f" % (profileBins[1]-profileBins[0],))
    if pulsarPeriod >= tDump:
        print("WARNING:  Pulsar period is longer than the integration time!")
    print(" ")
    
    pulsarDM, pulsarDoppler = refSrc.dm, refSrc.doppler
    oFreq = numpy.fft.fftfreq(drxLFFT, d=1.0/srate[-1]) + cFreqs[-1][vdifPivot-1]
    oFreq = numpy.fft.fftshift(oFreq)*pulsarDoppler
    tDelay = dispDelay(oFreq, pulsarDM)
    tDiff = numpy.diff(tDelay)
    while abs(tDiff.max()) > tSub:
        vdifLFFT *= 2
        drxLFFT *= 2
        oFreq = numpy.fft.fftfreq(drxLFFT, d=1.0/srate[-1]) + cFreqs[-1][vdifPivot-1]
        oFreq = numpy.fft.fftshift(oFreq)*pulsarDoppler
        tDelay = dispDelay(oFreq, pulsarDM)
        tDiff = numpy.diff(tDelay)
        
    subIntTimes = [[] for i in xrange(nProfileBins)]
    subIntCount = [0 for i in xrange(nProfileBins)]
    subIntWeight = [0 for i in xrange(nProfileBins)]
    fileCount   = [0 for i in xrange(nProfileBins)]
    visXX = [0 for i in xrange(nProfileBins)]
    visXY = [0 for i in xrange(nProfileBins)]
    visYX = [0 for i in xrange(nProfileBins)]
    visYY = [0 for i in xrange(nProfileBins)]
    wallStart = time.time()
    done = False
    oldStartRel = [0 for i in xrange(nVDIFInputs+nDRXInputs)]
    delayStepApplied = [False for step in delaySteps]
    currentDM, currentDoppler = -1.0, -1.0
    for i in xrange(nChunks):
        wallTime = time.time()
        
        tStart = []
        tStartB = []
        
        vdifRef = [0 for j in xrange(nVDIFInputs*2)]
        drxRef  = [0 for j in xrange(nDRXInputs*2) ]
        
        # Read in the data
        try:
            dataV *= 0.0
            dataD *= 0.0
        except NameError:
            dataV = numpy.zeros((len(vdifRef), readers[ 0].DATA_LENGTH*nFramesV), dtype=numpy.float32)
            dataD = numpy.zeros((len(drxRef),  readers[-1].DATA_LENGTH*nFramesD), dtype=numpy.complex64)
        for j,f in enumerate(fh):
            if readers[j] is vdif:
                ## VDIF
                k = 0
                while k < beampols[j]*nFramesV:
                    try:
                        cFrame = readers[j].read_frame(f, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
                        buffers[j].append( cFrame )
                    except errors.SyncError:
                        print("Error - VDIF @ %i, %i" % (i, j))
                        f.seek(vdif.FRAME_SIZE, 1)
                        continue
                    except errors.EOFError:
                        done = True
                        break
                        
                    frames = buffers[j].get()
                    if frames is None:
                        continue
                        
                    for cFrame in frames:
                        std,pol = cFrame.id
                        sid = 2*j + pol
                        
                        if k == 0:
                            tStart.append( sum(cFrame.time, 0.0) )
                            tStart[-1] += grossOffsets[j]
                            tStartB.append( get_better_time(cFrame) )
                            tStartB[-1][0] += grossOffsets[j]
                            
                            for p in (0,1):
                                psid = 2*j + p
                                vdifRef[psid] = cFrame.header.seconds_from_epoch*framesPerSecondV + cFrame.header.frame_in_second
                                
                        count = cFrame.header.seconds_from_epoch*framesPerSecondV + cFrame.header.frame_in_second
                        count -= vdifRef[sid]
                        dataV[sid, count*readers[j].DATA_LENGTH:(count+1)*readers[j].DATA_LENGTH] = cFrame.payload.data
                        k += 1
                        
            elif readers[j] is guppi:
                ## GUPPI
                k = 0
                while k < beampols[j]*nFramesV:
                    try:
                        cFrame = readers[j].read_frame(f)
                        buffers[j].append( cFrame )
                    except errors.SyncError:
                        print("Error - GUPPI @ %i, %i" % (i, j))
                        continue
                    except errors.EOFError:
                        done = True
                        break
                        
                    frames = buffers[j].get()
                    if frames is None:
                        continue
                        
                    for cFrame in frames:
                        std,pol = cFrame.id
                        sid = 2*j + pol
                        
                        if k == 0:
                            tStart.append( sum(cFrame.time, 0.0) )
                            tStart[-1] += grossOffsets[j]
                            tStartB.append( get_better_time(cFrame) )
                            tStartB[-1][0] += grossOffsets[j]
                            
                            for p in (0,1):
                                psid = 2*j + p
                                vdifRef[psid] = cFrame.header.offset // readers[j].DATA_LENGTH
                                
                        count = cFrame.header.offset // readers[j].DATA_LENGTH
                        count -= vdifRef[sid]
                        dataV[sid, count*readers[j].DATA_LENGTH:(count+1)*readers[j].DATA_LENGTH] = cFrame.payload.data
                        k += 1
                        
            elif readers[j] is drx:
                ## DRX
                k = 0
                while k < beampols[j]*nFramesD:
                    try:
                        cFrame = readers[j].read_frame(f)
                        buffers[j].append( cFrame )
                    except errors.SyncError:
                        print("Error - DRX @ %i, %i" % (i, j))
                        continue
                    except errors.EOFError:
                        done = True
                        break
                        
                    frames = buffers[j].get()
                    if frames is None:
                        continue
                        
                    for cFrame in frames:
                        beam,tune,pol = cFrame.id
                        if tune != vdifPivot:
                            continue
                        bid = 2*(j-nVDIFInputs) + pol
                        
                        if k == 0:
                            tStart.append( sum(cFrame.time, 0.0) )
                            tStart[-1] += grossOffsets[j]
                            tStartB.append( get_better_time(cFrame) )
                            tStartB[-1][0] += grossOffsets[j]
                            
                            for p in (0,1):
                                pbid = 2*(j-nVDIFInputs) + p
                                drxRef[pbid] = cFrame.payload.timetag
                                
                        count = cFrame.payload.timetag
                        count -= drxRef[bid]
                        count //= (4096*int(196e6/srate[-1]))
                        ### Fix from some LWA-SV files that seem to cause the current LSL
                        ### ring buffer problems
                        if count < 0:
                            continue
                        try:
                            dataD[bid, count*readers[j].DATA_LENGTH:(count+1)*readers[j].DATA_LENGTH] = cFrame.payload.data
                            k += beampols[j]//2
                        except ValueError:
                            k = beampols[j]*nFramesD
                            break
        if done:
            break
            
        print('RR - Read finished in %.3f s for %.3fs of data' % (time.time()-wallTime, tRead))
        
        # Figure out which DRX tuning corresponds to the VDIF data
        if nDRXInputs > 0:
            dataD /= 7.0
            
        # Time tag alignment (sample based)
        ## Initial time tags for each stream and the relative start time for each stream
        if args.verbose:
            ### TT = time tag
            print('TT - Start', tStartB)
        tStartMin = min([sec for sec,frac in tStartB])
        tStartRel = [(sec-tStartMin)+frac for sec,frac in tStartB]
        
        ## Sample offsets between the streams
        offsets = []
        for j in xrange(nVDIFInputs+nDRXInputs):
            offsets.append( int( round(nsround(max(tStartRel) - tStartRel[j])*srate[j]) ) )
        if args.verbose:
            print('TT - Offsets', offsets)
            
        ## Roll the data to apply the sample offsets and then trim the ends to get rid 
        ## of the rolled part
        for j,offset in enumerate(offsets):
            if j < nVDIFInputs:
                if offset != 0:
                    idx0 = 2*j + 0
                    idx1 = 2*j + 1
                    tStart[j] += offset/(srate[j])
                    tStartB[j][1] += offset/(srate[j])
                    dataV[idx0,:] = numpy.roll(dataV[idx0,:], -offset)
                    dataV[idx1,:] = numpy.roll(dataV[idx1,:], -offset)
                    
            else:
                if offset != 0:
                    idx0 = 2*(j - nVDIFInputs) + 0
                    idx1 = 2*(j - nVDIFInputs) + 1
                    tStart[j] += offset/(srate[j])
                    tStartB[j][1] += offset/(srate[j])
                    dataD[idx0,:] = numpy.roll(dataD[idx0,:], -offset)
                    dataD[idx1,:] = numpy.roll(dataD[idx1,:], -offset)
                    
        vdifOffsets = offsets[:nVDIFInputs]
        drxOffsets = offsets[nVDIFInputs:]
        
        ## Apply the corrections to the original time tags and report on the sub-sample
        ## residuals
        if args.verbose:
            print('TT - Adjusted', tStartB)
        tStartMinSec  = min([sec  for sec,frac in tStartB])
        tStartMinFrac = min([frac for sec,frac in tStartB])
        tStartRel = [(sec-tStartMinSec)+(frac-tStartMinFrac) for sec,frac in tStartB]
        if args.verbose:
            print('TT - Residual', ["%.1f ns" % (r*1e9,) for r in tStartRel])
        for k in xrange(len(tStartRel)):
            antennas[2*k+0].cable.clock_offset -= tStartRel[k] - oldStartRel[k]
            antennas[2*k+1].cable.clock_offset -= tStartRel[k] - oldStartRel[k]
        oldStartRel = tStartRel
        
        # Setup everything we need to loop through the sub-integrations
        nSub = int(tRead/tSub)
        nSampV = int(srate[ 0]*tSub)
        nSampD = int(srate[-1]*tSub)
        
        #tV = i*tRead + numpy.arange(dataV.shape[1]-max(vdifOffsets), dtype=numpy.float64)/srate[ 0]
        if nDRXInputs > 0:
            tD = i*tRead + numpy.arange(dataD.shape[1]-max(drxOffsets), dtype=numpy.float64)/srate[-1]
            
        # Loop over sub-integrations
        for j in xrange(nSub):
            ## Select the data to work with
            tSubInt = tStart[0] + (j+1)*nSampV/srate[0] - nSampV//2/srate[0]
            tSubIntB = (tStartB[0][0], tStartB[0][1] + (j+1)*nSampV/srate[0] - nSampV//2/srate[0])
            #tVSub    = tV[j*nSampV:(j+1)*nSampV]
            if nDRXInputs > 0:
                tDSub    = tD[j*nSampD:(j+1)*nSampD]
            dataVSub = dataV[:,j*nSampV:(j+1)*nSampV]
            #if dataVSub.shape[1] != tVSub.size:
            #	dataVSub = dataVSub[:,:tVSub.size]
            #if tVSub.size == 0:
            #	continue
            dataDSub = dataD[:,j*nSampD:(j+1)*nSampD]
            if nDRXInputs > 0:
                if dataDSub.shape[1] != tDSub.size:
                    dataDSub = dataDSub[:,:tDSub.size]
                if tDSub.size == 0:
                    continue
                    
            ## Update antennas for any delay steps
            for k in xrange(len(delaySteps)):
                if delaySteps[k] is None:
                    ## Skip over null entries
                    continue
                elif delaySteps[k][0][0] > tStart[0]:
                    ## Skip over antennas where the next step is in the future
                    continue
                    
                ## Find the next step
                nextStep = numpy.where( (tStart[0] - delaySteps[k][0]) >= 0.0 )[0][0]
                step = delaySteps[k][1][nextStep]
                if step != 0.0:
                    ## Report on the delay step
                    print("DS - Applying delay step of %.3f ns to antenna %i" % (step*1e9, antennas[2*k+0].stand.id))
                    print("DS - Step corresponds to %.1f deg at band center" % (360*cFreqs[0][0]*step,))
                    ## Apply the step
                    antennas[2*k+0].cable.clock_offset += step
                    antennas[2*k+1].cable.clock_offset += step
                    ## Update the delay step flag
                    delayStepApplied[k] = True
                ## Clenup so we don't re-apply the step at the next iteration
                if nextStep+1 < delaySteps[k][0].size:
                    ### There are still more we can apply
                    delaySteps[k] = (delaySteps[k][0][nextStep+1:], delaySteps[k][1][nextStep+1:])
                else:
                    ### There is not left to apply
                    delaySteps[k] = None
                    
            ## Update the observation
            observer.date = astro.unix_to_utcjd(tSubInt) - astro.DJD_OFFSET
            refSrc.compute(observer)
            
            ## Correct for the LWA dipole power pattern
            if nDRXInputs > 0:
                dipoleX, dipoleY = jones.get_lwa_antenna_gain(observer, refSrc)
                dataDSub[0::2,:] /= numpy.sqrt(dipoleX)
                dataDSub[1::2,:] /= numpy.sqrt(dipoleY)
                
            ## Get the Jones matrices and apply
            ## NOTE: This moves the LWA into the frame of the VLA
            if nVDIFInputs*nDRXInputs > 0:
                lwaToSky = jones.get_matrix_lwa(observer, refSrc)
                skyToVLA = jones.get_matrix_vla(observer, refSrc, inverse=True)
                dataDSub = jones.apply_matrix(dataDSub, numpy.matrix(skyToVLA)*numpy.matrix(lwaToSky))
                
            ## Correlate
            delayPadding = multirate.get_optimal_delay_padding(antennas[:2*nVDIFInputs], antennas[2*nVDIFInputs:],
                                                               LFFT=drxLFFT, sample_rate=srate[-1], 
                                                               central_freq=cFreqs[-1][vdifPivot-1], 
                                                               Pol='*', phase_center=refSrc)
            if nVDIFInputs > 0:
                freqV, feoV, veoV, deoV = multirate.fengine(dataVSub, antennas[:2*nVDIFInputs], LFFT=vdifLFFT,
                                                            sample_rate=srate[0], central_freq=cFreqs[0][0]-srate[0]/4,
                                                            Pol='*', phase_center=refSrc, 
                                                            delayPadding=delayPadding)
                
            if nDRXInputs > 0:
                freqD, feoD, veoD, deoD = multirate.fengine(dataDSub, antennas[2*nVDIFInputs:], LFFT=drxLFFT,
                                                            sample_rate=srate[-1], central_freq=cFreqs[-1][vdifPivot-1], 
                                                            Pol='*', phase_center=refSrc, 
                                                            delayPadding=delayPadding)
                
            ## Rotate the phase in time to deal with frequency offset between the VLA and LWA
            if nDRXInputs*nVDIFInputs > 0:
                subChanFreqOffset = (cFreqs[0][0]-cFreqs[-1][vdifPivot-1]) % (freqD[1]-freqD[0])
                
                if i == 0 and j == 0:
                    ## FC = frequency correction
                    tv,tu = bestFreqUnits(subChanFreqOffset)
                    print("FC - Applying fringe rotation rate of %.3f %s to the DRX data" % (tv,tu))
                    
                for w in xrange(feoD.shape[2]):
                    feoD[:,:,w] *= numpy.exp(-2j*numpy.pi*subChanFreqOffset*tDSub[w*drxLFFT])
                    
            ## Sort out what goes where (channels and antennas) if we don't already know
            try:
                if nVDIFInputs > 0:
                    freqV = freqV[goodV]
                    feoV = numpy.roll(feoV, -goodV[0], axis=1)[:,:len(goodV),:]
                if nDRXInputs > 0:
                    freqD = freqD[goodD]
                    feoD = numpy.roll(feoD, -goodD[0], axis=1)[:,:len(goodD),:]
                    
            except NameError:
                ### Frequency overlap
                fMin, fMax = -1e12, 1e12
                if nVDIFInputs > 0:
                    fMin, fMax = max([fMin, freqV.min()]), min([fMax, freqV.max()])
                if nDRXInputs > 0:
                    fMin, fMax = max([fMin, freqD.min()]), min([fMax, freqD.max()])
                    
                ### Channels and antennas (X vs. Y)
                if nVDIFInputs > 0:
                    goodV = numpy.where( (freqV >= fMin) & (freqV <= fMax) )[0]
                    aXV = [k for (k,a) in enumerate(antennas[:2*nVDIFInputs]) if a.pol == 0]
                    aYV = [k for (k,a) in enumerate(antennas[:2*nVDIFInputs]) if a.pol == 1]
                if nDRXInputs > 0:
                    goodD = numpy.where( (freqD >= fMin) & (freqD <= fMax) )[0]
                    aXD = [k for (k,a) in enumerate(antennas[2*nVDIFInputs:]) if a.pol == 0]
                    aYD = [k for (k,a) in enumerate(antennas[2*nVDIFInputs:]) if a.pol == 1]
                    
                ### Validate the channel alignent and fix it if needed
                if nVDIFInputs*nDRXInputs != 0:
                    pd = freqV[goodV[0]] - freqD[goodD[0]]
                    # Need to shift?
                    if abs(pd) >= 1.01*abs(subChanFreqOffset):
                        ## Need to shift
                        if pd < 0.0:
                            goodV = goodV[1:]
                        else:
                            goodD = goodD[1:]
                            
                    # Need to trim?
                    if len(goodV) > len(goodD):
                        ## Yes, goodV is too long
                        goodV = goodV[:len(goodD)]
                    elif len(goodD) > len(goodV):
                        ## Yes, goodD is too long
                        goodD = goodD[:len(goodV)]
                    else:
                        ## No, nothing needs to be done
                        pass
                        
                    # Validate
                    fd = freqV[goodV] - freqD[goodD]
                    try:
                        assert(fd.min() >= 0.99*subChanFreqOffset)
                        assert(fd.max() <= 1.01*subChanFreqOffset)
                        
                        ## FS = frequency selection
                        tv,tu = bestFreqUnits(freqV[1]-freqV[0])
                        print("FS - Found %i, %.3f %s overalapping channels" % (len(goodV), tv, tu))
                        tv,tu = bestFreqUnits(freqV[goodV[-1]]-freqV[goodV[0]])
                        print("FS - Bandwidth is %.3f %s" % (tv, tu))
                        print("FS - Channels span %.3f MHz to %.3f MHz" % (freqV[goodV[0]]/1e6, freqV[goodV[-1]]/1e6))
                            
                    except AssertionError:
                        raise RuntimeError("Cannot find a common frequency set between the input data: offsets range between %.3f Hz and %.3f Hz, expected %.3f Hz" % (fd.min(), fd.max(), subChanFreqOffset))
                        
                ### Apply
                if nVDIFInputs > 0:
                    freqV = freqV[goodV]
                    feoV = numpy.roll(feoV, -goodV[0], axis=1)[:,:len(goodV),:]
                if nDRXInputs > 0:
                    freqD = freqD[goodD]
                    feoD = numpy.roll(feoD, -goodD[0], axis=1)[:,:len(goodD),:]
            try:
                nchan = freqV.size
                fdt = feoV.dtype
                vdt = veoV.dtype
            except NameError:
                nchan = freqD.size
                fdt = feoD.dtype
                vdt = veoD.dtype
            ## Setup the intermediate F-engine products and trim the data
            ### Figure out the minimum number of windows
            nWin = 1e12
            if nVDIFInputs > 0:
                nWin = min([nWin, feoV.shape[2]])
            if nDRXInputs > 0:
                nWin = min([nWin, feoD.shape[2]])
                
            ### Initialize the intermediate arrays
            try:
                assert(feoX.shape[2] == nWin)
            except (NameError, AssertionError):
                feoX = numpy.zeros((nVDIFInputs+nDRXInputs, nchan, nWin), dtype=fdt)
                feoY = numpy.zeros((nVDIFInputs+nDRXInputs, nchan, nWin), dtype=fdt)
                veoX = numpy.zeros((nVDIFInputs+nDRXInputs, nWin), dtype=vdt)
                veoY = numpy.zeros((nVDIFInputs+nDRXInputs, nWin), dtype=vdt)
                
            ### Trim
            if nVDIFInputs > 0:
                feoV = feoV[:,:,:nWin]
                veoV = veoV[:,:nWin]
            if nDRXInputs > 0:
                feoD = feoD[:,:,:nWin]
                veoD = veoD[:,:nWin]
                
            ## Sort it all out by polarization
            for k in xrange(nVDIFInputs):
                feoX[k,:,:] = feoV[aXV[k],:,:]
                feoY[k,:,:] = feoV[aYV[k],:,:]
                veoX[k,:] = veoV[aXV[k],:]
                veoY[k,:] = veoV[aYV[k],:]
            for k in xrange(nDRXInputs):
                feoX[k+nVDIFInputs,:,:] = feoD[aXD[k],:,:]
                feoY[k+nVDIFInputs,:,:] = feoD[aYD[k],:,:]
                veoX[k+nVDIFInputs,:] = veoD[aXD[k],:]
                veoY[k+nVDIFInputs,:] = veoD[aYD[k],:]
                
            ## Cross multiply
            try:
                sfreqXX = freqV
                sfreqYY = freqV
            except NameError:
                sfreqXX = freqD
                sfreqYY = freqD
            svisXX = multirate.xengine(feoX, veoX, feoX, veoX)
            svisXY = multirate.xengine(feoX, veoX, feoY, veoY)
            svisYX = multirate.xengine(feoY, veoY, feoX, veoX)
            svisYY = multirate.xengine(feoY, veoY, feoY, veoY)
            
            # Get a most precise representation of the current time
            mjdi = tSubIntB[0] / 86400 + 40587		
            mjdf = (tSubIntB[0] % 86400 + tSubIntB[1]) / 86400.0
            
            # Determine the pulsar phase as a function of frequency
            refSrc.compute_pulsar(mjdi, mjdf)
            currentPeriod = refSrc.period
            ## Dispersion
            if currentDM != refSrc.dm or currentDoppler != refSrc.doppler:
                currentDM = refSrc.dm*1.0
                currentDoppler = refSrc.doppler*1.0
                tDisp = dispDelay(sfreqXX*currentDoppler, currentDM)
            phaseDispersion = tDisp / currentPeriod
            phaseDispersion %= 1.0
            ## Folding
            phaseProfile = refSrc.phase
            ## Combined
            profilePhase = (phaseProfile - phaseDispersion) % 1.0
            
            ## Map the phases to bins
            bestBins = {}
            for b,phs in enumerate(profilePhase):
                bestBin = numpy.where( phs >= profileBins )[0][-1] % nProfileBins
                try:
                    bestBins[bestBin].append( b )
                except KeyError:
                    bestBins[bestBin] = [b,]
                    
            #summary = [None for i in profileBins[:-2]]
            #for bestBin in bestBins:
            #	summary[bestBin] = (len(bestBins[bestBin]), subIntCount[bestBin])
            #print(summary)
            
            ### Accumulate
            for bestBin in bestBins:
                bestFreq = bestBins[bestBin]
                if subIntCount[bestBin] == 0:
                    subIntTimes[bestBin] = []
                    freqXX = sfreqXX
                    freqYY = sfreqYY
                    try:
                        okToZero = numpy.where( subIntWeight[bestBin] ==  subIntWeight[bestBin].max() )[0]
                        subIntWeight[bestBin] *= 0
                    except AttributeError:
                        subIntWeight[bestBin] = numpy.zeros(freqXX.size)
                        
                    visXX[bestBin] = svisXX*0.0
                    visXY[bestBin] = svisXY*0.0
                    visYX[bestBin] = svisYX*0.0
                    visYY[bestBin] = svisYY*0.0
                    
                subIntTimes[bestBin].append( tSubInt )
                visXX[bestBin][:,bestFreq] += svisXX[:,bestFreq] / nDump
                visXY[bestBin][:,bestFreq] += svisXY[:,bestFreq] / nDump
                visYX[bestBin][:,bestFreq] += svisYX[:,bestFreq] / nDump
                visYY[bestBin][:,bestFreq] += svisYY[:,bestFreq] / nDump
                subIntCount[bestBin] += 1
                subIntWeight[bestBin][bestFreq] += 1
            
            ## Save
            anyFilesSaved = False
            for bestBin in bestBins:
                if subIntCount[bestBin] == nDump:
                    subIntCount[bestBin] = 0
                    fileCount[bestBin] += 1
                    
                    visXX[bestBin] *= nDump / subIntWeight[bestBin]
                    visXY[bestBin] *= nDump / subIntWeight[bestBin]
                    visYX[bestBin] *= nDump / subIntWeight[bestBin]
                    visYY[bestBin] *= nDump / subIntWeight[bestBin]
                    
                    ### Compute the effective integration time - this should be
                    ### tDump/nProfileBins but let's use the actual median number
                    ### of accumulations at tSub instead
                    try:
                        tDumpAct
                    except NameError:
                        tDumpAct = numpy.median(subIntWeight[bestBin]) * tSub
                        
                    ### CD = correlator dump
                    outfile = "%s-vis2-bin%03i-%05i.npz" % (outbase, bestBin, fileCount[bestBin])
                    numpy.savez(outfile, config=rawConfig, polycos=rawPolycos, 
                                srate=srate[0]/2.0, freq1=freqXX, 
                                vis1XX=visXX[bestBin], vis1XY=visXY[bestBin], 
                                vis1YX=visYX[bestBin], vis1YY=visYY[bestBin], 
                                tStart=numpy.mean(subIntTimes[bestBin]), tInt=tDumpAct, 
                                delayStepApplied=delayStepApplied)
                    anyFilesSaved = True
                    print("CD - writing integration %i, bin %i to disk, timestamp is %.3f s" % (fileCount[bestBin], bestBin, numpy.mean(subIntTimes[bestBin])))
                    if bestBin == 0:
                        if fileCount[0] == 1:
                            print("CD - each integration is %.1f MB on disk" % (os.path.getsize(outfile)/1024.0**2,))
                            print("CD - effective integration time is %.3f s" % tDumpAct)
                        if (fileCount[0]-1) % 25 == 0:
                            print("CD - average processing time per integration is %.3f s" % ((time.time() - wallStart)/max(fileCount),))
                            etc = (nInt - max(fileCount)) * (time.time() - wallStart)/max(fileCount)
                            eth = int(etc/60.0) / 60
                            etm = int(etc/60.0) % 60
                            ets = etc % 60
                            print("CD - estimated time to completion is %i:%02i:%04.1f" % (eth, etm, ets))
            if anyFilesSaved:
                delayStepApplied = [False for step in delaySteps]
                
    # Cleanup
    etc = time.time() - wallStart
    eth = int(etc/60.0) / 60
    etm = int(etc/60.0) % 60
    ets = etc % 60
    print("Processing finished after %i:%02i:%04.1f" % (eth, etm, ets))
    print("Average time per integration was %.3f s" % (etc/max(fileCount),))
    for f in fh:
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='the next generation of correlator for LWA/VLA/eLWA pulsar data', 
        epilog='NOTE:  The actual dump time for the correlator is a function of the requested dump time and number of profile bins used for folding.',
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
    parser.add_argument('-b', '--profile-bins', type=int, default=0, 
                        help='number of bins to use when folding')
    parser.add_argument('-d', '--duration', type=float, default=0.0, 
                        help='duration in seconds of the file to correlate; 0 = everything')
    parser.add_argument('-g', '--tag', type=str, 
                        help='tag to use for the output file')
    parser.add_argument('-j', '--jit', action='store_true', 
                        help='enable experimental just-in-time optimizations')
    parser.add_argument('-w', '--which', type=int, default=0, 
                        help='for LWA-only observations, which tuning to use for correlation; 0 = auto-select')
    args = parser.parse_args()
    main(args)
    
