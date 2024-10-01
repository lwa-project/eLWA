#!/usr/bin/env python3

"""
Given a VDIF file, plot the time averaged spectra for each beam output over some 
period.
"""

import os
import sys
import h5py
import math
import numpy as np
import argparse
from datetime import datetime

import lsl.reader.vdif as vdif
import lsl.reader.drspec as drspec
import lsl.reader.errors as errors
import lsl.statistics.robust as robust
import lsl.correlator.fx as fxc
from lsl.astro import unix_to_utcjd, DJD_OFFSET
from lsl.common import progress, stations
from lsl.common import mcs, metabundle
from lsl.misc import parser as aph

import matplotlib.pyplot as plt

from utils import *

import data as hdfData


def bestFreqUnits(freq):
    """Given a numpy array of frequencies in Hz, return a new array with the
    frequencies in the best units possible (kHz, MHz, etc.)."""
    
    # Figure out how large the data are
    scale = int(math.log10(freq.max()))
    if scale >= 9:
        divis = 1e9
        units = 'GHz'
    elif scale >= 6:
        divis = 1e6
        units = 'MHz'
    elif scale >= 3:
        divis = 1e3
        units = 'kHz'
    else:
        divis = 1
        units = 'Hz'
        
    # Convert the frequency
    newFreq = freq / divis
    
    # Return units and freq
    return (newFreq, units)


def processDataBatchLinear(fh, header, antennas, tStart, duration, sample_rate, args, dataSets, obsID=1, clip1=0, clip2=0):
    """
    Process a chunk of data in a raw vdif file into linear polarization 
    products and add the contents to an HDF5 file.
    """
    
    # Length of the FFT
    LFFT = args.fft_length
    
    # Find the start of the observation
    junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
    srate = junkFrame.sample_rate
    t0 = junkFrame.time
    fh.seek(-vdif.FRAME_SIZE, 1)
    
    print('Looking for #%i at %s with sample rate %.1f Hz...' % (obsID, tStart, sample_rate))
    while t0.datetime < tStart or srate != sample_rate:
        junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
        srate = junkFrame.sample_rate
        t0 = junkFrame.time
    print('... Found #%i at %s with sample rate %.1f Hz' % (obsID, junkFrame.time.datetime, srate))
    tDiff = t0.datetime - tStart
    try:
        duration = duration - tDiff.total_seconds()
    except:
        duration = duration - (tDiff.seconds + tDiff.microseconds/1e6)
    
    beam,pol = junkFrame.id
    beams = vdif.get_thread_count(fh)
    tunepols = vdif.get_thread_count(fh)
    tunepol = tunepols
    beampols = tunepol
    
    # Make sure that the file chunk size contains is an integer multiple
    # of the FFT length so that no data gets dropped.  This needs to
    # take into account the number of beampols in the data, the FFT length,
    # and the number of samples per frame.
    maxFrames = int(1.0*28000/beampols*vdif.DATA_LENGTH/float(2*LFFT))*2*LFFT//vdif.DATA_LENGTH*beampols
    
    # Number of frames per second 
    nFramesSecond = int(srate) // vdif.DATA_LENGTH
    
    # Number of frames to integrate over
    nFramesAvg = int(round(args.average * srate / vdif.DATA_LENGTH * beampols))
    nFramesAvg = int(1.0 * nFramesAvg / beampols*vdif.DATA_LENGTH/float(2*LFFT))*2*LFFT//vdif.DATA_LENGTH*beampols
    args.average = 1.0 * nFramesAvg / beampols * vdif.DATA_LENGTH / srate
    maxFrames = nFramesAvg
    
    # Number of remaining chunks (and the correction to the number of
    # frames to read in).
    nChunks = int(round(duration / args.average))
    if nChunks == 0:
        nChunks = 1
    nFrames = nFramesAvg*nChunks
    
    # Line up the time tags for the various tunings/polarizations
    timetags = []
    for i in range(16):
        junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
        timetags.append(junkFrame.header.seconds_from_epoch*nFramesSecond + junkFrame.header.frame_in_second)
    fh.seek(-16*vdif.FRAME_SIZE, 1)
    
    i = 0
    if beampols == 4:
        while (timetags[i+0] != timetags[i+1]) or (timetags[i+0] != timetags[i+2]) or (timetags[i+0] != timetags[i+3]):
            i += 1
            fh.seek(vdif.FRAME_SIZE, 1)
            
    elif beampols == 2:
        while timetags[i+0] != timetags[i+1]:
            i += 1
            fh.seek(vdif.FRAME_SIZE, 1)
            
    # Date & Central Frequency
    beginDate = junkFrame.time.datetime
    central_freq1 = 0.0
    central_freq2 = 0.0
    for i in range(4):
        junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
        b,p = junkFrame.id
        if p == 0:
            central_freq1 = junkFrame.central_freq
        elif p == 0:
            central_freq2 = junkFrame.central_freq
        else:
            pass
    fh.seek(-4*vdif.FRAME_SIZE, 1)
    freq = np.fft.fftshift(np.fft.fftfreq(LFFT, d=2/srate))
    
    dataSets['obs%i-freq1' % obsID][:] = freq + central_freq1
    dataSets['obs%i-freq2' % obsID][:] = freq + central_freq2
    
    obs = dataSets['obs%i' % obsID]
    obs.attrs['tInt'] = args.average
    obs.attrs['tInt_Unit'] = 's'
    obs.attrs['LFFT'] = LFFT
    obs.attrs['nchan'] = LFFT
    obs.attrs['RBW'] = freq[1]-freq[0]
    obs.attrs['RBW_Units'] = 'Hz'
    
    # Create the progress bar so that we can keep up with the conversion.
    pbar = progress.ProgressBarPlus(max=nChunks)
    
    data_products = ['XX', 'YY']
    done = False
    for i in range(nChunks):
        # Find out how many frames remain in the file.  If this number is larger
        # than the maximum of frames we can work with at a time (maxFrames),
        # only deal with that chunk
        framesRemaining = nFrames - i*maxFrames
        if framesRemaining > maxFrames:
            framesWork = maxFrames
        else:
            framesWork = framesRemaining
            
        count = {0:0, 1:0, 2:0, 3:0}
        data = np.zeros((4,framesWork*vdif.DATA_LENGTH//beampols), dtype=np.csingle)
        # If there are fewer frames than we need to fill an FFT, skip this chunk
        if data.shape[1] < LFFT:
            break
            
        # Inner loop that actually reads the frames into the data array
        for j in range(framesWork):
            # Read in the next frame and anticipate any problems that could occur
            try:
                cFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0, verbose=False)
            except errors.EOFError:
                done = True
                break
            except errors.SyncError:
                continue

            beam,pol = cFrame.id
            aStand = pol
            if j is 0:
                cTime = cFrame.time
                
            try:
                data[aStand, count[aStand]*vdif.DATA_LENGTH:(count[aStand]+1)*vdif.DATA_LENGTH] = cFrame.payload.data
                count[aStand] +=  1
            except ValueError:
                raise RuntimeError("Invalid Shape")
                
        # Save out some easy stuff
        dataSets['obs%i-time' % obsID][i] = float(cTime)        # pylint: disable=possibly-used-before-assignment,used-before-assignment
        
        if not args.without_sats:
            sats = ((data.real**2 + data.imag**2) >= 49).sum(axis=1)
            dataSets['obs%i-Saturation1' % obsID][i,:] = sats[0:2]
            dataSets['obs%i-Saturation2' % obsID][i,:] = sats[2:4]
        else:
            dataSets['obs%i-Saturation1' % obsID][i,:] = -1
            dataSets['obs%i-Saturation2' % obsID][i,:] = -1
        
        # Calculate the spectra for this block of data and then weight the results by 
        # the total number of frames read.  This is needed to keep the averages correct.
        if clip1 == clip2:
            freq, tempSpec1 = fxc.SpecMaster(data, LFFT=2*LFFT, window=args.window, verbose=args.verbose, sample_rate=srate, clip_level=clip1)
            freq, tempSpec1 = freq[LFFT:], tempSpec1[:,LFFT:]
            
            l = 0
            for t in (1,2):
                for p in data_products:
                    dataSets['obs%i-%s%i' % (obsID, p, t)][i,:] = tempSpec1[l,:]
                    l += 1
                    
        else:
            freq, tempSpec1 = fxc.SpecMaster(data[:2,:], LFFT=2*LFFT, window=args.window, verbose=args.verbose, sample_rate=srate, clip_level=clip1)
            freq, tempSpec2 = fxc.SpecMaster(data[2:,:], LFFT=2*LFFT, window=args.window, verbose=args.verbose, sample_rate=srate, clip_level=clip2)
            freq, tempSpec1, tempSpec2 = freq[LFFT:], tempSpec1[:,LFFT:], tempSpec2[:,LFFT:]
            
            for l,p in enumerate(data_products):
                dataSets['obs%i-%s%i' % (obsID, p, 1)][i,:] = tempSpec1[l,:]
                dataSets['obs%i-%s%i' % (obsID, p, 2)][i,:] = tempSpec2[l,:]
                
        # We don't really need the data array anymore, so delete it
        del(data)
        
        # Are we done yet?
        if done:
            break
            
        ## Update the progress bar and remaining time estimate
        pbar.inc()
        sys.stdout.write('%s\r' % pbar.show())
        sys.stdout.flush()
        
    pbar.amount = pbar.max
    sys.stdout.write('%s\n' % pbar.show())
    sys.stdout.flush()
    
    return True


def processDataBatchStokes(fh, header, antennas, tStart, duration, sample_rate, args, dataSets, obsID=1, clip1=0, clip2=0):
    """
    Process a chunk of data in a raw vdif file into Stokes parameters and 
    add the contents to an HDF5 file.
    """
    
    # Length of the FFT
    LFFT = args.fft_length
    
    # Find the start of the observation
    junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
    srate = junkFrame.sample_rate
    t0 = junkFrame.time
    fh.seek(-vdif.FRAME_SIZE, 1)
    
    print('Looking for #%i at %s with sample rate %.1f Hz...' % (obsID, tStart, sample_rate))
    while t0.datetime < tStart or srate != sample_rate:
        junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
        srate = junkFrame.sample_rate
        t0 = junkFrame.time
    print('... Found #%i at %s with sample rate %.1f Hz' % (obsID, datetime.utcfromtimestamp(t0), srate))
    tDiff = t0.datetime - tStart
    try:
        duration = duration - tDiff.total_seconds()
    except:
        duration = duration - (tDiff.seconds + tDiff.microseconds/1e6)
    
    beam,pol = junkFrame.id
    beams = vdif.get_thread_count(fh)
    tunepols = vdif.get_thread_count(fh)
    tunepol = tunepols
    beampols = tunepol
    
    # Make sure that the file chunk size contains is an integer multiple
    # of the FFT length so that no data gets dropped.  This needs to
    # take into account the number of beampols in the data, the FFT length,
    # and the number of samples per frame.
    maxFrames = int(1.0*28000/beampols*vdif.DATA_LENGTH/float(2*LFFT))*2*LFFT//vdif.DATA_LENGTH*beampols
    
    # Number of frames per second 
    nFramesSecond = int(srate) // vdif.DATA_LENGTH
    
    # Number of frames to integrate over
    nFramesAvg = int(round(args.average * srate / vdif.DATA_LENGTH * beampols))
    nFramesAvg = int(1.0 * nFramesAvg / beampols*vdif.DATA_LENGTH/float(2*LFFT))*2*LFFT//vdif.DATA_LENGTH*beampols
    args.average = 1.0 * nFramesAvg / beampols * vdif.DATA_LENGTH / srate
    maxFrames = nFramesAvg
    
    # Number of remaining chunks (and the correction to the number of
    # frames to read in).
    nChunks = int(round(duration / args.average))
    if nChunks == 0:
        nChunks = 1
    nFrames = nFramesAvg*nChunks
    
    # Line up the time tags for the various tunings/polarizations
    timetags = []
    for i in range(16):
        junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
        timetags.append(junkFrame.header.seconds_from_epoch*nFramesSecond + junkFrame.header.frame_in_second)
    fh.seek(-16*vdif.FRAME_SIZE, 1)
    
    i = 0
    if beampols == 4:
        while (timetags[i+0] != timetags[i+1]) or (timetags[i+0] != timetags[i+2]) or (timetags[i+0] != timetags[i+3]):
            i += 1
            fh.seek(vdif.FRAME_SIZE, 1)
            
    elif beampols == 2:
        while timetags[i+0] != timetags[i+1]:
            i += 1
            fh.seek(vdif.FRAME_SIZE, 1)
            
    # Date & Central Frequency
    beginDate = junkFrame.time.datetime
    central_freq1 = 0.0
    central_freq2 = 0.0
    for i in range(4):
        junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
        b,p = junkFrame.id
        if p == 0:
            central_freq1 = junkFrame.central_freq
        elif p == 0:
            central_freq2 = junkFrame.central_freq
        else:
            pass
    fh.seek(-4*vdif.FRAME_SIZE, 1)
    freq = np.fft.fftshift(np.fft.fftfreq(LFFT, d=2/srate))
    
    dataSets['obs%i-freq1' % obsID][:] = freq + central_freq1
    dataSets['obs%i-freq2' % obsID][:] = freq + central_freq2
    
    obs = dataSets['obs%i' % obsID]
    obs.attrs['tInt'] = args.average
    obs.attrs['tInt_Unit'] = 's'
    obs.attrs['LFFT'] = LFFT
    obs.attrs['nchan'] = LFFT
    obs.attrs['RBW'] = freq[1]-freq[0]
    obs.attrs['RBW_Units'] = 'Hz'
    
    # Create the progress bar so that we can keep up with the conversion.
    pbar = progress.ProgressBarPlus(max=nChunks)
    
    data_products = ['I', 'Q', 'U', 'V']
    done = False
    for i in range(nChunks):
        # Find out how many frames remain in the file.  If this number is larger
        # than the maximum of frames we can work with at a time (maxFrames),
        # only deal with that chunk
        framesRemaining = nFrames - i*maxFrames
        if framesRemaining > maxFrames:
            framesWork = maxFrames
        else:
            framesWork = framesRemaining
            
        count = {0:0, 1:0, 2:0, 3:0}
        data = np.zeros((4,framesWork*vdif.DATA_LENGTH//beampols), dtype=np.csingle)
        # If there are fewer frames than we need to fill an FFT, skip this chunk
        if data.shape[1] < LFFT:
            break
            
        # Inner loop that actually reads the frames into the data array
        for j in range(framesWork):
            # Read in the next frame and anticipate any problems that could occur
            try:
                cFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0, verbose=False)
            except errors.EOFError:
                done = True
                break
            except errors.SyncError:
                continue
                
            beam,pol = cFrame.id
            aStand = pol
            if j is 0:
                cTime = cFrame.time
                
            try:
                data[aStand, count[aStand]*vdif.DATA_LENGTH:(count[aStand]+1)*vdif.DATA_LENGTH] = cFrame.payload.data
                count[aStand] +=  1
            except ValueError:
                raise RuntimeError("Invalid Shape")
                
        # Save out some easy stuff
        dataSets['obs%i-time' % obsID][i] = float(cTime)        # pylint: disable=possibly-used-before-assignment,used-before-assignment
        
        if not args.without_sats:
            sats = ((data.real**2 + data.imag**2) >= 49).sum(axis=1)
            dataSets['obs%i-Saturation1' % obsID][i,:] = sats[0:2]
            dataSets['obs%i-Saturation2' % obsID][i,:] = sats[2:4]
        else:
            dataSets['obs%i-Saturation1' % obsID][i,:] = -1
            dataSets['obs%i-Saturation2' % obsID][i,:] = -1
            
        # Calculate the spectra for this block of data and then weight the results by 
        # the total number of frames read.  This is needed to keep the averages correct.
        if clip1 == clip2:
            freq, tempSpec1 = fxc.StokesMaster(data, antennas, LFFT=2*LFFT, window=args.window, verbose=args.verbose, sample_rate=srate, clip_level=clip1)
            freq, tempSpec1 = freq[LFFT:], tempSpec1[:,:,LFFT:]
            
            for t in (1,2):
                for l,p in enumerate(data_products):
                    dataSets['obs%i-%s%i' % (obsID, p, t)][i,:] = tempSpec1[l,t-1,:]
                    
        else:
            freq, tempSpec1 = fxc.StokesMaster(data[:2,:], antennas[:2], LFFT=2*LFFT, window=args.window, verbose=args.verbose, sample_rate=srate, clip_level=clip1)
            freq, tempSpec2 = fxc.StokesMaster(data[2:,:], antennas[2:], LFFT=2*LFFT, window=args.window, verbose=args.verbose, sample_rate=srate, clip_level=clip2)
            freq, tempSpec1, tempSpec2 = freq[LFFT:], tempSpec1[:,:,LFFT:], tempSpec2[:,:,LFFT:]
            
            for l,p in enumerate(data_products):
                dataSets['obs%i-%s%i' % (obsID, p, 1)][i,:] = tempSpec1[l,0,:]
                dataSets['obs%i-%s%i' % (obsID, p, 2)][i,:] = tempSpec2[l,0,:]

        # We don't really need the data array anymore, so delete it
        del(data)
        
        # Are we done yet?
        if done:
            break
            
        ## Update the progress bar and remaining time estimate
        pbar.inc()
        sys.stdout.write('%s\r' % pbar.show())
        sys.stdout.flush()
        
    pbar.amount = pbar.max
    sys.stdout.write('%s\n' % pbar.show())
    sys.stdout.flush()
    
    return True


def main(args):
    # Length of the FFT
    LFFT = args.fft_length
    if args.bartlett:
        window = np.bartlett
    elif args.blackman:
        window = np.blackman
    elif args.hanning:
        window = np.hanning
    else:
        window = fxc.null_window
    args.window = window
    
    # Open the file and find good data (not spectrometer data)
    filename = args.filename
    fh = open(filename, "rb")
    header = vdif.read_guppi_header(fh)
    vdif.FRAME_SIZE = vdif.get_frame_size(fh)
    nFramesFile = os.path.getsize(filename) // vdif.FRAME_SIZE
    
    while True:
        try:
            junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
            try:
                srate = junkFrame.sample_rate
                t0 = junkFrame.time
                vdif.DATA_LENGTH = junkFrame.payload.data.size
                break
            except ZeroDivisionError:
                pass
        except errors.SyncError:
            fh.seek(-vdif.FRAME_SIZE+1, 1)
            
    fh.seek(-vdif.FRAME_SIZE, 1)
    
    beam,pol = junkFrame.id
    beams = 1
    tunepols = vdif.get_thread_count(fh)
    tunepol = tunepols
    beampols = tunepol

    # Offset in frames for beampols beam/tuning/pol. sets
    offset = int(args.skip * srate / vdif.DATA_LENGTH * beampols)
    offset = int(1.0 * offset / beampols) * beampols
    fh.seek(offset*vdif.FRAME_SIZE, 1)
    
    # Iterate on the offsets until we reach the right point in the file.  This
    # is needed to deal with files that start with only one tuning and/or a 
    # different sample rate.  
    while True:
        ## Figure out where in the file we are and what the current tuning/sample 
        ## rate is
        junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
        srate = junkFrame.sample_rate
        t1 = junkFrame.time
        tunepols = (vdif.get_thread_count(fh),)
        tunepol = tunepols[0]
        beampols = tunepol
        fh.seek(-vdif.FRAME_SIZE, 1)
        
        ## See how far off the current frame is from the target
        tDiff = t1 - (t0 + args.skip)
        
        ## Half that to come up with a new seek parameter
        tCorr = -tDiff / 2.0
        cOffset = int(tCorr * srate / vdif.DATA_LENGTH * beampols)
        cOffset = int(1.0 * cOffset / beampols) * beampols
        offset += cOffset
        
        ## If the offset is zero, we are done.  Otherwise, apply the offset
        ## and check the location in the file again/
        if cOffset is 0:
            break
        fh.seek(cOffset*vdif.FRAME_SIZE, 1)
    
    # Update the offset actually used
    args.skip = t1 - t0
    offset = int(round(args.skip * srate / vdif.DATA_LENGTH * beampols))
    offset = int(1.0 * offset / beampols) * beampols

    # Make sure that the file chunk size contains is an integer multiple
    # of the FFT length so that no data gets dropped.  This needs to
    # take into account the number of beampols in the data, the FFT length,
    # and the number of samples per frame.
    maxFrames = int(1.0*28000/beampols*vdif.DATA_LENGTH/float(2*LFFT))*2*LFFT/vdif.DATA_LENGTH*beampols

    # Number of frames to integrate over
    nFramesAvg = int(args.average * srate / vdif.DATA_LENGTH * beampols)
    nFramesAvg = int(1.0 * nFramesAvg / beampols*vdif.DATA_LENGTH/float(2*LFFT))*2*LFFT/vdif.DATA_LENGTH*beampols
    args.average = 1.0 * nFramesAvg / beampols * vdif.DATA_LENGTH / srate
    maxFrames = nFramesAvg
    
    # Number of remaining chunks (and the correction to the number of
    # frames to read in).
    if args.duration == 0:
        args.duration = 1.0 * nFramesFile / beampols * vdif.DATA_LENGTH / srate
        args.duration -= args.skip
    else:
        args.duration = int(round(args.duration * srate * beampols / vdif.DATA_LENGTH) / beampols * vdif.DATA_LENGTH / srate)
    nChunks = int(round(args.duration / args.average))
    if nChunks == 0:
        nChunks = 1
    nFrames = nFramesAvg*nChunks
    
    # Date & Central Frequency
    t1  = junkFrame.time
    beginDate = junkFrame.time.datetime
    central_freq1 = 0.0
    central_freq2 = 0.0
    for i in range(4):
        junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
        b,p = junkFrame.id
        if p == 0:
            central_freq1 = junkFrame.central_freq
        elif p == 0:
            central_freq2 = junkFrame.central_freq
        else:
            pass
    fh.seek(-4*vdif.FRAME_SIZE, 1)
    
    # File summary
    print("Filename: %s" % filename)
    print("Date of First Frame: %s" % str(beginDate))
    print("Beams: %i" % beams)
    print("Tune/Pols: %i" % tunepols)
    print("Sample Rate: %i Hz" % srate)
    print("Bit Depth: %i" % junkFrame.header.bits_per_sample)
    print("Tuning Frequency: %.3f Hz (1); %.3f Hz (2)" % (central_freq1, central_freq2))
    print("Frames: %i (%.3f s)" % (nFramesFile, 1.0 * nFramesFile / beampols * vdif.DATA_LENGTH / srate))
    print("---")
    print("Offset: %.3f s (%i frames)" % (args.skip, offset))
    print("Integration: %.3f s (%i frames; %i frames per beam/tune/pol)" % (args.average, nFramesAvg, nFramesAvg // beampols))
    print("Duration: %.3f s (%i frames; %i frames per beam/tune/pol)" % (args.average*nChunks, nFrames, nFrames // beampols))
    print("Chunks: %i" % nChunks)
    print(" ")
    
    # Get the clip levels
    clip1 = args.clip_level
    clip2 = args.clip_level
        
    # Make the pseudo-antennas for Stokes calculation
    antennas = []
    for i in range(4):
        if i // 2 == 0:
            newAnt = stations.Antenna(1)
        else:
            newAnt = stations.Antenna(2)
            
        if i % 2 == 0:
            newAnt.pol = 0
        else:
            newAnt.pol = 1
            
        antennas.append(newAnt)
        
    # Setup the output file
    outname = os.path.split(filename)[1]
    outname = os.path.splitext(outname)[0]
    outname = '%s-waterfall.hdf5' % outname
    
    if os.path.exists(outname):
        if not args.force:
            yn = input("WARNING: '%s' exists, overwrite? [Y/n] " % outname)
        else:
            yn = 'y'
            
        if yn not in ('n', 'N'):
            os.unlink(outname)
        else:
            raise RuntimeError("Output file '%s' already exists" % outname)
            
    f = hdfData.create_new_file(outname)
    
    # Look at the metadata and come up with a list of observations.  If 
    # there are no metadata, create a single "observation" that covers the
    # whole file.
    obsList = {}
    obsList[1] = (datetime.utcfromtimestamp(t1), datetime(2222,12,31,23,59,59), args.duration, srate)
    hdfData.fill_minimum(f, 1, beam, srate)
        
    if (not args.stokes):
        data_products = ['XX', 'YY']
    else:
        data_products = ['I', 'Q', 'U', 'V']
        
    for o in sorted(obsList.keys()):
        for t in (1,2):
            hdfData.create_observation_set(f, o, t, np.arange(LFFT, dtype=np.float32), int(round(obsList[o][2]/args.average)), data_products)
            
    f.attrs['FileGenerator'] = 'hdfWaterfall.py'
    f.attrs['InputData'] = os.path.basename(filename)
    
    # Create the various HDF group holders
    ds = {}
    for o in sorted(obsList.keys()):
        obs = hdfData.get_observation_set(f, o)
        
        ds['obs%i' % o] = obs
        ds['obs%i-time' % o] = obs.create_dataset('time', (int(round(obsList[o][2]/args.average)),), 'f8')
        
        for t in (1,2):
            ds['obs%i-freq%i' % (o, t)] = hdfData.get_data_set(f, o, t, 'freq')
            for p in data_products:
                ds["obs%i-%s%i" % (o, p, t)] = hdfData.get_data_set(f, o, t, p)
            ds['obs%i-Saturation%i' % (o, t)] = hdfData.get_data_set(f, o, t, 'Saturation')
            
    # Load in the correct analysis function
    if (not args.stokes):
        processDataBatch = processDataBatchLinear
    else:
        processDataBatch = processDataBatchStokes
        
    # Go!
    for o in sorted(obsList.keys()):
        try:
            processDataBatch(fh, header, antennas, obsList[o][0], obsList[o][2], obsList[o][3], args, ds, obsID=o, clip1=clip1, clip2=clip2)
        except RuntimeError as e:
            print("Observation #%i: %s, abandoning this observation" % (o, str(e)))

    # Save the output to a HDF5 file
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='read in a VDIF file and create a collection of time-averaged spectra stored as an HDF5 file called <filename>-waterfall.hdf5', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument('filename', type=str, 
                        help='filename to process')
    wgroup = parser.add_mutually_exclusive_group(required=False)
    wgroup.add_argument('-t', '--bartlett', action='store_true', 
                        help='apply a Bartlett window to the data')
    wgroup.add_argument('-b', '--blackman', action='store_true', 
                        help='apply a Blackman window to the data')
    wgroup.add_argument('-n', '--hanning', action='store_true', 
                        help='apply a Hanning window to the data')
    parser.add_argument('-s', '--skip', type=aph.positive_or_zero_float, default=0.0, 
                        help='skip the specified number of seconds at the beginning of the file')
    parser.add_argument('-a', '--average', type=aph.positive_float, default=1.0, 
                        help='number of seconds of data to average for spectra')
    parser.add_argument('-d', '--duration', type=aph.positive_or_zero_float, default=0.0, 
                        help='number of seconds to calculate the waterfall for; 0 for everything') 
    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false',
                        help='run %(prog)s in silent mode')
    parser.add_argument('-l', '--fft-length', type=aph.positive_int, default=4096, 
                        help='set FFT length')
    parser.add_argument('-c', '--clip-level', type=aph.positive_or_zero_int, default=0,  
                        help='FFT blanking clipping level in counts; 0 disables')
    parser.add_argument('-e', '--estimate-clip-level', action='store_true', 
                        help='use robust statistics to estimate an appropriate clip level; overrides -c/--clip-level')
    parser.add_argument('-f', '--force', action='store_true', 
                        help='force overwritting of existing HDF5 files')
    parser.add_argument('-k', '--stokes', action='store_true', 
                        help='generate Stokes parameters instead of XX and YY')
    parser.add_argument('-w', '--without-sats', action='store_true',
                        help='do not generate saturation counts')
    args = parser.parse_args()
    main(args)
    
