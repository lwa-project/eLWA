#!/usr/bin/env python

"""
Given a VDIF file, plot the time averaged spectra for each beam output over some 
period.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    raw_input = input
    
import os
import sys
import h5py
import math
import numpy
import ephem
import getopt
from datetime import datetime

import lsl.reader.vdif as vdif
import lsl.reader.drspec as drspec
import lsl.reader.errors as errors
import lsl.statistics.robust as robust
import lsl.correlator.fx as fxc
from lsl.astro import unix_to_utcjd, DJD_OFFSET
from lsl.common import progress, stations
from lsl.common import mcs, metabundle

import matplotlib.pyplot as plt

from utils import *

import data as hdfData


def usage(exitCode=None):
    print("""vdifWaterfall.py - Read in VDIF files and create a collection of 
time-averaged spectra.  These spectra are saved to a HDF5 file called <filename>-waterfall.hdf5.

Usage: vdifWaterfall.py [OPTIONS] file

Options:
-h, --help                  Display this help information
-t, --bartlett              Apply a Bartlett window to the data
-b, --blackman              Apply a Blackman window to the data
-n, --hanning               Apply a Hanning window to the data
-s, --skip                  Skip the specified number of seconds at the beginning
                            of the file (default = 0)
-a, --average               Number of seconds of data to average for spectra 
                            (default = 1)
-d, --duration              Number of seconds to calculate the waterfall for 
                            (default = 0; run the entire file)
-q, --quiet                 Run vdifSpectra in silent mode and do not show the plots
-l, --fft-length            Set FFT length (default = 4096)
-c, --clip-level            FFT blanking clipping level in counts (default = 0, 
                            0 disables)
-f, --force                 Force overwritting of existing HDF5 files
-k, --stokes                Generate Stokes parameters instead of XX and YY
-w, --without-sats          Do not generate saturation counts

Note:  Both the -m/--metadata and -i/--sdf options provide the same additional
    observation information to hdfWaterfall.py so only one needs to be provided.

Note:  Specifying the -m/--metadata or -i/--sdf optiosn overrides the 
    -d/--duration setting and the entire file is reduced.
""")
    
    if exitCode is not None:
        sys.exit(exitCode)
    else:
        return True


def parseOptions(args):
    config = {}
    # Command line flags - default values
    config['offset'] = 0.0
    config['average'] = 1.0
    config['LFFT'] = 4096
    config['freq1'] = 0
    config['freq2'] = 0
    config['maxFrames'] = 28000
    config['window'] = fxc.null_window
    config['duration'] = 0.0
    config['verbose'] = True
    config['clip'] = 0
    config['metadata'] = None
    config['sdf'] = None
    config['force'] = False
    config['linear'] = True
    config['countSats'] = True
    config['args'] = []
    
    # Read in and process the command line flags
    try:
        opts, args = getopt.getopt(args, "hqtbnl:s:a:d:c:fkw", ["help", "quiet", "bartlett", "blackman", "hanning", "fft-length=", "skip=", "average=", "duration=", "freq1=", "freq2=", "clip-level=", "force", "stokes", "without-sats"])
    except getopt.GetoptError as err:
        # Print help information and exit:
        print(str(err)) # will print something like "option -a not recognized"
        usage(exitCode=2)
        
    # Work through opts
    for opt, value in opts:
        if opt in ('-h', '--help'):
            usage(exitCode=0)
        elif opt in ('-q', '--quiet'):
            config['verbose'] = False
        elif opt in ('-t', '--bartlett'):
            config['window'] = numpy.bartlett
        elif opt in ('-b', '--blackman'):
            config['window'] = numpy.blackman
        elif opt in ('-n', '--hanning'):
            config['window'] = numpy.hanning
        elif opt in ('-l', '--fft-length'):
            config['LFFT'] = int(value)
        elif opt in ('-s', '--skip'):
            config['offset'] = float(value)
        elif opt in ('-a', '--average'):
            config['average'] = float(value)
        elif opt in ('-d', '--duration'):
            config['duration'] = float(value)
        elif opt in ('-c', '--clip-level'):
            config['clip'] = int(value)
        elif opt in ('-e', '--estimate-clip'):
            config['estimate'] = True
        elif opt in ('-m', '--metadata'):
            config['metadata'] = value
        elif opt in ('-i', '--sdf'):
            config['sdf'] = value
        elif opt in ('-f', '--force'):
            config['force'] = True
        elif opt in ('-k', '--stokes'):
            config['linear'] = False
        elif opt in ('-w', '--without-sats'):
            config['countSats'] = False
        else:
            assert False
            
    # Add in arguments
    config['args'] = args
    
    # Return configuration
    return config


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


def processDataBatchLinear(fh, header, antennas, tStart, duration, sample_rate, config, dataSets, obsID=1, clip1=0, clip2=0):
    """
    Process a chunk of data in a raw vdif file into linear polarization 
    products and add the contents to an HDF5 file.
    """
    
    # Length of the FFT
    LFFT = config['LFFT']
    
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
    maxFrames = int(1.0*config['maxFrames']/beampols*vdif.DATA_LENGTH/float(2*LFFT))*2*LFFT//vdif.DATA_LENGTH*beampols
    
    # Number of frames per second 
    nFramesSecond = int(srate) // vdif.DATA_LENGTH
    
    # Number of frames to integrate over
    nFramesAvg = int(round(config['average'] * srate / vdif.DATA_LENGTH * beampols))
    nFramesAvg = int(1.0 * nFramesAvg / beampols*vdif.DATA_LENGTH/float(2*LFFT))*2*LFFT//vdif.DATA_LENGTH*beampols
    config['average'] = 1.0 * nFramesAvg / beampols * vdif.DATA_LENGTH / srate
    maxFrames = nFramesAvg
    
    # Number of remaining chunks (and the correction to the number of
    # frames to read in).
    nChunks = int(round(duration / config['average']))
    if nChunks == 0:
        nChunks = 1
    nFrames = nFramesAvg*nChunks
    
    # Line up the time tags for the various tunings/polarizations
    timetags = []
    for i in xrange(16):
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
    for i in xrange(4):
        junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
        b,p = junkFrame.id
        if p == 0:
            central_freq1 = junkFrame.central_freq
        elif p == 0:
            central_freq2 = junkFrame.central_freq
        else:
            pass
    fh.seek(-4*vdif.FRAME_SIZE, 1)
    freq = numpy.fft.fftshift(numpy.fft.fftfreq(LFFT, d=2/srate))
    if float(fxc.__version__) < 0.8:
        freq = freq[1:]
        
    dataSets['obs%i-freq1' % obsID][:] = freq + central_freq1
    dataSets['obs%i-freq2' % obsID][:] = freq + central_freq2
    
    obs = dataSets['obs%i' % obsID]
    obs.attrs['tInt'] = config['average']
    obs.attrs['tInt_Unit'] = 's'
    obs.attrs['LFFT'] = LFFT
    obs.attrs['nchan'] = LFFT-1 if float(fxc.__version__) < 0.8 else LFFT
    obs.attrs['RBW'] = freq[1]-freq[0]
    obs.attrs['RBW_Units'] = 'Hz'
    
    data_products = ['XX', 'YY']
    done = False
    for i in xrange(nChunks):
        # Find out how many frames remain in the file.  If this number is larger
        # than the maximum of frames we can work with at a time (maxFrames),
        # only deal with that chunk
        framesRemaining = nFrames - i*maxFrames
        if framesRemaining > maxFrames:
            framesWork = maxFrames
        else:
            framesWork = framesRemaining
        print("Working on chunk %i, %i frames remaining" % (i+1, framesRemaining))
        
        count = {0:0, 1:0, 2:0, 3:0}
        data = numpy.zeros((4,framesWork*vdif.DATA_LENGTH//beampols), dtype=numpy.csingle)
        # If there are fewer frames than we need to fill an FFT, skip this chunk
        if data.shape[1] < LFFT:
            break
            
        # Inner loop that actually reads the frames into the data array
        print("Working on %.1f ms of data" % ((framesWork*vdif.DATA_LENGTH/beampols/srate)*1000.0))
        
        for j in xrange(framesWork):
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
        dataSets['obs%i-time' % obsID][i] = float(cTime)
        
        if config['countSats']:
            sats = ((data.real**2 + data.imag**2) >= 49).sum(axis=1)
            dataSets['obs%i-Saturation1' % obsID][i,:] = sats[0:2]
            dataSets['obs%i-Saturation2' % obsID][i,:] = sats[2:4]
        else:
            dataSets['obs%i-Saturation1' % obsID][i,:] = -1
            dataSets['obs%i-Saturation2' % obsID][i,:] = -1
        
        # Calculate the spectra for this block of data and then weight the results by 
        # the total number of frames read.  This is needed to keep the averages correct.
        if clip1 == clip2:
            freq, tempSpec1 = fxc.SpecMaster(data, LFFT=2*LFFT, window=config['window'], verbose=config['verbose'], sample_rate=srate, clip_level=clip1)
            freq, tempSpec1 = freq[LFFT:], tempSpec1[:,LFFT:]
            
            l = 0
            for t in (1,2):
                for p in data_products:
                    dataSets['obs%i-%s%i' % (obsID, p, t)][i,:] = tempSpec1[l,:]
                    l += 1
                    
        else:
            freq, tempSpec1 = fxc.SpecMaster(data[:2,:], LFFT=2*LFFT, window=config['window'], verbose=config['verbose'], sample_rate=srate, clip_level=clip1)
            freq, tempSpec2 = fxc.SpecMaster(data[2:,:], LFFT=2*LFFT, window=config['window'], verbose=config['verbose'], sample_rate=srate, clip_level=clip2)
            freq, tempSpec1, tempSpec2 = freq[LFFT:], tempSpec1[:,LFFT:], tempSpec2[:,LFFT:]
            
            for l,p in enumerate(data_products):
                dataSets['obs%i-%s%i' % (obsID, p, 1)][i,:] = tempSpec1[l,:]
                dataSets['obs%i-%s%i' % (obsID, p, 2)][i,:] = tempSpec2[l,:]
                
        # We don't really need the data array anymore, so delete it
        del(data)
        
        # Are we done yet?
        if done:
            break
            
    return True


def processDataBatchStokes(fh, header, antennas, tStart, duration, sample_rate, config, dataSets, obsID=1, clip1=0, clip2=0):
    """
    Process a chunk of data in a raw vdif file into Stokes parameters and 
    add the contents to an HDF5 file.
    """
    
    # Length of the FFT
    LFFT = config['LFFT']
    
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
    maxFrames = int(1.0*config['maxFrames']/beampols*vdif.DATA_LENGTH/float(2*LFFT))*2*LFFT//vdif.DATA_LENGTH*beampols
    
    # Number of frames per second 
    nFramesSecond = int(srate) // vdif.DATA_LENGTH
    
    # Number of frames to integrate over
    nFramesAvg = int(round(config['average'] * srate / vdif.DATA_LENGTH * beampols))
    nFramesAvg = int(1.0 * nFramesAvg / beampols*vdif.DATA_LENGTH/float(2*LFFT))*2*LFFT//vdif.DATA_LENGTH*beampols
    config['average'] = 1.0 * nFramesAvg / beampols * vdif.DATA_LENGTH / srate
    maxFrames = nFramesAvg
    
    # Number of remaining chunks (and the correction to the number of
    # frames to read in).
    nChunks = int(round(duration / config['average']))
    if nChunks == 0:
        nChunks = 1
    nFrames = nFramesAvg*nChunks
    
    # Line up the time tags for the various tunings/polarizations
    timetags = []
    for i in xrange(16):
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
    for i in xrange(4):
        junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
        b,p = junkFrame.id
        if p == 0:
            central_freq1 = junkFrame.central_freq
        elif p == 0:
            central_freq2 = junkFrame.central_freq
        else:
            pass
    fh.seek(-4*vdif.FRAME_SIZE, 1)
    freq = numpy.fft.fftshift(numpy.fft.fftfreq(LFFT, d=2/srate))
    if float(fxc.__version__) < 0.8:
        freq = freq[1:]
        
    dataSets['obs%i-freq1' % obsID][:] = freq + central_freq1
    dataSets['obs%i-freq2' % obsID][:] = freq + central_freq2
    
    obs = dataSets['obs%i' % obsID]
    obs.attrs['tInt'] = config['average']
    obs.attrs['tInt_Unit'] = 's'
    obs.attrs['LFFT'] = LFFT
    obs.attrs['nchan'] = LFFT-1 if float(fxc.__version__) < 0.8 else LFFT
    obs.attrs['RBW'] = freq[1]-freq[0]
    obs.attrs['RBW_Units'] = 'Hz'
    
    data_products = ['I', 'Q', 'U', 'V']
    done = False
    for i in xrange(nChunks):
        # Find out how many frames remain in the file.  If this number is larger
        # than the maximum of frames we can work with at a time (maxFrames),
        # only deal with that chunk
        framesRemaining = nFrames - i*maxFrames
        if framesRemaining > maxFrames:
            framesWork = maxFrames
        else:
            framesWork = framesRemaining
        print("Working on chunk %i, %i frames remaining" % (i+1, framesRemaining))
        
        count = {0:0, 1:0, 2:0, 3:0}
        data = numpy.zeros((4,framesWork*vdif.DATA_LENGTH//beampols), dtype=numpy.csingle)
        # If there are fewer frames than we need to fill an FFT, skip this chunk
        if data.shape[1] < LFFT:
            break
            
        # Inner loop that actually reads the frames into the data array
        print("Working on %.1f ms of data" % ((framesWork*vdif.DATA_LENGTH/beampols/srate)*1000.0))
        
        for j in xrange(framesWork):
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
        dataSets['obs%i-time' % obsID][i] = float(cTime)
        
        if config['countSats']:
            sats = ((data.real**2 + data.imag**2) >= 49).sum(axis=1)
            dataSets['obs%i-Saturation1' % obsID][i,:] = sats[0:2]
            dataSets['obs%i-Saturation2' % obsID][i,:] = sats[2:4]
        else:
            dataSets['obs%i-Saturation1' % obsID][i,:] = -1
            dataSets['obs%i-Saturation2' % obsID][i,:] = -1
            
        # Calculate the spectra for this block of data and then weight the results by 
        # the total number of frames read.  This is needed to keep the averages correct.
        if clip1 == clip2:
            freq, tempSpec1 = fxc.StokesMaster(data, antennas, LFFT=2*LFFT, window=config['window'], verbose=config['verbose'], sample_rate=srate, clip_level=clip1)
            freq, tempSpec1 = freq[LFFT:], tempSpec1[:,:,LFFT:]
            
            for t in (1,2):
                for l,p in enumerate(data_products):
                    dataSets['obs%i-%s%i' % (obsID, p, t)][i,:] = tempSpec1[l,t-1,:]
                    
        else:
            freq, tempSpec1 = fxc.StokesMaster(data[:2,:], antennas[:2], LFFT=2*LFFT, window=config['window'], verbose=config['verbose'], sample_rate=srate, clip_level=clip1)
            freq, tempSpec2 = fxc.StokesMaster(data[2:,:], antennas[2:], LFFT=2*LFFT, window=config['window'], verbose=config['verbose'], sample_rate=srate, clip_level=clip2)
            freq, tempSpec1, tempSpec2 = freq[LFFT:], tempSpec1[:,:,LFFT:], tempSpec2[:,:,LFFT:]
            
            for l,p in enumerate(data_products):
                dataSets['obs%i-%s%i' % (obsID, p, 1)][i,:] = tempSpec1[l,0,:]
                dataSets['obs%i-%s%i' % (obsID, p, 2)][i,:] = tempSpec2[l,0,:]

        # We don't really need the data array anymore, so delete it
        del(data)
        
        # Are we done yet?
        if done:
            break
            
    return True


def main(args):
    # Parse command line options
    config = parseOptions(args)
    
    # Length of the FFT
    LFFT = config['LFFT']
    
    # Open the file and find good data (not spectrometer data)
    filename = config['args'][0]
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
    offset = int(config['offset'] * srate / vdif.DATA_LENGTH * beampols)
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
        tDiff = t1 - (t0 + config['offset'])
        
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
    config['offset'] = t1 - t0
    offset = int(round(config['offset'] * srate / vdif.DATA_LENGTH * beampols))
    offset = int(1.0 * offset / beampols) * beampols

    # Make sure that the file chunk size contains is an integer multiple
    # of the FFT length so that no data gets dropped.  This needs to
    # take into account the number of beampols in the data, the FFT length,
    # and the number of samples per frame.
    maxFrames = int(1.0*config['maxFrames']/beampols*vdif.DATA_LENGTH/float(2*LFFT))*2*LFFT/vdif.DATA_LENGTH*beampols

    # Number of frames to integrate over
    nFramesAvg = int(config['average'] * srate / vdif.DATA_LENGTH * beampols)
    nFramesAvg = int(1.0 * nFramesAvg / beampols*vdif.DATA_LENGTH/float(2*LFFT))*2*LFFT/vdif.DATA_LENGTH*beampols
    config['average'] = 1.0 * nFramesAvg / beampols * vdif.DATA_LENGTH / srate
    maxFrames = nFramesAvg
    
    # Number of remaining chunks (and the correction to the number of
    # frames to read in).
    if config['metadata'] is not None:
        config['duration'] = 0
    if config['duration'] == 0:
        config['duration'] = 1.0 * nFramesFile / beampols * vdif.DATA_LENGTH / srate
        config['duration'] -= config['offset']
    else:
        config['duration'] = int(round(config['duration'] * srate * beampols / vdif.DATA_LENGTH) / beampols * vdif.DATA_LENGTH / srate)
    nChunks = int(round(config['duration'] / config['average']))
    if nChunks == 0:
        nChunks = 1
    nFrames = nFramesAvg*nChunks
    
    # Date & Central Frequency
    t1  = junkFrame.time
    beginDate = junkFrame.time.datetime
    central_freq1 = 0.0
    central_freq2 = 0.0
    for i in xrange(4):
        junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
        b,p = junkFrame.id
        if p == 0:
            central_freq1 = junkFrame.central_freq
        elif p == 0:
            central_freq2 = junkFrame.central_freq
        else:
            pass
    fh.seek(-4*vdif.FRAME_SIZE, 1)
    
    config['freq1'] = central_freq1
    config['freq2'] = central_freq2

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
    print("Offset: %.3f s (%i frames)" % (config['offset'], offset))
    print("Integration: %.3f s (%i frames; %i frames per beam/tune/pol)" % (config['average'], nFramesAvg, nFramesAvg // beampols))
    print("Duration: %.3f s (%i frames; %i frames per beam/tune/pol)" % (config['average']*nChunks, nFrames, nFrames // beampols))
    print("Chunks: %i" % nChunks)
    print(" ")
    
    # Get the clip levels
    clip1 = config['clip']
    clip2 = config['clip']
        
    # Make the pseudo-antennas for Stokes calculation
    antennas = []
    for i in xrange(4):
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
        if not config['force']:
            yn = raw_input("WARNING: '%s' exists, overwrite? [Y/n] " % outname)
        else:
            yn = 'y'
            
        if yn not in ('n', 'N'):
            os.unlink(outname)
        else:
            raise RuntimeError("Output file '%s' already exists" % outname)
            
    f = hdfData.createNewFile(outname)
    
    # Look at the metadata and come up with a list of observations.  If 
    # there are no metadata, create a single "observation" that covers the
    # whole file.
    obsList = {}
    if config['metadata'] is not None:
        sdf = metabundle.get_sdf(config['metadata'])
        
        sdfBeam  = sdf.sessions[0].vdifBeam
        spcSetup = sdf.sessions[0].spcSetup
        if sdfBeam != beam:
            raise RuntimeError("Metadata is for beam #%i, but data is from beam #%i" % (sdfBeam, beam))
            
        for i,obs in enumerate(sdf.sessions[0].observations):
            sdfStart = mcs.mjdmpm_to_datetime(obs.mjd, obs.mpm)
            sdfStop  = mcs.mjdmpm_to_datetime(obs.mjd, obs.mpm + obs.dur)
            obsDur   = obs.dur/1000.0
            obsSR    = srate
            
            obsList[i+1] = (sdfStart, sdfStop, obsDur, obsSR)
            
        print("Observations:")
        for i in sorted(obsList.keys()):
            obs = obsList[i]
            print(" #%i: %s to %s (%.3f s) at %.3f MHz" % (i, obs[0], obs[1], obs[2], obs[3]/1e6))
        print(" ")
            
        hdfData.fillFromMetabundle(f, config['metadata'])
        
    elif config['sdf'] is not None:
        from lsl.common import mcs
        from lsl.common.sdf import parse_sdf
        sdf = parse_sdf(config['sdf'])
        
        sdfBeam  = sdf.sessions[0].vdifBeam
        spcSetup = sdf.sessions[0].spcSetup
        if sdfBeam != beam:
            raise RuntimeError("Metadata is for beam #%i, but data is from beam #%i" % (sdfBeam, beam))
            
        for i,obs in enumerate(sdf.sessions[0].observations):
            sdfStart = mcs.mjdmpm_to_datetime(obs.mjd, obs.mpm)
            sdfStop  = mcs.mjdmpm_to_datetime(obs.mjd, obs.mpm + obs.dur)
            obsChunks = int(numpy.ceil(obs.dur/1000.0 * srate / (spcSetup[0]*spcSetup[1])))
            
            obsList[i+1] = (sdfStart, sdfStop, obsChunks)
            
        hdfData.fillFromSDF(f, config['sdf'])
        
    else:
        obsList[1] = (datetime.utcfromtimestamp(t1), datetime(2222,12,31,23,59,59), config['duration'], srate)
        
        hdfData.fillMinimum(f, 1, beam, srate)
        
    if config['linear']:
        data_products = ['XX', 'YY']
    else:
        data_products = ['I', 'Q', 'U', 'V']
        
    for o in sorted(obsList.keys()):
        for t in (1,2):
            hdfData.createDataSets(f, o, t, numpy.arange(LFFT-1 if float(fxc.__version__) < 0.8 else LFFT, dtype=numpy.float32), int(round(obsList[o][2]/config['average'])), data_products)
            
    f.attrs['FileGenerator'] = 'hdfWaterfall.py'
    f.attrs['InputData'] = os.path.basename(filename)
    
    # Create the various HDF group holders
    ds = {}
    for o in sorted(obsList.keys()):
        obs = hdfData.getObservationSet(f, o)
        
        ds['obs%i' % o] = obs
        ds['obs%i-time' % o] = obs.create_dataset('time', (int(round(obsList[o][2]/config['average'])),), 'f8')
        
        for t in (1,2):
            ds['obs%i-freq%i' % (o, t)] = hdfData.get_data_set(f, o, t, 'freq')
            for p in data_products:
                ds["obs%i-%s%i" % (o, p, t)] = hdfData.get_data_set(f, o, t, p)
            ds['obs%i-Saturation%i' % (o, t)] = hdfData.get_data_set(f, o, t, 'Saturation')
            
    # Load in the correct analysis function
    if config['linear']:
        processDataBatch = processDataBatchLinear
    else:
        processDataBatch = processDataBatchStokes
        
    # Go!
    for o in sorted(obsList.keys()):
        try:
            processDataBatch(fh, header, antennas, obsList[o][0], obsList[o][2], obsList[o][3], config, ds, obsID=o, clip1=clip1, clip2=clip2)
        except RuntimeError as e:
            print("Observation #%i: %s, abandoning this observation" % (o, str(e)))

    # Save the output to a HDF5 file
    f.close()


if __name__ == "__main__":
    main(sys.argv[1:])
