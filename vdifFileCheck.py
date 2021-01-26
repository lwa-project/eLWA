#!/usr/bin/env python

"""
Run through a VDIF file and determine if it is bad or not.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import os
import sys
import numpy
import argparse
from datetime import datetime

from lsl import astro
from lsl.reader import vdif, errors
from lsl.misc import parser as aph

from utils import *


def main(args):
    # Parse the command line
    filename = args.filename
    
    fh = open(filename, 'rb')
    is_vlite = is_vlite_vdif(fh)
    if is_vlite:
        ## TODO:  Clean this up
        header = {'OBSFREQ': 352e6,
                  'OBSBW':   64e6}
    else:
        header = vdif.read_guppi_header(fh)
    vdif.FRAME_SIZE = vdif.get_frame_size(fh)
    nFramesFile = os.path.getsize(filename) // vdif.FRAME_SIZE
    
    junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
    srate = junkFrame.sample_rate
    vdif.DATA_LENGTH = junkFrame.payload.data.size
    beam, pol = junkFrame.id
    tunepols = vdif.get_thread_count(fh)
    beampols = tunepols
    
    # Get the frequencies
    cFreq = 0.0
    for j in xrange(4):
        junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
        s,p = junkFrame.id
        if p == 0:
            cFreq = junkFrame.central_freq
            
    # Date
    junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
    fh.seek(-vdif.FRAME_SIZE, 1)
    beginDate = junkFrame.time.datetime
        
    # Report
    print("Filename: %s" % os.path.basename(filename))
    print("  Date of First Frame: %s" % beginDate)
    print("  Station: %i" % beam)
    print("  Sample Rate: %i Hz" % srate)
    print("  Bit Depth: %i" % junkFrame.header.bits_per_sample)
    print("  Tuning 1: %.1f Hz" % cFreq)
    print(" ")
    
    # Determine the clip level
    if args.trim_level is None:
        if junkFrame.header.bits_per_sample == 1:
            args.trim_level = abs(1.0)**2
        elif junkFrame.header.bits_per_sample == 2:
            args.trim_level = abs(3.3359)**2
        elif junkFrame.header.bits_per_sample == 4:
            args.trim_level = abs(7/2.95)**2
        elif junkFrame.header.bits_per_sample == 8:
            args.trim_level = abs(255/256.)**2
        else:
            args.trim_level = 1.0
        print("Setting clip level to %.3f" % args.trim_level)
        print(" ")
        
    # Convert chunk length to total frame count
    chunkLength = int(args.length * srate / vdif.DATA_LENGTH * tunepols)
    chunkLength = int(1.0 * chunkLength / tunepols) * tunepols
    
    # Convert chunk skip to total frame count
    chunkSkip = int(args.skip * srate / vdif.DATA_LENGTH * tunepols)
    chunkSkip = int(1.0 * chunkSkip / tunepols) * tunepols
    
    # Output arrays
    clipFraction = []
    meanPower = []
    meanRMS = []
    
    # Go!
    i = 1
    done = False
    print("    |      Clipping   |     Power     |      RMS      |")
    print("    |      1X      1Y |     1X     1Y |     1X     1Y |")
    print("----+-----------------+---------------+---------------+")
    
    while True:
        count = {0:0, 1:0}
        data = numpy.empty((2,chunkLength*vdif.DATA_LENGTH//tunepols), dtype=numpy.float32)
        for j in xrange(chunkLength):
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
            
            try:
                data[aStand, count[aStand]*vdif.DATA_LENGTH:(count[aStand]+1)*vdif.DATA_LENGTH] = cFrame.payload.data
                
                # Update the counters so that we can average properly later on
                count[aStand] += 1
            except ValueError:
                pass
        
        if done:
            break
            
        else:
            rms = numpy.sqrt( (data**2).mean(axis=1) )
            data = numpy.abs(data)**2
            
            clipFraction.append( numpy.zeros(2) )
            meanPower.append( data.mean(axis=1) )
            meanRMS.append( rms )
            for j in xrange(2):
                bad = numpy.nonzero(data[j,:] > args.trim_level)[0]
                clipFraction[-1][j] = 1.0*len(bad) / data.shape[1]
                
            clip = clipFraction[-1]
            power = meanPower[-1]
            print("%3i | %6.2f%% %6.2f%% | %6.3f %6.3f | %6.3f %6.3f |" % (i, clip[0]*100.0, clip[1]*100.0, power[0], power[1], rms[0], rms[1]))
        
            i += 1
            fh.seek(vdif.FRAME_SIZE*chunkSkip, 1)
            
    clipFraction = numpy.array(clipFraction)
    meanPower = numpy.array(meanPower)
    meanRMS = numpy.array(meanRMS)
    
    clip = clipFraction.mean(axis=0)
    power = meanPower.mean(axis=0)
    rms = meanRMS.mean(axis=0)
    
    print("----+-----------------+---------------+---------------+")
    print("%3s | %6.2f%% %6.2f%% | %6.3f %6.3f | %6.3f %6.3f |" % ('M', clip[0]*100.0, clip[1]*100.0, power[0], power[1], rms[0], rms[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='run through a VDIF file and determine if it is bad or not', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, 
                        help='filename to check')
    parser.add_argument('-l', '--length', type=aph.positive_float, default=1.0, 
                        help='length of time in seconds to analyze')
    parser.add_argument('-s', '--skip', type=aph.positive_float, default=900.0, 
                        help='skip period in seconds between chunks')
    parser.add_argument('-t', '--trim-level', type=aph.positive_float, 
                        help='trim level for power analysis with clipping')
    args = parser.parse_args()
    main(args)
    
