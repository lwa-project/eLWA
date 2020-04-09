#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run through a DRX file and determine if it is bad or not.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import sys
import ephem
import numpy
import getopt
from datetime import datetime

from lsl import astro
from lsl.reader import errors

import guppi
from utils import *


def usage(exitCode=None):
    print """guppiFileCheck.py - Run through a GUPPI raw file and determine if it is bad or not.

Usage: guppiFileCheck.py [OPTIONS] filename

Options:
-h, --help         Display this help information
-l, --length       Length of time in seconds to analyze (default 1 s)
-s, --skip         Skip period in seconds between chunks (default 900 s)
-t, --trim-level   Trim level for power analysis with clipping (default is 
                set by bit depth)
"""
    
    if exitCode is not None:
        sys.exit(exitCode)
    else:
        return None


def parseConfig(args):
    config = {}
    config['length'] = 1.0
    config['skip'] = 900.0
    config['trim'] = None
    
    try:
        opts, args = getopt.getopt(args, "hl:s:t:", ["help", "length=", "skip=", "trim-level="])
    except getopt.GetoptError, err:
        # Print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage(exitCode=2)

    # Work through opts
    for opt, value in opts:
        if opt in ('-h', '--help'):
            usage(exitCode=0)
        elif opt in ('-l', '--length'):
            config['length'] = float(value)
        elif opt in ('-s', '--skip'):
            config['skip'] = float(value)
        elif opt in ('-t', '--trim-level'):
            config['trim'] = float(value)
        else:
            assert False
            
    # Add in arguments
    config['args'] = args
    
    # Return
    return config


def main(args):
    # Parse the command line
    config = parseConfig(args)
    filename = config['args'][0]
    
    fh = open(filename, 'rb')
    header = guppi.read_guppi_header(fh)
    guppi.FRAME_SIZE = guppi.get_frame_size(fh)
    nFramesFile = os.path.getsize(filename) / guppi.FRAME_SIZE
    
    junkFrame = guppi.read_frame(fh)
    srate = junkFrame.sample_rate
    guppi.DataLength = junkFrame.data.data.size
    beam, pol = junkFrame.id
    tunepols = guppi.get_thread_count(fh)
    beampols = tunepols
    
    # Get the frequencies
    cFreq = 0.0
    for j in xrange(4):
        junkFrame = guppi.read_frame(fh)
        s,p = junkFrame.id
        if p == 0:
            cFreq = junkFrame.central_freq
            
    # Date
    junkFrame = guppi.read_frame(fh)
    fh.seek(-guppi.FRAME_SIZE, 1)
    beginDate = datetime.utcfromtimestamp(junkFrame.get_time())
        
    # Report
    print "Filename: %s" % os.path.basename(filename)
    print "  Date of First Frame: %s" % beginDate
    print "  Station: %i" % beam
    print "  Sample Rate: %i Hz" % srate
    print "  Bit Depth: %i" % junkFrame.header.bits_per_sample
    print "  Tuning 1: %.1f Hz" % cFreq
    print " "
    
    # Determine the clip level
    if config['trim'] is None:
        if junkFrame.header.bits_per_sample == 1:
            config['trim'] = 1
        elif junkFrame.header.bits_per_sample == 2:
            config['trim'] = 1
        elif junkFrame.header.bits_per_sample == 4:
            config['trim'] = 49
        elif junkFrame.header.bits_per_sample == 8:
            config['trim'] = 256
        else:
            config['trim'] = 1.0
        print "Setting clip level to %.3f" % config['trim']
        print " "
        
    # Convert chunk length to total frame count
    chunkLength = int(config['length'] * srate / guppi.DataLength * tunepols)
    chunkLength = int(1.0 * chunkLength / tunepols) * tunepols
    
    # Convert chunk skip to total frame count
    chunkSkip = int(config['skip'] * srate / guppi.DataLength * tunepols)
    chunkSkip = int(1.0 * chunkSkip / tunepols) * tunepols
    
    # Output arrays
    clipFraction = []
    meanPower = []
    meanRMS = []
    
    # Go!
    i = 1
    done = False
    print "    |      Clipping   |     Power     |      RMS      |"
    print "    |      1X      1Y |     1X     1Y |     1X     1Y |"
    print "----+-----------------+---------------+---------------+"
    
    while True:
        count = {0:0, 1:0}
        data = numpy.empty((2,chunkLength*guppi.DataLength/tunepols), dtype=numpy.float32)
        for j in xrange(chunkLength):
            # Read in the next frame and anticipate any problems that could occur
            try:
                cFrame = guppi.read_frame(fh, Verbose=False)
            except errors.EOFError:
                done = True
                break
            except errors.SyncError:
                continue
            
            beam,pol = cFrame.id
            aStand = pol
            
            try:
                data[aStand, count[aStand]*guppi.DataLength:(count[aStand]+1)*guppi.DataLength] = cFrame.data.data
                
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
                bad = numpy.nonzero(data[j,:] > config['trim'])[0]
                clipFraction[-1][j] = 1.0*len(bad) / data.shape[1]
                
            clip = clipFraction[-1]
            power = meanPower[-1]
            print "%3i | %6.2f%% %6.2f%% | %6.3f %6.3f | %6.3f %6.3f |" % (i, clip[0]*100.0, clip[1]*100.0, power[0], power[1], rms[0], rms[1])
        
            i += 1
            if fh.tell() + guppi.FRAME_SIZE*chunkSkip >= os.path.getsize(filename):
                break
            fh.seek(guppi.FRAME_SIZE*chunkSkip, 1)
            
    clipFraction = numpy.array(clipFraction)
    meanPower = numpy.array(meanPower)
    meanRMS = numpy.array(meanRMS)
    
    clip = clipFraction.mean(axis=0)
    power = meanPower.mean(axis=0)
    rms = meanRMS.mean(axis=0)
    
    print "----+-----------------+---------------+---------------+"
    print "%3s | %6.2f%% %6.2f%% | %6.3f %6.3f | %6.3f %6.3f |" % ('M', clip[0]*100.0, clip[1]*100.0, power[0], power[1], rms[0], rms[1])


if __name__ == "__main__":
    main(sys.argv[1:])
    
