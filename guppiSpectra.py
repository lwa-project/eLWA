#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy
import getopt
from datetime import datetime

from lsl.reader import errors
from lsl.correlator import fx as fxc

import guppi
from utils import *

from matplotlib import pyplot as plt


def usage(exitCode=None):
    print """guppiSpectra.py - Read in a GUPPI file and plot spectra

Usage:
guppiSpectra.py [OPTIONS] <guppi_file>

Options:
-h, --help                  Display this help information
-l, --fft-length            Set FFT length (default = 4096)
-s, --skip                  Amount of time in to skip into the files (seconds; 
                            default = 0 s)
-t, --avg-time              Window to average visibilities in time (seconds; 
                            default = 1 s)
"""
    
    if exitCode is not None:
        sys.exit(exitCode)
    else:
        return True


def parseConfig(args):
    config = {}
    # Command line flags - default values
    config['avgTime'] = 1.0
    config['LFFT'] = 4096
    config['skip'] = 0.0
    config['args'] = []
    
    # Read in and process the command line flags
    try:
        opts, args = getopt.getopt(args, "hl:t:s:", ["help", "fft-length=", "avg-time=", "skip="])
    except getopt.GetoptError, err:
        # Print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage(exitCode=2)
        
    # Work through opts
    for opt, value in opts:
        if opt in ('-h', '--help'):
            usage(exitCode=0)
        elif opt in ('-l', '--fft-length'):
            config['LFFT'] = int(value)
        elif opt in ('-t', '--avg-time'):
            config['avgTime'] = float(value)
        elif opt in ('-s', '--skip'):
            config['skip'] = float(value)
        else:
            assert False
            
    # Add in arguments
    config['args'] = args
    
    # Return configuration
    return config


def main(args):
    # Parse the command line
    config = parseConfig(args)
    filename = config['args'][0]
    
    # Length of the FFT
    LFFT = config['LFFT']
    
    fh = open(filename, 'rb')
    header = read_guppi_header(fh)
    guppi.FrameSize = guppi.getFrameSize(fh)
    nFramesFile = os.path.getsize(filename) / guppi.FrameSize
    
    junkFrame = guppi.readFrame(fh)
    srate = junkFrame.getSampleRate()
    guppi.DataLength = junkFrame.data.data.size
    beam, pol = junkFrame.parseID()
    tunepols = guppi.getThreadCount(fh)
    beampols = tunepols
    
    if config['skip'] != 0:
        print "Skipping forward %.3f s" % config['skip']
        print "-> %.6f (%s)" % (junkFrame.getTime(), datetime.utcfromtimestamp(junkFrame.getTime()))
        
        offset = int(config['skip']*srate / guppi.DataLength )
        fh.seek(guppi.FrameSize*beampols*(offset - 1), 1)
        junkFrame = guppi.readFrame(fh)
        fh.seek(-guppi.FrameSize, 1)
        
        print "-> %.6f (%s)" % (junkFrame.getTime(), datetime.utcfromtimestamp(junkFrame.getTime()))
        tStart = junkFrame.getTime()
        
    # Get the frequencies
    cFreq = 0.0
    for j in xrange(4):
        junkFrame = guppi.readFrame(fh)
        s,p = junkFrame.parseID()
        if p == 0:
            cFreq = junkFrame.getCentralFreq()
            
    # Set integration time
    tInt = config['avgTime']
    nFrames = int(round(tInt*srate/guppi.DataLength))
    tInt = nFrames*guppi.DataLength/srate
    
    nFrames = int(round(tInt*srate/guppi.DataLength))
    
    # Read in some data
    tFile = nFramesFile / beampols * guppi.DataLength / srate
    
    # Date
    junkFrame = guppi.readFrame(fh)
    fh.seek(-guppi.FrameSize, 1)
    beginDate = datetime.utcfromtimestamp(junkFrame.getTime())
        
    # Report
    print "Filename: %s" % os.path.basename(filename)
    print "  Date of First Frame: %s" % beginDate
    print "  Station: %i" % beam
    print "  Sample Rate: %i Hz" % srate
    print "  Tuning 1: %.1f Hz" % cFreq
    print "  Bit Depth: %i" % junkFrame.header.bitsPerSample
    print "  Integration Time: %.3f s" % tInt
    print "  Integrations in File: %i" % int(tFile/tInt)
    print " "

    # Go!
    data = numpy.zeros((beampols, guppi.DataLength*nFrames), dtype=numpy.complex64)
    count = [0 for i in xrange(data.shape[0])]
    for i in xrange(beampols*nFrames):
        try:
            cFrame = guppi.readFrame(fh)
        except errors.syncError:
            print "Error @ %i, %i" % (i, j)
            f.seek(guppi.FrameSize, 1)
            continue
        std,pol = cFrame.parseID()
        sid = pol
        
        data[sid, count[sid]*guppi.DataLength:(count[sid]+1)*guppi.DataLength] = cFrame.data.data
        count[sid] += 1
        
    # Transform and trim off the negative frequencies
    freq, psd = fxc.SpecMaster(data, LFFT=2*LFFT, SampleRate=srate, CentralFreq=header['OBSFREQ']-srate/4)
    freq, psd = freq[LFFT:], psd[:,LFFT:]
    
    # Plot
    fig = plt.figure()
    ax = fig.gca()
    for i in xrange(psd.shape[0]):
        ax.plot(freq/1e6, numpy.log10(psd[i,:])*10, label='%i' % i)
    ax.set_title('%i' % beam)
    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('PSD [arb. dB]')
    ax.legend(loc=0)
    plt.show()
    
    # Done
    fh.close()


if __name__ == "__main__":
    main(sys.argv[1:])
    
