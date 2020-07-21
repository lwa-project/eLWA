#!/usr/bin/env python

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import os
import sys
import numpy
import getopt
from datetime import datetime

from lsl.reader import vdif, errors
from lsl.correlator import fx as fxc

from utils import *

from matplotlib import pyplot as plt


def usage(exitCode=None):
    print("""vdifHistogram.py - Read in a VDIF file and plot sample histograms

Usage:
vdifHistogram.py [OPTIONS] <vdif_file>

Options:
-h, --help                  Display this help information
-s, --skip                  Amount of time in to skip into the files (seconds; 
                            default = 0 s)
-t, --avg-time              Window to average visibilities in time (seconds; 
                            default = 1 s)
""")
    
    if exitCode is not None:
        sys.exit(exitCode)
    else:
        return True


def parseConfig(args):
    config = {}
    # Command line flags - default values
    config['avgTime'] = 1.0
    config['skip'] = 0.0
    config['args'] = []
    
    # Read in and process the command line flags
    try:
        opts, args = getopt.getopt(args, "ht:s:", ["help", "avg-time=", "skip="])
    except getopt.GetoptError as err:
        # Print help information and exit:
        print(str(err)) # will print something like "option -a not recognized"
        usage(exitCode=2)
        
    # Work through opts
    for opt, value in opts:
        if opt in ('-h', '--help'):
            usage(exitCode=0)
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
    
    if config['skip'] != 0:
        print("Skipping forward %.3f s" % config['skip'])
        print("-> %.6f (%s)" % (junkFrame.time, junkFrame.time.datetime))
        
        offset = int(config['skip']*srate / vdif.DATA_LENGTH)
        fh.seek(beampols*vdif.FRAME_SIZE*offset, 1)
        junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
        fh.seek(-vdif.FRAME_SIZE, 1)
        
        print("-> %.6f (%s)" % (junkFrame.time, junkFrame.time.datetime))
        tStart = junkFrame.time
        
    # Get the frequencies
    cFreq = 0.0
    for j in xrange(4):
        junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
        s,p = junkFrame.id
        if p == 0:
            cFreq = junkFrame.central_freq
            
    # Set integration time
    tInt = config['avgTime']
    nFrames = int(round(tInt*srate/vdif.DATA_LENGTH))
    tInt = nFrames*vdif.DATA_LENGTH/srate
    
    nFrames = int(round(tInt*srate/vdif.DATA_LENGTH))
    
    # Read in some data
    tFile = nFramesFile / beampols * vdif.DATA_LENGTH / srate
    
    # Date
    junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
    fh.seek(-vdif.FRAME_SIZE, 1)
    beginDate = junkFrame.time.datetime
        
    # Report
    print("Filename: %s" % os.path.basename(filename))
    print("  Date of First Frame: %s" % beginDate)
    print("  Station: %i" % beam)
    print("  Sample Rate: %i Hz" % srate)
    print("  Tuning 1: %.1f Hz" % cFreq)
    print("  Bit Depth: %i" % junkFrame.header.bits_per_sample)
    print("  Integration Time: %.3f s" % tInt)
    print("  Integrations in File: %i" % int(tFile/tInt))
    print(" ")

    # Go!
    data = numpy.zeros((beampols, vdif.DATA_LENGTH*nFrames), dtype=numpy.complex64)
    count = [0 for i in xrange(data.shape[0])]
    for i in xrange(beampols*nFrames):
        try:
            cFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
        except errors.SyncError:
            print("Error @ %i" % i)
            fh.seek(vdif.FRAME_SIZE, 1)
            continue
        std,pol = cFrame.id
        sid = pol
        
        data[sid, count[sid]*vdif.DATA_LENGTH:(count[sid]+1)*vdif.DATA_LENGTH] = cFrame.payload.data
        count[sid] += 1
        
    # Plot
    nBins = 2**junkFrame.header.bits_per_sample
    weights = numpy.ones(data.shape[1], dtype=numpy.float32) * 100.0/data.shape[1]
    
    fig = plt.figure()
    ## X
    ax = fig.add_subplot(2, 1, 1)
    c,b,o = ax.hist(data[0,:].real, bins=nBins, weights=weights)
    ax.set_xlim((b[0],b[-1]))
    ax.set_xlabel('Value - X')
    ax.set_ylabel('Fraction [%]')
    ## Y
    ax = fig.add_subplot(2, 1, 2)
    c,b,o = ax.hist(data[1,:].real, bins=nBins, weights=weights)
    ax.set_xlim((b[0],b[-1]))
    ax.set_xlabel('Value - Y')
    ax.set_ylabel('Fraction [%]')
    plt.show()
    
    # Done
    fh.close()


if __name__ == "__main__":
    main(sys.argv[1:])
    
