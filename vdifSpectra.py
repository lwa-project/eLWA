#!/usr/bin/env python

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

from lsl.reader import vdif, errors
from lsl.correlator import fx as fxc
from lsl.misc import parser as aph

from utils import *

from matplotlib import pyplot as plt


def main(args):
    # Parse the command line
    filename = args.filename
    
    # Length of the FFT
    LFFT = args.fft_length
    
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
    
    if args.skip != 0:
        print("Skipping forward %.3f s" % args.skip)
        print("-> %.6f (%s)" % (junkFrame.time, junkFrame.time.datetime))
        
        offset = int(args.skip*srate / vdif.DATA_LENGTH)
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
    tInt = args.avg_time
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
        
    # Transform and trim off the negative frequencies
    freq, psd = fxc.SpecMaster(data, LFFT=2*LFFT, sample_rate=srate, central_freq=header['OBSFREQ']-srate/4)
    freq, psd = freq[LFFT:], psd[:,LFFT:]
    if is_vlite:
        psd = psd[:,::-1]
        
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
    parser = argparse.ArgumentParser(
        description='read in a VDIF file and plot spectra', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, 
                        help='filename to analyze')
    parser.add_argument('-l', '--fft-length', type=aph.positive_int, default=4096, 
                        help='set FFT length')
    parser.add_argument('-s', '--skip', type=aph.positive_or_zero_float, default=0.0, 
                        help='amount of time in seconds to skip into the file')
    parser.add_argument('-t', '--avg-time', type=aph.positive_float, default=1.0,
                        help='window to average over in seconds')
    args = parser.parse_args()
    main(args)
    