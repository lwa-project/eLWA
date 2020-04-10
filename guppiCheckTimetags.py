#!/usr/bin/env python

"""
Check the time in a GUPPI file for flow.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import os
import sys
import ephem
import gc

from lsl import astro
from lsl.reader import errors
from lsl.common.dp import fS

import guppi
from utils import *


def main(args):
    if args[0] == '-s':
        skip = float(args[1])
        filename = args[2]
    else:
        skip = 0
        filename = args[0]
        
    fh = open(filename, 'rb')
    header = guppi.read_guppi_header(fh)
    guppi.FRAME_SIZE = guppi.get_frame_size(fh)
    nThreads = guppi.get_thread_count(fh)
    nFramesFile = os.path.getsize(filename) / guppi.FRAME_SIZE
    nFramesSecond = guppi.get_frames_per_second(fh)
    
    junkFrame = guppi.read_frame(fh)
    sample_rate = junkFrame.sample_rate
    guppi.DataLength = junkFrame.data.data.size
    nSampsFrame = guppi.DataLength
    station, thread = junkFrame.id
    tunepols = nThreads
    beampols = tunepols
    fh.seek(-guppi.FRAME_SIZE, 1)
    
    # Store the information about the first frame and convert the timetag to 
    # an ephem.Date object.
    prevDate = ephem.Date(astro.unix_to_utcjd(junkFrame.get_time()) - astro.DJD_OFFSET)
    prevTime = junkFrame.header.offset / int(sample_rate)
    prevFrame = junkFrame.header.offset % int(sample_rate) / guppi.DataLength

    # Skip ahead
    fh.seek(int(skip*sample_rate/guppi.DataLength)*guppi.FRAME_SIZE, 1)
    
    # Report on the file
    print("Filename: %s" % os.path.basename(args[0]))
    print("  Station: %i" % station)
    print("  Thread count: %i" % nThreads)
    print("  Date of first frame: %i -> %s" % (prevTime, str(prevDate)))
    print("  Samples per frame: %i" % guppi.DataLength)
    print("  Frames per second: %i" % nFramesSecond)
    print("  Sample rate: %i Hz" % sample_rate)
    print("  Bit Depth: %i" % junkFrame.header.bits_per_sample)
    print(" ")
    if skip != 0:
        print("Skipping ahead %i frames (%.6f seconds)" % (int(skip*sample_rate/guppi.DataLength)*4, int(skip*sample_rate/guppi.DataLength)*guppi.DataLength/sample_rate))
        print(" ")
        
    prevDate = ['' for i in xrange(nThreads)]
    prevTime = [0 for i in xrange(nThreads)]
    prevFrame = [0 for i in xrange(nThreads)]
    for i in xrange(nThreads):
        currFrame = guppi.read_frame(fh)
        
        station, thread = currFrame.id		
        prevDate[thread] = ephem.Date(astro.unix_to_utcjd(currFrame.get_time()) - astro.DJD_OFFSET)
        prevTime[thread] = currFrame.header.offset / int(sample_rate)
        prevFrame[thread] = currFrame.header.offset % int(sample_rate) / guppi.DataLength
        
    inError = False
    while True:
        try:
            currFrame = guppi.read_frame(fh)
            if inError:
                print("ERROR: sync. error cleared @ byte %i" % (fh.tell() - guppi.FRAME_SIZE,))
            inError = False
        except errors.SyncError:
            if not inError:
                print("ERROR: invalid frame (sync. word error) @ byte %i" % fh.tell())
                inError = True
            fh.seek(guppi.FRAME_SIZE, 1)
            continue
        except errors.EOFError:
            break
        
            
        station, thread = currFrame.id
        currDate = ephem.Date(astro.unix_to_utcjd(currFrame.get_time()) - astro.DJD_OFFSET)
        print("->", thread, get_better_time(currFrame), currFrame.header.offset % int(sample_rate) / guppi.DataLength)
        currTime = currFrame.header.offset / int(sample_rate)
        currFrame = currFrame.header.offset % int(sample_rate) / guppi.DataLength
        
        if thread == 0 and currTime % 10 == 0 and currFrame == 0:
            print("station %i, thread %i: t.t. %i @ frame %i -> %s" % (station, thread, currTime, currFrame, currDate))
            
            
        deltaT = (currTime - prevTime[thread])*nFramesSecond + (currFrame - prevFrame[thread])
        #print(station, thread, deltaT, fh.tell())

        if deltaT < 1:
            print("ERROR: t.t. %i @ frame %i < t.t. %i @ frame %i + 1" % (currTime, currFrame, prevTime[thread], prevFrame[thread]))
            print("       -> difference: %i (%.5f seconds); %s" % (deltaT, deltaT*nSampsFrame/sample_rate, str(currDate)))
            print("       -> station %i, thread %i" % (station, thread))
        elif deltaT > 1:
            print("ERROR: t.t. %i @ frame %i > t.t. %i @ frame %i + 1" % (currTime, currFrame, prevTime[thread], prevFrame[thread]))
            print("       -> difference: %i (%.5f seconds); %s" % (deltaT, deltaT*nSampsFrame/sample_rate, str(currDate)))
            print("       -> station %i, thread %i" % (station, thread))
        else:
            pass
        
        prevDate[thread] = currDate
        prevTime[thread] = currTime
        prevFrame[thread] = currFrame
        
        del currFrame
        
    fh.close()


if __name__ == "__main__":
    main(sys.argv[1:])
