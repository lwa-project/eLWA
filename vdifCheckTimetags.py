#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check the time times in a VDIF file for flow.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import sys
import ephem
import gc

from lsl import astro
from lsl.reader import vdif
from lsl.reader import errors
from lsl.common.dp import fS

from utils import *


def get_thread_count(filehandle):
    """
    Find out how many thrads are present by examining the first 1024
    records.  Return the number of threads found.
    """
    
    # Save the current position in the file so we can return to that point
    fhStart = filehandle.tell()
    
    # Make sure we have a frame size to work with
    try:
        vdif.FRAME_SIZE
    except AttributeError:
        vdif.FRAME_SIZE = vdif.get_frame_size(filehandle)
        
    # Build up the list-of-lists that store ID codes and loop through 1024
    # frames.  In each case, parse pull the thread ID and append the thread 
    # ID to the relevant thread array if it is not already there.
    threads = []
    i = 0
    while i < 1024:
        try:
            cFrame = vdif.read_frame(filehandle)
        except errors.SyncError:
            filehandle.seek(vdif.FRAME_SIZE, 1)
            continue
        except errors.EOFError:
            continue
            
        cID = cFrame.header.thread_id
        if cID not in threads:
            threads.append(cID)
        i += 1
        
    # Return to the place in the file where we started
    filehandle.seek(fhStart)
    
    # Return the number of threads found
    return len(threads)


def get_frames_per_second(filehandle):
    """
    Find out the number of frames per second in a file by watching how the 
    headers change.  Returns the number of frames in a second.
    """
    
    # Save the current position in the file so we can return to that point
    fhStart = filehandle.tell()
    
    # Get the number of threads
    nThreads = get_thread_count(filehandle)
    
    # Get the current second counts for all threads
    ref = {}
    i = 0
    while i < nThreads:
        try:
            cFrame = vdif.read_frame(filehandle)
        except errors.SyncError:
            filehandle.seek(vdif.FRAME_SIZE, 1)
            continue
        except EOFError:
            continue
            
        cID = cFrame.header.thread_id
        cSC = cFrame.header.seconds_from_epoch
        ref[cID] = cSC
        i += 1
        
    # Read frames until we see a change in the second counter
    cur = {}
    fnd = []
    while True:
        ## Get a frame
        try:
            cFrame = vdif.read_frame(filehandle)
        except errors.SyncError:
            filehandle.seek(vdif.FRAME_SIZE, 1)
            continue
        except EOFError:
            break
            
        ## Pull out the relevant metadata
        cID = cFrame.header.thread_id
        cSC = cFrame.header.seconds_from_epoch
        cFC = cFrame.header.frame_in_second
        
        ## Figure out what to do with it
        if cSC == ref[cID]:
            ### Same second as the reference, save the frame number
            cur[cID] = cFC
        else:
            ### Different second than the reference, we've found something
            ref[cID] = cSC
            if cID not in fnd:
                fnd.append( cID )
                
        if len(fnd) == nThreads:
            break
            
    # Return to the place in the file where we started
    filehandle.seek(fhStart)
    
    # Pull out the mode
    mode = {}
    for key,value in cur.iteritems():
        try:
            mode[value] += 1
        except KeyError:
            mode[value] = 1
    best, bestValue = 0, 0
    for key,value in mode.iteritems():
        if value > bestValue:
            best = key
            bestValue = value
            
    # Correct for a zero-based counter and return
    best += 1
    return best


def main(args):
    if args[0] == '-s':
        skip = float(args[1])
        filename = args[2]
    else:
        skip = 0
        filename = args[0]
        
    fh = open(filename, 'rb')
    header = vdif.read_guppi_header(fh)
    vdif.FRAME_SIZE = vdif.get_frame_size(fh)
    nThreads = get_thread_count(fh)
    nFramesFile = os.path.getsize(filename) / vdif.FRAME_SIZE
    nFramesSecond = get_frames_per_second(fh)
    
    junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
    sample_rate = junkFrame.sample_rate
    vdif.DataLength = junkFrame.data.data.size
    nSampsFrame = vdif.DataLength
    station, thread = junkFrame.id
    tunepols = nThreads
    beampols = tunepols
    fh.seek(-vdif.FRAME_SIZE, 1)
    
    # Store the information about the first frame and convert the timetag to 
    # an ephem.Date object.
    prevDate = ephem.Date(astro.unix_to_utcjd(junkFrame.get_time()) - astro.DJD_OFFSET)
    prevTime = junkFrame.header.seconds_from_epoch
    prevFrame = junkFrame.header.frame_in_second

    # Skip ahead
    fh.seek(int(skip*sample_rate/vdif.DataLength)*vdif.FRAME_SIZE, 1)
    
    # Report on the file
    print "Filename: %s" % os.path.basename(args[0])
    print "  Station: %i" % station
    print "  Thread count: %i" % nThreads
    print "  Date of first frame: %i -> %s" % (prevTime, str(prevDate))
    print "  Samples per frame: %i" % vdif.DataLength
    print "  Frames per second: %i" % nFramesSecond
    print "  Sample rate: %i Hz" % sample_rate
    print "  Bit Depth: %i" % junkFrame.header.bits_per_sample
    print " "
    if skip != 0:
        print "Skipping ahead %i frames (%.6f seconds)" % (int(skip*sample_rate/vdif.DataLength)*4, int(skip*sample_rate/vdif.DataLength)*vdif.DataLength/sample_rate)
        print " "
        
    prevDate = ['' for i in xrange(nThreads)]
    prevTime = [0 for i in xrange(nThreads)]
    prevFrame = [0 for i in xrange(nThreads)]
    for i in xrange(nThreads):
        currFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
        
        station, thread = currFrame.id		
        prevDate[thread] = ephem.Date(astro.unix_to_utcjd(currFrame.get_time()) - astro.DJD_OFFSET)
        prevTime[thread] = currFrame.header.seconds_from_epoch
        prevFrame[thread] = currFrame.header.frame_in_second
        
    inError = False
    while True:
        try:
            currFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
            if inError:
                print "ERROR: sync. error cleared @ byte %i" % (fh.tell() - vdif.FRAME_SIZE,)
            inError = False
        except errors.SyncError:
            if not inError:
                print "ERROR: invalid frame (sync. word error) @ byte %i" % fh.tell()
                inError = True
            fh.seek(vdif.FRAME_SIZE, 1)
            continue
        except errors.EOFError:
            break
        
            
        station, thread = currFrame.id
        currDate = ephem.Date(astro.unix_to_utcjd(currFrame.get_time()) - astro.DJD_OFFSET)
        currTime = currFrame.header.seconds_from_epoch
        currFrame = currFrame.header.frame_in_second
        
        if thread == 0 and currTime % 10 == 0 and currFrame == 0:
            print "station %i, thread %i: t.t. %i @ frame %i -> %s" % (station, thread, currTime, currFrame, currDate)
            
        deltaT = (currTime - prevTime[thread])*nFramesSecond + (currFrame - prevFrame[thread])
        #print station, thread, deltaT, fh.tell()

        if deltaT < 1:
            print "ERROR: t.t. %i @ frame %i < t.t. %i @ frame %i + 1" % (currTime, currFrame, prevTime[thread], prevFrame[thread])
            print "       -> difference: %i (%.5f seconds); %s" % (deltaT, deltaT*nSampsFrame/sample_rate, str(currDate))
            print "       -> station %i, thread %i" % (station, thread)
        elif deltaT > 1:
            print "ERROR: t.t. %i @ frame %i > t.t. %i @ frame %i + 1" % (currTime, currFrame, prevTime[thread], prevFrame[thread])
            print "       -> difference: %i (%.5f seconds); %s" % (deltaT, deltaT*nSampsFrame/sample_rate, str(currDate))
            print "       -> station %i, thread %i" % (station, thread)
        else:
            pass
        
        prevDate[thread] = currDate
        prevTime[thread] = currTime
        prevFrame[thread] = currFrame
        
        del currFrame
        
    fh.close()


if __name__ == "__main__":
    main(sys.argv[1:])
