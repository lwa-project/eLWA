#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check the time times in a guppi file for flow.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

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
	header = readGUPPIHeader(fh)
	guppi.FrameSize = guppi.getFrameSize(fh)
	nThreads = guppi.getThreadCount(fh)
	nFramesFile = os.path.getsize(filename) / guppi.FrameSize
	nFramesSecond = guppi.getFramesPerSecond(fh)
	
	junkFrame = guppi.readFrame(fh)
	sampleRate = junkFrame.getSampleRate()
	guppi.DataLength = junkFrame.data.data.size
	nSampsFrame = guppi.DataLength
	station, thread = junkFrame.parseID()
	tunepols = nThreads
	beampols = tunepols
	fh.seek(-guppi.FrameSize, 1)
	
	# Store the information about the first frame and convert the timetag to 
	# an ephem.Date object.
	prevDate = ephem.Date(astro.unix_to_utcjd(junkFrame.getTime()) - astro.DJD_OFFSET)
	prevTime = junkFrame.header.offset / int(sampleRate)
	prevFrame = junkFrame.header.offset % int(sampleRate) / guppi.DataLength

	# Skip ahead
	fh.seek(int(skip*sampleRate/guppi.DataLength)*guppi.FrameSize, 1)
	
	# Report on the file
	print "Filename: %s" % os.path.basename(args[0])
	print "  Station: %i" % station
	print "  Thread count: %i" % nThreads
	print "  Date of first frame: %i -> %s" % (prevTime, str(prevDate))
	print "  Samples per frame: %i" % guppi.DataLength
	print "  Frames per second: %i" % nFramesSecond
	print "  Sample rate: %i Hz" % sampleRate
	print "  Bit Depth: %i" % junkFrame.header.bitsPerSample
	print " "
	if skip != 0:
		print "Skipping ahead %i frames (%.6f seconds)" % (int(skip*sampleRate/guppi.DataLength)*4, int(skip*sampleRate/guppi.DataLength)*guppi.DataLength/sampleRate)
		print " "
		
	prevDate = ['' for i in xrange(nThreads)]
	prevTime = [0 for i in xrange(nThreads)]
	prevFrame = [0 for i in xrange(nThreads)]
	for i in xrange(nThreads):
		currFrame = guppi.readFrame(fh)
		
		station, thread = currFrame.parseID()		
		prevDate[thread] = ephem.Date(astro.unix_to_utcjd(currFrame.getTime()) - astro.DJD_OFFSET)
		prevTime[thread] = currFrame.header.offset / int(sampleRate)
		prevFrame[thread] = currFrame.header.offset % int(sampleRate) / guppi.DataLength
		
	inError = False
	while True:
		try:
			currFrame = guppi.readFrame(fh)
			if inError:
				print "ERROR: sync. error cleared @ byte %i" % (fh.tell() - guppi.FrameSize,)
			inError = False
		except errors.syncError:
			if not inError:
				print "ERROR: invalid frame (sync. word error) @ byte %i" % fh.tell()
				inError = True
			fh.seek(guppi.FrameSize, 1)
			continue
		except errors.eofError:
			break
		
			
		station, thread = currFrame.parseID()
		currDate = ephem.Date(astro.unix_to_utcjd(currFrame.getTime()) - astro.DJD_OFFSET)
		print "->", thread, getBetterTime(currFrame), currFrame.header.offset % int(sampleRate) / guppi.DataLength
		currTime = currFrame.header.offset / int(sampleRate)
		currFrame = currFrame.header.offset % int(sampleRate) / guppi.DataLength
		
		if thread == 0 and currTime % 10 == 0 and currFrame == 0:
			print "station %i, thread %i: t.t. %i @ frame %i -> %s" % (station, thread, currTime, currFrame, currDate)
			
			
		deltaT = (currTime - prevTime[thread])*nFramesSecond + (currFrame - prevFrame[thread])
		#print station, thread, deltaT, fh.tell()

		if deltaT < 1:
			print "ERROR: t.t. %i @ frame %i < t.t. %i @ frame %i + 1" % (currTime, currFrame, prevTime[thread], prevFrame[thread])
			print "       -> difference: %i (%.5f seconds); %s" % (deltaT, deltaT*nSampsFrame/sampleRate, str(currDate))
			print "       -> station %i, thread %i" % (station, thread)
		elif deltaT > 1:
			print "ERROR: t.t. %i @ frame %i > t.t. %i @ frame %i + 1" % (currTime, currFrame, prevTime[thread], prevFrame[thread])
			print "       -> difference: %i (%.5f seconds); %s" % (deltaT, deltaT*nSampsFrame/sampleRate, str(currDate))
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
