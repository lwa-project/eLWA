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


def getThreadCount(filehandle):
	"""
	Find out how many thrads are present by examining the first 1024
	records.  Return the number of threads found.
	"""
	
	# Save the current position in the file so we can return to that point
	fhStart = filehandle.tell()
	
	# Make sure we have a frame size to work with
	try:
		vdif.FrameSize
	except AttributeError:
		vdif.FrameSize = vdif.getFrameSize(filehandle)
		
	# Build up the list-of-lists that store ID codes and loop through 1024
	# frames.  In each case, parse pull the thread ID and append the thread 
	# ID to the relevant thread array if it is not already there.
	threads = []
	i = 0
	while i < 1024:
		try:
			cFrame = vdif.readFrame(filehandle)
		except errors.syncError:
			filehandle.seek(vdif.FrameSize, 1)
			continue
		except errors.eofError:
			continue
			
		cID = cFrame.header.threadID
		if cID not in threads:
			threads.append(cID)
		i += 1
		
	# Return to the place in the file where we started
	filehandle.seek(fhStart)
	
	# Return the number of threads found
	return len(threads)


def getFramesPerSecond(filehandle):
	"""
	Find out the number of frames per second in a file by watching how the 
	headers change.  Returns the number of frames in a second.
	"""
	
	# Save the current position in the file so we can return to that point
	fhStart = filehandle.tell()
	
	# Get the number of threads
	nThreads = getThreadCount(filehandle)
	
	# Get the current second counts for all threads
	ref = {}
	i = 0
	while i < nThreads:
		try:
			cFrame = vdif.readFrame(filehandle)
		except errors.syncError:
			filehandle.seek(vdif.FrameSize, 1)
			continue
		except eofError:
			continue
			
		cID = cFrame.header.threadID
		cSC = cFrame.header.secondsFromEpoch
		ref[cID] = cSC
		i += 1
		
	# Read frames until we see a change in the second counter
	cur = {}
	fnd = []
	while True:
		## Get a frame
		try:
			cFrame = vdif.readFrame(filehandle)
		except errors.syncError:
			filehandle.seek(vdif.FrameSize, 1)
			continue
		except eofError:
			break
			
		## Pull out the relevant metadata
		cID = cFrame.header.threadID
		cSC = cFrame.header.secondsFromEpoch
		cFC = cFrame.header.frameInSecond
		
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
	header = readGUPPIHeader(fh)
	vdif.FrameSize = vdif.getFrameSize(fh)
	nThreads = getThreadCount(fh)
	nFramesFile = os.path.getsize(filename) / vdif.FrameSize
	nFramesSecond = getFramesPerSecond(fh)
	
	junkFrame = vdif.readFrame(fh, centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0)
	sampleRate = junkFrame.getSampleRate()
	vdif.DataLength = junkFrame.data.data.size
	nSampsFrame = vdif.DataLength
	station, thread = junkFrame.parseID()
	tunepols = nThreads
	beampols = tunepols
	fh.seek(-vdif.FrameSize, 1)
	
	# Store the information about the first frame and convert the timetag to 
	# an ephem.Date object.
	prevDate = ephem.Date(astro.unix_to_utcjd(junkFrame.getTime()) - astro.DJD_OFFSET)
	prevTime = junkFrame.header.secondsFromEpoch
	prevFrame = junkFrame.header.frameInSecond

	# Skip ahead
	fh.seek(int(skip*sampleRate/vdif.DataLength)*vdif.FrameSize, 1)
	
	# Report on the file
	print "Filename: %s" % os.path.basename(args[0])
	print "  Station: %i" % station
	print "  Thread count: %i" % nThreads
	print "  Date of first frame: %i -> %s" % (prevTime, str(prevDate))
	print "  Samples per frame: %i" % vdif.DataLength
	print "  Frames per second: %i" % nFramesSecond
	print "  Sample rate: %i Hz" % sampleRate
	print "  Bit Depth: %i" % junkFrame.header.bitsPerSample
	print " "
	if skip != 0:
		print "Skipping ahead %i frames (%.6f seconds)" % (int(skip*sampleRate/vdif.DataLength)*4, int(skip*sampleRate/vdif.DataLength)*vdif.DataLength/sampleRate)
		print " "
		
	prevDate = ['' for i in xrange(nThreads)]
	prevTime = [0 for i in xrange(nThreads)]
	prevFrame = [0 for i in xrange(nThreads)]
	for i in xrange(nThreads):
		currFrame = vdif.readFrame(fh, centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0)
		
		station, thread = currFrame.parseID()		
		prevDate[thread] = ephem.Date(astro.unix_to_utcjd(currFrame.getTime()) - astro.DJD_OFFSET)
		prevTime[thread] = currFrame.header.secondsFromEpoch
		prevFrame[thread] = currFrame.header.frameInSecond
		
	inError = False
	while True:
		try:
			currFrame = vdif.readFrame(fh, centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0)
			if inError:
				print "ERROR: sync. error cleared @ byte %i" % (fh.tell() - vdif.FrameSize,)
			inError = False
		except errors.syncError:
			if not inError:
				print "ERROR: invalid frame (sync. word error) @ byte %i" % fh.tell()
				inError = True
			fh.seek(vdif.FrameSize, 1)
			continue
		except errors.eofError:
			break
		
			
		station, thread = currFrame.parseID()
		currDate = ephem.Date(astro.unix_to_utcjd(currFrame.getTime()) - astro.DJD_OFFSET)
		currTime = currFrame.header.secondsFromEpoch
		currFrame = currFrame.header.frameInSecond
		
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
