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
from lsl.reader import vdif, errors

from utils import *


def usage(exitCode=None):
	print """vdifFileCheck.py - Run through a VDIF file and determine if it is bad or not.

Usage: vdifFileCheck.py [OPTIONS] filename

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
	header = readGUPPIHeader(fh)
	vdif.FrameSize = vdif.getFrameSize(fh)
	nFramesFile = os.path.getsize(filename) / vdif.FrameSize
	
	junkFrame = vdif.readFrame(fh, centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0)
	srate = junkFrame.getSampleRate()
	vdif.DataLength = junkFrame.data.data.size
	beam, pol = junkFrame.parseID()
	tunepols = vdif.getThreadCount(fh)
	beampols = tunepols
	
	# Get the frequencies
	cFreq = 0.0
	for j in xrange(4):
		junkFrame = vdif.readFrame(fh, centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0)
		s,p = junkFrame.parseID()
		if p == 0:
			cFreq = junkFrame.getCentralFreq()
			
	# Date
	junkFrame = vdif.readFrame(fh, centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0)
	fh.seek(-vdif.FrameSize, 1)
	beginDate = datetime.utcfromtimestamp(junkFrame.getTime())
		
	# Report
	print "Filename: %s" % os.path.basename(filename)
	print "  Date of First Frame: %s" % beginDate
	print "  Station: %i" % beam
	print "  Sample Rate: %i Hz" % srate
	print "  Bit Depth: %i" % junkFrame.header.bitsPerSample
	print "  Tuning 1: %.1f Hz" % cFreq
	print " "
	
	# Determine the clip level
	if config['trim'] is None:
		if junkFrame.header.bitsPerSample == 1:
			config['trim'] = abs(1.0)**2
		elif junkFrame.header.bitsPerSample == 2:
			config['trim'] = abs(3.3359)**2
		elif junkFrame.header.bitsPerSample == 4:
			config['trim'] = abs(7/2.95)**2
		elif junkFrame.header.bitsPerSample == 8:
			config['trim'] = abs(255/256.)**2
		else:
			config['trim'] = 1.0
		print "Setting clip level to %.3f" % config['trim']
		print " "
		
	# Convert chunk length to total frame count
	chunkLength = int(config['length'] * srate / vdif.DataLength * tunepols)
	chunkLength = int(1.0 * chunkLength / tunepols) * tunepols
	
	# Convert chunk skip to total frame count
	chunkSkip = int(config['skip'] * srate / vdif.DataLength * tunepols)
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
		data = numpy.empty((2,chunkLength*vdif.DataLength/tunepols), dtype=numpy.float32)
		for j in xrange(chunkLength):
			# Read in the next frame and anticipate any problems that could occur
			try:
				cFrame = vdif.readFrame(fh, centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0, Verbose=False)
			except errors.eofError:
				done = True
				break
			except errors.syncError:
				continue
			
			beam,pol = cFrame.parseID()
			aStand = pol
			
			try:
				data[aStand, count[aStand]*vdif.DataLength:(count[aStand]+1)*vdif.DataLength] = cFrame.data.data
				
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
			fh.seek(vdif.FrameSize*chunkSkip, 1)
			
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
	