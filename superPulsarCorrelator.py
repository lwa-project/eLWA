#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Correlator for LWA and/or VLA data.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import re
import sys
import math
import time
import ephem
import numpy
import getopt
from datetime import datetime

from lsl import astro
from lsl.common import stations, metabundle
from lsl.statistics import robust
from lsl.correlator import uvUtils
from lsl.correlator import fx as fxc
from lsl.writer import fitsidi
from lsl.correlator.uvUtils import computeUVW
from lsl.common.constants import c as vLight

from lsl.reader import drx, vdif, errors
from lsl.reader.buffer import DRXFrameBuffer

import guppi
from lsl.misc.dedispersion import delay as dispDelay

import jones
from utils import *
from buffer import VDIFFrameBuffer


def usage(exitCode=None):
	print """superPulsarCorrelator.py - The next generation of correlator for pulsar
data

Usage:
superPulsarCorrelator.py [OPTIONS] <config_file>

Options:
-h, --help                  Display this help information
-q, --quiet                 Disable verbose time tag information
-l, --fft-length            Set FFT length (default = 512)
-s, --skip                  Amount of time in to skip into the files (seconds; 
                            default = 0 s)
-u, --subint-time           Sub-integration time for the data (seconds; 
                            default = 0.020 s)
-t, --dump-time             Correlator dump time for saving the visibilties
                            (seconds; default = 1 s)
-d, --duration              Duration of the file to correlate (seconds; 
                            default = 0 -> everything)
-g, --tag                   Tag to use for the output file (default = first eight
                            characters of the first input file)
-j, --jit                   Enable experimental just-in-time optimizations 
                            (default = no and you should keep it that way)
-w, --which                 For LWA-only observations, which tuning to use for
                            correlation (1 or 2; default = auto-select)
"""
	
	if exitCode is not None:
		sys.exit(exitCode)
	else:
		return True


def parseConfig(args):
	config = {}
	# Command line flags - default values
	config['verbose'] = True
	config['readTime'] = 1.0
	config['subTime'] = 0.020
	config['dumpTime'] = 1.0
	config['LFFT'] = 512
	config['skip'] = 0.0
	config['duration'] = 0.0
	config['tag'] = None
	config['withJIT'] = False
	config['which'] = None
	config['args'] = []
	
	# Read in and process the command line flags
	try:
		opts, args = getopt.getopt(args, "hql:u:t:s:d:g:jw:", ["help", "quiet", "fft-length=", "subint-time=", "dump-time=", "skip=", "duration=", "tag=", "jit", "which="])
	except getopt.GetoptError, err:
		# Print help information and exit:
		print str(err) # will print something like "option -a not recognized"
		usage(exitCode=2)
		
	# Work through opts
	for opt, value in opts:
		if opt in ('-h', '--help'):
			usage(exitCode=0)
		elif opt in ('-q', '--quiet'):
			config['verbose'] = False
		elif opt in ('-l', '--fft-length'):
			config['LFFT'] = int(value)
		elif opt in ('-r', '--read-time'):
			config['readTime'] = float(value)
		elif opt in ('-u', '--subint-time'):
			config['subTime'] = float(value)
		elif opt in ('-t', '--dump-time'):
			config['dumpTime'] = float(value)
		elif opt in ('-s', '--skip'):
			config['skip'] = float(value)
		elif opt in ('-d', '--duration'):
			config['duration'] = float(value)
		elif opt in ('-g', '--tag'):
			config['tag'] = value
		elif opt in ('-j', '--jit'):
			config['withJIT'] = True
		elif opt in ('-w', '--which'):
			config['which'] = int(value, 10)
		else:
			assert False
			
	# Add in arguments
	config['args'] = args
	
	# Validate
	if config['which'] is not None:
		if config['which'] not in (1, 2):
			raise RuntimeError("Invalid LWA tuning: %i" % config['which'])
			
	# Return configuration
	return config


def bestFreqUnits(freq):
	"""Given a numpy array of frequencies in Hz, return a new array with the
	frequencies in the best units possible (kHz, MHz, etc.)."""
	
	# Figure out how large the data are
	try:
		scale = int(math.log10(max(freq)))
	except TypeError:
		scale = int(math.log10(freq))
	if scale >= 9:
		divis = 1e9
		units = 'GHz'
	elif scale >= 6:
		divis = 1e6
		units = 'MHz'
	elif scale >= 3:
		divis = 1e3
		units = 'kHz'
	elif scale >= 0:
		divis = 1
		units = 'Hz'
	elif scale >= -3:
		divis = 1e-3
		units = 'mHz'
	else:
		divis = 1e-6
		units = 'uHz'
		
	# Convert the frequency
	newFreq = freq / divis
	
	# Return units and freq
	return (newFreq, units)


def main(args):
	# Parse the command line
	config = parseConfig(args)
	
	# Select the multirate module to use
	if config['withJIT']:
		from jit import multirate
	else:
		import multirate
		
	# Length of the FFT
	LFFT = config['LFFT']
	
	# Build up the station
	site = stations.lwa1
	## Updated 2017/1/17 with solution for the 12/3 run
	site.lat = 34.0687955336 * numpy.pi/180
	site.long = -107.628427469 * numpy.pi/180
	site.elev = 2129.62500168
	observer = site.getObserver()
	
	# Parse the correlator configuration
	refSrc, filenames, metanames, foffsets, readers, antennas = readCorrelatorConfiguration(config['args'][0])
	config['duration'] = refSrc.duration
	
	# Get the raw configuration
	fh = open(config['args'][0], 'r')
	rawConfig = fh.readlines()
	fh.close()
	
	# Antenna report
	print "Antennas:"
	for ant in antennas:
		print "  Antenna %i: Stand %i, Pol. %i (%.3f us offset)" % (ant.id, ant.stand.id, ant.pol, ant.cable.clockOffset*1e6)
		
	# Open and align the files
	fh = []
	nFramesFile = []
	srate = []
	beams = []
	tunepols = []
	beampols = []
	tStart = []
	cFreqs = []
	bitDepths = []
	delaySteps = []
	buffers = []
	grossOffsets = []
	for i,(filename,metaname,foffset) in enumerate(zip(filenames, metanames, foffsets)):
		fh.append( open(filename, "rb") )
		
		go = numpy.int32(antennas[2*i].cable.clockOffset)
		antennas[2*i+0].cable.clockOffset -= go
		antennas[2*i+1].cable.clockOffset -= go
		grossOffsets.append( go )
		if go != 0:
			print "Correcting time tags for gross offset of %i s" % grossOffsets[i]
			print "  Antenna clock offsets are now at %.3f us, %.3f us" % (antennas[2*i+0].cable.clockOffset*1e6, antennas[2*i+1].cable.clockOffset*1e6)
		
		if readers[i] in (vdif, guppi):
			header = readGUPPIHeader(fh[i])
			readers[i].FrameSize = readers[i].getFrameSize(fh[i])
			
		nFramesFile.append( os.path.getsize(filename) / readers[i].FrameSize )
		if readers[i] is vdif:
			junkFrame = readers[i].readFrame(fh[i], centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0)
			readers[i].DataLength = junkFrame.data.data.size
			beam, pol = junkFrame.parseID()
		elif readers[i] is guppi:
			junkFrame = readers[i].readFrame(fh[i])
			readers[i].DataLength = junkFrame.data.data.size
			beam, pol = junkFrame.parseID()
		elif readers[i] is drx:
			junkFrame = readers[i].readFrame(fh[i])
			readers[i].DataLength = junkFrame.data.iq.size
			beam, tune, pol = junkFrame.parseID()
		fh[i].seek(-readers[i].FrameSize, 1)
		
		beams.append( beam )
		srate.append( junkFrame.getSampleRate() )
		
		if readers[i] in (vdif, guppi):
			tunepols.append( readers[i].getThreadCount(fh[i]) )
			beampols.append( tunepols[i] )
		elif readers[i] is drx:
			beampols.append( 4 if antennas[2*i].stand.id == 51 else 4 )	# CHANGE THIS FOR SINGLE TUNING LWA-SV DATA
			
		skip = config['skip'] + foffset
		if skip != 0:
			print "Skipping forward %.3f s" % skip
			print "-> %.6f (%s)" % (junkFrame.getTime(), datetime.utcfromtimestamp(junkFrame.getTime()))
			
			offset = int(skip*srate[i] / readers[i].DataLength)
			fh[i].seek(beampols[i]*readers[i].FrameSize*offset, 1)
			if readers[i] is vdif:
				junkFrame = readers[i].readFrame(fh[i], centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0)
			else:
				junkFrame = readers[i].readFrame(fh[i])
			fh[i].seek(-readers[i].FrameSize, 1)
			
			print "-> %.6f (%s)" % (junkFrame.getTime(), datetime.utcfromtimestamp(junkFrame.getTime()))
			
		tStart.append( junkFrame.getTime() + grossOffsets[i] )
		
		# Get the frequencies
		cFreq1 = 0.0
		cFreq2 = 0.0
		for j in xrange(4):
			if readers[i] is vdif:
				junkFrame = readers[i].readFrame(fh[i], centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0)
				s,p = junkFrame.parseID()
				if p == 0:
					cFreq1 = junkFrame.getCentralFreq()
				else:
					pass
			elif readers[i] is guppi:
				junkFrame = readers[i].readFrame(fh[i])
				s,p = junkFrame.parseID()
				if p == 0:
					cFreq1 = junkFrame.getCentralFreq()
				else:
					pass
			elif readers[i] is drx:
				junkFrame = readers[i].readFrame(fh[i])
				b,t,p = junkFrame.parseID()
				if p == 0:
					if t == 1:
						cFreq1 = junkFrame.getCentralFreq()
					else:
						cFreq2 = junkFrame.getCentralFreq()
				else:
					pass
		fh[i].seek(-4*readers[i].FrameSize, 1)
		cFreqs.append( [cFreq1,cFreq2] )
		try:
			bitDepths.append( junkFrame.header.bitsPerSample )
		except AttributeError:
			bitDepths.append( 8 )
			
		# Parse the metadata to get the delay steps
		delayStep = None
		if readers[i] is drx and metaname is not None:
			delayStep = parseLWAMetaData(metaname)
		delaySteps.append( delayStep )
		
		# Setup the frame buffers
		if readers[i] is vdif:
			buffers.append( VDIFFrameBuffer(threads=[0,1]) )
		elif readers[i] is guppi:
			buffers.append( GUPPIFrameBuffer(threads=[0,1]) )
		elif readers[i] is drx:
			if beampols[i] == 4:
				buffers.append( DRXFrameBuffer(beams=[beam,], tunes=[1,2], pols=[0,1]) )
			else:
				buffers.append( DRXFrameBuffer(beams=[beam,], tunes=[1,], pols=[0,1]) )
	for i in xrange(len(filenames)):
		# Align the files as close as possible by the time tags
		if readers[i] is vdif:
			timeTags = []
			for k in xrange(16):
				junkFrame = readers[i].readFrame(fh[i])
				timeTags.append(junkFrame.header.frameInSecond)
			fh[i].seek(-16*readers[i].FrameSize, 1)
			
			j = 0
			while (timeTags[j+0] != timeTags[j+1]):
				j += 1
				fh[i].seek(readers[i].FrameSize, 1)
			
			nFramesFile[i] -= j
			
		elif readers[i] is guppi:
			pass
			
		elif readers[i] is drx:
			timeTags = []
			for k in xrange(16):
				junkFrame = readers[i].readFrame(fh[i])
				timeTags.append(junkFrame.data.timeTag)
			fh[i].seek(-16*readers[i].FrameSize, 1)
			
			j = 0
			if beampols[i] == 4:
				while (timeTags[j+0] != timeTags[j+1]) or (timeTags[j+0] != timeTags[j+2]) or (timeTags[j+0] != timeTags[j+3]):
					j += 1
					fh[i].seek(readers[i].FrameSize, 1)
			else:
				while (timeTags[j+0] != timeTags[j+1]):
					j += 1
					fh[i].seek(readers[i].FrameSize, 1)
					
			nFramesFile[i] -= j
			
		# Align the files as close as possible by the time tags
		if readers[i] is vdif:
			junkFrame = readers[i].readFrame(fh[i], centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0)
		else:
			junkFrame = readers[i].readFrame(fh[i])
		fh[i].seek(-readers[i].FrameSize, 1)
			
		j = 0
		while junkFrame.getTime() + grossOffsets[i] < max(tStart):
			if readers[i] is vdif:
				for k in xrange(beampols[i]):
					junkFrame = readers[i].readFrame(fh[i], centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0)
			else:
				for k in xrange(beampols[i]):
					junkFrame = readers[i].readFrame(fh[i])
			j += beampols[i]
			
		jTime = j*readers[i].DataLength/srate[i]/beampols[i]
		print "Shifted beam %i data by %i frames (%.4f s)" % (beams[i], j, jTime)
		
	# Set integration time
	tRead = config['readTime']
	nFrames = int(round(tRead*srate[-1]/readers[-1].DataLength))
	tRead = nFrames*readers[-1].DataLength/srate[-1]
	
	nFramesV = tRead*srate[0]/readers[0].DataLength
	nFramesD = nFrames
	while nFramesV != int(nFramesV):
		nFrames += 1
		tRead = nFrames*readers[-1].DataLength/srate[-1]
		
		nFramesV = tRead*srate[0]/readers[0].DataLength
		nFramesD = nFrames
	nFramesV = int(nFramesV)
	
	# Read in some data
	tFileV = nFramesFile[ 0] / beampols[ 0] * readers[ 0].DataLength / srate[ 0]
	tFileD = nFramesFile[-1] / beampols[-1] * readers[-1].DataLength / srate[-1]
	tFile = min([tFileV, tFileD])
	if config['duration'] > 0.0:
		duration = config['duration']
		duration = tRead * int(round(duration / tRead))
		tFile = duration
		
	# Date
	beginMJDs = []
	beginDates = []
	for i in xrange(len(filenames)):
		if readers[i] is vdif:
			junkFrame = readers[i].readFrame(fh[i], centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0)
		else:
			junkFrame = readers[i].readFrame(fh[i])
		fh[i].seek(-readers[i].FrameSize, 1)
		
		beginMJDs.append( astro.unix_to_utcjd(junkFrame.getTime()) - astro.MJD_OFFSET)
		beginDates.append( datetime.utcfromtimestamp(junkFrame.getTime()) )
		
	# Set the output base filename
	if config['tag'] is None:
		outbase = os.path.basename(filenames[0])
		outbase = os.path.splitext(outbase)[0][:8]
	else:
		outbase = config['tag']
		
	# Report
	for i in xrange(len(filenames)):
		print "Filename: %s" % os.path.basename(filenames[i])
		print "  Type/Reader: %s" % readers[i].__name__
		print "  Delay Steps Avaliable: %s" % ('No' if delaySteps[i] is None else 'Yes',)
		print "  Date of First Frame: %s" % beginDates[i]
		print "  Sample Rate: %i Hz" % srate[i]
		print "  Tuning 1: %.3f Hz" % cFreqs[i][0]
		print "  Tuning 2: %.3f Hz" % cFreqs[i][1]
		print "  Bit Depth: %i" % bitDepths[i]
	print "  ==="
	print "  Phase Center:"
	print "    Name: %s" % refSrc.name
	print "    RA: %s" % refSrc._ra
	print "    Dec: %s" % refSrc._dec
	print "  ==="
	print "  Data Read Time: %.3f s" % tRead
	print "  Data Reads in File: %i" % int(tFile/tRead)
	print " "
	
	nVDIFInputs = sum([1 for reader in readers if reader is vdif]) + sum([1 for reader in readers if reader is guppi])
	nDRXInputs = sum([1 for reader in readers if reader is drx])
	print "Processing %i VDIF and %i DRX input streams" % (nVDIFInputs, nDRXInputs)
	print " "
	
	nFramesV = int(round(tRead*srate[0]/readers[0].DataLength))
	framesPerSecondV = int(srate[0] / readers[0].DataLength)
	nFramesB = nFrames
	framesPerSecondB = srate[-1] / readers[-1].DataLength
	print "VDIF Frames/s: %.6f" % framesPerSecondV
	print "VDIF Frames/Integration: %i" % nFramesV
	print "DRX Frames/s: %.6f" % framesPerSecondB
	print "DRX Frames/Integration: %i" % nFramesB
	print "Sample Count Ratio: %.6f" % (1.0*(nFramesV*readers[0].DataLength)/(nFramesB*4096),)
	print "Sample Rate Ratio: %.6f" % (srate[0]/srate[-1],)
	print " "
	
	vdifLFFT = LFFT * 2
	drxLFFT = vdifLFFT * srate[-1] / srate[0]
	while drxLFFT != int(drxLFFT):
		vdifLFFT += 1
		drxLFFT = vdifLFFT * srate[-1] / srate[0]
	drxLFFT = int(drxLFFT)
	print "VDIF Transform Size: %i" % vdifLFFT
	print "DRX Transform Size: %i" % drxLFFT
	print " "
	
	vdifPivot = 1
	if abs(cFreqs[0][0] - cFreqs[-1][1]) < 10:
		vdifPivot = 2
	if nVDIFInputs == 0 and config['which'] is not None:
		vdifPivot = config['which']
	print "VDIF appears to correspond to tuning #%i in DRX" % vdifPivot
	print " "
	
	nChunks = int(tFile/tRead)
	tSub = config['subTime']
	tSub = tRead / int(round(tRead/tSub))
	tDump = config['dumpTime']
	tDump = tSub * int(round(tDump/tSub))
	nDump = int(tDump / tSub)
	tDump = nDump * tSub
	nInt = int((nChunks*tRead) / tDump)
	print "Sub-integration time is: %.3f s" % tSub
	print "Integration (dump) time is: %.3f s" % tDump
	print " "
	
	# Solve for the pulsar binning
	observer.date = beginMJDs[0] + astro.MJD_OFFSET - astro.DJD_OFFSET
	refSrc.compute(observer)
	pulsarPeriod = refSrc.period
	nProfileBins = int(pulsarPeriod / tSub)
	nProfileBins = min([nProfileBins, 64])
	profileBins = numpy.linspace(0, 1+1.0/nProfileBins, nProfileBins+2)
	profileBins -= (profileBins[1]-profileBins[0])/2.0
	print "Pulsar frequency: %.6f Hz" % refSrc.frequency
	print "Pulsar period: %.6s seconds" % pulsarPeriod
	print "Number of profile bins:  %i" % nProfileBins
	print "Phase coverage per bin: %.3f" % (profileBins[1]-profileBins[0],)
	print " "
	
	subIntTimes = [[] for i in xrange(nProfileBins)]
	subIntCount = [0 for i in xrange(nProfileBins)]
	subIntWeight = [0 for i in xrange(nProfileBins)]
	fileCount   = [0 for i in xrange(nProfileBins)]
	visXX = [0 for i in xrange(nProfileBins)]
	visXY = [0 for i in xrange(nProfileBins)]
	visYX = [0 for i in xrange(nProfileBins)]
	visYY = [0 for i in xrange(nProfileBins)]
	wallStart = time.time()
	done = False
	firstPass = True
	for i in xrange(nChunks):
		wallTime = time.time()
		
		tStart = []
		tStartB = []
		
		vdifRef = [0 for j in xrange(nVDIFInputs*2)]
		drxRef  = [0 for j in xrange(nDRXInputs*4) ]
		
		# Read in the data
		dataV = numpy.zeros((len(vdifRef), readers[ 0].DataLength*nFramesV), dtype=numpy.complex64)
		dataD = numpy.zeros((len(drxRef),  readers[-1].DataLength*nFramesD), dtype=numpy.complex64)
		for j,f in enumerate(fh):
			if readers[j] is vdif:
				## VDIF
				k = 0
				while k < beampols[j]*nFramesV:
					try:
						cFrame = readers[j].readFrame(f, centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0)
						buffers[j].append( cFrame )
					except errors.syncError:
						print "Error @ %i, %i" % (i, j)
						f.seek(vdif.FrameSize, 1)
						continue
					except errors.eofError:
						done = True
						break
						
					frames = buffers[j].get()
					if frames is None:
						continue
						
					for cFrame in frames:
						std,pol = cFrame.parseID()
						sid = 2*j + pol
						
						if k == 0:
							tStart.append( cFrame.getTime() )
							tStart[-1] += grossOffsets[j]
							tStartB.append( getBetterTime(cFrame) )
							tStartB[-1][0] += grossOffsets[j]
							
							for p in (0,1):
								psid = 2*j + p
								vdifRef[psid] = cFrame.header.secondsFromEpoch*framesPerSecondV + cFrame.header.frameInSecond
								
						count = cFrame.header.secondsFromEpoch*framesPerSecondV + cFrame.header.frameInSecond
						count -= vdifRef[sid]
						dataV[sid, count*readers[j].DataLength:(count+1)*readers[j].DataLength] = cFrame.data.data
						k += 1
						
			elif readers[j] is guppi:
				## GUPPI
				k = 0
				while k < beampols[j]*nFramesV:
					try:
						cFrame = readers[j].readFrame(f)
						buffers[j].append( cFrame )
					except errors.syncError:
						print "Error @ %i, %i" % (i, j)
						continue
					except errors.eofError:
						done = True
						break
						
					frames = buffers[j].get()
					if frames is None:
						continue
						
					for cFrame in frames:
						std,pol = cFrame.parseID()
						sid = 2*j + pol
						
						if k == 0:
							tStart.append( cFrame.getTime() )
							tStart[-1] += grossOffsets[j]
							tStartB.append( getBetterTime(cFrame) )
							tStartB[-1][0] += grossOffsets[j]
							
							for p in (0,1):
								psid = 2*j + p
								vdifRef[psid] = cFrame.header.offset / readers[j].DataLength
								
						count = cFrame.header.offset / readers[j].DataLength
						count -= vdifRef[sid]
						dataV[sid, count*readers[j].DataLength:(count+1)*readers[j].DataLength] = cFrame.data.data
						k += 1
						
			elif readers[j] is drx:
				## DRX
				k = 0
				while k < beampols[j]*nFramesD:
					try:
						cFrame = readers[j].readFrame(f)
						buffers[j].append( cFrame )
					except errors.eofError:
						done = True
						break
						
					frames = buffers[j].get()
					if frames is None:
						continue
						
					for cFrame in frames:
						beam,tune,pol = cFrame.parseID()
						bid = 4*(j-nVDIFInputs) + 2*(tune-1) + pol
						
						if k == 0:
							tStart.append( cFrame.getTime() )
							tStart[-1] += grossOffsets[j]
							tStartB.append( getBetterTime(cFrame) )
							tStartB[-1][0] += grossOffsets[j]
							
							for t in (1,2):
								for p in (0,1):
									pbid = 4*(j-nVDIFInputs) + 2*(t-1) + p
									drxRef[pbid] = cFrame.data.timeTag
									
						count = cFrame.data.timeTag
						count -= drxRef[bid]
						count /= (4096*int(196e6/srate[-1]))
						dataD[bid, count*readers[j].DataLength:(count+1)*readers[j].DataLength] = cFrame.data.iq
						k += 1
		if done:
			break
			
		# Figure out which DRX tuning corresponds to the VDIF data
		if nDRXInputs > 0:
			sel = []
			for l in xrange(4*nDRXInputs):
				if vdifPivot == 1 and (l/2)%2 == 0:
					sel.append( l )
				elif vdifPivot == 2 and (l/2)%2 == 1:
					sel.append( l )
			dataD = dataD[sel,:]
			dataD /= 7.0
			
		# Time tag alignment (sample based)
		## Initial time tags for each stream and the relative start time for each stream
		if config['verbose']:
			### TT = time tag
			print 'TT - Start', tStartB
		tStartMin = min([sec for sec,frac in tStartB])
		tStartRel = [(sec-tStartMin)+frac for sec,frac in tStartB]
		
		## Sample offsets between the streams
		offsets = []
		for j in xrange(nVDIFInputs+nDRXInputs):
			offsets.append( int( math.floor((max(tStartRel) - tStartRel[j])*srate[j]) ) )
		if config['verbose']:
			print 'TT - Offsets', offsets
			
		## Roll the data to apply the sample offsets and then trim the ends to get rid 
		## of the rolled part
		for j,offset in enumerate(offsets):
			if j < nVDIFInputs:
				if offset != 0:
					idx0 = 2*j + 0
					idx1 = 2*j + 1
					tStart[j] += offset/(srate[j])
					tStartB[j][1] += offset/(srate[j])
					dataV[idx0,:] = numpy.roll(dataV[idx0,:], -offset)
					dataV[idx1,:] = numpy.roll(dataV[idx1,:], -offset)
					
			else:
				if offset != 0:
					idx0 = 2*(j - nVDIFInputs) + 0
					idx1 = 2*(j - nVDIFInputs) + 1
					tStart[j] += offset/(srate[j])
					tStartB[j][1] += offset/(srate[j])
					dataD[idx0,:] = numpy.roll(dataD[idx0,:], -offset)
					dataD[idx1,:] = numpy.roll(dataD[idx1,:], -offset)
					
		vdifOffsets = offsets[:nVDIFInputs]
		if nVDIFInputs > 0:
			if max(vdifOffsets) != 0:
				dataV = dataV[:,:-max(vdifOffsets)]
				
		drxOffsets = offsets[nVDIFInputs:]
		if nDRXInputs > 0:
			if max(drxOffsets) != 0:
				dataD = dataD[:,:-max(drxOffsets)]
				
		## Apply the corrections to the original time tags and report on the sub-sample
		## residuals
		if config['verbose']:
			print 'TT - Adjusted', tStartB
		tStartMinSec  = min([sec  for sec,frac in tStartB])
		tStartMinFrac = min([frac for sec,frac in tStartB])
		tStartRel = [(sec-tStartMinSec)+(frac-tStartMinFrac) for sec,frac in tStartB]
		if config['verbose']:
			print 'TT - Residual', ["%.1f ns" % (r*1e9,) for r in tStartRel]
		if firstPass:
			for k in xrange(len(tStartRel)):
				antennas[2*k+0].cable.clockOffset += tStartRel[k]
				antennas[2*k+1].cable.clockOffset += tStartRel[k]
			firstPass = False
			
		# Setup everything we need to loop through the sub-integrations
		nSub = int(tRead/tSub)
		nSampV = int(srate[ 0]*tSub)
		nSampD = int(srate[-1]*tSub)
		
		#tV = i*tRead + numpy.arange(dataV.shape[1], dtype=numpy.float64)/srate[ 0]
		tD = i*tRead + numpy.arange(dataD.shape[1], dtype=numpy.float64)/srate[-1]
		
		# Loop over sub-integrations
		for j in xrange(nSub):
			## Select the data to work with
			tSubInt = tStart[0] + (j+1)*nSampV/srate[0] - nSampV/2/srate[0]
			#tVSub    = tV[j*nSampV:(j+1)*nSampV]
			tDSub    = tD[j*nSampD:(j+1)*nSampD]
			dataVSub = dataV[:,j*nSampV:(j+1)*nSampV]
			dataDSub = dataD[:,j*nSampD:(j+1)*nSampD]
			
			## Update antennas for any delay steps
			for k in xrange(len(delaySteps)):
				if delaySteps[k] is None:
					## Skip over null entries
					continue
				elif delaySteps[k][0][0] > tStart[0]:
					## Skip over antennas where the next step is in the future
					continue
					
				## Find the next step
				nextStep = numpy.where( (tStart[0] - delaySteps[k][0]) >= 0.0 )[0][0]
				step = delaySteps[k][1][nextStep]
				if step != 0.0:
					## Report on the delay step
					print "DS - Applying delay step of %.3f ns to antenna %i" % (step*1e9, antennas[2*k+0].stand.id)
					print "DS - Step corresponds to %.1f deg at band center" % (360*cFreqs[0][0]*step,)
					## Apply the step
					antennas[2*k+0].cable.clockOffset += step
					antennas[2*k+1].cable.clockOffset += step
				## Clenup so we don't re-apply the step at the next iteration
				if nextStep+1 < delaySteps[k][0].size:
					### There are still more we can apply
					delaySteps[k] = (delaySteps[k][0][nextStep+1:], delaySteps[k][1][nextStep+1:])
				else:
					### There is not left to apply
					delaySteps[k] = None
					
			## Update the observation
			observer.date = astro.unix_to_utcjd(tSubInt) - astro.DJD_OFFSET
			refSrc.compute(observer)
			
			## Get the Jones matrices and apply
			## NOTE: This moves the LWA into the frame of the VLA
			if nDRXInputs > 0:
				lwaToSky = jones.getMatrixLWA(site, refSrc)
				dataDSub = jones.applyMatrix(dataDSub, lwaToSky)
				skyToVLA = jones.getMatrixVLA(site, refSrc, inverse=True)
				dataDSub = jones.applyMatrix(dataDSub, skyToVLA)
				
			## Correlate
			if nVDIFInputs > 0:
				freqV, feoV, veoV, deoV = multirate.MRF(dataVSub, antennas[:2*nVDIFInputs], LFFT=vdifLFFT,
											SampleRate=srate[0], CentralFreq=cFreqs[0][0]-srate[0]/4,
											Pol='*', phaseCenter=refSrc)
											
			if nDRXInputs > 0:
				freqD, feoD, veoD, deoD = multirate.MRF(dataDSub, antennas[2*nVDIFInputs:], LFFT=drxLFFT,
											SampleRate=srate[-1], CentralFreq=cFreqs[-1][vdifPivot-1], 
											Pol='*', phaseCenter=refSrc)
			
			## Rotate the phase in time to deal with frequency offset between the VLA and LWA
			if nDRXInputs*nVDIFInputs > 0:
				subChanFreqOffset = (cFreqs[0][0]-cFreqs[-1][vdifPivot-1]) % (freqD[1]-freqD[0])
				
				if i == 0 and j == 0:
					## FC = frequency correction
					tv,tu = bestFreqUnits(subChanFreqOffset)
					print "FC - Applying fringe rotation rate of %.3f %s to the DRX data" % (tv,tu)
					
				for s in xrange(feoD.shape[0]):
					for w in xrange(feoD.shape[2]):
						feoD[s,:,w] *= numpy.exp(-2j*numpy.pi*subChanFreqOffset*tDSub[w*drxLFFT])
						
			## Sort out what goes where (channels and antennas) if we don't already know
			try:
				if nVDIFInputs > 0:
					freqV = freqV[goodV]
					feoV = feoV[:,goodV,:]
				if nDRXInputs > 0:
					freqD = freqD[goodD]
					feoD = feoD[:,goodD,:]
					
			except NameError:
				### Frequency overlap
				fMin, fMax = -1e12, 1e12
				if nVDIFInputs > 0:
					fMin, fMax = max([fMin, freqV[vdifLFFT/2:].min()]), min([fMax, freqV[vdifLFFT/2:].max()])
				if nDRXInputs > 0:
					fMin, fMax = max([fMin, freqD.min()]), min([fMax, freqD.max()])
					
				### Channels and antennas (X vs. Y)
				if nVDIFInputs > 0:
					goodV = numpy.where( (freqV >= fMin) & (freqV <= fMax) )[0]
					aXV = [k for (k,a) in enumerate(antennas[:2*nVDIFInputs]) if a.pol == 0]
					aYV = [k for (k,a) in enumerate(antennas[:2*nVDIFInputs]) if a.pol == 1]
				if nDRXInputs > 0:
					goodD = numpy.where( (freqD >= fMin) & (freqD <= fMax) )[0]
					aXD = [k for (k,a) in enumerate(antennas[2*nVDIFInputs:]) if a.pol == 0]
					aYD = [k for (k,a) in enumerate(antennas[2*nVDIFInputs:]) if a.pol == 1]
					
				### Validate the channel alignent and fix it if needed
				if nVDIFInputs*nDRXInputs != 0:
					pd = freqV[goodV[0]] - freqD[goodD[0]]
					# Need to shift?
					if abs(pd) >= 1.01*abs(subChanFreqOffset):
						## Need to shift
						if pd < 0.0:
							goodV = goodV[1:]
						else:
							goodD = goodD[1:]
							
					# Need to trim?
					if len(goodV) > len(goodD):
						## Yes, goodV is too long
						goodV = goodV[:len(goodD)]
					elif len(goodD) > len(goodV):
						## Yes, goodD is too long
						goodD = goodD[:len(goodV)]
					else:
						## No, nothing needs to be done
						pass
						
					# Validate
					fd = freqV[goodV] - freqD[goodD]
					try:
						assert(fd.min() >= 0.99*subChanFreqOffset)
						assert(fd.max() <= 1.01*subChanFreqOffset)
						
						## FS = frequency selection
						tv,tu = bestFreqUnits(freqV[1]-freqV[0])
						print "FS - Found %i, %.3f %s overalapping channels" % (len(goodV), tv, tu)
						tv,tu = bestFreqUnits(freqV[goodV[-1]]-freqV[goodV[0]])
						print "FS - Bandwidth is %.3f %s" % (tv, tu)
						print "FS - Channels span %.3f MHz to %.3f MHz" % (freqV[goodV[0]]/1e6, freqV[goodV[-1]]/1e6)
							
					except AssertionError:
						raise RuntimeError("Cannot find a common frequency set between the input data: offsets range between %.3f Hz and %.3f Hz, expected %.3f Hz" % (fd.min(), fd.max(), subChanFreqOffset))
						
				### Apply
				if nVDIFInputs > 0:
					freqV = freqV[goodV]
					feoV = feoV[:,goodV,:]
				if nDRXInputs > 0:
					freqD = freqD[goodD]
					feoD = feoD[:,goodD,:]
			try:
				nChan = feoV.shape[1]
				fdt = feoV.dtype
				vdt = veoV.dtype
			except NameError:
				nChan = feoD.shape[1]
				fdt = feoD.dtype
				vdt = veoD.dtype
			## Setup the intermediate F-engine products and trim the data
			### Figure out the minimum number of windows
			nWin = 1e12
			if nVDIFInputs > 0:
				nWin = min([nWin, feoV.shape[2]])
			if nDRXInputs > 0:
				nWin = min([nWin, feoD.shape[2]])
				
			### Initialize the intermediate arrays
			feoX = numpy.zeros((nVDIFInputs+nDRXInputs, nChan, nWin), dtype=fdt)
			feoY = numpy.zeros((nVDIFInputs+nDRXInputs, nChan, nWin), dtype=fdt)
			veoX = numpy.zeros((nVDIFInputs+nDRXInputs, nWin), dtype=vdt)
			veoY = numpy.zeros((nVDIFInputs+nDRXInputs, nWin), dtype=vdt)
			
			### Trim
			if nVDIFInputs > 0:
				feoV = feoV[:,:,:nWin]
				veoV = veoV[:,:nWin]
			if nDRXInputs > 0:
				feoD = feoD[:,:,:nWin]
				veoD = veoD[:,:nWin]
				
			## Sort it all out by polarization
			if nVDIFInputs > 0:
				feoX[:nVDIFInputs,:,:] = feoV[aXV,:,:]
				feoY[:nVDIFInputs,:,:] = feoV[aYV,:,:]
				veoX[:nVDIFInputs,:] = veoV[aXV,:]
				veoY[:nVDIFInputs,:] = veoV[aYV,:]
			if nDRXInputs > 0:
				feoX[nVDIFInputs:,:,:] = feoD[aXD,:,:]
				feoY[nVDIFInputs:,:,:] = feoD[aYD,:,:]
				veoX[nVDIFInputs:,:] = veoD[aXD,:]
				veoY[nVDIFInputs:,:] = veoD[aYD,:]
				
			## Cross multiply
			try:
				sfreqXX = freqV
				sfreqYY = freqV
			except NameError:
				sfreqXX = freqD
				sfreqYY = freqD
			svisXX = multirate.MRX(feoX, veoX, feoX, veoX)
			svisXY = multirate.MRX(feoX, veoX, feoY, veoY)
			svisYX = multirate.MRX(feoY, veoY, feoX, veoX)
			svisYY = multirate.MRX(feoY, veoY, feoY, veoY)
			
			# Determine the pulsar phase as a function of frequency
			currentPeriod = refSrc.period
			## Dispersion
			try:
				phaseDispersion = tDisp / currentPeriod
			except NameError:
				tDisp = dispDelay(sfreqXX, refSrc.dm)
				phaseDispersion = tDisp / currentPeriod
			phaseDispersion -= numpy.floor(phaseDispersion)
			## Folding
			phaseProfile = refSrc.phase
			phaseProfile -= int(phaseProfile)
			## Combined
			profilePhase = (phaseProfile - phaseDispersion) % 1
			
			## Map the phases to bins
			bestBins = {}
			for b,phs in enumerate(profilePhase):
				bestBin = numpy.where( phs >= profileBins )[0][-1] % nProfileBins
				try:
					bestBins[bestBin].append( b )
				except KeyError:
					bestBins[bestBin] = [b,]
					
			summary = [None for i in profileBins[:-2]]
			for bestBin in bestBins:
				summary[bestBin] = (len(bestBins[bestBin]), subIntCount[bestBin])
			print summary
			
			### Accumulate
			for bestBin in bestBins:
				bestFreq = bestBins[bestBin]
				if subIntCount[bestBin] == 0:
					subIntTimes[bestBin] = []
					freqXX = sfreqXX
					freqYY = sfreqYY
					try:
						okToZero = numpy.where( subIntWeight[bestBin] ==  subIntWeight[bestBin].max() )[0]
						subIntWeight[bestBin] *= 0
					except AttributeError:
						subIntWeight[bestBin] = numpy.zeros(freqXX.size)
						
					visXX[bestBin] = svisXX*0.0
					visXY[bestBin] = svisXY*0.0
					visYX[bestBin] = svisYX*0.0
					visYY[bestBin] = svisYY*0.0
					
				subIntTimes[bestBin].append( tSubInt )
				visXX[bestBin][:,bestFreq] += svisXX[:,bestFreq] / nDump
				visXY[bestBin][:,bestFreq] += svisXY[:,bestFreq] / nDump
				visYX[bestBin][:,bestFreq] += svisYX[:,bestFreq] / nDump
				visYY[bestBin][:,bestFreq] += svisYY[:,bestFreq] / nDump
				subIntCount[bestBin] += 1
				subIntWeight[bestBin][bestFreq] += 1
			
			## Save
			for bestBin in bestBins:
				if subIntCount[bestBin] == nDump:
					subIntCount[bestBin] = 0
					fileCount[bestBin] += 1
					
					visXX[bestBin] *= nDump / subIntWeight[bestBin]
					visXY[bestBin] *= nDump / subIntWeight[bestBin]
					visYX[bestBin] *= nDump / subIntWeight[bestBin]
					visYY[bestBin] *= nDump / subIntWeight[bestBin]
					
					outfile = "%s-vis2-bin%03i-%05i.npz" % (outbase, bestBin, fileCount[bestBin])
					numpy.savez(outfile, config=rawConfig, srate=srate[0]/2.0, freq1=freqXX, 
								vis1XX=visXX[bestBin], vis1XY=visXY[bestBin], 
								vis1YX=visYX[bestBin], vis1YY=visYY[bestBin], 
								tStart=numpy.mean(subIntTimes[bestBin]), tInt=tDump)
					### CD = correlator dump
					print "CD - writing integration %i, bin %i to disk, timestamp is %.3f s" % (fileCount[bestBin], bestBin, numpy.mean(subIntTimes[bestBin]))
					if bestBin == 0:
						if fileCount[0] == 1:
							print "CD - each integration is %.1f MB on disk" % (os.path.getsize(outfile)/1024.0**2,)
						if (fileCount[0]-1) % 25 == 0:
							print "CD - average processing time per integration is %.3f s" % ((time.time() - wallStart)/sum(fileCount),)
							etc = (nInt - sum(fileCount)) * (time.time() - wallStart)/sum(fileCount)
							eth = int(etc/60.0) / 60
							etm = int(etc/60.0) % 60
							ets = etc % 60
							print "CD - estimated time to completion is %i:%02i:%04.1f" % (eth, etm, ets)
							
	# Cleanup
	etc = time.time() - wallStart
	eth = int(etc/60.0) / 60
	etm = int(etc/60.0) % 60
	ets = etc % 60
	print "Processing finished after %i:%02i:%04.1f" % (eth, etm, ets)
	print "Average time per integration was %.3f s" % (etc/sum(fileCount),)
	for f in fh:
		f.close()


if __name__ == "__main__":
	main(sys.argv[1:])
	
