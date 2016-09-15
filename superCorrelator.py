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

from utils import *
import multirate


def usage(exitCode=None):
	print """superCorrelator.py - The next generation of correlator

Usage:
superCorrelator.py [OPTIONS] <config_file>

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
	config['args'] = []
	
	# Read in and process the command line flags
	try:
		opts, args = getopt.getopt(args, "hql:u:t:s:d:g:", ["help", "quiet", "fft-length=", "subint-time=", "dump-time=", "skip=", "duration=", "tag="])
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
		else:
			assert False
			
	# Add in arguments
	config['args'] = args
	
	# Return configuration
	return config


def main(args):
	# Parse the command line
	config = parseConfig(args)
	
	# Length of the FFT
	LFFT = config['LFFT']
	
	# Build up the station
	site = stations.lwa1
	observer = site.getObserver()
	
	# Parse the correlator configuration
	refSrc, filenames, foffsets, readers, antennas = readCorrelatorConfiguration(config['args'][0])
	
	# Get the raw configuration
	fh = open(config['args'][0], 'r')
	rawConfig = fh.readlines()
	fh.close()
	
	# Antenna report
	print "Antennas:"
	for ant in antennas:
		print "  Antenna %i: Stand %i, Pol. %i" % (ant.id, ant.stand.id, ant.pol)
		
	# Open and align the files
	fh = []
	nFramesFile = []
	srate = []
	beams = []
	tunepols = []
	beampols = []
	tStart = []
	cFreqs = []
	for i,(filename,foffset) in enumerate(zip(filenames, foffsets)):
		fh.append( open(filename, "rb") )
		
		if readers[i] is vdif:
			header = readGUPPIHeader(fh[i])
			readers[i].FrameSize = readers[i].getFrameSize(fh[i])
			
		nFramesFile.append( os.path.getsize(filename) / readers[i].FrameSize )
		if readers[i] is vdif:
			junkFrame = readers[i].readFrame(fh[i], centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0)
			readers[i].DataLength = junkFrame.data.data.size
			beam, pol = junkFrame.parseID()
		elif readers[i] is drx:
			junkFrame = readers[i].readFrame(fh[i])
			readers[i].DataLength = junkFrame.data.iq.size
			beam, tune, pol = junkFrame.parseID()
		fh[i].seek(-readers[i].FrameSize, 1)
		
		beams.append( beam )
		srate.append( junkFrame.getSampleRate() )
		
		if readers[i] is vdif:
			tunepols.append( readers[i].getThreadCount(fh[i]) )
			beampols.append( tunepols[i] )
		elif readers[i] is drx:
			beampols.append( 4 )
			
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
			
		tStart.append( junkFrame.getTime() )
		
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
				
		elif readers[i] is drx:
			timeTags = []
			for k in xrange(16):
				junkFrame = readers[i].readFrame(fh[i])
				timeTags.append(junkFrame.data.timeTag)
			fh[i].seek(-16*readers[i].FrameSize, 1)
			
			j = 0
			while (timeTags[j+0] != timeTags[j+1]) or (timeTags[j+0] != timeTags[j+2]) or (timeTags[j+0] != timeTags[j+3]):
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
		while junkFrame.getTime()  < max(tStart):
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
	beginDates = []
	for i in xrange(len(filenames)):
		if readers[i] is vdif:
			junkFrame = readers[i].readFrame(fh[i], centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0)
		elif readers[i] is drx:
			junkFrame = readers[i].readFrame(fh[i])
		fh[i].seek(-readers[i].FrameSize, 1)
		
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
		print "  Date of First Frame: %s" % beginDates[i]
		print "  Sample Rate: %i Hz" % srate[i]
		print "  Tuning 1: %.1f Hz" % cFreqs[i][0]
		print "  Tuning 2: %.1f Hz" % cFreqs[i][1]
	print "  ==="
	print "  Phase Center:"
	print "    Name: %s" % refSrc.name
	print "    RA: %s" % refSrc._ra
	print "    Dec: %s" % refSrc._dec
	print "  ==="
	print "  Data Read Time: %.3f s" % tRead
	print "  Data Reads in File: %i" % int(tFile/tRead)
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
	
	vdifLFFT = LFFT
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
	print "VDIF appears to correspond to tuning #%i in DRX" % vdifPivot
	print " "
	
	nChunks = int(tFile/tRead)
	tSub = config['subTime']
	tDump = config['dumpTime']
	nDump = int(tDump / tSub)
	tDump = nDump * tSub
	print "Sub-integration time is: %.3f s" % tSub
	print "Integration (dump) time is: %.3f s" % tDump
	print " "
	
	nVDIFInputs = sum([1 for reader in readers if reader is vdif])
	nDRXInputs = sum([1 for reader in readers if reader is drx])
	print "Processing %i VDIF and %i DRX input streams" % (nVDIFInputs, nDRXInputs)
	print " "
	
	subIntTimes = []
	subIntCount = 0
	fileCount   = 0
	for i in xrange(nChunks):
		wallTime = time.time()
		
		tStart = []
		tStartB = []
		
		vdifRef = [0 for j in xrange(nVDIFInputs*2)]
		drxRef  = [0 for j in xrange(nDRXInputs*4) ]
				
		for j,f in enumerate(fh):
			if readers[j] is vdif:
				try:
					junkFrame = readers[j].readFrame(f, centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0)
				except errors.syncError:
					print "Error @ %i, %i" % (i, j)
					f.seek(vdif.FrameSize, 1)
					junkFrame = readers[j].readFrame(f, centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0)
					
				std,pol = junkFrame.parseID()
				for p in (0,1):
					sid = 2*j + p
					vdifRef[sid] = junkFrame.header.secondsFromEpoch*framesPerSecondV + junkFrame.header.frameInSecond
					
			elif readers[j] is drx:
				junkFrame = readers[j].readFrame(f)
				
				beam,tune,pol = junkFrame.parseID()
				for t in (1,2):
					for p in (0,1):
						bid = 4*(j-nVDIFInputs) + 2*(t-1) + p
						drxRef[bid] = junkFrame.data.timeTag
						
			tStart.append( junkFrame.getTime() )
			tStartB.append( getBetterTime(junkFrame) )
			f.seek(-readers[j].FrameSize, 1)
			
		dataV = numpy.zeros((len(vdifRef), readers[ 0].DataLength*nFramesV), dtype=numpy.complex64)
		dataD = numpy.zeros((len(drxRef),  readers[-1].DataLength*nFramesD), dtype=numpy.complex64)
		for j,f in enumerate(fh):
			if readers[j] is vdif:
				for k in xrange(beampols[j]*nFramesV):
					try:
						cFrame = readers[j].readFrame(f, centralFreq=header['OBSFREQ'], sampleRate=header['OBSBW']*2.0)
					except errors.syncError:
						print "Error @ %i, %i" % (i, j)
						f.seek(vdif.FrameSize, 1)
						continue
					std,pol = cFrame.parseID()
					sid = 2*j + pol
					
					count = cFrame.header.secondsFromEpoch*framesPerSecondV + cFrame.header.frameInSecond
					count -= vdifRef[sid]
					dataV[sid, count*readers[j].DataLength:(count+1)*readers[j].DataLength] = cFrame.data.data
					
			elif readers[j] is drx:
				for k in xrange(beampols[j]*nFramesD):
					cFrame = readers[j].readFrame(f)
					beam,tune,pol = cFrame.parseID()
					bid = 4*(j-nVDIFInputs) + 2*(tune-1) + pol
					
					count = cFrame.data.timeTag
					count -= drxRef[bid]
					count /= (4096*int(196e6/srate[-1]))
					dataD[bid, count*readers[j].DataLength:(count+1)*readers[j].DataLength] = cFrame.data.iq
					
		sel = []
		for j in xrange(nDRXInputs):
			for k in xrange(beampols[-1]):
				l = j*beampols[-1] + k
				if vdifPivot == 1 and (l/2)%2 == 0:
					sel.append( l )
				elif vdifPivot == 2 and (l/2)%2 == 1:
					sel.append( l )
		dataD = dataD[sel,:]
		dataD /= 7.0
		
		if config['verbose']:
			print 'TT - Start', tStartB
		tStartMin = min([sec for sec,frac in tStartB])
		tStartRel = [(sec-tStartMin)+frac for sec,frac in tStartB]
		
		offsets = []
		for j in xrange(nVDIFInputs+nDRXInputs):
			offsets.append( int( round((max(tStartRel) - tStartRel[j])*srate[j]) ) )
		if config['verbose']:
			print 'TT - Offsets', offsets
			
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
				
		if config['verbose']:
			print 'TT - Adjusted', tStartB
		tStartMinSec  = min([sec  for sec,frac in tStartB])
		tStartMinFrac = min([frac for sec,frac in tStartB])
		tStartRel = [(sec-tStartMinSec)+(frac-tStartMinFrac) for sec,frac in tStartB]
		if config['verbose']:
			print 'TT - Residual', ["%.1f ns" % (r*1e9,) for r in tStartRel]
			
		nSub = int(tRead/tSub)
		nSampV = int(srate[ 0]*tSub)
		nSampD = int(srate[-1]*tSub)
		
		tV = i*tRead + numpy.arange(dataV.shape[1], dtype=numpy.float64)/srate[ 0]
		tD = i*tRead + numpy.arange(dataD.shape[1], dtype=numpy.float64)/srate[-1]
		
		for j in xrange(nSub):
			tSubInt = tStart[0] + (j+1)*nSampV/srate[0] - nSampV/srate[0]
			tVSub    = tV[j*nSampV:(j+1)*nSampV]
			tDSub    = tD[j*nSampD:(j+1)*nSampD]
			dataVSub = dataV[:,j*nSampV:(j+1)*nSampV]
			dataDSub = dataD[:,j*nSampD:(j+1)*nSampD]
			
			# Update the observation
			observer.date = astro.unix_to_utcjd(tSubInt) - astro.DJD_OFFSET
			refSrc.compute(observer)
			
			# Correlate
			if nVDIFInputs > 0:
				freqV, feoXV, veoXV, deoXV = multirate.MRF(dataVSub, antennas[:2*nVDIFInputs], LFFT=vdifLFFT, SampleRate=srate[0], 
									CentralFreq=cFreqs[0][0]-srate[0]/4, Pol='XX', phaseCenter=refSrc)
				freqV, feoYV, veoYV, deoYV = multirate.MRF(dataVSub, antennas[:2*nVDIFInputs], LFFT=vdifLFFT, SampleRate=srate[0], 
									CentralFreq=cFreqs[0][0]-srate[0]/4, Pol='YY', phaseCenter=refSrc)
			if nDRXInputs > 0:
				freqD, feoXD, veoXD, deoXD = multirate.MRF(dataDSub, antennas[2*nVDIFInputs:], LFFT=drxLFFT, SampleRate=srate[-1], 
									CentralFreq=cFreqs[0][0], Pol='XX', phaseCenter=refSrc)
				freqD, feoYD, veoYD, deoYD = multirate.MRF(dataDSub, antennas[2*nVDIFInputs:], LFFT=drxLFFT, SampleRate=srate[-1], 
									CentralFreq=cFreqs[0][0], Pol='YY', phaseCenter=refSrc)
			
			# Rotate the phase in time to deal with frequency offset between the VLA and LWA
			if nDRXInputs > 0:
				for s in xrange(feoYD.shape[0]):
					for w in xrange(feoXD.shape[2]):
						feoXD[s,:,w] *= numpy.exp(-2j*numpy.pi*(cFreqs[0][0]-cFreqs[-1][vdifPivot-1])*tDSub[w*drxLFFT])
					for w in xrange(feoYD.shape[2]):
						feoYD[s,:,w] *= numpy.exp(-2j*numpy.pi*(cFreqs[0][0]-cFreqs[-1][vdifPivot-1])*tDSub[w*drxLFFT])
						
			fMin, fMax = -1e12, 1e12
			if nVDIFInputs > 0:
				fMin, fMax = max([fMin, freqV.min()]), min([fMax, freqV.max()])
			if nDRXInputs > 0:
				fMin, fMax = max([fMin, freqD.min()]), min([fMax, freqD.max()])
			if nVDIFInputs > 0:
				goodV = numpy.where( (freqV >= fMin) & (freqV <= fMax) )[0]
				freqV = freqV[goodV]
				feoXV, feoYV = feoXV[:,goodV,:], feoYV[:,goodV,:]
			if nDRXInputs > 0:
				goodD = numpy.where( (freqD >= fMin) & (freqD <= fMax) )[0]
				freqD = freqD[goodD]
				feoXD, feoYD = feoXD[:,goodD,:], feoYD[:,goodD,:]
			#print freqV[0], freqV[1]-freqV[0], freqV.size, freqD[0], freqD[1]-freqD[0], freqD.size
			
			nWin = 1e12
			if nVDIFInputs > 0:
				nWin = min([nWin, feoXV.shape[2], feoYV.shape[2]])
			if nDRXInputs > 0:
				nWin = min([nWin, feoXD.shape[2], feoYD.shape[2]])
			feoX = numpy.zeros((nVDIFInputs+nDRXInputs, feoXV.shape[1], nWin), dtype=feoXV.dtype)
			if nVDIFInputs > 0:
				feoX[:nVDIFInputs,:,:] = feoXV[:,:,:nWin]
			if nDRXInputs > 0:
				feoX[nVDIFInputs:,:,:] = feoXD[:,:,:nWin]
			veoX = numpy.zeros((nVDIFInputs+nDRXInputs, nWin), dtype=veoXV.dtype)
			if nVDIFInputs > 0:
				veoX[:nVDIFInputs,:] = veoXV[:,:nWin]
			if nDRXInputs > 0:
				veoX[nVDIFInputs:,:] = veoXD[:,:nWin]
			feoY = numpy.zeros((nVDIFInputs+nDRXInputs, feoYV.shape[1], nWin), dtype=feoYV.dtype)
			if nVDIFInputs > 0:
				feoY[:nVDIFInputs,:,:] = feoYV[:,:,:nWin]
			if nDRXInputs > 0:
				feoY[nVDIFInputs:,:,:] = feoYD[:,:,:nWin]
			veoY = numpy.zeros((nVDIFInputs+nDRXInputs, nWin), dtype=veoYV.dtype)
			if nVDIFInputs > 0:
				veoY[:nVDIFInputs,:] = veoYV[:,:nWin]
			if nDRXInputs > 0:
				veoY[nVDIFInputs:,:] = veoYD[:,:nWin]
			
			sfreqXX = freqV
			sfreqYY = freqV
			svisXX = multirate.MRX(feoX, veoX, feoX, veoX)
			svisXY = multirate.MRX(feoX, veoX, feoY, veoY)
			svisYX = multirate.MRX(feoY, veoY, feoX, veoX)
			svisYY = multirate.MRX(feoY, veoY, feoY, veoY)
			
			if subIntCount == 0:
				subIntTimes = [tSubInt,]
				freqXX = sfreqXX
				freqYY = sfreqYY
				visXX  = svisXX / nDump
				visXY  = svisXY / nDump
				visYX  = svisYX / nDump
				visYY  = svisYY / nDump
			else:
				subIntTimes.append( tSubInt )
				visXX += svisXX / nDump
				visXY += svisXY / nDump
				visYX += svisYX / nDump
				visYY += svisYY / nDump
			subIntCount += 1
			
			if subIntCount == nDump:
				subIntCount = 0
				fileCount += 1
				
				if nChunks != 1:
					outfile = "%s-vis2-%05i.npz" % (outbase, fileCount)
				else:
					outfile = "%s-vis2.npz" % outbase
				print '->', fileCount, j, numpy.mean(subIntTimes), fileCount*nDump*tSub
				numpy.savez(outfile, config=rawConfig, srate=srate[0]/2.0, freq1=freqXX, 
							vis1XX=visXX, vis1XY=visXY, vis1YX=visYX, vis1YY=visYY, 
							tStart=numpy.mean(subIntTimes), tInt=nDump*tSub)


if __name__ == "__main__":
	main(sys.argv[1:])
	