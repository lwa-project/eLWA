#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A FITS-IDI compatible version of fringeSearch.py to finding course delays and 
rates.

NOTE:  This script does not try to fringe search only a single source.  Rather,
       it searches the file as a whole.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import sys
import numpy
import getopt
import pyfits
from datetime import datetime

from lsl.astro import utcjd_to_unix
from lsl.statistics import robust
from lsl.misc.mathutil import to_dB

from matplotlib import pyplot as plt


def usage(exitCode=None):
	print """fringeSearchIDI.py - Given a FITS-IDI file, search for fringes.

Usage:
fringeSearchIDI.py [OPTIONS] fits

Options:
-h, --help                  Display this help information
-r, --ref-ant               Reference antenna (default = first antenna)
-b, --baseline              Search only the specified baseline in 'ANT-ANT' format
-y, --y-only                Limit the search on VLA-LWA baselines to the VLA
                            Y pol. only
-e, --delay-window          Delay search window in us (default = -inf,inf 
                            = maximum allowed by spectral resolution)
-a, --rate-window           Rate search window in mHz (default = -inf,inf
                            = maximum allowed by temporal resolution)
-p, --plot                  Show search plots at the end (default = no)
"""
	
	if exitCode is not None:
		sys.exit(exitCode)
	else:
		return True


def parseConfig(args):
	config = {}
	# Command line flags - default values
	config['refAnt'] = None
	config['baseline'] = None
	config['yOnlyVLALWA'] = False
	config['delayWindow'] = [-numpy.inf, numpy.inf]
	config['rateWindow'] = [-numpy.inf, numpy.inf]
	config['plot'] = False
	config['args'] = []
	
	# Read in and process the command line flags
	try:
		opts, args = getopt.getopt(args, "hr:b:ye:a:p", ["help", "baseline=", "ref-ant=", "y-only", "delay-window=", "rate-window=", "plot"])
	except getopt.GetoptError, err:
		# Print help information and exit:
		print str(err) # will print something like "option -a not recognized"
		usage(exitCode=2)
		
	# Work through opts
	for opt, value in opts:
		if opt in ('-h', '--help'):
			usage(exitCode=0)
		elif opt in ('-r', '--ref-ant'):
			config['refAnt'] = int(value, 10)
		elif opt in ('-b', '--baseline'):
			config['baseline'] = [(int(v0,10),int(v1,10)) for v0,v1 in [v.split('-') for v in value.split(',')]]
		elif opt in ('-y', '--y-only'):
			config['yOnlyVLALWA'] = True
		elif opt in ('-e', '--delay-window'):
			config['delayWindow'] = [float(v) for v in value.split(',', 1)]
		elif opt in ('-a', '--rate-window'):
			config['rateWindow'] = [float(v) for v in value.split(',', 1)]
		elif opt in ('-p', '--plot'):
			config['plot'] = True
		else:
			assert False
			
	# Add in arguments
	config['args'] = args
	
	# Fill the baseline list with the conjugates
	if config['baseline'] is not None:
		newBaselines = []
		for pair in config['baseline']:
			newBaselines.append( (pair[1],pair[0]) )
		config['baseline'].extend(newBaselines)
		
	# Validate
	if len(config['args']) != 1:
		raise RuntimeError("Must provide at a single FITS-IDI file to plot")
	if config['delayWindow'][0] > config['delayWindow'][1]:
		raise RuntimeError("Invalid delay search window: %.3f to %.3f" % tuple(config['delayWindow']))
	if config['rateWindow'][0] > config['rateWindow'][1]:
		raise RuntimeError("Invalid rate search window: %.3f to %.3f" % tuple(config['rateWindow']))
		
	# Return configuration
	return config


def main(args):
	# Parse the command line
	config = parseConfig(args)
	filename = config['args'][0]
	
	print "Working on '%s'" % os.path.basename(filename)
	# Open the FITS IDI file and access the UV_DATA extension
	hdulist = pyfits.open(filename, mode='readonly')
	andata = hdulist['ANTENNA']
	fqdata = hdulist['FREQUENCY']
	uvdata = hdulist['UV_DATA']
	
	# Verify we can flag this data
	if uvdata.header['STK_1'] > 0:
		raise RuntimeError("Cannot flag data with STK_1 = %i" % uvdata.header['STK_1'])
	if uvdata.header['NO_STKD'] < 4:
		raise RuntimeError("Cannot flag data with NO_STKD = %i" % uvdata.header['NO_STKD'])
		
	# Pull out various bits of information we need to flag the file
	## Antenna look-up table
	antLookup = {}
	for an, ai in zip(andata.data['ANNAME'], andata.data['ANTENNA_NO']):
		antLookup[an] = ai
	## Frequency and polarization setup
	nBand, nFreq, nStk = uvdata.header['NO_BAND'], uvdata.header['NO_CHAN'], uvdata.header['NO_STKD']
	## Baseline list
	bls = uvdata.data['BASELINE']
	## Time of each integration
	obsdates = uvdata.data['DATE']
	obstimes = uvdata.data['TIME']
	## Source list
	srcs = uvdata.data['SOURCE']
	## Band information
	fqoffsets = fqdata.data['BANDFREQ'].ravel()
	## Frequency channels
	freq = (numpy.arange(nFreq)-(uvdata.header['CRPIX3']-1))*uvdata.header['CDELT3']
	freq += uvdata.header['CRVAL3']
	## UVW coordinates
	u, v, w = uvdata.data['UU'], uvdata.data['VV'], uvdata.data['WW']
	uvw = numpy.array([u, v, w]).T
	## The actual visibility data
	flux = uvdata.data['FLUX'].astype(numpy.float32)
	
	# Convert the visibilities to something that we can easily work with
	nComp = flux.shape[1] / nBand / nFreq / nStk
	if nComp == 2:
		## Case 1) - Just real and imaginary data
		flux = flux.view(numpy.complex64)
	else:
		## Case 2) - Real, imaginary data + weights (drop the weights)
		flux = flux[:,0::nComp] + 1j*flux[:,1::nComp]
	flux.shape = (flux.shape[0], nBand, nFreq, nStk)
	
	# Find unique baselines, times, and sources to work with
	ubls = numpy.unique(bls)
	utimes = numpy.unique(obstimes)
	usrc = numpy.unique(srcs)
	
	# Convert times to real times
	times = utcjd_to_unix(obsdates + obstimes)
	times = numpy.unique(times)
	
	# Find unique scans to work on
	blocks = []
	for src in usrc:
		valid = numpy.where( src == srcs )[0]
		
		blocks.append( [valid[0],valid[0]] )
		for v in valid[1:]:
			if v == blocks[-1][1] + 1:
				blocks[-1][1] = v
			else:
				blocks.append( [v,v] )
	blocks.sort()
	
	search_bls = []
	cross = []
	for i in xrange(len(ubls)):
		bl = ubls[i]
		ant1, ant2 = (bl>>8)&0xFF, bl&0xFF 
		if ant1 != ant2:
			search_bls.append( bl )
			cross.append( i )
	nBL = len(cross)
	
	iTimes = numpy.zeros(times.size-1, dtype=times.dtype)
	for i in xrange(1, len(times)):
		iTimes[i-1] = times[i] - times[i-1]
	print " -> Interval: %.3f +/- %.3f seconds (%.3f to %.3f seconds)" % (iTimes.mean(), iTimes.std(), iTimes.min(), iTimes.max())
	
	print "Number of frequency channels: %i (~%.1f Hz/channel)" % (len(freq), freq[1]-freq[0])

	dTimes = times - times[0]
	
	dMax = 1.0/(freq[1]-freq[0])/4
	dMax = int(dMax*1e6)*1e-6
	if -dMax*1e6 > config['delayWindow'][0]:
		config['delayWindow'][0] = -dMax*1e6
	if dMax*1e6 < config['delayWindow'][1]:
		config['delayWindow'][1] = dMax*1e6
	rMax = 1.0/iTimes.mean()/4
	rMax = int(rMax*1e2)*1e-2
	if -rMax*1e3 > config['rateWindow'][0]:
		config['rateWindow'][0] = -rMax*1e3
	if rMax*1e3 < config['rateWindow'][1]:
		config['rateWindow'][1] = rMax*1e3
		
	dres = 1.0
	nDelays = int((config['delayWindow'][1]-config['delayWindow'][0])/dres)
	while nDelays < 50:
		dres /= 10
		nDelays = int((config['delayWindow'][1]-config['delayWindow'][0])/dres)
	while nDelays > 5000:
		dres *= 10
		nDelays = int((config['delayWindow'][1]-config['delayWindow'][0])/dres)
	nDelays += (nDelays + 1) % 2
	
	rres = 10.0
	nRates = int((config['rateWindow'][1]-config['rateWindow'][0])/rres)
	while nRates < 50:
		rres /= 10
		nRates = int((config['rateWindow'][1]-config['rateWindow'][0])/rres)
	while nRates > 5000:
		rres *= 10
		nRates = int((config['rateWindow'][1]-config['rateWindow'][0])/rres)
	nRates += (nRates + 1) % 2
	
	print "Searching delays %.1f to %.1f us in steps of %.2f us" % (config['delayWindow'][0], config['delayWindow'][1], dres)
	print "           rates %.1f to %.1f mHz in steps of %.2f mHz" % (config['rateWindow'][0], config['rateWindow'][1], rres)
	print " "
	
	delay = numpy.linspace(config['delayWindow'][0]*1e-6, config['delayWindow'][1]*1e-6, nDelays)		# s
	drate = numpy.linspace(config['rateWindow'][0]*1e-3,  config['rateWindow'][1]*1e-3,  nRates )		# Hz
	
	# Find RFI and trim it out.  This is done by computing average visibility 
	# amplitudes (a "spectrum") and running a median filter in frequency to extract
	# the bandpass.  After the spectrum has been bandpassed, 3sigma features are 
	# trimmed.  Additionally, area where the bandpass fall below 10% of its mean
	# value are also masked.
	spec  = numpy.median(numpy.abs(flux[:,0,:,0]), axis=0)
	spec += numpy.median(numpy.abs(flux[:,0,:,1]), axis=0)
	smth = spec*0.0
	winSize = int(250e3/(freq[1]-freq[0]))
	winSize += ((winSize+1)%2)
	for i in xrange(smth.size):
		mn = max([0, i-winSize/2])
		mx = min([i+winSize/2+1, smth.size])
		smth[i] = numpy.median(spec[mn:mx])
	smth /= robust.mean(smth)
	bp = spec / smth
	good = numpy.where( (smth > 0.1) & (numpy.abs(bp-robust.mean(bp)) < 3*robust.std(bp)) )[0]
	nBad = nFreq - len(good)
	print "Masking %i of %i channels (%.1f%%)" % (nBad, nFreq, 100.0*nBad/nFreq)
	if config['plot']:
		fig = plt.figure()
		ax = fig.gca()
		ax.plot(freq/1e6, numpy.log10(spec)*10)
		ax.plot(freq[good]/1e6, numpy.log10(spec[good])*10)
		ax.set_title('Mean Visibility Amplitude')
		ax.set_xlabel('Frequency [MHz]')
		ax.set_ylabel('PSD [arb. dB]')
		plt.draw()
	
	freq2 = freq*1.0
	freq2.shape += (1,)
	dTimes2 = dTimes*1.0
	dTimes2.shape += (1,)
	
	# NOTE: Assumed linear data
	polMapper = {'XX':0, 'YY':1, 'XY':2, 'YX':3}
	
	print "%3s  %9s  %2s  %6s  %9s  %11s" % ('#', 'BL', 'Pl', 'S/N', 'Delay', 'Rate')
	for b in xrange(len(search_bls)):
		bl = search_bls[b]
		ant1, ant2 = (bl>>8)&0xFF, bl&0xFF
		
		## Skip over baselines that are not in the baseline list (if provided)
		if config['baseline'] is not None:
			if (ant1, ant2) not in config['baseline']:
				continue
		## Skip over baselines that don't include the reference antenna
		elif ant1 != config['refAnt'] and ant2 != config['refAnt']:
			continue
			
		## Check and see if we need to conjugate the visibility, i.e., switch from
		## baseline (*,ref) to baseline (ref,*)
		doConj = False
		if ant2 == config['refAnt']:
			doConj = True
			
		## Figure out which polarizations to process
		if ant1 not in (51, 52) and ant2 not in (51, 52):
			### Standard VLA-VLA baseline
			polToUse = ('XX', 'YY')
		else:
			### LWA-LWA or LWA-VLA baseline
			if config['yOnlyVLALWA']:
				polToUse = ('YX', 'YY')
			else:
				polToUse = ('XX', 'XY', 'YX', 'YY')
				
		if config['plot']:
			fig = plt.figure()
			axs = {}
			axs['XX'] = fig.add_subplot(2, 2, 1)
			axs['YY'] = fig.add_subplot(2, 2, 2)
			axs['XY'] = fig.add_subplot(2, 2, 3)
			axs['YX'] = fig.add_subplot(2, 2, 4)
			
		valid = numpy.where( bls == bl )[0]
		for pol in polToUse:
			subData = flux[valid,0,:,polMapper[pol]]*1.0
			subData = subData[:,good]
			if doConj:
				subData = subData.conj()
			subData = numpy.dot(subData, numpy.exp(-2j*numpy.pi*freq2[good,:]*delay))
			subData /= freq2[good,:].size
			amp = numpy.dot(subData.T, numpy.exp(-2j*numpy.pi*dTimes2*drate))
			amp = numpy.abs(amp / dTimes2.size)
			
			blName = (ant1, ant2)
			if doConj:
				blName = (ant2, ant1)
			blName = '%s-%s' % ('EA%02i' % blName[0] if blName[0] < 51 else 'LWA%i' % (blName[0]-50), 
						'EA%02i' % blName[1] if blName[1] < 51 else 'LWA%i' % (blName[1]-50))
						
			best = numpy.where( amp == amp.max() )
			if amp.max() > 0:
				bsnr = (amp[best]-amp.mean())[0]/amp.std()
				bdly = delay[best[0][0]]*1e6
				brat = drate[best[1][0]]*1e3
				print "%3i  %9s  %2s  %6.2f  %6.2f us  %7.2f mHz" % (b, blName, pol, bsnr, bdly, brat)
			else:
				print "%3i  %9s  %2s  %6s  %9s  %11s" % (b, blName, pol, '----', '----', '----')
				
			if config['plot']:
				axs[pol].imshow(amp, origin='lower', interpolation='nearest', 
							extent=(drate[0]*1e3, drate[-1]*1e3, delay[0]*1e6, delay[-1]*1e6), 
							cmap='gray_r')
				axs[pol].plot(drate[best[1][0]]*1e3, delay[best[0][0]]*1e6, linestyle='', marker='x', color='r', ms=15, alpha=0.75)
				
		if config['plot']:
			fig.suptitle(os.path.basename(filename))
			for pol,ax in axs.iteritems():
				ax.axis('auto')
				ax.set_title(pol)
				ax.set_xlabel('Rate [mHz]')
				ax.set_ylabel('Delay [$\\mu$s]')
			fig.suptitle("%s" % blName)
			plt.draw()
			
	if config['plot']:
		plt.show()


if __name__ == "__main__":
	main(sys.argv[1:])
	