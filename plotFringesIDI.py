#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A FITS-IDI compatible version of plotFringes2.py.

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

from scipy.stats import scoreatpercentile as percentile

from lsl.astro import utcjd_to_unix

from matplotlib import pyplot as plt


def usage(exitCode=None):
	print """plotFringesIDI.py - Given a FITS-IDI file, create plots of the visibilities

Usage:
plotFringesIDI.py [OPTIONS] fits

Options:
-h, --help                  Display this help information
-r, --ref-ant               Limit plots to baselines containing the reference 
                            antenna (default = plot everything)
-b, --baseline              Limit plots to the specified baseline in 'ANT-ANT' 
                            format
-x, --xx                    Plot XX data (default)
-z, --xy                    Plot XY data
-w, --yx                    Plot YX data
-y, --yy                    Plot YY data
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
	config['polToPlot'] = 'XX'
	config['args'] = []
	
	# Read in and process the command line flags
	try:
		opts, args = getopt.getopt(args, "hr:b:xzwy", ["help", "ref-ant=", "baseline=", "xx", "xy", "yx", "yy"])
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
		elif opt in ('-x', '--xx'):
			config['polToPlot'] = 'XX'
		elif opt in ('-z', '--xy'):
			config['polToPlot'] = 'XY'
		elif opt in ('-w', '--yx'):
			config['polToPlot'] = 'YX'
		elif opt in ('-y', '--yy'):
			config['polToPlot'] = 'YY'
		else:
			assert False
			
	# Add in arguments
	config['args'] = args
	
	# Fill the baseline list with the conjugates, if needed
	if config['baseline'] is not None:
		newBaselines = []
		for pair in config['baseline']:
			newBaselines.append( (pair[1],pair[0]) )
		config['baseline'].extend(newBaselines)
		
	# Validate
	if len(config['args']) != 1:
		raise RuntimeError("Must provide at a single FITS-IDI file to plot")
		
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
	
	plot_bls = []
	cross = []
	for i in xrange(len(ubls)):
		bl = ubls[i]
		ant1, ant2 = (bl>>8)&0xFF, bl&0xFF 
		if ant1 != ant2:
			if config['baseline'] is not None:
				if (ant1,ant2) in config['baseline']:
					plot_bls.append( bl )
					cross.append( i )
			elif config['refAnt'] is not None:
				if ant1 == config['refAnt'] or ant2 == config['refAnt']:
					plot_bls.append( bl )
					cross.append( i )
			else:
				plot_bls.append( bl )
				cross.append( i )
	nBL = len(cross)
	
	good = numpy.arange(freq.size/8, freq.size*7/8)		# Inner 75% of the band
	
	# NOTE: Assumed linear data
	polMapper = {'XX':0, 'YY':1, 'XY':2, 'YX':3}
	
	fig1 = plt.figure()
	fig2 = plt.figure()
	fig3 = plt.figure()
	fig4 = plt.figure()
	
	k = 0
	nRow = int(numpy.sqrt( len(plot_bls) ))
	nCol = int(numpy.ceil(len(plot_bls)*1.0/nRow))
	for b in xrange(len(plot_bls)):
		bl = plot_bls[b]
		valid = numpy.where( bls == bl )[0]
		i,j = (bl>>8)&0xFF, bl&0xFF
		vis = flux[valid,0,:,polMapper[config['polToPlot']]]
		dTimes = obstimes[valid]*86400.0
		dTimes -= dTimes[0]
		
		ax = fig1.add_subplot(nRow, nCol, k+1)
		ax.imshow(numpy.angle(vis), extent=(freq[0]/1e6, freq[-1]/1e6, dTimes[0], dTimes[-1]), origin='lower', vmin=-numpy.pi, vmax=numpy.pi, interpolation='nearest')
		ax.axis('auto')
		ax.set_xlabel('Frequency [MHz]')
		ax.set_ylabel('Elapsed Time [s]')
		ax.set_title("%i,%i - %s" % (i,j,config['polToPlot']))
		ax.set_xlim((freq[0]/1e6, freq[-1]/1e6))
		ax.set_ylim((dTimes[0], dTimes[-1]))
		
		ax = fig2.add_subplot(nRow, nCol, k+1)
		amp = numpy.abs(vis)
		vmin, vmax = percentile(amp, 1), percentile(amp, 99)
		ax.imshow(amp, extent=(freq[0]/1e6, freq[-1]/1e6, dTimes[0], dTimes[-1]), origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
		ax.axis('auto')
		ax.set_xlabel('Frequency [MHz]')
		ax.set_ylabel('Elapsed Time [s]')
		ax.set_title("%i,%i - %s" % (i,j,config['polToPlot']))
		ax.set_xlim((freq[0]/1e6, freq[-1]/1e6))
		ax.set_ylim((dTimes[0], dTimes[-1]))
				
		ax = fig3.add_subplot(nRow, nCol, k+1)
		ax.plot(freq/1e6, numpy.abs(vis.mean(axis=0)))
		ax.set_xlabel('Frequency [MHz]')
		ax.set_ylabel('Mean Vis. Amp. [lin.]')
		ax.set_title("%i,%i - %s" % (i,j,config['polToPlot']))
		ax.set_xlim((freq[0]/1e6, freq[-1]/1e6))
		
		ax = fig4.add_subplot(nRow, nCol, k+1)
		ax.plot(dTimes, numpy.angle(vis[:,good].mean(axis=1))*180/numpy.pi, linestyle='', marker='+')
		ax.set_ylim((-180, 180))
		ax.set_xlabel('Elapsed Time [s]')
		ax.set_ylabel('Mean Vis. Phase [deg]')
		ax.set_title("%i,%i - %s" % (i,j,config['polToPlot']))
		ax.set_xlim((dTimes[0], dTimes[-1]))
		
		k += 1
		
	fig1.suptitle("%s to %s UTC" % (datetime.utcfromtimestamp(times[0]).strftime("%Y/%m/%d %H:%M"), datetime.utcfromtimestamp(times[-1]).strftime("%Y/%m/%d %H:%M")))
	fig2.suptitle("%s to %s UTC" % (datetime.utcfromtimestamp(times[0]).strftime("%Y/%m/%d %H:%M"), datetime.utcfromtimestamp(times[-1]).strftime("%Y/%m/%d %H:%M")))
	fig3.suptitle("%s to %s UTC" % (datetime.utcfromtimestamp(times[0]).strftime("%Y/%m/%d %H:%M"), datetime.utcfromtimestamp(times[-1]).strftime("%Y/%m/%d %H:%M")))
	fig4.suptitle("%s to %s UTC" % (datetime.utcfromtimestamp(times[0]).strftime("%Y/%m/%d %H:%M"), datetime.utcfromtimestamp(times[-1]).strftime("%Y/%m/%d %H:%M")))
	
	plt.show()


if __name__ == "__main__":
	main(sys.argv[1:])
	