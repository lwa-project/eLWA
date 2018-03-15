#!/usr/bin/env python

"""
RFI flagger for FITS-IDI files containing eLWA data.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import sys
import numpy
import getopt
import pyfits

from flagger import *


def usage(exitCode=None):
	print """flagIDI.py - Flag RFI in FITS-IDI files

Usage:
flagIDI.py [OPTIONS] <fits_file>

Options:
-h, --help                  Display this help information
"""
	
	if exitCode is not None:
		sys.exit(exitCode)
	else:
		return True


def parseConfig(args):
	config = {}
	# Command line flags - default values
	config['args'] = []
	
	# Read in and process the command line flags
	try:
		opts, args = getopt.getopt(args, "h", ["help",])
	except getopt.GetoptError, err:
		# Print help information and exit:
		print str(err) # will print something like "option -a not recognized"
		usage(exitCode=2)
		
	# Work through opts
	for opt, value in opts:
		if opt in ('-h', '--help'):
			usage(exitCode=0)
		else:
			assert False
			
	# Add in arguments
	config['args'] = args
	
	# Return configuration
	return config


def main(args):
	# Parse the command line
	config = parseConfig(args)
	filename = config['args'][0]
	
	print "Working on '%s'" % os.path.basename(filename)
	# Open the FITS IDI file and access the UV_DATA extension
	hdulist = pyfits.open(filename, mode='readonly')
	uvdata = hdulist['UV_DATA']
	
	# Pull out various bits of information we need to flag the file
	## Frequency and polarization setup
	nFreq, nStk = uvdata.header['NO_CHAN'], uvdata.header['NO_STKD']
	## Baseline list
	bls = uvdata.data['BASELINE']
	## Time of each integration
	obsdates = uvdata.data['TIME']
	## Source list
	srcs = uvdata.data['SOURCE']
	## Frequency channels
	freq = (numpy.arange(nFreq)-(uvdata.header['CRPIX3']-1))*uvdata.header['CDELT3']
	freq += uvdata.header['CRVAL3']
	## UVW coordinates
	u, v, w = uvdata.data['UU'], uvdata.data['VV'], uvdata.data['WW']
	uvw = numpy.array([u, v, w]).T
	## The actual visibility data
	flux = uvdata.data['FLUX'].astype(numpy.float32)
	
	# Convert the visibilities to something that we can easily work with
	nComp = flux.shape[1] / nFreq / nStk
	if nComp == 2:
		## Case 1) - Just real and imaginary data
		flux = flux.view(numpy.complex64)
		flux.shape = (flux.shape[0], nFreq, nStk)
	else:
		## Case 2) - Real, imaginary data + weights (drop the weights)
		flux = flux[:,0::nComp] + 1j*flux[:,1::nComp]
		
	# Find unique baselines, times, and sources to work with
	ubls = numpy.unique(bls)
	utimes = numpy.unique(obsdates)
	usrc = numpy.unique(srcs)
	
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
	
	# Build up the mask
	mask = numpy.zeros(flux.shape, dtype=numpy.bool)
	for i,block in enumerate(blocks):
		print "Working on scan %i of %i" % (i+1, len(blocks))
		match = range(block[0],block[1]+1)
		
		bbls = numpy.unique(bls[match])
		times = obsdates[match] * 86400.0
		crd = uvw[match,:]
		visXX = flux[match,:,0]
		visYY = flux[match,:,3]
		
		nBL = len(bbls)
		times = times[0::nBL]
		crd.shape = (crd.shape[0]/nBL, nBL, 1, 3)
		visXX.shape = (visXX.shape[0]/nBL, nBL, visXX.shape[1])
		visYY.shape = (visYY.shape[0]/nBL, nBL, visYY.shape[1])
		
		antennas = []
		for j in xrange(nBL):
			ant1, ant2 = (bbls[j]>>8)&0xFF, bbls[j]&0xFF
			if ant1 not in antennas:
				antennas.append(ant1)
			if ant2 not in antennas:
				antennas.append(ant2)
				
		maskXX = mask_bandpass(antennas, times, freq, visXX)
		maskYY = mask_bandpass(antennas, times, freq, visYY)
		
		visXX = numpy.ma.array(visXX, mask=maskXX)
		visYY = numpy.ma.array(visYY, mask=maskYY)
		
		visXX.mask = mask_spurious(antennas, times, crd, freq, visXX)
		visYY.mask = mask_spurious(antennas, times, crd, freq, visYY)
		
		visXX.mask = cleanup_mask(visXX.mask)
		visYY.mask = cleanup_mask(visYY.mask)
		
		submask = visXX.mask
		submask.shape = (len(match), flux.shape[1])
		mask[match,:,0] = submask
		submask = visYY.mask
		submask.shape = (len(match), flux.shape[1])
		mask[match,:,3] = submask
		submask = visXX.mask | visYY.mask
		submask.shape = (len(match), flux.shape[1])
		mask[match,:,1] = submask
		mask[match,:,2] = submask
		
		print '  Flagged %.1f%% of this scan' % (100.0*mask[match,:,:].sum()/mask[match,:,:].size,)
		
	# Convert the masks into a format suitable for writing to a FLAG table
	obsdates.shape = (obsdates.shape[0]/nBL, nBL)
	mask.shape = (mask.shape[0]/nBL, nBL, nFreq, nStk)
	ants, times, chans, pols = [], [], [], []
	for i in xrange(nBL):
		ant1, ant2 = (bls[i]>>8)&0xFF, bls[i]&0xFF
		maskXX = mask[:,i,:,0]
		maskXY = mask[:,i,:,1]
		maskYY = mask[:,i,:,3]
		
		print ant1, ant2
		flagsXX, _ = create_flag_groups(obsdates[:,i], freq, maskXX)
		flagsXY, _ = create_flag_groups(obsdates[:,i], freq, maskXY)
		flagsYY, _ = create_flag_groups(obsdates[:,i], freq, maskYY)
		print ' ', len(flagsXX), len(flagsXY), len(flagsYY)
		
		for flag in flagsXX:
			ants.append( (ant1,ant2) )
			times.append( (obsdates[flag[0],i], obsdates[flag[1],i]) )
			chans.append( (flag[2]+1, flag[3]+1) )
			pols.append( (1, 0, 0, 0) )
		for flag in flagsXY:
			ants.append( (ant1,ant2) )
			times.append( (obsdates[flag[0],i], obsdates[flag[1],i]) )
			chans.append( (flag[2]+1, flag[3]+1) )
			pols.append( (0, 1, 1, 0) )
		for flag in flagsYY:
			ants.append( (ant1,ant2) )
			times.append( (obsdates[flag[0],i], obsdates[flag[1],i]) )
			chans.append( (flag[2]+1, flag[3]+1) )
			pols.append( (0, 0, 0, 1) )
			
	## Build the FLAG table
	### Columns
	nFlags = len(ants)
	c1 = pyfits.Column(name='SOURCE_ID', format='1J',  array=numpy.zeros((nFlags,), dtype=numpy.int32))
	c2 = pyfits.Column(name='ARRAY',     format='1J',  array=numpy.zeros((nFlags,), dtype=numpy.int32))
	c3 = pyfits.Column(name='ANTS',      format='2J',  array=numpy.array(ants, dtype=numpy.int32))
	c4 = pyfits.Column(name='FREQID',    format='1J',  array=numpy.zeros((nFlags,), dtype=numpy.int32))
	c5 = pyfits.Column(name='TIMERANG',  format='2E',  array=numpy.array(times, dtype=numpy.float32))
	c6 = pyfits.Column(name='BANDS',     format='1J',  array=numpy.ones((nFlags,), dtype=numpy.int32))
	c7 = pyfits.Column(name='CHANS',     format='2J',  array=numpy.array(chans, dtype=numpy.int32))
	c8 = pyfits.Column(name='PFLAGS',    format='4J',  array=numpy.array(pols, dtype=numpy.int32))
	c9 = pyfits.Column(name='REASON',    format='A40', array=numpy.array(['FLAGIDI.PY' for i in xrange(nFlags)]))
	c10 = pyfits.Column(name='SEVERITY', format='1J',  array=numpy.zeros((nFlags,), dtype=numpy.int32)-1)
	colDefs = pyfits.ColDefs([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
	### The table itself
	flags = pyfits.new_table(colDefs)
	### The header
	flags.header['EXTNAME'] = ('FLAG', 'FITS-IDI table name')
	flags.header['EXTVER'] = (1, 'table instance number') 
	flags.header['TABREV'] = (2, 'table format revision number')
	for key in ('NO_STKD', 'STK_1', 'NO_BAND', 'NO_CHAN', 'REF_FREQ', 'CHAN_BW', 'REF_PIXL', 'OBSCODE', 'ARRNAM', 'RDATE'):
		flags.header[key] = (uvdata.header[key], uvdata.header.comments[key])
		
	# Clean up the old FLAG tables, if any, and then insert the new table where it needs to be
	## Find old tables
	toRemove = []
	for hdu in hdulist:
		try:
			if hdu.header['EXTNAME'] == 'FLAG':
				toRemove.append( hdu )
		except KeyError:
			pass
	## Remove old tables
	for hdu in toRemove:
		ver = hdu.header['EXTVER']
		del hdulist[hdulist.index(hdu)]
		print "  WARNING: removing old FLAG table - version %i" % ver
	## Insert the new table right before UV_DATA
	hdulist.insert(-1, flags)
	
	# Save
	
	outname = os.path.basename(filename)
	outname, outext = os.path.splitext(outname)
	outname = '%s_flagged%s' % (outname, outext)
	hdulist2 = pyfits.open(outname, mode='append')
	primary =	pyfits.PrimaryHDU()
	for key in hdulist[0].header:
		primary.header[key] = (hdulist[0].header[key], hdulist[0].header.comments[key])
	hdulist2.append(primary)
	hdulist2.flush()
	for hdu in hdulist[1:]:
		hdulist2.append(hdu)
		hdulist2.flush()
	hdulist2.close()
	print "  Flagged FITS IDI file is '%s'" % outname


if __name__ == "__main__":
	numpy.seterr(all='ignore')
	main(sys.argv[1:])
	
