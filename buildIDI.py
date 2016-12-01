#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Given a collection of .npz files create a FITS-IDI file that can be read in by
AIPS.

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
import tempfile
from datetime import datetime, timedelta, tzinfo

from lsl import astro
from lsl.common import stations, metabundle
from lsl.statistics import robust
from lsl.correlator import uvUtils
from lsl.correlator import fx as fxc
from lsl.writer import fitsidi
from lsl.correlator.uvUtils import computeUVW
from lsl.common.constants import c as vLight
from lsl.common.mcs import datetime2mjdmpm

from utils import readCorrelatorConfiguration


def usage(exitCode=None):
	print """buildIDI.py - Given a collection of .npz files generated by "the next 
generation of correlator", create one or more FITS IDI files containing the
data.

Usage:
buildIDI.py [OPTIONS] npz [npz [...]]

Options:
-h, --help                  Display this help information
-d, --decimate              Frequency decimation factor (default = 1)
-l, --limit                 Limit the data loaded to the first N files
                            (default = -1 = load all)
-s, --split                 Maximum number of integrations in a FITS IDI file
                            (default = 3000)
-t, --tag                   Optional tag to add to the filename
-f, --force                  Force overwritting of existing FITS-IDI files
"""
	
	if exitCode is not None:
		sys.exit(exitCode)
	else:
		return True


def parseConfig(args):
	config = {}
	# Command line flags - default values
	config['freqDecimation'] = 1
	config['lastFile'] = -1
	config['maxIntsInIDI'] = 3000
	config['outnameTag'] = None
	config['force'] = False
	config['args'] = []
	
	# Read in and process the command line flags
	try:
		opts, args = getopt.getopt(args, "hd:l:s:t:f", ["help", "decimate=", "limit=", "split=", "tag=", "force"])
	except getopt.GetoptError, err:
		# Print help information and exit:
		print str(err) # will print something like "option -a not recognized"
		usage(exitCode=2)
		
	# Work through opts
	for opt, value in opts:
		if opt in ('-h', '--help'):
			usage(exitCode=0)
		elif opt in ('-d', '--decimate'):
			config['freqDecimation'] = int(value, 10)
		elif opt in ('-l', '--limit'):
			config['lastFile'] = int(value, 10)
		elif opt in ('-s', '--split'):
			config['maxIntsInIDI'] = int(value, 10)
		elif opt in ('-t', '--tag'):
			config['outnameTag'] = value
		elif opt in ('-f', '--force'):
			config['force'] = True
		else:
			assert False
			
	# Add in arguments
	config['args'] = args
	
	# Validate
	if len(config['args']) == 0:
		raise RuntimeError("Must provide at least one .npz file to plot")
	if config['freqDecimation'] <= 0:
		raise RuntimeError("Invalid value for the frequency decimation factor")
	if config['lastFile'] <= 0 and config['lastFile'] != -1:
		raise RuntimeError("Invalid value for the last file to process")
		
	# Return configuration
	return config


def main(args):
	# Parse the command line
	config = parseConfig(args)
	
	filenames = config['args']
	filenames.sort()
	if config['lastFile'] != -1:
		filenames = filenames[:config['lastFile']]
		
	# Build up the station
	site = stations.lwa1
	observer = site.getObserver()
	
	# Load in the file file to figure out what to do
	dataDict = numpy.load(filenames[0])
	tStart = dataDict['tStart'].item()
	tInt = dataDict['tInt']
	
	freq = dataDict['freq1']
	
	cConfig = dataDict['config']
	fh, tempConfig = tempfile.mkstemp(suffix='.txt', prefix='config-')
	fh = open(tempConfig, 'w')
	for line in cConfig:
		fh.write('%s\n' % line)
	fh.close()
	refSrc, junk1, junk2, junk3, junk4, antennas = readCorrelatorConfiguration(tempConfig)
	os.unlink(tempConfig)
	
	visXX = dataDict['vis1XX'].astype(numpy.complex64)
	visXY = dataDict['vis1XY'].astype(numpy.complex64)
	visYX = dataDict['vis1YX'].astype(numpy.complex64)
	visYY = dataDict['vis1YY'].astype(numpy.complex64)
	dataDict.close()
			
	print "Antennas:"
	for ant in antennas:
		print "  Antenna %i: Stand %i, Pol. %i" % (ant.id, ant.stand.id, ant.pol)
		
	nChan = visXX.shape[1]
	blList = uvUtils.getBaselines([ant for ant in antennas if ant.pol == 0], IncludeAuto=True)
	
	
	if config['freqDecimation'] > 1:
		if nChan % config['freqDecimation'] != 0:
			raise RuntimeError("Invalid freqeunce decimation factor:  %i %% %i = %i" % (nChan, config['freqDecimation'], nChan%config['freqDecimation']))

		nChan /= config['freqDecimation']
		freq.shape = (freq.size/config['freqDecimation'], config['freqDecimation'])
		freq = freq.mean(axis=1)
		
	# Fill in the data
	for i,filename in enumerate(filenames):
		## Load in the integration
		dataDict = numpy.load(filename)

		tStart = dataDict['tStart'].item()
		tInt = dataDict['tInt'].item()
		visXX = dataDict['vis1XX'].astype(numpy.complex64)
		visXY = dataDict['vis1XY'].astype(numpy.complex64)
		visYX = dataDict['vis1YX'].astype(numpy.complex64)
		visYY = dataDict['vis1YY'].astype(numpy.complex64)
		
		dataDict.close()
		
		if config['freqDecimation'] > 1:
			visXX.shape = (visXX.shape[0], visXX.shape[1]/config['freqDecimation'], config['freqDecimation'])
			visXX = visXX.mean(axis=2)
			visXY.shape = (visXY.shape[0], visXY.shape[1]/config['freqDecimation'], config['freqDecimation'])
			visXY = visXY.mean(axis=2)
			visYX.shape = (visYX.shape[0], visYX.shape[1]/config['freqDecimation'], config['freqDecimation'])
			visYX = visYX.mean(axis=2)
			visYY.shape = (visYY.shape[0], visYY.shape[1]/config['freqDecimation'], config['freqDecimation'])
			visYY = visYY.mean(axis=2)
			
		if i % config['maxIntsInIDI'] == 0:
			## Clean up the previous file
			try:
				fits.write()
				fits.close()
			except NameError:
				pass
				
			## Create the FITS-IDI file as needed
			### What to call it
			if config['outnameTag'] is None:
				outname = 'buildIDI_%s.FITS_%i' % (refSrc.name, i/config['maxIntsInIDI']+1,)
			else:
				outname = 'buildIDI_%s_%s.FITS_%i' % (refSrc.name, config['outnameTag'], i/config['maxIntsInIDI']+1,)
				
			### Does it already exist or not
			if os.path.exists(outname):
				if not config['force']:
					yn = raw_input("WARNING: '%s' exists, overwrite? [Y/n] " % outname)
				else:
					yn = 'y'
					
				if yn not in ('n', 'N'):
					os.unlink(outname)
				else:
					raise RuntimeError("Output file '%s' already exists" % outname)
					
			### Create the file
			fits = fitsidi.IDI(outname, refTime=tStart)
			fits.setStokes(['XX', 'XY', 'YX', 'YY'])
			fits.setFrequency(freq)
			fits.setGeometry(stations.lwa1, [a for a in antennas if a.pol == 0])
			print "Opening %s for writing" % outname

		if i % 10 == 0:
			print i
			
		## Update the observation
		observer.date = datetime.utcfromtimestamp(tStart).strftime('%Y/%m/%d %H:%M:%S.%f')
		refSrc.compute(observer)
		
		## Convert the setTime to a MJD and save the visibilities to the FITS IDI file
		obsTime = astro.unix_to_taimjd(tStart)
		fits.addDataSet(obsTime, tInt, blList, visXX, pol='XX', source=refSrc)
		fits.addDataSet(obsTime, tInt, blList, visXY, pol='XY', source=refSrc)
		fits.addDataSet(obsTime, tInt, blList, visYX, pol='YX', source=refSrc)
		fits.addDataSet(obsTime, tInt, blList, visYY, pol='YY', source=refSrc)
		
	# Cleanup the last file
	fits.write()
	fits.close()


if __name__ == "__main__":
	main(sys.argv[1:])
	