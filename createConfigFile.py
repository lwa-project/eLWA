#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to take in a collection of observation files and build up a 
superCorrelator.py configuration script.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import sys
import numpy
import getopt
from datetime import datetime

from lsl.reader import drx, vdif, errors

from utils import *
from get_vla_ant_pos import database


def usage(exitCode=None):
	print """createConfigFile.py - Given a collection of LWA/VLA data files, or a directory
containing LWA/VLA data files, generate a configuration file for 
superCorrelator.py.

Usage:
createConfigFile.py [OPTIONS] file [file [...]]
 or 
createConfigFile.py [OPTIONS] directory

Options:
-h, --help                  Display this help information
-o, --output                Write the configuration to the specified file
                            (default = standard out)
"""
	
	if exitCode is not None:
		sys.exit(exitCode)
	else:
		return True


def parseConfig(args):
	config = {}
	# Command line flags - default values
	config['output'] = None
	config['args'] = []
	
	# Read in and process the command line flags
	try:
		opts, args = getopt.getopt(args, "ho:", ["help", "output="])
	except getopt.GetoptError, err:
		# Print help information and exit:
		print str(err) # will print something like "option -a not recognized"
		usage(exitCode=2)
		
	# Work through opts
	for opt, value in opts:
		if opt in ('-h', '--help'):
			usage(exitCode=0)
		elif opt in ('-o','--output'):
			config['output'] = value
		else:
			assert False
			
	# Add in arguments
	config['args'] = args
	
	# Return configuration
	return config


VLA_ECEF = numpy.array((-1601185.4, -5041977.5, 3554875.9))

LWA_ECEF = numpy.array((-1602206.5890935822, -5042244.2888975842, 3554076.3184691621))
LWA_LAT = 34.070 * numpy.pi/180
LWA_LON = -107.628 * numpy.pi/180
LWA_ROT = numpy.array([[ numpy.sin(LWA_LAT)*numpy.cos(LWA_LON), numpy.sin(LWA_LAT)*numpy.sin(LWA_LON), -numpy.cos(LWA_LAT)], 
				   [-numpy.sin(LWA_LON),                    numpy.cos(LWA_LON),                    0                  ],
				   [ numpy.cos(LWA_LAT)*numpy.cos(LWA_LON), numpy.cos(LWA_LAT)*numpy.sin(LWA_LON),  numpy.sin(LWA_LAT)]])


def main(args):
	# Parse the command line
	config = parseConfig(args)
	filenames = config['args']
	
	# Check if the first argument on the command line is a directory.  If so, 
	# use what is in that directory
	if os.path.isdir(filenames[0]):
		filenames = [os.path.join(filenames[0], filename) for filename in os.listdir(filenames[0])]
		filenames.sort()
		
	# Open the database connection to NRAO to find the antenna locations
	db = database('params')
	
	# Setup what we need to write out a configration file
	corrConfig = {'source': {'name':'', 'ra2000':'', 'dec2000':''}, 
			    'inputs': []}
	
	for filename in filenames:
		#print "%s:" % os.path.basename(filename)

		# Open the file
		fh = open(filename, 'rb')
		
		# Figure out what to do with the file
		ext = os.path.splitext(filename)[1]
		if ext == '':
			## DRX
			## Read in the first few frames to get the start time
			frame0 = drx.readFrame(fh)
			frame1 = drx.readFrame(fh)
			frame2 = drx.readFrame(fh)
			frame3 = drx.readFrame(fh)
			freq1, freq2 = None, None
			for frame in (frame0, frame1, frame2, frame3):
				if frame.parseID()[1] == 1:
					freq1 = frame.getCentralFreq()
				else:
					freq2 = frame.getCentralFreq()
			tStart = datetime.utcfromtimestamp(frame0.getTime())
			
			## Save
			corrConfig['inputs'].append( {'file': filename, 'type': 'DRX', 
									'antenna': 'LWA1', 'pols': 'X, Y', 
									'location': (0.0, 0.0, 0.0), 
									'clockoffset': (0.0, 0.0), 'fileoffset': 0, 
									'tstart': tStart, 'freq':(freq1,freq2)} )
									
		elif ext == '.vdif':
			## VDIF
			## Read in the GUPPI header
			header = readGUPPIHeader(fh)
			#print "  Source: %s" % header['SRC_NAME']
			#print "  Frequency: %.3f MHz" % (header['OBSFREQ']/1e6,)
			#print "  Bandwidth: %.3f MHz" % header['CHAN_BW']
		
			## Read in the first frame
			vdif.FrameSize = vdif.getFrameSize(fh)
			frame = vdif.readFrame(fh)
			antID = frame.parseID()[0] - 12300
			tStart =  datetime.utcfromtimestamp(frame.getTime())
			nThread = vdif.getThreadCount(fh)
			#print "  Station: %s" % antID
			#print "  Start Time: %s" % tStart
		
			## Find the antenna location
			pad, edate = db.get_pad('EA%02i' % antID, tStart)
	    		x,y,z = db.get_xyz(pad, tStart)
			#print "  Pad: %s" % pad
			#print "  VLA relative XYZ: %.3f, %.3f, %.3f" % (x,y,z)
			
			## Move into the LWA1 coodinate system
			### relative to ECEF
			xyz = numpy.array([x,y,z])
			xyz += VLA_ECEF
			### ECEF to LWA1
			rho = xyz - LWA_ECEF
			sez = numpy.dot(LWA_ROT, rho)
			enz = sez[[1,0,2]]
			enz[1] *= -1
			
			## Save
			corrConfig['source']['name'] = header['SRC_NAME']
			corrConfig['source']['ra2000'] = header['RA_STR']
			corrConfig['source']['dec2000'] = header['DEC_STR']
			corrConfig['inputs'].append( {'file': filename, 'type': 'VDIF', 
									'antenna': 'EA%02i' % antID, 'pols': 'Y, X', 
									'location': (enz[0], enz[1], enz[2]),
									'clockoffset': (0.0, 0.0), 'fileoffset': 0, 
									'pad': pad, 'tstart': tStart, 'freq':header['OBSFREQ']} )
									
		# Done
		fh.close()
		
	# Close out the connection to NRAO
	db.close()
	
	# Update the file offsets to get things lined up better
	tMax = max([input['tstart'] for input in corrConfig['inputs']])
	for input in corrConfig['inputs']:
		diff = tMax - input['tstart']
		offset = diff.days*86400 + diff.seconds + diff.microseconds/1e6
		input['fileoffset'] = max([0, offset])
		
	# Render the configuration
	## Setup
	if config['output'] is None:
		fh = sys.stdout
	else:
		fh = open(config['output'], 'w')
	## Preample
	fh.write("# Created\n")
	fh.write("#  on %s\n" % datetime.now())
	fh.write("#  using %s, revision $Rev$\n" % os.path.basename(__file__))
	fh.write("\n")
	## Source
	fh.write("Source\n")
	fh.write("  Name     %s\n" % corrConfig['source']['name'])
	fh.write("  RA2000   %s\n" % corrConfig['source']['ra2000'])
	fh.write("  Dec2000  %s\n" % corrConfig['source']['dec2000'])
	fh.write("SourceDone\n")
	fh.write("\n")
	## Input files
	for input in corrConfig['inputs']:
		fh.write("Input\n")
		fh.write("# Start time is %s\n" % input['tstart'])
		try:
			fh.write("# VLA pad is %s\n" % input['pad'])
		except KeyError:
			pass
		try:
			fh.write("# Frequency tuning 1 is %.3f Hz\n" % input['freq'][0])
			fh.write("# Frequency tuning 2 is %.3f Hz\n" % input['freq'][1])
		except TypeError:
			fh.write("# Frequency tuning is %.3f Hz\n" % input['freq'])
		fh.write("  File         %s\n" % input['file'])
		fh.write("  Type         %s\n" % input['type'])
		fh.write("  Antenna      %s\n" % input['antenna'])
		fh.write("  Pols         %s\n" % input['pols'])
		fh.write("  Location     %.6f, %.6f, %.6f\n" % input['location'])
		fh.write("  ClockOffset  %.3f, %.3f\n" % input['clockoffset'])
		fh.write("  FileOffset   %.3f\n" % input['fileoffset'])
		fh.write("InputDone\n")
		fh.write("\n")
	if fh != sys.stdout:
		fh.close()


if __name__ == "__main__":
	main(sys.argv[1:])
	