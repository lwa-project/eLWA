#!/usr/bin/env python

"""
Simple script to take in a collection of observation files and build up a 
superCorrelator.py configuration script.
"""

import os
import sys
import numpy
from datetime import datetime

from lsl.reader import drx, vdif, errors

from utils import *
from get_vla_ant_pos import database


VLA_ECEF = numpy.array((-1601185.4, -5041977.5, 3554875.9))

LWA_ECEF = numpy.array((-1602206.5890935822, -5042244.2888975842, 3554076.3184691621))
LWA_LAT = 34.070 * numpy.pi/180
LWA_LON = -107.628 * numpy.pi/180
LWA_ROT = numpy.array([[ numpy.sin(LWA_LAT)*numpy.cos(LWA_LON), numpy.sin(LWA_LAT)*numpy.sin(LWA_LON), -numpy.cos(LWA_LAT)], 
				   [-numpy.sin(LWA_LON),                    numpy.cos(LWA_LON),                    0                  ],
				   [ numpy.cos(LWA_LAT)*numpy.cos(LWA_LON), numpy.cos(LWA_LAT)*numpy.sin(LWA_LON),  numpy.sin(LWA_LAT)]])


def main(args):
	filenames = args
	
	# Check if the first argument on the command line is a directory.  If so, 
	# use what is in that directory
	if os.path.isdir(filenames[0]):
		filenames = [os.path.join(filenames[0], filename) for filename in os.listdir(filenames[0])]
		filenames.sort()
		
	# Open the database connection to NRAO to find the antenna locations
	db = database('params')
	
	# Setup what we need to write out a configration file
	config = {'source': {'name':'', 'ra2000':'', 'dec2000':''}, 
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
			config['inputs'].append( {'file': filename, 'type': 'DRX', 
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
			config['source']['name'] = header['SRC_NAME']
			config['source']['ra2000'] = header['RA_STR']
			config['source']['dec2000'] = header['DEC_STR']
			config['inputs'].append( {'file': filename, 'type': 'VDIF', 
								 'antenna': 'EA%02i' % antID, 'pols': 'Y, X', 
								 'location': (enz[0], enz[1], enz[2]),
								 'clockoffset': (0.0, 0.0), 'fileoffset': 0, 
								 'pad': pad, 'tstart': tStart, 'freq':header['OBSFREQ']} )
		
		# Done
		fh.close()
		
	# Close out the connection to NRAO
	db.close()
	
	# Update the file offsets to get things lined up better
	tMax = max([input['tstart'] for input in config['inputs']])
	for input in config['inputs']:
		diff = tMax - input['tstart']
		offset = diff.days*86400 + diff.seconds + diff.microseconds/1e6
		input['fileoffset'] = max([0, offset])
		
	# Render the configuration
	## Preample
	print "# Created"
	print "#  on %s" % datetime.now()
	print "#  using %s" % os.path.basename(__file__)
	print ""
	## Source
	print "Source"
	print "  Name     %s" % config['source']['name']
	print "  RA2000   %s" % config['source']['ra2000']
	print "  Dec2000  %s" % config['source']['dec2000']
	print "SourceDone"
	print ""
	## Input files
	for input in config['inputs']:
		print "Input"
		print "# Start time is %s" % input['tstart']
		try:
			print "# VLA pad is %s" % input['pad']
		except KeyError:
			pass
		try:
			print "# Frequency tuning 1 is %.3f Hz" % input['freq'][0]
			print "# Frequency tuning 2 is %.3f Hz" % input['freq'][1]
		except TypeError:
			print "# Frequency tuning is %.3f Hz" % input['freq']
		print "  File         %s" % input['file']
		print "  Type         %s" % input['type']
		print "  Antenna      %s" % input['antenna']
		print "  Pols         %s" % input['pols']
		print "  Location     %.6f, %.6f, %.6f" % input['location']
		print "  ClockOffset  %.3f, %.3f" % input['clockoffset']
		print "  FileOffset   %.3f" % input['fileoffset']
		print "InputDone"
		print ""


if __name__ == "__main__":
	main(sys.argv[1:])
	
