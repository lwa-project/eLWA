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
import ephem
import numpy
import getopt
from datetime import datetime, timedelta

from lsl.reader import drx, vdif, errors
from lsl.common import metabundle, metabundleADP
from lsl.common.mcs import mjdmpm2datetime

import guppi
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
-l, --lwa1-offset           LWA1 clock offset (default = 0)
-s, --lwasv-offset          LWA-SV clock offset (default = 0)
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
	config['lwa1Offset'] = 0.0
	config['lwasvOffset'] = 0.0
	config['args'] = []
	
	# Read in and process the command line flags
	try:
		opts, args = getopt.getopt(args, "hl:s:o:", ["help", "lwa1-offset=", "lwasv-offset=", "output="])
	except getopt.GetoptError, err:
		# Print help information and exit:
		print str(err) # will print something like "option -a not recognized"
		usage(exitCode=2)
		
	# Work through opts
	for opt, value in opts:
		if opt in ('-h', '--help'):
			usage(exitCode=0)
		elif opt in ('-l', '--lwa1-offset'):
			config['lwa1Offset'] = value
		elif opt in ('-s', '--lwasv-offset'):
			config['lwasvOffset'] = value
		elif opt in ('-o','--output'):
			config['output'] = value
		else:
			assert False
			
	# Add in arguments
	config['args'] = args
	
	# Return configuration
	return config


VLA_ECEF = numpy.array((-1601185.4, -5041977.5, 3554875.9))

## Derived from the 2017 Oct 31 LWA1 SSMIF
LWA1_ECEF = numpy.array((-1602258.2104158669, -5042300.0220439518, 3553974.6599673284))
LWA1_LAT =   34.068894 * numpy.pi/180
LWA1_LON = -107.628350 * numpy.pi/180
LWA1_ROT = numpy.array([[ numpy.sin(LWA1_LAT)*numpy.cos(LWA1_LON), numpy.sin(LWA1_LAT)*numpy.sin(LWA1_LON), -numpy.cos(LWA1_LAT)], 
                        [-numpy.sin(LWA1_LON),                     numpy.cos(LWA1_LON),                      0                  ],
                        [ numpy.cos(LWA1_LAT)*numpy.cos(LWA1_LON), numpy.cos(LWA1_LAT)*numpy.sin(LWA1_LON),  numpy.sin(LWA1_LAT)]])

## Derived from the 2017 Oct 27 LWA-SV SSMIF
LWASV_ECEF = numpy.array((-1531554.7717322097, -5045440.9839560054, 3579249.988606174))
LWASV_LAT =   34.348358 * numpy.pi/180
LWASV_LON = -106.885783 * numpy.pi/180
LWASV_ROT = numpy.array([[ numpy.sin(LWASV_LAT)*numpy.cos(LWASV_LON), numpy.sin(LWASV_LAT)*numpy.sin(LWASV_LON), -numpy.cos(LWASV_LAT)], 
                         [-numpy.sin(LWASV_LON),                      numpy.cos(LWASV_LON),                       0                   ],
                         [ numpy.cos(LWASV_LAT)*numpy.cos(LWASV_LON), numpy.cos(LWASV_LAT)*numpy.sin(LWASV_LON),  numpy.sin(LWASV_LAT)]])


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
	
	# Pass 1 - Get the LWA metadata so we know where we are pointed
	sources = []
	metadata = {}
	lwasite = {}
	for filename in filenames:
		# Figure out what to do with the file
		ext = os.path.splitext(filename)[1]
		if ext == '.tgz':
			## LWA Metadata
			try:
				## Extract the SDF
				if len(sources) == 0:
					try:
						sdf = metabundle.getSessionDefinition(filename)
					except Exception as e:
						sdf = metabundleADP.getSessionDefinition(filename)
					for obs in sdf.sessions[0].observations:
						ra = ephem.hours(str(obs.ra))
						dec = ephem.hours(str(obs.dec))
						tStart = mjdmpm2datetime(obs.mjd, obs.mpm)
						tStop  = mjdmpm2datetime(obs.mjd, obs.mpm+obs.dur)
						sources.append( {'name':obs.target, 'ra2000':ra, 'dec2000':dec, 'start':tStart, 'stop':tStop} )
						
				## Extract the file information so that we can pair things together
				fileInfo = metabundle.getSessionMetaData(filename)
				for obsID in fileInfo.keys():
					metadata[fileInfo[obsID]['tag']] = filename
					
				## Figure out LWA1 vs LWA-SV
				try:
					cs = metabundle.getCommandScript(filename)
					for c in cs:
						if c['subsystemID'] == 'DP':
							site = 'LWA1'
							break
						elif c['subsystemID'] == 'ADP':
							site = 'LWA-SV'
							break
				except ValueError:
					site = 'LWA-SV'
				for obsID in fileInfo.keys():
					lwasite[fileInfo[obsID]['tag']] = site
					
			except Exception as e:
				sys.stderr.write("ERROR reading metadata file: %s\n" % str(e))
				sys.stderr.flush()
				
	# Setup what we need to write out a configration file
	corrConfig = {'source': {'name':'', 'ra2000':'', 'dec2000':''}, 
			    'inputs': []}
	
	metadata = {}
	for filename in filenames:
		#print "%s:" % os.path.basename(filename)
		
		# Skip over empty files
		if os.path.getsize(filename) == 0:
			continue
			
		# Open the file
		fh = open(filename, 'rb')
		
		# Figure out what to do with the file
		ext = os.path.splitext(filename)[1]
		if ext == '':
			## DRX
			try:
				## Get the site
				try:
					sitename = lwasite[os.path.basename(filename)]
				except KeyError:
					sitename = 'LWA1'
					
				## Get the location so that we can set site-specific parameters
				if sitename == 'LWA1':
					xyz = LWA1_ECEF
					off = config['lwa1Offset']
				elif sitename == 'LWA-SV':
					xyz = LWASV_ECEF
					off = config['lwasvOffset']
				else:
					raise RuntimeError("Unknown LWA site '%s'" % site)
					
				## Move into the LWA1 coodinate system
				### ECEF to LWA1
				rho = xyz - LWA1_ECEF
				sez = numpy.dot(LWA1_ROT, rho)
				enz = sez[[1,0,2]]
				enz[1] *= -1
				
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
				
				## Read in the last few frames to find the end time
				fh.seek(os.path.getsize(filename) - 4*drx.FrameSize)
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
				tStop = datetime.utcfromtimestamp(frame0.getTime())
				
				## Save
				corrConfig['inputs'].append( {'file': filename, 'type': 'DRX', 
										'antenna': sitename, 'pols': 'X, Y', 
										'location': (enz[0], enz[1], enz[2]), 
										'clockoffset': (off, off), 'fileoffset': 0, 
										'tstart': tStart, 'tstop': tStop, 'freq':(freq1,freq2)} )
										
			except Exception as e:
				sys.stderr.write("ERROR reading DRX file: %s\n" % str(e))
				sys.stderr.flush()
				
		elif ext == '.vdif':
			## VDIF
			try:
				## Read in the GUPPI header
				header = readGUPPIHeader(fh)
				
				## Read in the first frame
				vdif.FrameSize = vdif.getFrameSize(fh)
				frame = vdif.readFrame(fh)
				antID = frame.parseID()[0] - 12300
				tStart =  datetime.utcfromtimestamp(frame.getTime())
				nThread = vdif.getThreadCount(fh)
				
				## Read in the last frame
				nJump = int(os.path.getsize(filename)/vdif.FrameSize)
				nJump -= 4
				fh.seek(nJump*vdif.FrameSize, 1)
				mark = fh.tell()
				frame = vdif.readFrame(fh)
				tStop = datetime.utcfromtimestamp(frame.getTime())
			
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
				rho = xyz - LWA1_ECEF
				sez = numpy.dot(LWA1_ROT, rho)
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
										'pad': pad, 'tstart': tStart, 'tstop': tStop, 'freq':header['OBSFREQ']} )
										
			except Exception as e:
				sys.stderr.write("ERROR reading VDIF file: %s\n" % str(e))
				sys.stderr.flush()
				
		elif ext == '.raw':
			## GUPPI Raw
			try:
				## Read in the GUPPI header
				header = readGUPPIHeader(fh)
				
				## Read in the first frame
				guppi.FrameSize = guppi.getFrameSize(fh)
				frame = guppi.readFrame(fh)
				antID = frame.parseID()[0] - 12300
				tStart =  datetime.utcfromtimestamp(frame.getTime())
				nThread = guppi.getThreadCount(fh)
				
				## Read in the last frame
				nJump = int(os.path.getsize(filename)/guppi.FrameSize)
				nJump -= 4
				fh.seek(nJump*guppi.FrameSize, 1)
				mark = fh.tell()
				frame = guppi.readFrame(fh)
				tStop = datetime.utcfromtimestamp(frame.getTime())
			
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
				rho = xyz - LWA1_ECEF
				sez = numpy.dot(LWA1_ROT, rho)
				enz = sez[[1,0,2]]
				enz[1] *= -1
				### z offset from pad height to elevation bearing
				enz[2] += 11.0
				
				## Save
				corrConfig['source']['name'] = header['SRC_NAME']
				corrConfig['source']['ra2000'] = header['RA_STR']
				corrConfig['source']['dec2000'] = header['DEC_STR']
				corrConfig['inputs'].append( {'file': filename, 'type': 'GUPPI', 
										'antenna': 'EA%02i' % antID, 'pols': 'Y, X', 
										'location': (enz[0], enz[1], enz[2]),
										'clockoffset': (0.0, 0.0), 'fileoffset': 0, 
										'pad': pad, 'tstart': tStart, 'tstop': tStop, 'freq':header['OBSFREQ']} )
										
			except Exception as e:
				sys.stderr.write("ERROR reading GUPPI file: %s\n" % str(e))
				sys.stderr.flush()
				
		elif ext == '.tgz':
			## LWA Metadata
			try:
				## Extract the file information so that we can pair things together
				fileInfo = metabundle.getSessionMetaData(filename)
				for obsID in fileInfo.keys():
					metadata[fileInfo[obsID]['tag']] = filename
					
			except Exception as e:
				sys.stderr.write("ERROR reading metadata file: %s\n" % str(e))
				sys.stderr.flush()
				
		# Done
		fh.close()
		
	# Close out the connection to NRAO
	db.close()
	
	# Choose a VDIF reference file, if there is one, and mark whether or 
	# not DRX files were found
	vdifRefFile = None
	isDRX = False
	for input in corrConfig['inputs']:
		if input['type'] in ('VDIF', 'GUPPI'):
			if vdifRefFile is None:
				vdifRefFile = input
		elif input['type'] == 'DRX':
				isDRX = True
			
	# Set a state variable so that we can generate a warning about missing
	# DRX files
	drxFound = False
	
	# Purge DRX files that don't make sense
	toPurge = []
	drxFound = False
	lwasvFound = False
	for input in corrConfig['inputs']:
		### Sort out multiple DRX files - this only works if we have only one LWA station
		if input['type'] == 'DRX' and vdifRefFile is not None:
			l0, l1 = input['tstart'], input['tstop']
			v0, v1 = vdifRefFile['tstart'], vdifRefFile['tstop']
			ve = (v1 - v0).total_seconds()
			overlapWithVDIF = (v0>=l0 and v0<l1) or (l0>=v0 and l0<v1)
			lvo = (min([v1,l1]) - max([v0,l0])).total_seconds()
			if not overlapWithVDIF or lvo < 0.25*ve:
				toPurge.append( input )
			drxFound = True
			if input['antenna'] == 'LWA-SV':
				lwasvFound = True
	for input in toPurge:
		del corrConfig['inputs'][corrConfig['inputs'].index(input)]
		
	# VDIF/DRX warning check/report
	if isDRX and not drxFound:
		sys.stderr.write("WARNING: DRX files provided but none overlapped with VDIF data")
		
	# Update the file offsets to get things lined up better
	tMax = max([input['tstart'] for input in corrConfig['inputs']])
	for input in corrConfig['inputs']:
		diff = tMax - input['tstart']
		offset = diff.days*86400 + diff.seconds + diff.microseconds/1e6
		input['fileoffset'] = max([0, offset])
		
	# Reconcile the source lists for when we have eLWA data.  This is needed so
	# that we use the source information contained in the VDIF files rather than
	# the stub information contained in the SDFs
	if len(sources) <= 1:
		if corrConfig['source']['name'] != '':
			## Update the source information with what comes from the VLA
			try:
				sources[0] = corrConfig['source']
			except IndexError:
				sources.append( corrConfig['source'] )
	# Update the dwell time using the minimum on-source time for all inputs if 
	# there is only one source, i.e., for full eLWA runs
	if len(sources) == 1:
		sources[0]['start'] = max([input['tstart'] for input in corrConfig['inputs']])
		sources[0]['stop'] = min([input['tstop'] for input in corrConfig['inputs']])
		
	# Render the configuration
	startRef = sources[0]['start']
	for s,source in enumerate(sources):
		startOffset = source['start'] - startRef
		startOffset = startOffset.total_seconds()
		
		dur = source['stop'] - source['start']
		dur = dur.total_seconds()
		
		## Small correction for the first scan to compenstate for stale data at LWA-SV
		if lwasvFound and s == 0:
			startOffset += 10.0
			dur -= 10.0
			
		## Setup
		if config['output'] is None:
			fh = sys.stdout
		else:
			outname = config['output']
			if len(sources) > 1:
				outname += str(s+1)
			fh = open(outname, 'w')
			
		## Preample
		fh.write("# Created\n")
		fh.write("#  on %s\n" % datetime.now())
		fh.write("#  using %s, revision $Rev$\n" % os.path.basename(__file__))
		fh.write("\n")
		## Source
		fh.write("Source\n")
		fh.write("# Observation start is %s\n" % source['start'])
		fh.write("# Duration is %s\n" % (source['stop'] - source['start'],))
		fh.write("  Name     %s\n" % source['name'])
		fh.write("  RA2000   %s\n" % source['ra2000'])
		fh.write("  Dec2000  %s\n" % source['dec2000'])
		fh.write("  Duration %.3f\n" % dur)
		fh.write("SourceDone\n")
		fh.write("\n")
		## Input files
		for input in corrConfig['inputs']:
			fh.write("Input\n")
			fh.write("# Start time is %s\n" % input['tstart'])
			fh.write("# Stop time is %s\n" % input['tstop'])
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
			try:
				metaname = metadata[os.path.basename(input['file'])]
				fh.write("  MetaData     %s\n" % metaname)
			except KeyError:
				pass
			fh.write("  Type         %s\n" % input['type'])
			fh.write("  Antenna      %s\n" % input['antenna'])
			fh.write("  Pols         %s\n" % input['pols'])
			fh.write("  Location     %.6f, %.6f, %.6f\n" % input['location'])
			fh.write("  ClockOffset  %s, %s\n" % input['clockoffset'])
			fh.write("  FileOffset   %.3f\n" % (startOffset + input['fileoffset'],))
			fh.write("InputDone\n")
			fh.write("\n")
		if fh != sys.stdout:
			fh.close()


if __name__ == "__main__":
	main(sys.argv[1:])
	
