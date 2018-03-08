# -*- coding: utf-8 -*-

"""
Utility module for the various scripts needed to correlate LWA and VLA data.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import re
import ephem
import numpy
from datetime import datetime

from lsl import astro
from lsl.common import stations
from lsl.reader import drx, vdif
from lsl.common.dp import fS
from lsl.common.mcs import datetime2mjdmpm, delaytoMCSD, MCSDtodelay
from lsl.common.metabundle import getCommandScript
from lsl.common.metabundleADP import getCommandScript as getCommandScriptADP
from lsl.misc.beamformer import calcDelay

import guppi


__version__ = '0.3'
__revision__ = '$Rev$'
__all__ = ['EnhancedFixedBody', 'multiColumnPrint', 'parseTimeString', 'readCorrelatorConfiguration', 
		 'getBetterTime', 'readGUPPIHeader', 'parseLWAMetaData', 'PolyCo', 'PolyCos', 
		 '__version__', '__revision__', '__all__']


# List of bright radio sources and pulsars in PyEphem format
_srcs = ["ForA,f|J,03:22:41.70,-37:12:30.0,1",
         "TauA,f|J,05:34:32.00,+22:00:52.0,1", 
         "VirA,f|J,12:30:49.40,+12:23:28.0,1",
         "HerA,f|J,16:51:08.15,+04:59:33.3,1", 
         "SgrA,f|J,17:45:40.00,-29:00:28.0,1", 
         "CygA,f|J,19:59:28.30,+40:44:02.0,1", 
         "CasA,f|J,23:23:27.94,+58:48:42.4,1",
	    "3C196,f|J,08:13:36.06,+48:13:02.6,1",
	    "3C286,f|J,13:31:08.29,+30:30:33.0,1",
	    "3C295,f|J,14:11:20.47,+52:12:09.5,1", ]


def _topo2baryMJD(src, observer):
	"""
	Given an EnhancedFixedBody instance, use TEMPO to determine an 
	interpolating function that maps topocentric time to barycentric time.  
	"""
	
	from residuals import read_residuals
	
	# Source position
	ra = str(src.a_ra)
	dec = str(src.a_dec)
	
	# Observatory for the observation - fixed on LWA1 until I can figure out how to do better
	obs = 'LW'
	
	# Topocentric times to compute the barycentric times for
	# NOTE:  This includes padding at the end for the fitting in TEMPO
	if type(observer) is ephem.Observer:
		topoMJD = observer.date + (astro.DJD_OFFSET - astro.MJD_OFFSET)
	elif type(observer) is ephem.Date:
		topoMJD = observer + (astro.DJD_OFFSET - astro.MJD_OFFSET)
	else:
		topoMJD = observer
	topoMJD = topoMJD + numpy.linspace(0, 0.1, 11)
	
	# Write the TEMPO file for the conversion from topocentric to barycentric times
	fh = open('bary.tmp', 'w')
	fh.write("""C  Header Section
  HEAD                   
  PSR                 bary
  NPRNT                  2
  P0                   1.0 1
  P1                   0.0
  CLK            UTC(NIST)
  PEPOCH           %19.13f
  COORD              J2000
  RA                    %s
  DEC                   %s
  DM                   0.0
  EPHEM              DE200
C  TOA Section (uses ITAO Format)
C  First 8 columns must have + or -!
  TOA\n""" % (topoMJD[0], ra, dec))
	for t in topoMJD:
		fh.write("topocen+ %19.13f  0.00     0.0000  0.000000  %s\n" % (t, obs))
	fh.close()
	
	# Run TEMPO
	status = os.system('tempo bary.tmp > barycorr.out')
	if status != 0:
		## This didn't work, skipping
		print "WARNING: Could not run TEMPO, skipping conversion function calculation"
		baryMJD = topoMJD[0]
	else:
		## Read in the barycentric times
		resids = read_residuals()
		baryMJD = resids.bary_TOA[0]
		
	# Cleanup
	for filename in ('bary.tmp', 'bary.par', 'barycorr.out', 'resid2.tmp', 'tempo.lis', 'matrix.tmp'):
		try:
			os.unlink(filename)
		except OSError:
			pass
			
	# Return
	return baryMJD


class EnhancedFixedBody(ephem.FixedBody):
	"""
	Sub-class of ephem.FixedBody that allows for pulsar phase and frequency 
	calculation.  This is done through a new '_polycos' attribute that is set
	after initilization to a PolyCos instance.  On calling the 'compute' 
	method, the 'phase' and 'frequency' attributes are set.
	"""
	
	def __setattr__(self, name, value):
		# Validate that the _polycos attribute is set to a PolyCo or PolyCos instance
		if name == '_polycos':
			if type(value) not in (PolyCo, PolyCos):
				raise ValueError("Must set _polycos with a PolyCo or PolyCos instance")
				
		# Set the attribute if everything is ok
		super(self.__class__, self).__setattr__(name, value)
		
	def __getattr__(self, name):
		if name in ('phase', 'frequency', 'period'):
			## Is this even valid?
			if getattr(self, '_polycos', None) is None:
				raise ValueError("Pulsar parameters cannot be determined for a non-pulsar body")
				
		# Get the attribute if every is ok
		super(self.__class__, self).__getattr__(name)
		
	def compute(self, when=None, epoch=ephem.J2000):
		# Basic validation
		if when is None:
			when = ephem.now()
			
		# Compute the basic source parameters via PyEphem
		if type(when) is ephem.Observer:
			super(self.__class__, self).compute(when)
		else:
			super(self.__class__, self).compute(when, epoch=epoch)
			
		# If there are polycos, compute the phase and frequency as well
		if getattr(self, '_polycos', None) is not None:
			## We can only really do this if we have an observer so
			## that we can move to the barycenter
			if type(when) is ephem.Observer:
				mjd = _topo2baryMJD(self, when)
				self.dm = self._polycos.getDM(mjd)
				self.phase = self._polycos.getPhase(mjd)
				self.frequency = self._polycos.getFrequency(mjd)
				self.period = 1.0/self.frequency
			else:
				delattr(self, 'dm')
				delattr(self, 'phase')
				delattr(self, 'frequency')
				delattr(self, 'period')


def multiColumnPrint(items, sep=';  ', width=86):
	"""
	Multi-column print statement for lists.
	"""
	
	# Find the longest item in the list so that we can make aligned columns.
	maxLen = 0
	for i in items:
		l = len(str(i))
		if l > maxLen:
			maxLen = l
	formatter = "%%%is" % maxLen
	
	# Find out how many columns to make using the width the print over, the 
	# maximum item size, and the size of the separator.  When doing this also
	# make sure that we have at least one column
	nCol = width / (maxLen+len(sep))
	if nCol < 1:
		nCol = 1
		
	# Figure out how many rows to use.  This needs to take into acount partial
	# rows with len(items) % nCol != 0.
	nRow = len(items) / nCol + (0 if (len(items) % nCol) == 0 else 1)
	
	# Print
	for r in xrange(nRow):
		## Build up the line
		out = sep.join([formatter % str(i) for i in items[r*nCol:(r+1)*nCol]])
		## Add the separator at the end if this isn't the last line
		if r != nRow-1:
			out += sep
		## Print
		print out


_timeRE = re.compile('[ \t]*(?P<value>[+-]?\d*\.?\d*([Ee][+-]?\d*)?)[ \t]*(?P<unit>[mun]?s)?[ \t]*')
def parseTimeString(value):
	"""
	Given a time in the format of "decimal_value unit", convert the string 
	to a floating point time in seconds.  Valid units are:
	 * ks - kiloseconds
	 * s - seconds
	 * ms - milliseconds
	 * us - microseconds
	 * ns - nanoseconds
	 
	If no units are provided, the value is assumed to be in seconds.
	"""
	
	try:
		value = float(value)
	except ValueError:
		mtch = _timeRE.match(value)
		if mtch is None:
			raise ValueError("Invalid literal for parseTimeString(): %s" % value)
		value = float(mtch.group('value'))
		unit = mtch.group('unit')
		if unit is not None:
			if unit == 'ks':
				value *= 1e3
			elif unit == 'ms':
				value *= 1e-3
			elif unit == 'us':
				value *= 1e-6
			elif unit == 'ns':
				value *= 1e-9
				
	return value


def readCorrelatorConfiguration(filename):
	"""
	Parse a correlator configuration file generated by createConfigFile.py and
	return a five-element tuple of:
	 * the reference source as a ephem.FixedBody-compatible instance
	 * a list of filenames, 
	 * a list of metadata tarball names, 
	 * a list of file offsets in seconds, 
	 * a list of lsl.reader modules to use, and
	 * a list of lsl.common.stations.Antenna instances for each file
	"""
	
	sources = []
	blocks = []
	
	fh = open(filename, 'r')
	for line in fh:
		if line[0] == '#':
			continue
		if len(line) < 3:
			continue
			
		line = line.strip().rstrip()
		
		if line == 'Source':
			source = {'duration':0.0}
		elif line[:4] == 'Name':
			source['name'] = line.split(None, 1)[1]
		elif line[:6] == 'RA2000':
			source['ra'] = line.split(None, 1)[1]
		elif line[:7] == 'Dec2000':
			source['dec'] = line.split(None, 1)[1]
		elif line[:6] == 'Polyco':
			source['polyco'] = line.split(None, 1)[1]
		elif line[:8] == 'Duration':
			source['duration'] = float(line.split(None, 1)[1])
		elif line == 'SourceDone':
			sources.append( source )
		elif line == 'Input':
			block = {'fileOffset':0.0}
		elif line[:4] == 'File' and line[4] != 'O':
			block['filename'] = line.split(None, 1)[1]
		elif line[:8] == 'MetaData':
			## Optional
			block['metadata'] = line.split(None, 1)[1]
		elif line[:4] == 'Type':
			block['type'] = line.split(None, 1)[1].lower()
		elif line[:7] == 'Antenna':
			block['antenna'] = line.split(None, 1)[1]
		elif line[:4] == 'Pols':
			block['pols'] = [v.strip().rstrip() for v in line.split(None, 1)[1].split(',')]
		elif line[:8] == 'Location':
			block['location'] = [float(v) for v in line.split(None, 1)[1].split(',')]
		elif line[:11] == 'ClockOffset':
			block['clockOffset'] = [parseTimeString(v) for v in line.split(None, 1)[1].split(',')]
		elif line[:10] == 'FileOffset':
			block['fileOffset'] = parseTimeString(line.split(None, 1)[1])
		elif line == 'InputDone':
			## Make sure we have a metaData key since it is optional
			if 'metadata' not in block:
				block['metadata'] = None
			blocks.append( block )
	fh.close()
	
	# Find the reference source
	if 'ra' in sources[0].keys() and 'dec' in sources[0].keys():
		refSource = EnhancedFixedBody()
		refSource.name = sources[0]['name']
		refSource._ra = sources[0]['ra']
		refSource._dec = sources[0]['dec']
		refSource._epoch = ephem.J2000
		refSource.duration = sources[0]['duration']
		try:
			refSource._polycos = PolyCos(sources[0]['polyco'])
		except KeyError:
			pass
	else:
		srcs = [ephem.Sun(),]
		for line in _srcs:
			srcs.append( ephem.readdb(line) )
		
		refSource = None
		for i in xrange(len(srcs)):
			if srcs[i].name == sources[0]['name']:
				refSource = srcs[i]
				break
			
	# Sort everything out so that the VDIF files come first
	order = sorted(range(len(blocks)), key=lambda x: blocks[x]['type'][-1])
	blocks = [blocks[o] for o in order]

	# Build up a list of filenames	
	filenames = [block['filename'] for block in blocks]
	
	# Build up a list of metadata filenames
	metanames = [block['metadata'] for block in blocks]
	
	# Build up a list of file offsets
	offsets = [block['fileOffset'] for block in blocks]
	
	# Build up a list of readers
	readers = []
	for block in blocks:
		if block['type'] == 'vdif':
			readers.append( vdif )
		elif block['type'] == 'guppi':
			readers.append( guppi )
		elif block['type'] == 'drx':
			readers.append( drx )
		else:
			readers.append( None )
			
	# Build up a list of antennas
	antennas = []
	i = 1
	for block in blocks:
		aid = None
		name = block['antenna']
		if name.lower() in ('lwa1', 'lwa-1'):
			aid = 51
		elif name.lower() in ('lwasv', 'lwa-sv'):
			aid = 52
		else:
			for j in xrange(len(name)):
				try:
					aid = int(name[j:], 10)
					if name[:j].lower() == 'lwa':
						aid += 50
					break
				except ValueError:
					pass
		pols = block['pols']
		location = block['location']
		clockOffsets = block['clockOffset']
		
		if aid is None:
			raise RuntimeError("Cannot convert antenna name '%s' to a number" % name)		
			
		stand = stations.Stand(aid, *location)
		for pol,offset in zip(pols, clockOffsets):
			cable = stations.Cable('%s-%s' % (name, pol), 0.0, vf=1.0, dd=0.0)
			cable.setClockOffset(offset)
			
			if pol.lower() == 'x':
				antenna = stations.Antenna(i, stand=stand, cable=cable, pol=0)
			else:
				antenna = stations.Antenna(i, stand=stand, cable=cable, pol=1)
						
			antennas.append( antenna )
			i += 1
			
	# Done
	return refSource, filenames, metanames, offsets, readers, antennas


def getBetterTime(frame):
	"""
	Given a lsl.reader.vdif.Frame, guppi.Frame, or lsl.reader.drx.Frame 
	instance, return a more accurate time for the frame.  Unlike the 
	Frame.getTime() functions, this function returns a two-element tuple of:
	 * integer seconds since the UNIX epoch and
	 * fractional second
	"""
	
	if type(frame) == vdif.Frame:
		epochDT = datetime(2000+frame.header.refEpoch/2, (frame.header.refEpoch % 2)*6+1, 1, 0, 0, 0, 0)
		epochMJD, epochMPM = datetime2mjdmpm(epochDT)
		sec = int(round(astro.utcjd_to_unix(epochMJD + astro.MJD_OFFSET)))
		sec += frame.header.secondsFromEpoch

		dataSize = frame.header.frameLength*8 - 32 + 16*frame.header.isLegacy
		samplesPerWord = 32 / frame.header.bitsPerSample                                # dimensionless
		nSamples = dataSize / 4 * samplesPerWord                                # bytes -> words -> samples
		frameRate = frame.header.sampleRate / nSamples
		frac = frame.header.frameInSecond/frameRate
		
	elif type(frame) == guppi.Frame:
		mjd = frame.header.imjd + frame.header.smjd / 86400.0
		sec = int(round(astro.utcjd_to_unix(mjd + astro.MJD_OFFSET)))
		
		offset = frame.header.offset
		whol = offset / int(frame.getSampleRate())
		frac = (offset - whol*int(frame.getSampleRate())) / float(frame.getSampleRate())
		
		sec += whol
		frac += frame.header.fmjd
		if frac >= 1.0:
			sec += 1
			frac -= 1.0
			
	elif type(frame) == drx.Frame:
		# HACK - What should T_NOM really be at LWA1???
		tt = frame.data.timeTag - (6660 if frame.header.timeOffset else 0)
		sec = tt/196000000
		frac = (tt - sec*196000000)/196e6
		
	else:
		raise TypeError("Unknown frame type: %s" % type(frame).__name__)
		
	return [sec, frac]


def readGUPPIHeader(filehandle):
	"""
	Read in a GUPPI header at the start of a VDIF file from the VLA.  The 
	contents of the header are returned as a dictionary.
	"""
	
	# Read in the GUPPI header
	header = {}
	while True:
		line = filehandle.read(80)
		if line[:3] == 'END':
			break
		elif line[:8] == 'CONTINUE':
			junk, value2 = line.split(None, 1)
			value = "%s%s" % (value[:-1], value2[:-1])
		else:
			name, value = line.split('=', 1)
			name = name.strip()
		try:
			value = int(value, 10)
		except:
			try:
				value = float(value)
			except:
				value = value.strip().replace("'", '')
		header[name.strip()] = value
	header['OBSBW'] *= 1e6
	header['OBSFREQ'] *= 1e6
	
	return header


def parseLWAMetaData(filename):
	"""
	Read in a LWA metadata tarball and return a two-element tuple of the
	BAM command times and the relative change in delay for the beamformer
	is s.
	"""
	
	t = []
	d = []
	
	# Load in the command script and walk through the commands
	try:
		cs = getCommandScript(filename)
	except ValueError:
		cs = getCommandScriptADP(filename)
	for c in cs:
		## Jump over any command that is not a BAM
		if c['commandID'] != 'BAM':
			continue
			
		## Figure out the station, antenna layout, and antenna closest to the array center
		try:
			refAnt
		except NameError:
			if c['subsystemID'] == 'DP':
				site = stations.lwa1
			else:
				site = stations.lwasv
			ants = site.getAntennas()
			
			refAnt = 0
			best = 1e20
			for i,a in enumerate(ants):
				r = a.stand.x**2 + a.stand.y**2 + a.stand.z**2
				if r < best:
					best = r
					refAnt = i
					
		## Parse the command to get the beamformer delays
		### Pointing and frequency
		beam, df, gf = c['data'].split(None, 2)
		freq, el, az = df.replace('.df','').split('_', 2)
		freq = float(freq)/10.0 * 1e6
		el = float(el)/10.0
		az = float(az)/10.0
		### Delay calculation
		b = calcDelay(ants, freq, az, el)
		b = b.max() - b
		
		## Figure out what it means and save it.  This includes a convertion to/from the MCS
		## delay that breaks things down into a course and fine delay for DP
		t.append( c['time'] )
		d.append( MCSDtodelay(delaytoMCSD(b[refAnt]*1e9))/1e9 )
		
	# Convert to NumPy arrays and adjust as needed
	t = numpy.array(t)
	d = numpy.array(d)
	t += 1.0					# BAM command are applied 1s after they are sent
	d0 = d[0]					# Get the initial offset
	d = numpy.diff(d)			# We want the relative delay change between steps
	if site == stations.lwa1:
		d = numpy.insert(d, 0, d0)	# ... and we need to start at the initial offset
	else:
		d = numpy.insert(d, 0, 0.0)	# ... and we need to start at zero
		
	# done
	return t, d


class PolyCo(object):
	"""
	Class for working with a pulsar PolyCo entry.
	"""
	
	def __init__(self, filename=None):
		if filename is not None:
			self.readFromFile(filename)
			
	def readFromFile(self, filename):
		"""
		Given a filename or open filehandle, read in that file and populate 
		the PolyCo instance with everything needed to get the pulsar phase 
		or frequency as a function of MJD.
		"""
		
		# Figure out what to do
		needToClose = False
		if type(filename) is not file:
			fh = open(filename, 'r')
			needToClose = True
		else:
			fh = filename
			
		# Go through the file line-by-line
		## First line
		line1 = fh.readline()
		if len(line1) < 3:
			raise IOError("readline() returned an empty string")
		name    = line1[ 0:10].lstrip().rstrip()
		date    = line1[10:20].lstrip().rstrip()
		time    = line1[20:31].lstrip().rstrip()
		tMid    = line1[31:51].lstrip().rstrip()
		DM      = line1[51:72].lstrip().rstrip()
		dShift  = line1[73:79].lstrip().rstrip()
		fitRMS  = line1[79:86].lstrip().rstrip()
		## Second line
		line2 = fh.readline()
		rPhase  = line2[ 0:20].lstrip().rstrip()
		rFreq   = line2[20:38].lstrip().rstrip()
		obsCode = line2[38:43].lstrip().rstrip()
		span    = line2[43:49].lstrip().rstrip()
		nCoeff  = line2[49:54].lstrip().rstrip()
		obsFreq = line2[54:75].lstrip().rstrip()
		binPhs  = line2[75:80].lstrip().rstrip()
		## Third and remaining lines
		coeffs = []
		for i in xrange(int(nCoeff)/3 + (0 if int(nCoeff)%3==0 else 1)):
			line = fh.readline()
			coeff1 = line[ 0:25].lstrip().rstrip()
			coeff2 = line[25:50].lstrip().rstrip()
			coeff3 = line[50:75].lstrip().rstrip()
			coeffs.append( coeff1 )
			coeffs.append( coeff2 )
			coeffs.append( coeff3 )
			
		if needToClose:
			fh.close()
			
		# Make what we've just read in useful
		## Type conversions - line 1
		while len(time) < 7:
			time = "0"+time
		self.date    = datetime.strptime("%s %s" % (date, time), "%d-%b-%y %H%M%S.%f")
		self.tMid    = float(tMid)
		self.DM      = float(DM)
		self.dShift  = float(dShift)*1e-4
		self.fitRMS  = 10**float(fitRMS)
		## Type conversions - line 2
		self.rPhase  = float(rPhase)
		self.rFreq   = float(rFreq)
		self.obsCode = obsCode
		self.span    = float(span)
		self.nCoeff  = int(nCoeff, 10)
		self.obsFreq = float(obsFreq)*1e6
		self.binPhs  = float(binPhs) if binPhs != '' else 0.0
		## Type conversions - coefficients
		self.coeffs = [float(c.replace('D', 'E')) for c in coeffs]
		
		# Fill in additional information about the valid MJD range
		self.validMinMJD = self.tMid - self.span/1440/2
		self.validMaxMJD = self.tMid + self.span/1440/2
		
	def getDM(self, mjd):
		"""
		Given a MJD value, return the dispersion measure of the pulsar in 
		pc cm^{-3}.
		"""
		
		# Are we ready to go?
		if getattr(self, 'tMid', None) is None:
			raise RuntimeError("Need to populated from a .polyco file before using")
			
		# Is the MJD valid for this polyco?
		if mjd < self.validMinMJD or mjd > self.validMaxMJD:
			raise ValueError("PolyCo is only valid for MJD %.6f to %.6f" % (self.validMinMJD, self.validMaxMJD))
			
		return self.DM
		
	def getPhase(self, mjd):
		"""
		Given a MJD value, compute the phase of the pulsar.
		"""
		
		# Are we ready to go?
		if getattr(self, 'tMid', None) is None:
			raise RuntimeError("Need to populated from a .polyco file before using")
			
		# Is the MJD valid for this polyco?
		if mjd < self.validMinMJD or mjd > self.validMaxMJD:
			raise ValueError("PolyCo is only valid for MJD %.6f to %.6f" % (self.validMinMJD, self.validMaxMJD))
			
		# Get the time difference and compute
		dt = (mjd - self.tMid)*1440.0
		phase = self.rPhase + dt*60.0*self.rFreq
		for i,c in enumerate(self.coeffs):
			phase += c*dt**i
			
		return phase
		
	def getFrequency(self, mjd):
		"""
		Given a MJD value, compute the frequency of the pulsar in Hz.
		"""
		
		# Are we ready to go?
		if getattr(self, 'tMid', None) is None:
			raise RuntimeError("Need to populated from a .polyco file before using")
			
		# Is the MJD valid for this polyco?
		if mjd < self.validMinMJD or mjd > self.validMaxMJD:
			raise ValueError("PolyCo is only valid for MJD %.6f to %.6f" % (self.validMinMJD, self.validMaxMJD))
			
		# Get the time difference and compute
		dt = (mjd - self.tMid)*1440.0
		freq = self.rFreq + 0.0
		for i,c in enumerate(self.coeffs):
			if i == 0:
				continue
			freq += i*c*dt**(i-1)/60.0
			
		return freq


class PolyCos(object):
	"""
	Class for working with pulsar PolyCos files.
	"""
	
	def __init__(self, filename=None):
		if filename is not None:
			self.readFromFile(filename)
			
	def readFromFile(self, filename):
		"""
		Given a filename or open filehandle, read in that file and populate 
		the PolyCos instance with everything needed to get the pulsar phase 
		or frequency as a function of MJD.
		"""
		
		# Figure out what to do
		needToClose = False
		if type(filename) is not file:
			fh = open(filename, 'r')
			needToClose = True
		else:
			fh = filename
			
		# Load in all of the coefficient sets
		m, p = [], []
		while True:
			try:
				c = PolyCo(fh)
				m.append( c.tMid )
				p.append( c )
			except IOError:
				break
				
		# Populate the instance with data
		self.tMids = numpy.array(m)
		self.polyCos = p
		
	def getDM(self, mjd):
		"""
		Given a MJD value, return the dispersion measure of the pulsar in 
		pc cm^{-3}.
		"""
		
		# Are we ready to go?
		if getattr(self, 'tMids', None) is None:
			raise RuntimeError("Need to populated from a .polycos file before using")
			
		# Find the best polynomial to use
		best = numpy.argmin( numpy.abs(self.tMids - mjd) )
		return self.polyCos[best].getDM(mjd)
		
	def getPhase(self, mjd):
		"""
		Given a MJD value, compute the phase of the pulsar.
		"""
		
		# Are we ready to go?
		if getattr(self, 'tMids', None) is None:
			raise RuntimeError("Need to populated from a .polycos file before using")
			
		# Find the best polynomial to use
		best = numpy.argmin( numpy.abs(self.tMids - mjd) )
		return self.polyCos[best].getPhase(mjd)
		
	def getFrequency(self, mjd):
		"""
		Given a MJD value, compute the frequency of the pulsar in Hz.
		"""
		
		# Are we ready to go?
		if getattr(self, 'tMids', None) is None:
			raise RuntimeError("Need to populated from a .polycos file before using")
			
		# Find the best polynomial to use
		best = numpy.argmin( numpy.abs(self.tMids - mjd) )
		return self.polyCos[best].getFrequency(mjd)
