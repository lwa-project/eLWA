import os
import sys
import ephem
from datetime import datetime

from lsl import astro
from lsl.common import stations
from lsl.reader import drx, vdif
from lsl.common.mcs import datetime2mjdmpm


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
		## Add the seperator at the end if this isnt' the last line
		if r != nRow-1:
			out += sep
		## Print
		print out


def readCorrelatorConfiguration(filename):
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
			source = {}
		elif line[:4] == 'Name':
			source['name'] = line.split(None, 1)[1]
		elif line[:6] == 'RA2000':
			source['ra'] = line.split(None, 1)[1]
		elif line[:7] == 'Dec2000':
			source['dec'] = line.split(None, 1)[1]
		elif line[:6] == 'Polyco':
			source['polyco'] = line.split(None, 1)[1]
		elif line == 'SourceDone':
			sources.append( source )
		elif line == 'Input':
			block = {'fileOffset':0.0}
		elif line[:4] == 'File' and line[4] != 'O':
			block['filename'] = line.split(None, 1)[1]
		elif line[:4] == 'Type':
			block['type'] = line.split(None, 1)[1].lower()
		elif line[:7] == 'Antenna':
			block['antenna'] = line.split(None, 1)[1]
		elif line[:4] == 'Pols':
			block['pols'] = [v.strip().rstrip() for v in line.split(None, 1)[1].split(',')]
		elif line[:8] == 'Location':
			block['location'] = [float(v) for v in line.split(None, 1)[1].split(',')]
		elif line[:11] == 'ClockOffset':
			block['clockOffset'] = [float(v) for v in line.split(None, 1)[1].split(',')]
		elif line[:10] == 'FileOffset':
			block['fileOffset'] = float(line.split(None, 1)[1])
		elif line == 'InputDone':
			blocks.append( block )
	fh.close()
	
	# Find the reference source
	if 'ra' in sources[0].keys() and 'dec' in sources[0].keys():
		refSource = ephem.FixedBody()
		refSource.name = sources[0]['name']
		refSource._ra = sources[0]['ra']
		refSource._dec = sources[0]['dec']
		refSource._epoch = ephem.J2000
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
	
	# Build up a list of file offsets
	offsets = [block['fileOffset'] for block in blocks]
	
	# Build up a list of readers
	readers = []
	for block in blocks:
		if block['type'] == 'vdif':
			readers.append( vdif )
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
	return refSource, filenames, offsets, readers, antennas


def getBetterTime(frame):
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
		
	else:
		tt = frame.data.timeTag - frame.header.timeOffset
		sec = tt/196000000
		frac = (tt - sec*196000000)/196e6
		
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


class PolyCo(object):
	def readFromFile(self, filename):
		"""
		Print given a filename, read in that file and populate the PolyCo 
		instance with everything needed to get the pulsar phase or frequency
		as a function of MJD.
		"""
		
		# Open the file and go through it line-by-line
		fh = open(filename, 'r')
		## First line
		line1 = fh.readline()
		name    = line1[ 0:10].lstrip().rstrip()
		date    = line1[10:19].lstrip().rstrip()
		time    = line1[19:31].lstrip().rstrip()
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
		lines = fh.readlines()
		for line in lines:
			coeff1 = line[ 0:25].lstrip().rstrip()
			coeff2 = line[25:50].lstrip().rstrip()
			coeff3 = line[50:75].lstrip().rstrip()
			coeffs.append( coeff1 )
			coeffs.append( coeff2 )
			coeffs.append( coeff3 )
			
		fh.close()
		
		# Make what we've just read in useful
		## Type conversions - line 1
		self.date    = datetime.strptime("%s %s" % (date, time), "%d-%m-%y %H%M%S.%f")
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
		self.binPhs  = float(binPhs)
		## Type conversions - coefficients
		self.coeffs = [float(c) for c in coeffs]
		
	def getPhase(self, mjd):
		"""
		Given a MJD value, compute the phase of the pulsar.
		"""
		
		# Are we ready to go?
		if getattr(self, 'tMid', None) is None:
			raise RuntimeError("Need to populated from a .polyco file before using")
			
		# Get the time difference and compute
		dt = (mjd - self.tMid)*1440.0
		phase = self.rPhase + dt*60.0*self.rFreq
		for i,c in enumerate(self.coeffs):
			phase += c*dt**i
			
		return phase
		
	def getFrequency(self, mjd):
		"""
		Given a MJD value, compute the frequency of the pulsar.
		"""
		
		# Are we ready to go?
		if getattr(self, 'tMid', None) is None:
			raise RuntimeError("Need to populated from a .polyco file before using")
			
		# Get the time difference and compute
		dt = (mjd - self.tMid)*1440.0
		freq = self.rFreq + 0.0
		for i,c in enumerate(self.coeffs):
			if i == 0:
				continue
			freq += i*c*dt**(i-1)/60.0
			
		return freq
		
