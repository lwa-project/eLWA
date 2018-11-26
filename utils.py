# -*- coding: utf-8 -*-

"""
Utility module for the various scripts needed to correlate LWA and VLA data.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import re
import time
import ephem
import fcntl
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


__version__ = '0.5'
__revision__ = '$Rev$'
__all__ = ['InterProcessLock', 'EnhancedFixedBody', 'EnhancedSun', 'EnhancedJupiter', 
           'multiColumnPrint', 'parseTimeString', 'nsround', 'readCorrelatorConfiguration', 
           'getBetterTime', 'readGUPPIHeader', 'parseLWAMetaData', 'PolyCos', 
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


class InterProcessLock(object):
    def __init__(self, name):
        self.name = name
        self.fh = open("%s.lock" % self.name, 'w+')
        self.locked = False
        
    def __del__(self):
        self.unlock()
        self.fh.close()
        
    def __enter__(self):
        self.lock()
        
    def __exit__(self, type, value, tb):
        self.unlock()
        
    def lock(self, block=True):	
        while not self.locked:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_EX)
                self.locked = True
            except IOError as e:
                if e.errno != errno.EAGAIN:
                    raise
                if not block:
                    break
                time.sleep(0.01)
                
        return self.locked
        
    def unlock(self):
        if not self.locked:
            return False
            
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.locked = False
        return True


class EnhancedFixedBody(ephem.FixedBody):
    """
    Sub-class of ephem.FixedBody that allows for pulsar phase and frequency 
    calculation.  This is done through a new '_polycos' attribute that is set
    after initilization to a PolyCos instance.  On calling the 'compute' 
    method, the 'phase' and 'frequency' attributes are set.
    """
    
    def __init__(self, body=None):
        super(self.__class__, self).__init__()
        
        if type(body) is ephem.FixedBody:
            for attr in ('name', '_ra', '_dec', '_epoch', '_pa', '_pmra', '_pmdec'):
                value = getattr(body, attr, None)
                if value is not None:
                    setattr(self, attr, value)
                    
    def __setattr__(self, name, value):
        # Validate that the _polycos attribute is set to a PolyCos instance
        if name == '_polycos':
            if type(value) != PolyCos:
                raise ValueError("Must set _polycos with a PolyCos instance")
                
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
            
        # Compute the pulsar parameters - if applicable
        self.compute_pulsar(when)
        
    def compute_pulsar(self, mjd, mjdf=0.0):
        """
        Compute the pulsar paramaters (if avaliable) with higher precision.
        """
        
        if getattr(self, '_polycos', None) is not None:
            if type(mjd) is ephem.Observer:
                mjd = mjd.date + (astro.DJD_OFFSET - astro.MJD_OFFSET)
                mjdf = 0.0
            elif type(mjd) is ephem.Date:
                mjd = mjd + (astro.DJD_OFFSET - astro.MJD_OFFSET)
                mjdf = 0.0
                
            self.dm = self._polycos.getDM(mjd, mjdf)
            self.phase = self._polycos.getPhase(mjd, mjdf)
            self.frequency = self._polycos.getFrequency(mjd, mjdf)
            self.doppler = self._polycos.getDoppler(mjd, mjdf)
            self.period = 1.0/self.frequency


class EnhancedSun(ephem.Sun):
    """
    Minimal sub-class of ephem.Sun to allow the 'duration' attribute to 
    be set.
    """
    
    def __getattr__(self, name):
        # Catch the _ra, _dec, _epoch, etc. attributes since they don't exist
        # for ephem.Planet sub-classes and we don't want to break things
        if name in ('_ra', '_dec', '_epoch', '_pa', '_pmra', '_pmdec'):
            return 'moving'
            
        # Get the attribute if every is ok
        super(self.__class__, self).__getattr__(name)
        
    def __setattr__(self, name, value):
        # Catch the _ra, _dec, _epoch, etc. attributes since they don't exist
        # for ephem.Planet sub-classes and we don't want to break things
        if name in ('_ra', '_dec', '_epoch', '_pa', '_pmra', '_pmdec'):
            raise AttributeError("Cannot set '%s' on this object" % value)
            
        # Set the attribute if everything is ok
        super(self.__class__, self).__setattr__(name, value)


class EnhancedJupiter(ephem.Jupiter):
    """
    Minimal sub-class of ephem.Jupiter to allow the 'duration' attribute to 
    be set.
    """
    
    def __getattr__(self, name):
        # Catch the _ra, _dec, _epoch, etc. attributes since they don't exist
        # for ephem.Planet sub-classes and we don't want to break things
        if name in ('_ra', '_dec', '_epoch', '_pa', '_pmra', '_pmdec'):
            return 'moving'
            
        # Get the attribute if every is ok
        super(self.__class__, self).__getattr__(name)
        
    def __setattr__(self, name, value):
        # Catch the _ra, _dec, _epoch, etc. attributes since they don't exist
        # for ephem.Planet sub-classes and we don't want to break things
        if name in ('_ra', '_dec', '_epoch', '_pa', '_pmra', '_pmdec'):
            raise AttributeError("Cannot set '%s' on this object" % value)
            
        # Set the attribute if everything is ok
        super(self.__class__, self).__setattr__(name, value)


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


_timeRE = re.compile('^[ \t]*(?P<value>[+-]?\d*\.?\d*([Ee][+-]?\d*)?)[ \t]*(?P<unit>(([kmun]?s)|h|m))?[ \t]*$')
def parseTimeString(value):
    """
    Given a time in the format of "decimal_value unit", convert the string 
    to a floating point time in seconds.  Valid units are:
      * h  - hours
      * ks - kiloseconds
      * m  - minutes
      * s  - seconds
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
            if unit == 'h':
                value *= 3600.0
            elif unit == 'ks':
                value *= 1e3
            elif unit == 'm':
                value *= 60.0
            elif unit == 'ms':
                value *= 1e-3
            elif unit == 'us':
                value *= 1e-6
            elif unit == 'ns':
                value *= 1e-9
                
    return value


def nsround(value):
    """
    Round a time in seconds to the nearest ns.
    """
    
    return round(value*1e9)/1e9


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
    else:
        srcs = [EnhancedSun(), EnhancedJupiter()]
        for line in _srcs:
            srcs.append( EnhancedFixedBody(ephem.readdb(line)) )
            
        refSource = None
        for i in xrange(len(srcs)):
            if srcs[i].name == sources[0]['name']:
                refSource = srcs[i]
                break
        if refSource is None:
            raise ValueError("Unknown source '%s'" % sources[0]['name'])
    refSource.duration = sources[0]['duration']
    try:
        refSource._polycos = PolyCos(sources[0]['polyco'], psrname=refSource.name.replace('PSR', '').replace('_', ''))
    except KeyError:
        pass
        
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
    d0 = d[0]				# Get the initial offset
    d = numpy.diff(d)			# We want the relative delay change between steps
    if site == stations.lwa1:
        d = numpy.insert(d, 0, d0)	# ... and we need to start at the initial offset
    else:
        d = numpy.insert(d, 0, d0)	# ... and we need to start at the initial offset
        
    # done
    return t, d


class PolyCos(object):
    """
    Class for working with pulsar PolyCos files.
    """
    
    def __init__(self, filename, psrname=None):
        if psrname is None:
            psrname = os.path.basename(filename)
            psrname = os.path.split('_', 1)[0]
            
        from polycos import polycos
        
        self._polycos_base = polycos(psrname, filename)
        
    def getDM(self, mjd, mjdf=0.0):
        """
        Given a MJD value, return the dispersion measure of the pulsar in 
        pc cm^{-3}.
        """
        
        goodpoly = self._polycos_base.select_polyco(mjd, mjdf)
        return self._polycos_base.polycos[goodpoly].DM
        
    def getPhase(self, mjd, mjdf=0.0):
        """
        Given a MJD value, compute the phase of the pulsar.
        """
        
        return self._polycos_base.get_phase(mjd, mjdf)
        
    def getFrequency(self, mjd, mjdf=0.0):
        """
        Given a MJD value, compute the frequency of the pulsar in Hz.
        """
        
        return self._polycos_base.get_freq(mjd, mjdf)
        
    def getDoppler(self, mjd, mjdf=0.0):
        """
        Given a MJD value, return the approximate topocentric Doppler shift of the 
        pulsar (~1 + vObs/c).
        """
        
        return 1.0 + self._polycos_base.get_voverc(mjd, mjdf)
