"""
Utility module for the various scripts needed to correlate LWA and VLA data.

"""

# Python2 compatibility
from __future__ import print_function, division, absolute_import

import os
import re
import time
import ephem
import errno
import fcntl
import numpy
import shutil
import tempfile
from datetime import datetime

from lsl import astro
from lsl.common import stations
from lsl.reader import base, drx, vdif
from lsl.common.dp import fS
from lsl.common.mcs import datetime_to_mjdmpm, delay_to_mcsd, mcsd_to_delay
from lsl.common.metabundle import get_command_script
from lsl.common.metabundleADP import get_command_script as get_command_scriptADP
from lsl.misc.beamformer import calc_delay


__version__ = '0.9'
__all__ = ['InterProcessLock', 'EnhancedFixedBody', 'EnhancedSun', 
           'EnhancedJupiter', 'multi_column_print', 'parse_time_string', 
           'nsround', 'read_correlator_configuration', 'get_better_time', 
           'parse_lwa_metadata', 'PolyCos']


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
        ephem.FixedBody.__init__(self)
        
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
        ephem.FixedBody.__setattr__(self, name, value)
        
    def __getattr__(self, name):
        if name in ('phase', 'frequency', 'period'):
            ## Is this even valid?
            if getattr(self, '_polycos', None) is None:
                raise ValueError("Pulsar parameters cannot be determined for a non-pulsar body")
                
        # Get the attribute if every is ok
        ephem.FixedBody.__getattr__(self, name)
        
    def compute(self, when=None, epoch=ephem.J2000):
        # Basic validation
        if when is None:
            when = ephem.now()
            
        # Compute the basic source parameters via PyEphem
        if type(when) is ephem.Observer:
            ephem.FixedBody.compute(self, when)
        else:
            ephem.FixedBody.compute(self, when, epoch=epoch)
            
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
        ephem.Sun.__getattr__(self, name)
        
    def __setattr__(self, name, value):
        # Catch the _ra, _dec, _epoch, etc. attributes since they don't exist
        # for ephem.Planet sub-classes and we don't want to break things
        if name in ('_ra', '_dec', '_epoch', '_pa', '_pmra', '_pmdec'):
            raise AttributeError("Cannot set '%s' on this object" % value)
            
        # Set the attribute if everything is ok
        ephem.Sun.__setattr__(self, name, value)


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
        ephem.Jupiter.__getattr__(self, name)
        
    def __setattr__(self, name, value):
        # Catch the _ra, _dec, _epoch, etc. attributes since they don't exist
        # for ephem.Planet sub-classes and we don't want to break things
        if name in ('_ra', '_dec', '_epoch', '_pa', '_pmra', '_pmdec'):
            raise AttributeError("Cannot set '%s' on this object" % value)
            
        # Set the attribute if everything is ok
        ephem.Jupiter.__setattr__(self, name, value)


def multi_column_print(items, sep=';  ', width=86):
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
    nCol = width // (maxLen+len(sep))
    if nCol < 1:
        nCol = 1
        
    # Figure out how many rows to use.  This needs to take into acount partial
    # rows with len(items) % nCol != 0.
    nRow = len(items) // nCol + (0 if (len(items) % nCol) == 0 else 1)
    
    # Print
    for r in range(nRow):
        ## Build up the line
        out = sep.join([formatter % str(i) for i in items[r*nCol:(r+1)*nCol]])
        ## Add the separator at the end if this isn't the last line
        if r != nRow-1:
            out += sep
        ## Print
        print(out)


_timeRE = re.compile('^[ \t]*(?P<value>[+-]?\d*\.?\d*([Ee][+-]?\d*)?)[ \t]*(?P<unit>(([kmun]?s)|h|m))?[ \t]*$')
def parse_time_string(value):
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
            raise ValueError("Invalid literal for parse_time_string(): %s" % value)
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


def _read_correlator_configuration(filename):
    """
    Backend function for read_correlator_configuration.
    """
    
    context = None
    config = {}
    sources = []
    blocks = []
    
    fh = open(filename, 'r')
    for line in fh:
        if line[0] == '#':
            continue
        if len(line) < 3:
            continue
            
        line = line.strip().rstrip()
        
        if line == 'Context':
            temp_context = {'observer':'Unknown', 'project':'Unknown', 'session':None, 'vlaref':None}
        elif line[:8] == 'Observer':
            temp_context['observer'] = line.split(None, 1)[1]
        elif line[:7] == 'Project':
            temp_context['project'] = line.split(None, 1)[1]
        elif line[:7] == 'Session':
            temp_context['session'] = line.split(None, 1)[1]
        elif line[:6] == 'VLARef':
            temp_context['vlaref'] = line.split(None, 1)[1]
        elif line == 'EndContext':
            context = temp_context
            
        elif line == 'Configuration':
            temp_config = {'inttime':None, 'channels':None, 'basis':None}
        elif line[:8] == 'Channels':
            temp_config['channels'] = int(line.split(None, 1)[1], 10)
        elif line[:7] == 'IntTime':
            temp_config['inttime'] = float(line.split(None, 1)[1])
        elif line[:8] == 'PolBasis':
            temp_config['basis'] = line.split(None, 1)[1]
        elif line == 'EndConfiguration':
            config = temp_config
            
        elif line == 'Source':
            source = {'intent':'target', 'duration':0.0}
        elif line[:4] == 'Name':
            source['name'] = line.split(None, 1)[1]
        elif line[:6] == 'Intent':
            source['intent'] = line.split(None, 1)[1].lower()
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
            block['clockOffset'] = [parse_time_string(v) for v in line.split(None, 1)[1].split(',')]
        elif line[:10] == 'FileOffset':
            block['fileOffset'] = parse_time_string(line.split(None, 1)[1])
        elif line == 'InputDone':
            ## Make sure we have a metaData key since it is optional
            if 'metadata' not in block:
                block['metadata'] = None
            blocks.append( block )
    fh.close()
    
    # Set the context
    config['context'] = context
    
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
        for i in range(len(srcs)):
            if srcs[i].name == sources[0]['name']:
                refSource = srcs[i]
                break
        if refSource is None:
            raise ValueError("Unknown source '%s'" % sources[0]['name'])
    refSource.intent = sources[0]['intent']
    refSource.duration = sources[0]['duration']
    try:
        if not os.path.exists(sources[0]['polyco']):
            # Maybe it is relative to the configuration file's path?
            sources[0]['polyco'] = os.path.join(os.path.dirname(filename), sources[0]['polyco'])
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
            for j in range(len(name)):
                try:
                    aid = int(name[j:], 10)
                    if name[:j].lower() == 'lwa':
                        aid += 50
                    break
                except ValueError:
                    pass
        pols = block['pols']
        location = block['location']
        clock_offsets = block['clockOffset']
        
        if aid is None:
            raise RuntimeError("Cannot convert antenna name '%s' to a number" % name)
            
        stand = stations.Stand(aid, *location)
        for pol,offset in zip(pols, clock_offsets):
            cable = stations.Cable('%s-%s' % (name, pol), 0.0, vf=1.0, dd=0.0)
            cable.clock_offset = offset
            
            if pol.lower() == 'x':
                antenna = stations.Antenna(i, stand=stand, cable=cable, pol=0)
            else:
                antenna = stations.Antenna(i, stand=stand, cable=cable, pol=1)
                        
            antennas.append( antenna )
            i += 1
            
    # Done
    return config, refSource, filenames, metanames, offsets, readers, antennas


def read_correlator_configuration(filename_or_npz):
    """
    Parse a correlator configuration file generated by createConfigFile.py and
    return a seven-element tuple of:
      * a correlator configuration dictionary or None if no configuraiton is found
      * the reference source as a ephem.FixedBody-compatible instance
      * a list of filenames, 
      * a list of metadata tarball names, 
      * a list of file offsets in seconds, 
      * a list of lsl.reader modules to use, and
      * a list of lsl.common.stations.Antenna instances for each file
    """
    
    # Sort out what to do depending on what we were given
    if isinstance(filename_or_npz, numpy.lib.npyio.NpzFile):
        ## An open .npz file, just work with it
        dataDict = filename_or_npz
        to_close = False
    elif os.path.splitext(filename_or_npz)[1] == '.npz':
        ## A .npz file, open and and then work with it
        dataDict = numpy.load(filename_or_npz)
        to_close = True
    else:
        ## Something else, just try some stuff
        try:
            ### Could it be a file that has already been opened?
            filename_or_npz = filename_or_npz.filename
        except AttributeError:
            pass
        return _read_correlator_configuration(filename_or_npz)
        
    # Make a temporary directory
    tempdir = tempfile.mkdtemp(prefix='config-')
    
    try:
        # Configuration file - oh, and check for a 'Polyco' definition
        cConfig = dataDict['config']
        tempConfig = os.path.join(tempdir, 'config.txt')
        fh = open(tempConfig, 'w')
        polycos = None
        for line in cConfig:
            fh.write('%s' % line)
            if line.find('Polyco') != -1:
                polycos = line.strip().rstrip()
        fh.close()
        
        # Polycos (optional)
        if polycos is not None:
            tempPolycos = os.path.basename(polycos.split(None, 1)[1])
            tempPolycos = os.path.join(tempdir, tempPolycos)
            try:
                cPolycos = dataDict['polycos']
                fh = open(tempPolycos, 'w')
                for line in cPolycos:
                    fh.write('%s' % line)
                fh.close()
            except KeyError:
                pass
                
        # Read
        full_config = _read_correlator_configuration(tempConfig)
        
    finally:
        # Cleanup
        shutil.rmtree(tempdir)
        if to_close:
            dataDict.close()
            
    # Done
    return full_config


def get_better_time(frame):
    """
    Given a lsl.reader.vdif.Frame or lsl.reader.drx.Frame instance, return a
    more accurate time for the frame.  Unlike the Frame.get_time() functions,
    this function returns a two-element tuple of:
      * integer seconds since the UNIX epoch and
      * fractional second
    """
    
    if isinstance(frame, drx.Frame):
        # HACK - What should T_NOM really be at LWA1???
        tt = frame.payload.timetag
        to = 6660 if frame.header.time_offset else 0
        return list(base.FrameTimestamp.from_dp_timetag(tt, to))
    else:
        return list(frame.time)


def parse_lwa_metadata(filename):
    """
    Read in a LWA metadata tarball and return a two-element tuple of the
    BAM command times and the relative change in delay for the beamformer
    is s.
    """
    
    t = []
    d = []
    
    # Load in the command script and walk through the commands
    try:
        cs = get_command_script(filename)
    except (RuntimeError, ValueError):
        cs = get_command_scriptADP(filename)
    for c in cs:
        ## Jump over any command that is not a BAM
        if c['commandID'] != 'BAM':
            continue
            
        ## Figure out the station, antenna layout, and antenna closest to the array center
        try:
            ref_ant
        except NameError:
            if c['subsystemID'] == 'DP':
                site = stations.lwa1
            else:
                site = stations.lwasv
            ants = site.antennas
            
            ref_ant = 0
            best = 1e20
            for i,a in enumerate(ants):
                r = a.stand.x**2 + a.stand.y**2 + a.stand.z**2
                if r < best:
                    best = r
                    ref_ant = i
                    
        ## Parse the command to get the beamformer delays
        ### Pointing and frequency
        beam, df, gf = c['data'].split(None, 2)
        freq, el, az = df.replace(b'.df',b'').split(b'_', 2)
        freq = float(freq)/10.0 * 1e6
        el = float(el)/10.0
        az = float(az)/10.0
        ### Delay calculation
        b = calc_delay(ants, freq, az, el)
        b = b.max() - b
        
        ## Figure out what it means and save it.  This includes a convertion to/from the MCS
        ## delay that breaks things down into a course and fine delay for DP
        t.append( c['time'] )
        d.append( mcsd_to_delay(delay_to_mcsd(b[ref_ant]*1e9))/1e9 )
        
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
            psrname = psrname.split('_', 1)[0]
            
        from polycos import polycos
        
        self.filename = filename
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
