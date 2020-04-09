# -*- coding: utf-8 -*-
"""
Module for working with GUPPI raw data from the VLA

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import copy
import numpy

from lsl import astro
from lsl.reader.errors import SyncError, EOFError


__version__ = '0.1'
__revision__ = '$Rev$'
__all__ = ['FrameHeader', 'FrameData', 'Frame', 'read_guppi_header', 'read_frame', 
        'get_frame_size', 'get_thread_count', 'getFramesPerBlock', 'get_frames_per_second', 
        'get_sample_rate', 'get_central_freq', '__version__', '__revision__', '__all__']


class FrameHeader(object):
    """
    Class that stores the information found in the header of a VDIF 
    frame.  Most fields in the VDIF version 1.1.1 header are stored.
    """
    
    def __init__(self, imjd=0, smjd=0, fmjd=0.0, offset=0.0, bits_per_sample=0, thread_id=0, station_id=0, sample_rate=0.0, central_freq=0.0):
        self.imjd = imjd
        self.smjd = smjd
        self.fmjd = fmjd
        self.offset = offset
        
        self.bits_per_sample = bits_per_sample
        self.thread_id = thread_id
        self.station_id = station_id
        
        self.sample_rate = sample_rate
        self.central_freq = central_freq
        
    def get_time(self):
        """
        Function to convert the time tag to seconds since the UNIX epoch.
        """
        
        mjd = self.imjd + (self.smjd + self.fmjd) / 86400.0
        seconds = astro.utcjd_to_unix(mjd + astro.MJD_OFFSET)
        seconds += self.offset / float(self.sample_rate)
        
        return seconds
        
    def parse_id(self):
        """
        Return a two-element tuple of the station ID and thread ID.
        
        .. note::
            The station ID is always returned as numeric.
        """
        return (self.station_id, self.thread_id)
        
    def get_sample_rate(self):
        """
        Return the sample rate of the data in samples/second.
        """
        
        return self.sample_rate*1.0
        
    def get_central_freq(self):
        """
        Function to get the central frequency of the VDIF data in Hz.
        """
        
        return self.central_freq*1.0


class FrameData(object):
    """
    Class that stores the information found in the data section of a VDIF
    frame.
    
    .. note::
        Unlike the other readers in the :mod:`lsl.reader` module the
        data are stored as numpy.float32 values.
    """
    
    def __init__(self, data=None):
        self.data = data


class Frame(object):
    """
    Class that stores the information contained within a single VDIF 
    frame.  It's properties are FrameHeader and FrameData objects.
    """

    def __init__(self, header=None, data=None):
        if header is None:
            self.header = FrameHeader()
        else:
            self.header = header
            
        if data is None:
            self.data = FrameData()
        else:
            self.data = data
            
        self.valid = True

    def parse_id(self):
        """
        Convenience wrapper for the Frame.FrameHeader.parse_id 
        function.
        """
        
        return self.header.id
        
    def parse_extended_user_data(self):
        """
        Convenience wrapper for the Frame.FrameHeader.parse_extended_user_data
        function.
        """
        
        return self.header.parse_extended_user_data()
        
    def get_time(self):
        """
        Convenience wrapper for the Frame.FrameHeader.get_time function.
        """
        
        return self.header.get_time()
        
    def get_sample_rate(self):
        """
        Convenience wrapper for the Frame.FrameHeader.get_sample_rate function.
        """
        
        return self.header.sample_rate
        
    def get_central_freq(self):
        """
        Convenience wrapper for the Frame.FrameHeader.get_central_freq function.
        """
        
        return self.header.central_freq
        
    def __add__(self, y):
        """
        Add the data sections of two frames together or add a number 
        to every element in the data section.
        """
    
        newFrame = copy.deepcopy(self)
        newFrame += y	
        return newFrame
            
    def __iadd__(self, y):
        """
        In-place add the data sections of two frames together or add 
        a number to every element in the data section.
        """
        
        try:
            self.data.data += y.data.data
        except AttributeError:
            self.data.data += y
        return self
        
    def __mul__(self, y):
        """
        Multiple the data sections of two frames together or multiply 
        a number to every element in the data section.
        """
        
        newFrame = copy.deepcopy(self)
        newFrame *= y
        return newFrame
        
    def __imul__(self, y):
        """
        In-place multiple the data sections of two frames together or 
        multiply a number to every element in the data section.
        """
        
        try:
            self.data.data *= y.data.data
        except AttributeError:
            self.data.data *= y
        return self


def _filenameToAntenna(filename, vdifID=False):
    try:
        ant = filename.split('BD-', 1)[1]
        ant = ant.split('.', 1)[0]
        ant = int(ant, 10)
    except (ValueError, IndexError):
        return 0
    
    lookup = {0: 1, 
            1: 3, 
            2: 5, 
            3: 9, 
            4: 10, 
            5: 11, 
            6: 12, 
            7: 13, 
            8: 14, 
            9: 18, 
            10: 19, 
            11: 23, 
            12: 27}
            
    ant = lookup[ant]
    if vdifID:
        ant += 12300
        
    return ant


_param_cache = {}


def read_guppi_header(filehandle):
    """
    Read in a GUPPI header at the start of a VDIF file from the VLA.  The 
    contents of the header are returned as a dictionary.
    """
    
    global _param_cache
    
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
    
    _param_cache[filehandle] = header
    _param_cache[filehandle]['ANTENNA']   = _filenameToAntenna(filehandle.name, vdifID=True)
    _param_cache[filehandle]['DATASTART'] = filehandle.tell()
    _param_cache[filehandle]['LAST']      = filehandle.tell()
    
    return header


def read_frame(filehandle, Verbose=False):
    try:
        _param_cache[filehandle]
    except KeyError:
        print "MISS"
        
        mark = filehandle.tell()
        filehandle.seek(0)
        read_guppi_header(filehandle)
        filehandle.seek(mark)
        
    last      = _param_cache[filehandle]['LAST']
    ant       = _param_cache[filehandle]['ANTENNA']
    imjd      = _param_cache[filehandle]['STT_IMJD']
    smjd      = _param_cache[filehandle]['STT_SMJD']
    fmjd      = _param_cache[filehandle]['STT_OFFS']
    ref       = _param_cache[filehandle]['DATASTART']
    blocksize = _param_cache[filehandle]['BLOCSIZE']
    pktsize   = _param_cache[filehandle]['PKTSIZE']
    npkt      = _param_cache[filehandle]['NPKT']
    nbits     = _param_cache[filehandle]['NBITS']
    npol      = 2 if _param_cache[filehandle]['PKTFMT'].rstrip() == 'VDIF' else 1
    srate     = _param_cache[filehandle]['OBSBW']
    srate    *= 2 if _param_cache[filehandle]['PKTFMT'].rstrip() == 'VDIF' else 1
    cfreq     = _param_cache[filehandle]['OBSFREQ']
    
    mark = filehandle.tell()
    if 'SPARE' in _param_cache[filehandle].keys() and mark == last:
        frame = _param_cache[filehandle]['SPARE'].pop(0)
        if len(_param_cache[filehandle]['SPARE']) == 0:
            del _param_cache[filehandle]['SPARE']
            
    else:
        try:
            del _param_cache[filehandle]['SPARE']
        except KeyError:
            pass
            
        try:
            raw = numpy.fromfile(filehandle, dtype=numpy.uint8, count=npol*pktsize)
            _param_cache[filehandle]['LAST'] = filehandle.tell()
        except IOError:
            raise EOFError
            
        if nbits == 4:
            data = numpy.zeros((2,raw.size), dtype=numpy.uint8)
            even = raw[0::2]
            odd  = raw[1::2]
            data[0,1::2] = (even & 0xF0) >> 4
            data[0,0::2] = (even & 0x0F)
            data[1,1::2] = (odd  & 0xF0) >> 4
            data[1,0::2] = (odd  & 0x0F)
            data = data.astype(numpy.float32) - 8
        elif nbits == 8:
            data = raw.astype(numpy.int8)
            data = data.reshape((npol,-1))
            
        offset = mark - ref
        offset /= pktsize*npol
        offset *= data.shape[1]
        
        pktsize = data.shape[1]
        frames = []
        for j in xrange(npol):
            fpkt = data[j,:]
            fhdr = FrameHeader(imjd=imjd, smjd=smjd, fmjd=fmjd, offset=offset, 
                        bits_per_sample=nbits, thread_id=j, station_id=ant, 
                        sample_rate=srate, central_freq=cfreq)
            fdat = FrameData(data=fpkt)
            frames.append( Frame(header=fhdr, data=fdat) )
            
        if len(frames) == 1:
            frame = frames[0]
        else:
            frame = frames.pop(0)
            _param_cache[filehandle]['SPARE'] = frames
            
    return frame


def get_frame_size(filehandle, nFrames=None):
    """
    Find out what the frame size is in bytes from a single observation.
    """
    
    try:
        _param_cache[filehandle]
    except KeyError:
        mark = filehandle.tell()
        filehandle.seek(0)
        read_guppi_header(filehandle)
        filehandle.seek(mark)
        
    frameSize = _param_cache[filehandle]['PKTSIZE']
    
    return frameSize


def get_thread_count(filehandle):
    """
    Find out how many thrads are present by examining the first 1024
    records.  Return the number of threads found.
    """
    
    try:
        _param_cache[filehandle]
    except KeyError:
        mark = filehandle.tell()
        filehandle.seek(0)
        read_guppi_header(filehandle)
        filehandle.seek(mark)
        
    nPol = 2 if _param_cache[filehandle]['PKTFMT'].rstrip() == 'VDIF' else 1
    
    # Return the number of threads found
    return nPol


def getFramesPerBlock(filehandle):
    """
    Find and return the number of frames per block.
    """
    
    try:
        _param_cache[filehandle]
    except KeyError:
        mark = filehandle.tell()
        filehandle.seek(0)
        read_guppi_header(filehandle)
        filehandle.seek(mark)
        
    nPol = 2 if _param_cache[filehandle]['PKTFMT'].rstrip() == 'VDIF' else 1
    
    return nPol


def get_frames_per_second(filehandle):
    """
    Find out the number of frames per second in a file by watching how the 
    headers change.  Returns the number of frames in a second.
    """
    
    # Save the current position in the file so we can return to that point
    fhStart = filehandle.tell()
    
    # Get the frame size
    FRAME_SIZE = get_frame_size(filehandle)
    
    # Get the number of threads
    nThreads = get_thread_count(filehandle)
    
    # Get the current second counts for all threads
    ref = {}
    i = 0
    while i < nThreads:
        try:
            cFrame = read_frame(filehandle)
        except SyncError:
            filehandle.seek(FRAME_SIZE, 1)
            continue
        except EOFError:
            break
            
        cID = cFrame.header.thread_id
        cSC = cFrame.header.offset / int(get_sample_rate(filehandle))
        ref[cID] = cSC
        i += 1
        
    # Read frames until we see a change in the second counter
    cur = {}
    fnd = []
    while True:
        ## Get a frame
        try:
            cFrame = read_frame(filehandle)
        except SyncError:
            filehandle.seek(FRAME_SIZE, 1)
            continue
        except EOFError:
            break
            
        ## Pull out the relevant metadata
        cID = cFrame.header.thread_id
        cSC = cFrame.header.offset / int(get_sample_rate(filehandle))
        cFC = cFrame.header.offset % int(get_sample_rate(filehandle)) / cFrame.data.data.size
        
        ## Figure out what to do with it
        if cSC == ref[cID]:
            ### Same second as the reference, save the frame number
            cur[cID] = cFC
        else:
            ### Different second than the reference, we've found something
            ref[cID] = cSC
            if cID not in fnd:
                fnd.append( cID )
                
        if len(fnd) == nThreads:
            break
            
    # Return to the place in the file where we started
    filehandle.seek(fhStart)
    
    # Pull out the mode
    mode = {}
    for key,value in cur.iteritems():
        try:
            mode[value] += 1
        except KeyError:
            mode[value] = 1
    best, bestValue = 0, 0
    for key,value in mode.iteritems():
        if value > bestValue:
            best = key
            bestValue = value
            
    # Correct for a zero-based counter and return
    best += 1
    return best


def get_sample_rate(filehandle):
    """
    Find and return the sample rate in Hz by looking at how many frames 
    there are per second and how many samples there are in a frame.
    """
    
    try:
        _param_cache[filehandle]
    except KeyError:
        mark = filehandle.tell()
        filehandle.seek(0)
        read_guppi_header(filehandle)
        filehandle.seek(mark)
        
    sample_rate = _param_cache[filehandle]['OBSBW']
    sample_rate *= 2 if _param_cache[filehandle]['PKTFMT'].rstrip() == 'VDIF' else 1
    
    # Return the sample rate
    return sample_rate


def get_central_freq(filehandle):
    """
    Find and return the central frequency in Hz.
    """
    
    try:
        _param_cache[filehandle]
    except KeyError:
        mark = filehandle.tell()
        filehandle.seek(0)
        read_guppi_header(filehandle)
        filehandle.seek(mark)
        
    central_freq = _param_cache[filehandle]['OBSFREQ']
    
    # Return the observing frequency
    return central_freq
