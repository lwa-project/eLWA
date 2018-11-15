# -*- coding: utf-8 -*-

"""
Module for working with VLA SDM files for flagging purposes

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import struct
from xml.etree import ElementTree

from lsl.astro import utcjd_to_unix, MJD_OFFSET


__version__ = '0.4'
__revision__ = '$Rev$'
__all__ = ['vla_to_utcmjd', 'vla_to_utcjd', 'vla_to_unix', 'get_antennas', 
        'get_flags', 'filter_flags', 'get_requantizer_gains', 'filter_requantizer_gains', 
        '__version__', '__revision__', '__all__']


def vla_to_utcmjd(timetag):
    """
    Convert a VLA timetag in MJD nanoseconds to an MJD.
    """
    
    mjd = timetag / 1e9 / 86400.0
    return mjd


def vla_to_utcjd(timetag):
    """
    Convert a VLA timetag in MJD nanoseconds to a JD.
    """
    
    return vla_to_utcmjd(timetag) + MJD_OFFSET


def vla_to_unix(timetag):
    """
    Convert a VLA timetag in MJD nanoseconds to a UNIX timestamp.
    """
    
    return utcjd_to_unix( vla_to_utcmjd(timetag) + MJD_OFFSET )


def _parse_convert(text):
    """
    Try a couple of different parsers for text data to get it into the
    correct datatype.
    """
    
    try:
        value = int(text, 10)
        return value
    except ValueError:
        pass
        
    try:
        value = float(text)
        return value
    except ValueError:
        pass
        
    return text


def _parse_compound(text):
    """
    Parse a packed text array of format "rows cols data" that is used in the
    text XML files for storing arrays.  If both 'rows' and 'cols' are one
    then a single value is returned.  If 'rows' is one and 'cols' is greater
    than one a list is returned.  Otherwise, a list of lists is returned.
    """
    
    rows, cols, data = text.split(None, 2)
    rows, cols = int(rows, 10), int(cols, 10)
    if rows == 1 and cols == 1:
        return data
    else:
        data = data.split(None, rows*cols-1)
        output = []
        for i,d in enumerate(data):
            if i % cols == 0:
                output.append( [] )
            output[-1].append( _parse_convert(d) )
        if rows == 1:
            output = output[0]
        return output


def get_antennas(filename):
    """
    Parse the Antenna.xml file in a VLA SDM set and return a dictionary
    of antenna information.
    """
    
    # If we are given a directory, assume that it is an SDM set and 
    # pull out the right file.  Otherwise, make sure we are given an
    # Antenna.xml file
    if os.path.isdir(filename):
        filename = os.path.join(filename, 'Antenna.xml')
    else:
        if os.path.basename(filename) != 'Antenna.xml':
            raise RuntimeError("Invalid filename: '%s'" % os.path.basename(filename))
            
    # Open up the XML file
    tree = ElementTree.parse(filename)
    root = tree.getroot()
    
    # Parse
    ants = {}
    for row in root.iter('row'):
        entry = {}
        for field in row:
            name, value = field.tag, field.text
            if name in ('position', 'offset'):
                value = _parse_compound(value)
            else:
                value = _parse_convert(value)
            entry[name] = value
            
        ants[entry['antennaId']] = entry['name'].upper()
        
    return ants


def get_flags(filename, skipSubreflector=True, skipFocus=True):
    """
    Parse the Flag.xml file in a VLA SDM set and return a list of flags
    that are not of type 'SUBREFLECTOR_ERROR' or 'FOCUS_ERROR'.  This
    list of flags has the antenna IDs converted to names via the 
    get_antennas() function.
    """
    
    # If we are given a directory, assume that it is an SDM set and 
    # pull out the right file.  Otherwise, make sure we are given an
    # Flag.xml file
    if os.path.isdir(filename):
        filename = os.path.join(filename, 'Flag.xml')
    else:
        if os.path.basename(filename) != 'Flag.xml':
            raise RuntimeError("Invalid filename: '%s'" % os.path.basename(filename))
            
    # Grab the antenna name mapper information
    antname = os.path.join(os.path.dirname(filename), 'Antenna.xml')
    ants = get_antennas(antname)
    
    # Open up the XML file
    tree = ElementTree.parse(filename)
    root = tree.getroot()
    
    # Parse
    flags = []
    for row in root.iter('row'):
        entry = {}
        for field in row:
            name, value = field.tag, field.text
            if name[-4:] == 'Time':
                ## Convert times to JD
                value = vla_to_utcjd( int(value, 10) )
            elif name == 'antennaId':
                ## Convert antenna IDs to names
                value = _parse_compound(value)
                value = ants[value]
            else:
                value = _parse_convert(value)
            entry[name] = value
            
        ## Skip over bogus flags
        if entry['reason'] == 'SUBREFLECTOR_ERROR' and skipSubreflector:
            continue
        if entry['reason'] == 'FOCUS_ERROR' and skipFocus:
            continue
            
        flags.append( entry )
        
    return flags


def filter_flags(flags, startJD, stopJD):
    """
    Given a list of flags returned by get_flags() and a start and stop JD, 
    filter the list to exclude flags that do not apply to that time range.
    """
    
    f = lambda x: False if x['endTime'] < startJD or x['startTime'] > stopJD else True
    return filter(f, flags)


def get_requantizer_gains(filename):
    """
    Parse the SysPower.bin file in a VLA SDM set and return a dictionary of
    requantizer gains as a function of time.  This dictionary is indexed with
    the antenna IDs converted to names via the get_antennas() function.  Each
    dictionary value is is a list of the requantizer gains with each gain
    being stored as a four-element list of [JD start, JD stop, gain0, gain1].
    """
    
    # If we are given a directory, assume that it is an SDM set and 
    # pull out the right file.  Otherwise, make sure we are given an
    # SysPower.bin file
    if os.path.isdir(filename):
        filename = os.path.join(filename, 'SysPower.bin')
    else:
        if os.path.basename(filename) != 'SysPower.bin':
            raise RuntimeError("Invalid filename: '%s'" % os.path.basename(filename))
            
    # Grab the antenna name mapper information
    antname = os.path.join(os.path.dirname(filename), 'Antenna.xml')
    ants = get_antennas(antname)
    
    # Open the file and begin parsing
    fh = open(filename, 'rb')
    gains = {}
    while fh.tell() < os.path.getsize(filename):
        ## Find the start of the next block of data we are working on
        block = fh.read(8)
        while block != 'Antenna_':
            fh.seek(-7, 1)
            block = fh.read(8)
        fh.seek(-8, 1)
        bStart = fh.tell()
        
        ## Find the end of the next block of data we are working on
        fh.seek(16*1024**2, 1)
        block = fh.read(8)
        while block != 'Antenna_' and fh.tell() < os.path.getsize(filename):
            fh.seek(-7, 1)
            block = fh.read(8)
        fh.seek(-4, 1)
        bStop = fh.tell()
        
        ## Go to the start of the block and read it in
        fh.seek(bStart)
        block = fh.read(bStop-bStart)
        
        ## Split by the 'Antenna_' key and parse each section
        block = block.split('Antenna_')
        for row in block:
            ### Skip over things that are too small
            if len(row) < 3:
                continue
                
            ### Split by the 'SpectralWindow_' key
            ant, row = row.split('SpectralWindow_', 1)
            
            ### Unpack and interperate the binary data - this is a fixed
            ### reading scheme that seems to work at least for the data 
            ### we have
            fields = struct.unpack('>siqqiBiffBiffBiff', row[:64])
            tMid, tInt, gain0, gain1 = fields[2], fields[3], fields[15], fields[16]
            tStart = tMid - tInt/2
            tStop  = tMid + tInt/2
            
            ### Convert the antenna ID to a name
            ant = 'Antenna_'+ant.split('\x00', 1)[0]
            ant = ants[ant]
            
            ### Save the data as necessary
            if ant not in gains:
                gains[ant] = {}
            if tStart not in gains[ant]:
                gains[ant][tStart] = [tStart, tStop, 0.0, 0.0]
            if gain0 != 0.0:
                gains[ant][tStart][2] = gain0
            if gain1 != 0.0:
                gains[ant][tStart][3] = gain1
                
    # Convert the dictionary of dictionaries to a dictionary of lists
    final = {}
    for ant in gains:
        final[ant] = []
        keys = sorted(gains[ant])
        for key in keys:
            tStart = vla_to_utcjd( gains[ant][key][0] )
            tStop  = vla_to_utcjd( gains[ant][key][1] )
            final[ant].append( [tStart, tStop, gains[ant][key][2], gains[ant][key][3]] )
            
    # Done
    return final


def filter_requantizer_gains(gains, startJD, stopJD):
    """
    Given a dictionary of requantizer gains returned by get_requantizer_gains()
    and a start and stop JD, filter the list to exclude gains that do not 
    apply to that time range.
    """
    
    f = lambda x: False if x[1] < startJD or x[0] > stopJD else True
    output = {}
    for key in gains:
        output[key] = filter(f, gains[key])
    return output


if __name__ == "__main__":
    import sys
    import numpy
    gains = get_requantizer_gains(sys.argv[1])
    
    import pylab
    for ant in ('EA20',):
        gain = numpy.array(gains[ant])
        
        try:
            norm
        except NameError:
            norm = numpy.median(gain[:,3])
        pylab.plot(gain[:,0]-gain[0,0], gain[:,2]/norm)
        pylab.plot(gain[:,0]-gain[0,0], gain[:,3]/norm)
    pylab.show()
    