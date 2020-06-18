"""
Module for working with VLA SDM files for flagging purposes.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import os
import numpy
import struct
from functools import reduce
from xml.etree import ElementTree

from lsl.astro import utcjd_to_unix, MJD_OFFSET
from lsl.misc.lru_cache import lru_cache


__version__ = '0.5'
__all__ = ['vla_to_utcmjd', 'vla_to_utcjd', 'vla_to_unix', 'get_antennas', 
           'get_flags', 'filter_flags', 
           'get_noise_diode_values', 'filter_noise_diode_values', 
           'get_switched_power_data', 'filter_switched_power_data', 
           'get_switched_power_sums', 'filter_switched_power_sums', 
           'get_switched_power_diffs', 'filter_switched_power_diffs', 
           'get_requantizer_gains', 'filter_requantizer_gains']


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


def _parse_interval(text):
    """
    Try a couple of different parsers for a text interval to get it into the
    correct datatype.
    """
    
    return [_parse_convert(v) for v in text.split(None, 1)]


def _parse_compound(text):
    """
    Parse a packed text array of format "ndim dim0 ... data" that is used in the
    text XML files for storing arrays.  If both 'rows' and 'cols' are one
    then a single value is returned.  If 'rows' is one and 'cols' is greater
    than one a list is returned.  Otherwise, a list of lists is returned.
    """
    
    ndim, data = text.split(None, 1)
    ndim = int(ndim, 10)
    fields = data.split(None, ndim)
    dims, data = [int(d, 10) for d in fields[:-1]], fields[-1]
    nentry = reduce(lambda x,y: x*y, dims)
    if nentry == 1:
        return data
    else:
        data = data.split(None, nentry-1)
        output = []
        for d in data:
            output.append( _parse_convert(d) )
        output = numpy.array(output)
        output.shape = tuple(dims)
        if dims[0] == 1:
            output = output[0]
        if len(output.shape) == 1:
            output = list(output)
        else:
            output = output.T
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


def get_noise_diode_values(filename):
    """
    Parse the CalDevice.xml file in a VLA SDM set and return a list of noise diode 
    power values.  This list of power values has the antenna IDs converted to names
    via the get_antennas() function.
    """
    
    # If we are given a directory, assume that it is an SDM set and 
    # pull out the right file.  Otherwise, make sure we are given an
    # CalDevice.xml file
    if os.path.isdir(filename):
        filename = os.path.join(filename, 'CalDevice.xml')
    else:
        if os.path.basename(filename) != 'CalDevice.xml':
            raise RuntimeError("Invalid filename: '%s'" % os.path.basename(filename))
            
    # Grab the antenna name mapper information
    antname = os.path.join(os.path.dirname(filename), 'Antenna.xml')
    ants = get_antennas(antname)
    
    # Open up the XML file
    tree = ElementTree.parse(filename)
    root = tree.getroot()
    
    # Parse
    powers = []
    for row in root.iter('row'):
        entry = {}
        for field in row:  
            name, value = field.tag, field.text
            if name == 'timeInterval':
                ## Convert times to JD
                tMid, tInt = _parse_interval(value)
                value = [tMid - tInt/2, tMid + tInt/2]
                value = [vla_to_utcjd(v) for v in value]
            elif name == 'antennaId':
                ## Convert antenna IDs to names
                value = _parse_convert(value)
                value = ants[value]
            elif name[:3] == 'cal' or name[-3:] == 'Cal':
                value = _parse_compound(value)
            else:
                value = _parse_convert(value)
            entry[name] = value
            
        # Cleanup the time
        entry['startTime'], entry['endTime'] = entry['timeInterval']
        del entry['timeInterval']
        
        # Pull out only the noise diode information
        idx = entry['calLoadNames'].index('NOISE_TUBE_LOAD')
        entry['calLoadNames'] = entry['calLoadNames'][idx]
        entry['noiseCal'] = entry['noiseCal'][idx]
        entry['coupledNoiseCal'] = entry['coupledNoiseCal'][idx,:]
        
        powers.append(entry)
        
    return powers


def filter_noise_diode_values(powers, startJD, stopJD):
    """
    Given a list of flags returned by get_flags() and a start and stop JD, 
    filter the list to exclude flags that do not apply to that time range.
    """
    
    f = lambda x: False if x['endTime'] < startJD or x['startTime'] > stopJD else True
    return filter(f, powers)


@lru_cache(2)
def get_switched_power_data(filename):
    """
    Parse the SysPower.bin file in a VLA SDM set and return a tuple of 
    dictionaries for (1) the switched power sum, (2) the switched power 
    difference, and (3) the requantizer gains.  These dictionaries are 
    indexed with the antenna IDs converted to names via the get_antennas() 
    function.  Each dictionary value is a list of the quantity being stored 
    as a four-element list of [JD start, JD stop, quantity0, quantity1].
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
    psums, pdiffs, gains = {}, {}, {}
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
            tMid,   tInt   = fields[ 2], fields[ 3]
            psum0,  psum1  = fields[11], fields[12]
            pdiff0, pdiff1 = fields[ 7], fields[ 8]
            gain0,  gain1  = fields[15], fields[16]
            tStart = tMid - tInt/2
            tStop  = tMid + tInt/2
            
            ### Convert the antenna ID to a name
            ant = 'Antenna_'+ant.split('\x00', 1)[0]
            ant = ants[ant]
            
            ### Save the data as necessary
            if ant not in psums:
                psums[ant]  = {}
                pdiffs[ant] = {}
                gains[ant]  = {}
            if tStart not in psums[ant]:
                psums[ant][tStart]  = [tStart, tStop, 0.0, 0.0]
                pdiffs[ant][tStart] = [tStart, tStop, 0.0, 0.0]
                gains[ant][tStart]  = [tStart, tStop, 0.0, 0.0]
            if psum0 != 0.0:
                psums[ant][tStart][2] = psum0
            if psum1 != 0.0:
                psums[ant][tStart][3] = psum1
            if pdiff0 != 0.0:
                pdiffs[ant][tStart][2] = pdiff0
            if pdiff1 != 0.0:
                pdiffs[ant][tStart][3] = pdiff1
            if gain0 != 0.0:
                gains[ant][tStart][2] = gain0
            if gain1 != 0.0:
                gains[ant][tStart][3] = gain1
                
    # Convert the dictionary of dictionaries to a dictionary of lists
    final_psums, final_pdiffs, final_gains = {}, {}, {}
    for ant in psums:
        final_psums[ant]  = []
        final_pdiffs[ant] = []
        final_gains[ant]  = []
        keys = sorted(psums[ant])
        for key in keys:
            tStart = vla_to_utcjd( psums[ant][key][0] )
            tStop  = vla_to_utcjd( psums[ant][key][1] )
            final_psums[ant].append( [tStart, tStop, psums[ant][key][2], psums[ant][key][3]] )
            final_pdiffs[ant].append( [tStart, tStop, pdiffs[ant][key][2], pdiffs[ant][key][3]] )
            final_gains[ant].append( [tStart, tStop, gains[ant][key][2], gains[ant][key][3]] )
            
    # Done
    return final_psums, final_pdiffs, final_gains


def filter_switched_power_data(psums, pdiffs, gains, startJD, stopJD):
    """
    Given the three dictionaries of power sum, power difference, and 
    requantizer gains returned by get_switched_power() and a start and 
    stop JD, filter the dictionaries to exclude values that do not apply
    to that time range.
    """
    
    f = lambda x: False if x[1] < startJD or x[0] > stopJD else True
    output = []
    for entry in (psums, pdiffs, gains):
        output.append({})
        for key in entry:
            output[-1] = filter(f, entry[key])
    return output


def get_switched_power_sums(filename):
    """
    Parse the SysPower.bin file in a VLA SDM set and return a dictionary of
    switched power sums as a function of time.  This dictionary is indexed with
    the antenna IDs converted to names via the get_antennas() function.  Each
    dictionary value is a list of the sums with each sum being stored as a 
    four-element list of [JD start, JD stop, sum0, sum1].
    """
    
    final, _, _ = get_switched_power_data(filename)
    return final


def filter_switched_power_sums(psums, startJD, stopJD):
    """
    Given a dictionary of switched power sums returned by get_switched_power_sums()
    and a start and stop JD, filter the list to exclude values that do not 
    apply to that time range.
    """
    
    f = lambda x: False if x[1] < startJD or x[0] > stopJD else True
    output = {}
    for key in psums:
        output[key] = filter(f, psums[key])
    return output


def get_switched_power_diffs(filename):
    """
    Parse the SysPower.bin file in a VLA SDM set and return a dictionary of
    switched power differences as a function of time.  This dictionary is 
    indexed with the antenna IDs converted to names via the get_antennas() 
    function.  Each dictionary value is a list of the differences with each 
    difference being stored as a four-element list of 
    [JD start, JD stop, diff0, diff1].
    """
    
    _, final, _ = get_switched_power_data(filename)
    return final


def filter_switched_power_diffs(pdiffs, startJD, stopJD):
    """
    Given a dictionary of switched power differences returned by 
    get_switched_power_diffs() and a start and stop JD, filter the list to 
    exclude values that do not apply to that time range.
    """
    
    f = lambda x: False if x[1] < startJD or x[0] > stopJD else True
    output = {}
    for key in pdiffs:
        output[key] = filter(f, pdiffs[key])
    return output


def get_requantizer_gains(filename):
    """
    Parse the SysPower.bin file in a VLA SDM set and return a dictionary of
    requantizer gains as a function of time.  This dictionary is indexed with
    the antenna IDs converted to names via the get_antennas() function.  Each
    dictionary value is a list of the requantizer gains with each gain
    being stored as a four-element list of [JD start, JD stop, gain0, gain1].
    """
    
    _, _, final = get_switched_power_data(filename)
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
    