#!/usr/bin/env python

"""
Given a collection of .npz files create a FITS-IDI file that can be read in by
AIPS.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    raw_input = input
    from functools import cmp_to_key
    
import os
import re
import sys
import glob
import time
import ephem
import numpy
import argparse
import tempfile
from datetime import datetime, timedelta, tzinfo

from astropy.constants import c as vLight
vLight = vLight.to('m/s').value

from lsl import astro
from lsl.common import stations, metabundle
from lsl.statistics import robust
from lsl.correlator import uvutil
from lsl.correlator import fx as fxc
from lsl.writer import fitsidi
from lsl.correlator.uvutil import compute_uvw
from lsl.common.mcs import datetime_to_mjdmpm

from utils import read_correlator_configuration

import fitsidi


_CMP_CACHE = {}
def cmpNPZ(x, y):
    try:
        xT = _CMP_CACHE[x]
    except KeyError:
        xDD = numpy.load(x)
        _CMP_CACHE[x] = xDD['tStart'].item()
        xDD.close()
        xT = _CMP_CACHE[x]
        
    try:
        yT = _CMP_CACHE[y]
    except KeyError:
        yDD = numpy.load(y)
        _CMP_CACHE[y] = yDD['tStart'].item()
        yDD.close()
        yT = _CMP_CACHE[y]
        
    return (xT > yT) - (xT < yT)


_SIMBAD_CACHE = {}
def getSourceName(src):
    """
    Function to take a ephem.FixedBody() instance and resolve it to a name 
    using Simbad.  This function returns the most popular (highest citation
    count) name within 2" of the provided position.
    """
    
    try:
        from urllib import quote_plus
        from urllib2 import urlopen
    except ImportError:
        from urllib.parse import quote_plus
        from urllib.request import urlopen
        
    # Pull out what we know about the source
    name = src.name
    ra = str(src._ra)
    dec = str(src._dec)
    epoch = (src._epoch - ephem.J2000) / 365.25 + 2000.0
    epoch = str(epoch)
    
    # See if it is in the lookup cache
    try:
        name = _SIMBAD_CACHE[(ra, dec, epoch)]
        return name
        
    except KeyError:
        pass
        
    # If not, we need to query Simbad to find out what to call it
    try:
        ## Query
        result = urlopen('http://simbad.u-strasbg.fr/simbad/sim-coo?Coord=%s&CooFrame=FK5&CooEpoch=%s&CooEqui=%s&CooDefinedFrames=none&Radius=2&Radius.unit=arcsec&submit=submit%%20query&CoordList=&output.format=ASCII' % (quote_plus('%s %s' % (ra, dec)), epoch, epoch))
        matches = result.readlines()
        
        ## Parse
        rank = 0
        for line in matches:
            ### Skip over blank lines, comments, and ASCII table headers
            if len(line) < 3:
                continue
            if line[0] in ('#', '-'):
                continue
                
            if line[:6] == 'Object':
                ### Case 1: There is only one object within the search radius
                fields = line.split('---')
                name = fields[0].replace('Object', '')
                _SIMBAD_CACHE[(ra, dec, epoch)] = name
                break
            else:
                ### Case 2: There are multiple objects and we need to parse 
                ### each entry to get the name and the citation count.  Highest
                ### citation count wins
                fields = line.split('|')
                if len(fields) < 13:
                    continue
                    
                if int(fields[-2], 10) > rank:
                    name = fields[2].replace('NAME ', '')
                    rank = int(fields[-2], 10)
                    _SIMBAD_CACHE[(ra, dec, epoch)] = name
                    
    except IOError:
        ## Fall-through for errors that should default to the original
        ## src.name
        pass
        
    except ValueError:
        ## Fall-through for errors that should default to the original
        ## src.name
        pass
        
    return name


def main(args):
    # Parse the command line
    filenames = args.filename
    # Special catch for when the command line is too short to list all of 
    # the files to combine.  This works by running glob.glob() using the
    # first input filename as the file regex
    if len(filenames) == 1 and filenames[0].find('*') != -1:
        filenames = glob.glob(filenames[0])
    try:
        filenames.sort(cmp=cmpNPZ)
    except TypeError:
        filenames.sort(key=cmp_to_key(cmpNPZ))
    if args.limit != -1:
        filenames = filenames[:args.limit]
        
    # Build up the station
    site = stations.lwa1
    observer = site.get_observer()
    
    # Load in the file file to figure out what to do
    dataDict = numpy.load(filenames[0])
    tStart = dataDict['tStart'].item()
    tInt = dataDict['tInt']
    
    freq = dataDict['freq1']
    
    config, refSrc, junk1, junk2, junk3, junk4, antennas = read_correlator_configuration(dataDict)
    if config is not None:
        if config['basis'] == 'linear':
            args.linear = True
            args.circular = False
            args.stokes = False
        elif config['basis'] == 'circular':
            args.linear = False
            args.circular = True
            args.stokes = False
        elif config['basis'] == 'stokes':
            args.linear = False
            args.circular = False
            args.stokes = True
        print("NOTE:  Set output polarization basis to '%s' per user defined configuration" % config['basis'])
        
    visXX = dataDict['vis1XX'].astype(numpy.complex64)
    visXY = dataDict['vis1XY'].astype(numpy.complex64)
    visYX = dataDict['vis1YX'].astype(numpy.complex64)
    visYY = dataDict['vis1YY'].astype(numpy.complex64)
    dataDict.close()
    
    # Build up the master list of antennas and report
    master_antennas = antennas
    obs_groups = [os.path.basename(filenames[0]).split('-vis2-', 1)[0],]
    for filename in filenames:
        group = os.path.basename(filename).split('-vis2-', 1)[0]
        if group not in obs_groups:
            dataDict = numpy.load(filename)
            config, refSrc, junk1, junk2, junk3, junk4, antennas = read_correlator_configuration(dataDict)
            del dataDict
            
            for ant in antennas:
                ## The FITS IDI writer only cares about the stand ID
                if (ant.stand.id, ant.pol) not in [(ma.stand.id, ma.pol) for ma in master_antennas]:
                    master_antennas.append(ant)
            obs_groups.append(group)
    master_antennas.sort(key=lambda x: (x.stand.id, x.pol))
    for i in range(len(master_antennas)):
        master_antennas[i].id = i+1
        
    print("Antennas:")
    for ant in master_antennas:
        print("  Antenna %i: Stand %i, Pol. %i" % (ant.id, ant.stand.id, ant.pol))
        
    nchan = visXX.shape[1]
    master_blList = uvutil.get_baselines([ant for ant in master_antennas if ant.pol == 0], include_auto=True)
    
    if args.decimate > 1:
        to_trim = (freq.size/args.decimate)*args.decimate
        to_drop = freq.size - to_trim
        if to_drop != 0:
            print("Warning: Dropping %i channels (%.1f%%; %.3f kHz)" % (to_drop, 100.0*to_drop/freq.size, to_drop*(freq[1]-freq[0])/1e3))
            
        nchan //= args.decimate
        if to_drop != 0:
            freq = freq[:to_trim]
        freq.shape = (freq.size//args.decimate, args.decimate)
        freq = freq.mean(axis=1)
        
    # Figure out the visibility conjugation problem in LSL, pre-1.1.4
    conjugateVis = False
    if float(fitsidi.__version__) < 0.9:
        print("Warning: Applying conjugate to visibility data")
        conjugateVis = True
        
    # Fill in the data
    for i,filename in enumerate(filenames):
        ## Load in the integration
        dataDict = numpy.load(filename)
        junk0, refSrc, junk1, junk2, junk3, junk4, antennas = read_correlator_configuration(dataDict)
        try:
            refSrc.name = refSrc.name.upper()	# For AIPS
            if refSrc.name[:12] == 'ELWA_SESSION':
                ## Convert ELWA_SESSION names to "real" source names
                refSrc.name = getSourceName(refSrc).replace(' ', '').upper()
        except AttributeError:
            ## Moving sources cannot have their names changed
            pass
        blList = uvutil.get_baselines([ant for ant in antennas if ant.pol == 0], include_auto=True)
        
        tStart = dataDict['tStart'].item()
        tInt = dataDict['tInt'].item()
        visXX = dataDict['vis1XX'].astype(numpy.complex64)
        visXY = dataDict['vis1XY'].astype(numpy.complex64)
        visYX = dataDict['vis1YX'].astype(numpy.complex64)
        visYY = dataDict['vis1YY'].astype(numpy.complex64)
        try:
            delayStepApplied = dataDict['delayStepApplied']
            try:
                len(delayStepApplied)
            except TypeError:
                delayStepApplied = [delayStepApplied if ant.stand.id > 50 else False for ant in antennas if ant.pol == 0]
        except KeyError:
            delayStepApplied = [False for ant in antennas if ant.pol == 0]
            
        dataDict.close()
        
        if args.decimate > 1:
            if to_drop != 0:
                visXX = visXX[:,:to_trim]
                visXY = visXY[:,:to_trim]
                visYX = visYX[:,:to_trim]
                visYY = visYY[:,:to_trim]
                
            visXX.shape = (visXX.shape[0], visXX.shape[1]//args.decimate, args.decimate)
            visXX = visXX.mean(axis=2)
            visXY.shape = (visXY.shape[0], visXY.shape[1]//args.decimate, args.decimate)
            visXY = visXY.mean(axis=2)
            visYX.shape = (visYX.shape[0], visYX.shape[1]//args.decimate, args.decimate)
            visYX = visYX.mean(axis=2)
            visYY.shape = (visYY.shape[0], visYY.shape[1]//args.decimate, args.decimate)
            visYY = visYY.mean(axis=2)
            
        if conjugateVis:
            visXX = visXX.conj()
            visXY = visXY.conj()
            visYX = visYX.conj()
            visYY = visYY.conj()
            
        if args.circular or args.stokes:
            visI = visXX + visYY
            visQ = visXX - visYY
            visU = visXY + visYX
            visV = (visXY - visYX) / 1.0j
            
            if args.circular:
                visRR = visI + visV
                visRL = visQ + 1j*visU
                visLR = visQ - 1j*visU
                visLL = visI - visV
                
        if i % args.split == 0:
            ## Clean up the previous file
            try:
                fits.write()
                fits.close()
            except NameError:
                pass
                
            ## Create the FITS-IDI file as needed
            ### What to call it
            if args.tag is None:
                outname = 'buildIDI.FITS_%i' % (i//args.split+1,)
            else:
                outname = 'buildIDI_%s.FITS_%i' % (args.tag, i//args.split+1,)
                
            ### Does it already exist or not
            if os.path.exists(outname):
                if not args.force:
                    yn = raw_input("WARNING: '%s' exists, overwrite? [Y/n] " % outname)
                else:
                    yn = 'y'
                    
                if yn not in ('n', 'N'):
                    os.unlink(outname)
                else:
                    raise RuntimeError("Output file '%s' already exists" % outname)
                    
            ### Create the file
            fits = fitsidi.Idi(outname, ref_time=tStart)
            if args.circular:
                fits.set_stokes(['RR', 'RL', 'LR', 'LL'])
            elif args.stokes:
                fits.set_stokes(['I', 'Q', 'U', 'V'])
            else:
                fits.set_stokes(['XX', 'XY', 'YX', 'YY'])
            fits.set_frequency(freq)
            fits.set_geometry(stations.lwa1, [a for a in master_antennas if a.pol == 0])
            fits.add_history('Created with %s, revision $Rev$' % os.path.basename(__file__))
            print("Opening %s for writing" % outname)
            
        if i % 10 == 0:
            print(i)
            
        ## Save any delay step information
        for step,ant in zip(delayStepApplied, [ant for ant in antennas if ant.pol == 0]):
            if step:
                fits.add_history("Delay step at %i %f" % (ant.stand.id, tStart))
                
        ## Update the observation
        observer.date = datetime.utcfromtimestamp(tStart).strftime('%Y/%m/%d %H:%M:%S.%f')
        refSrc.compute(observer)
        
        ## Convert the setTime to a MJD and save the visibilities to the FITS IDI file
        obsTime = astro.unix_to_taimjd(tStart)
        if args.circular:
            fits.add_data_set(obsTime, tInt, blList, visRR, pol='RR', source=refSrc)
            fits.add_data_set(obsTime, tInt, blList, visRL, pol='RL', source=refSrc)
            fits.add_data_set(obsTime, tInt, blList, visLR, pol='LR', source=refSrc)
            fits.add_data_set(obsTime, tInt, blList, visLL, pol='LL', source=refSrc)
        elif args.stokes:
            fits.add_data_set(obsTime, tInt, blList, visI, pol='I', source=refSrc)
            fits.add_data_set(obsTime, tInt, blList, visQ, pol='Q', source=refSrc)
            fits.add_data_set(obsTime, tInt, blList, visU, pol='U', source=refSrc)
            fits.add_data_set(obsTime, tInt, blList, visV, pol='V', source=refSrc)
        else:
            fits.add_data_set(obsTime, tInt, blList, visXX, pol='XX', source=refSrc)
            fits.add_data_set(obsTime, tInt, blList, visXY, pol='XY', source=refSrc)
            fits.add_data_set(obsTime, tInt, blList, visYX, pol='YX', source=refSrc)
            fits.add_data_set(obsTime, tInt, blList, visYY, pol='YY', source=refSrc)
            
    # Cleanup the last file
    fits.write()
    fits.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='given a collection of .npz files generated by "the next generation of correlator", create one or more FITS IDI files containing the data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('filename', type=str, nargs='+', 
                        help='filename to process')
    pgroup = parser.add_mutually_exclusive_group(required=False)
    pgroup.add_argument('-x', '--linear', action='store_true', default=True, 
                        help='write linear polarization data')
    pgroup.add_argument('-c', '--circular', action='store_true', 
                        help='convert to circular polarization')
    pgroup.add_argument('-k', '--stokes', action='store_true', 
                        help='convert to Stokes parameters')
    parser.add_argument('-d', '--decimate', type=int, default=1, 
                        help='frequency decimation factor')
    parser.add_argument('-l', '--limit', type=int, default=-1, 
                        help='limit the data loaded to the first N files, -1 = load all')
    parser.add_argument('-s', '--split', type=int, default=3000, 
                        help='maximum number of integrations in a FITS IDI file')
    parser.add_argument('-t', '--tag', type=str, 
                        help='optional tag to add to the filename')
    parser.add_argument('-f', '--force', action='store_true', 
                        help='force overwriting of existing FITS-IDI files')
    args = parser.parse_args()
    main(args)
    
