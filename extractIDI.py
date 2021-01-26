#!/usr/bin/env python

"""
Integration extractor for FITS-IDI files containing eLWA data.
"""

# Python2 compatibility
from __future__ import print_function
try:
    input = raw_input
except NameError:
    pass
    
import os
import sys
import time
import numpy
from astropy.io import fits as astrofits
import argparse
from datetime import datetime

import ubjson

from lsl.astro import utcjd_to_unix
from lsl.writer.fitsidi import NUMERIC_STOKES
from lsl.misc.dedispersion import delay as disp_delay


def main(args):
    # Parse the command line
    filename = args.filename
    
    t0 = time.time()
    print("Working on '%s'" % os.path.basename(filename))
    # Open the FITS IDI file and access the UV_DATA extension
    hdulist = astrofits.open(filename, mode='readonly')
    andata = hdulist['ANTENNA']
    fqdata = hdulist['FREQUENCY']
    fgcount = 0
    for hdu in hdulist[1:]:
        if hdu.header['EXTNAME'] == 'FLAG':
            fgcount += 1
    uvdata = hdulist['UV_DATA']
    
    # Verify we can flag this data
    if uvdata.header['STK_1'] > 0:
        raise RuntimeError("Cannot flag data with STK_1 = %i" % uvdata.header['STK_1'])
    if uvdata.header['NO_STKD'] < 4:
        raise RuntimeError("Cannot flag data with NO_STKD = %i" % uvdata.header['NO_STKD'])
        
    # NOTE: Assumes that the Stokes parameters increment by -1
    polMapper = {}
    for i in range(uvdata.header['NO_STKD']):
        stk = uvdata.header['STK_1'] - i
        polMapper[i] = NUMERIC_STOKES[stk]
        
    # Pull out various bits of information we need to flag the file
    ## Antenna look-up table
    antLookup = {}
    for an, ai in zip(andata.data['ANNAME'], andata.data['ANTENNA_NO']):
        antLookup[an] = ai
    ## Frequency and polarization setup
    nBand, nFreq, nStk = uvdata.header['NO_BAND'], uvdata.header['NO_CHAN'], uvdata.header['NO_STKD']
    ## Baseline list
    bls = uvdata.data['BASELINE']
    ## Time of each integration
    obsdates = uvdata.data['DATE']
    obstimes = uvdata.data['TIME']
    inttimes = uvdata.data['INTTIM']
    ## Source list
    srcs = uvdata.data['SOURCE']
    ## Band information
    fqoffsets = fqdata.data['BANDFREQ'].ravel()
    ## Frequency channels
    freq = (numpy.arange(nFreq)-(uvdata.header['CRPIX3']-1))*uvdata.header['CDELT3']
    freq += uvdata.header['CRVAL3']
    ## UVW coordinates
    try:
        u, v, w = uvdata.data['UU'], uvdata.data['VV'], uvdata.data['WW']
    except KeyError:
        u, v, w = uvdata.data['UU---SIN'], uvdata.data['VV---SIN'], uvdata.data['WW---SIN']
    uvw = numpy.array([u, v, w]).T
    ## The actual visibility data
    flux = uvdata.data['FLUX'].astype(numpy.float32)
    
    # Convert the visibilities to something that we can easily work with
    nComp = flux.shape[1] / nBand / nFreq / nStk
    if nComp == 2:
        ## Case 1) - Just real and imaginary data
        flux = flux.view(numpy.complex64)
    else:
        ## Case 2) - Real, imaginary data + weights (drop the weights)
        flux = flux[:,0::nComp] + 1j*flux[:,1::nComp]
    flux.shape = (flux.shape[0], nBand, nFreq, nStk)
    
    # Find unique baselines, times, and sources to work with
    ubls = numpy.unique(bls)
    utimes = numpy.unique(obstimes)
    usrc = numpy.unique(srcs)

    # Downselect
    try:
        start_int, stop_int = args.start_stop_or_meta.split('-', 1)
        start_int = int(start_int, 10)
        stop_int = int(stop_int, 10)
    except ValueError:
        print("Loading metadata from '%s'" % os.path.basename(args.start_stop_or_meta))
        with open(args.start_stop_or_meta, 'rb') as fh:
            meta = ubjson.load(fh)
        event_time = meta['t0'] + meta['peak_time']
        event_dm = meta['dm']
        event_width = meta['width']
        print("  t0: %f" % event_time)
        print("  width: %.3f ms" % (event_width*1e3,))
        print("  DM: %.3f pm/cm^3" % event_dm)
        
        data_time = utcjd_to_unix(obsdates[0] + obstimes[0])
        data_tint = inttimes[0]
        event_time -= disp_delay([361.941e6, freq.max()], event_dm)[0]
        start_int = int((event_time - data_time) / data_tint) - 1
        stop_int = start_int + max([1, int(event_width / data_tint)]) + 1
    to_keep = range(start_int*len(ubls), (stop_int+1)*len(ubls))
    print("UV_DATA entry selection is %i through %i" % (to_keep[0], to_keep[-1]))
    
    # Save
    print("  Saving to disk")
    ## What to call it
    outname = os.path.basename(filename)
    outname, outext = os.path.splitext(outname)
    outname = '%s_extracted%s' % (outname, outext)
    ## Does it already exist or not
    if os.path.exists(outname):
        if not args.force:
            yn = input("WARNING: '%s' exists, overwrite? [Y/n] " % outname)
        else:
            yn = 'y'
            
        if yn not in ('n', 'N'):
            os.unlink(outname)
        else:
            raise RuntimeError("Output file '%s' already exists" % outname)
    ## Open and create a new primary HDU
    hdulist2 = astrofits.open(outname, mode='append')
    primary =	astrofits.PrimaryHDU()
    processed = []
    for key in hdulist[0].header:
        if key in ('COMMENT', 'HISTORY'):
            if key not in processed:
                parts = str(hdulist[0].header[key]).split('\n')
                for part in parts:
                    primary.header[key] = part
                processed.append(key)
        else:
            primary.header[key] = (hdulist[0].header[key], hdulist[0].header.comments[key])
    hdulist2.append(primary)
    hdulist2.flush()
    ## Copy the extensions over to the new file
    for hdu in hdulist[1:]:
        ### The individual tables that need updating
        if hdu.header['EXTNAME'] == 'FLAG' and args.drop:
            if fgcount > 1:
                ## Drop this flag table
                fgcount -= 1
                ver = hdu.header['EXTVER'] 
                print("  WARNING: removing old FLAG table - version %i" % ver)
                continue
            else:
                ## Reset the EXTVER on the last FLAG table
                hdu.header['EXTVER'] = (1, 'table instance number')
                
        elif hdu.header['EXTNAME'] == 'UV_DATA':
            columns = []

            for col in hdu.data.columns:
                temp = hdu.data[col.name]
                temp = temp[to_keep]
                columns.append( astrofits.Column(name=col.name, unit=col.unit, format=col.format, array=temp) )
            colDefs = astrofits.ColDefs(columns)
            
            hduprime = astrofits.BinTableHDU.from_columns(colDefs)
            processed = ['NAXIS1', 'NAXIS2']
            for key in hdu.header:
                if key in ('COMMENT', 'HISTORY'):
                    if key not in processed:
                        parts = str(hdu.header[key]).split('\n')
                        for part in parts:
                            hduprime.header[key] = part
                        processed.append(key)
                elif key not in hduprime.header:
                    hduprime.header[key] = (hdu.header[key], hdu.header.comments[key])
            hdu = hduprime
            
        hdulist2.append(hdu)
        hdulist2.flush()
    hdulist2.close()
    hdulist.close()
    print("  -> Extracted FITS IDI file is '%s'" % outname)
    print("  Finished in %.3f s" % (time.time()-t0,))


if __name__ == "__main__":
    numpy.seterr(all='ignore')
    parser = argparse.ArgumentParser(
        description='Integration extractor for FITS-IDI files', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('start_stop_or_meta', type=str, 
                        help='start and stop integrations writen as "start-stop" or a .meta filename')
    parser.add_argument('filename', type=str, 
                        help='filename to process')
    parser.add_argument('-d', '--drop', action='store_true', 
                        help='drop all but the most recent FLAG tables')
    parser.add_argument('-f', '--force', action='store_true', 
                        help='force overwriting of existing FITS-IDI files')
    args = parser.parse_args()
    main(args)
    
