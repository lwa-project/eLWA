#!/usr/bin/env python

"""
Apply an amplitude scaling based on the VLA's switched power system 
to FITS-IDI files containing eLWA data.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    raw_input = input
    
import os
import sys
import time
import numpy
from astropy.io import fits as astrofits
import argparse
from datetime import datetime

from lsl.astro import utcjd_to_unix
from lsl.writer.fitsidi import NUMERIC_STOKES

from sdm import get_noise_diode_values, filter_noise_diode_values, \
                get_switched_power_diffs, filter_switched_power_diffs

from flagger import *


def main(args):
    # Parse the command line
    filenames = args.filename
    
    # Parse the SDM
    diodes = get_noise_diode_values(args.sdm)
    pdiffs = get_switched_power_diffs(args.sdm)
    
    for filename in filenames:
        t0 = time.time()
        print("Working on '%s'" % os.path.basename(filename))
        # Open the FITS IDI file and access the UV_DATA extension
        hdulist = astrofits.open(filename, mode='readonly')
        andata = hdulist['ANTENNA']
        fqdata = hdulist['FREQUENCY']
        fgdata = None
        for hdu in hdulist[1:]:
            if hdu.header['EXTNAME'] == 'FLAG':
                fgdata = hdu
        uvdata = hdulist['UV_DATA']
        
        # Verify we can flag this data - only linear for now
        if uvdata.header['STK_1'] > -5:
            raise RuntimeError("Cannot flag data with STK_1 = %i" % uvdata.header['STK_1'])
        if uvdata.header['NO_STKD'] < 4:
            raise RuntimeError("Cannot flag data with NO_STKD = %i" % uvdata.header['NO_STKD'])
            
        # Pull out various bits of information we need to flag the file
        ## Antenna look-up table
        antLookup, sdmLookup = {}, {}
        for an, ai in zip(andata.data['ANNAME'], andata.data['ANTENNA_NO']):
            antLookup[an] = ai
            sdmLookup[ai] = an.replace('LWA0', 'EA')
            
        # NOTE: Assumes that the Stokes parameters increment by -1
        polMapper = {}
        for i in xrange(uvdata.header['NO_STKD']):
            stk = uvdata.header['STK_1'] - i
            polMapper[i] = NUMERIC_STOKES[stk]
            
        ## Frequency and polarization setup
        nBand, nFreq, nStk = uvdata.header['NO_BAND'], uvdata.header['NO_CHAN'], uvdata.header['NO_STKD']
        ## Baseline list
        bls = uvdata.data['BASELINE']
        ## Time of each integration
        obsdates = uvdata.data['DATE']
        obstimes = uvdata.data['TIME']
        ## Source list
        srcs = uvdata.data['SOURCE']
        ## Band information
        fqoffsets = fqdata.data['BANDFREQ'].ravel()
        ## Frequency channels
        freq = (numpy.arange(nFreq)-(uvdata.header['CRPIX3']-1))*uvdata.header['CDELT3']
        freq += uvdata.header['CRVAL3']
        ## UVW coordinates
        u, v, w = uvdata.data['UU'], uvdata.data['VV'], uvdata.data['WW']
        uvw = numpy.array([u, v, w]).T
        ## The actual visibility data
        flux = uvdata.data['FLUX'].astype(numpy.float32)
        
        # Convert the visibilities to something that we can easily work with
        nComp = flux.shape[1] // nBand // nFreq // nStk
        if nComp == 2:
            ## Case 1) - Just real and imaginary data
            flux = flux.view(numpy.complex64)
        else:
            ## Case 2) - Real, imaginary data + weights (drop the weights)
            flux = flux[:,0::nComp] + 1j*flux[:,1::nComp]
        flux.shape = (flux.shape[0], nBand, nFreq, nStk)
        
        # Find unique baselines to work with
        ubls = numpy.unique(bls)
        nBL = len(ubls)
        
        # Filter the SDM values for what is relevant for this file
        fileStart = obsdates[ 0] + obstimes[ 0]
        fileStop  = obsdates[-1] + obstimes[-1]
        sub_diodes = filter_noise_diode_values(diodes, fileStart, fileStop)
        sub_pdiffs = filter_switched_power_diffs(pdiffs, fileStart, fileStop)
        
        # Apply
        nBL_Ints = len(bls)
        not_applied = []
        last_pdiff_idx = {}
        for an in antLookup:
            last_pdiff_idx[an.replace('LWA0', 'EA')] = 0
            
        print("  Scaling")
        for i in xrange(nBL_Ints):
            if i % 10000 == 0 or i+1 == nBL_Ints:
                print("    Baseline/integration %i of %i" % (i+1, nBL_Ints))
                
            intbl = bls[i]
            ant0, ant1 = sdmLookup[(intbl >> 8) & 0xFF], sdmLookup[intbl & 0xFF]
            intdate = obsdates[i] + obstimes[i]
            
            ## Get the noise diode power value
            diode0, diode1 = None, None
            for diode in sub_diodes:
                if diode['antennaId'] == ant0:
                    if intdate >= diode['startTime'] and intdate < diode['endTime']:
                        diode0 = diode['coupledNoiseCal']
                if diode['antennaId'] == ant1:
                    if intdate >= diode['startTime'] and intdate < diode['endTime']:
                        diode1 = diode['coupledNoiseCal']
                if diode0 is not None and diode1 is not None:
                    break
            if diode0 is None or diode1 is None:
                if diode0 is None and ant0 not in not_applied:
                    diode0 = [1.0, 1.0]
                    not_applied.append(ant0)
                if diode1 is None and ant1 not in not_applied:
                    diode1 = [1.0, 1.0]
                    not_applied.append(ant1)
                    
            ## Get the switched power difference value
            pdiff0, pdiff1 = None, None
            try:
                for j in xrange(last_pdiff_idx[ant0], len(sub_pdiffs[ant0])):
                    pdiff = sub_pdiffs[ant0][j]
                    if intdate >= pdiff[0] and intdate < pdiff[1]:
                        pdiff0 = pdiff[2:]
                        last_pdiff_idx[ant0] = j
                        break
            except KeyError:
                pass
            try:
                for j in xrange(last_pdiff_idx[ant1], len(sub_pdiffs[ant1])):
                    pdiff = sub_pdiffs[ant1][j]
                    if intdate >= pdiff[0] and intdate < pdiff[1]:
                        pdiff1 = pdiff[2:]
                        last_pdiff_idx[ant1] = j
                        break
            except KeyError:
                pass
            if pdiff0 is None or pdiff1 is None:
                if pdiff0 is None and ant0 not in not_applied:
                    pdiff0 = [1.0, 1.0]
                    not_applied.append(ant0)
                if pdiff1 is None and ant1 not in not_applied:
                    pdiff1 = [1.0, 1.0]
                    not_applied.append(ant1)
                continue
                
            ## Apply
            scales = []
            for pol in polMapper:
                polName = polMapper[pol]
                i0 = 1 if polName[0] == 'X' else 0  ## Backwards per our understanding of
                i1 = 1 if polName[0] == 'X' else 0  ## how LWA1 correlates with the VLA
                
                ### From EVLA memo 145
                scale = numpy.sqrt(diode0[i0]*diode1[i1]/pdiff0[i0]/pdiff1[i1])
                
                ### Adjust to keep LWA at about the same level
                scale /= args.norm     # A fudge factor
                scales.append(scale)
                #try:
                #    scaleRef
                #except NameError:
                #    scaleRef = scale
                #    print('ScaleRef:', scaleRef)
                #scale /= scaleRef
                
                flux[i,:,:,pol] *= scale
                
        # Report
        scales = numpy.array(scales)
        print("  Mean scale factor applied is %.6f" % (numpy.mean(scales),))
        print("  Median scale factor applied is %.6f" % (numpy.median(scales),))
        print("  Std. dev. scale factor applied is %.6f" % (numpy.std(scales),))
        if len(not_applied):
            print("  WARNING: amplitude scaling not applied for: %s" % ", ".join([ant.replace('EA', 'LWA0') for ant in not_applied]))
            
        # Build up the mask
        mask = numpy.where(numpy.isfinite(flux), False, True)
        
        # Convert the masks into a format suitable for writing to a FLAG table
        print("  Building FLAG table")
        ants, times, bands, chans, pols, reas, sevs = [], [], [], [], [], [], []
        if not args.drop:
            ## Old flags
            if fgdata is not None:
                for row in fgdata.data:
                    ants.append( row['ANTS'] )
                    times.append( row['TIMERANG'] )
                    try:
                        len(row['BANDS'])
                        bands.append( row['BANDS'] )
                    except TypeError:
                        bands.append( [row['BANDS'],] )
                    chans.append( row['CHANS'] )
                    pols.append( row['PFLAGS'] )
                    reas.append( row['REASON'] )
                    sevs.append( row['SEVERITY'] )
        ## New Flags
        obsdates.shape = (obsdates.shape[0]//nBL, nBL)
        obstimes.shape = (obstimes.shape[0]//nBL, nBL)
        mask.shape = (mask.shape[0]//nBL, nBL, nBand, nFreq, nStk)
        for i in xrange(nBL):
            ant1, ant2 = (bls[i]>>8)&0xFF, bls[i]&0xFF
            if i % 100 == 0 or i+1 == nBL:
                print("    Baseline %i of %i" % (i+1, nBL))
                
            for b,offset in enumerate(fqoffsets):
                maskXX = mask[:,i,b,:,0]
                maskYY = mask[:,i,b,:,1]
                
                flagsXX, _ = create_flag_groups(obstimes[:,i], freq+offset, maskXX)
                flagsYY, _ = create_flag_groups(obstimes[:,i], freq+offset, maskYY)
                
                for flag in flagsXX:
                    ants.append( (ant1,ant2) )
                    times.append( (obsdates[flag[0],i]+obstimes[flag[0],i]-obsdates[0,0], 
                                   obsdates[flag[1],i]+obstimes[flag[1],i]-obsdates[0,0]) )
                    bands.append( [1 if j == b else 0 for j in xrange(nBand)] )
                    chans.append( (flag[2]+1, flag[3]+1) )
                    pols.append( (1, 0, 1, 1) )
                    reas.append( 'AMPLITUDESCALE.PY' )
                    sevs.append( -1 )
                for flag in flagsYY:
                    ants.append( (ant1,ant2) )
                    times.append( (obsdates[flag[0],i]+obstimes[flag[0],i]-obsdates[0,0], 
                                   obsdates[flag[1],i]+obstimes[flag[1],i]-obsdates[0,0]) )
                    bands.append( [1 if j == b else 0 for j in xrange(nBand)] )
                    chans.append( (flag[2]+1, flag[3]+1) )
                    pols.append( (0, 1, 1, 1) )
                    reas.append( 'AMPLITUDESCALE.PY' )
                    sevs.append( -1 )
                    
        ## Build the FLAG table
        print('    FITS HDU')
        ### Columns
        nFlags = len(ants)
        c1 = astrofits.Column(name='SOURCE_ID', format='1J',           array=numpy.zeros((nFlags,), dtype=numpy.int32))
        c2 = astrofits.Column(name='ARRAY',     format='1J',           array=numpy.zeros((nFlags,), dtype=numpy.int32))
        c3 = astrofits.Column(name='ANTS',      format='2J',           array=numpy.array(ants, dtype=numpy.int32))
        c4 = astrofits.Column(name='FREQID',    format='1J',           array=numpy.zeros((nFlags,), dtype=numpy.int32))
        c5 = astrofits.Column(name='TIMERANG',  format='2E',           array=numpy.array(times, dtype=numpy.float32))
        c6 = astrofits.Column(name='BANDS',     format='%iJ' % nBand,  array=numpy.array(bands, dtype=numpy.int32).squeeze())
        c7 = astrofits.Column(name='CHANS',     format='2J',           array=numpy.array(chans, dtype=numpy.int32))
        c8 = astrofits.Column(name='PFLAGS',    format='4J',           array=numpy.array(pols, dtype=numpy.int32))
        c9 = astrofits.Column(name='REASON',    format='A40',          array=numpy.array(reas))
        c10 = astrofits.Column(name='SEVERITY', format='1J',           array=numpy.array(sevs, dtype=numpy.int32))
        colDefs = astrofits.ColDefs([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
        ### The table itself
        flags = astrofits.BinTableHDU.from_columns(colDefs)
        ### The header
        flags.header['EXTNAME'] = ('FLAG', 'FITS-IDI table name')
        flags.header['EXTVER'] = (1 if fgdata is None else fgdata.header['EXTVER']+1, 'table instance number') 
        flags.header['TABREV'] = (2, 'table format revision number')
        for key in ('NO_STKD', 'STK_1', 'NO_BAND', 'NO_CHAN', 'REF_FREQ', 'CHAN_BW', 'REF_PIXL', 'OBSCODE', 'ARRNAM', 'RDATE'):
            flags.header[key] = (uvdata.header[key], uvdata.header.comments[key])
        flags.header['HISTORY'] = 'Flagged with %s, revision $Rev$' % os.path.basename(__file__)
        
        # Clean up the old FLAG tables, if any, and then insert the new table where it needs to be 
        if args.drop:
            ## Reset the EXTVER on the new FLAG table
            flags.header['EXTVER'] = (1, 'table instance number')
            ## Find old tables 
            toRemove = [] 
            for hdu in hdulist: 
                try: 
                    if hdu.header['EXTNAME'] == 'FLAG': 
                        toRemove.append( hdu ) 
                except KeyError: 
                    pass 
            ## Remove old tables 
            for hdu in toRemove: 
                ver = hdu.header['EXTVER'] 
                del hdulist[hdulist.index(hdu)] 
                print("  WARNING: removing old FLAG table - version %i" % ver )
        ## Insert the new table right before UV_DATA 
        hdulist.insert(-1, flags)
        
        # Save
        print("  Saving to disk")
        ## What to call it
        outname = os.path.basename(filename)
        outname, outext = os.path.splitext(outname)
        outname = '%s_scaled%s' % (outname, outext)
        ## Does it already exist or not
        if os.path.exists(outname):
            if not args.force:
                yn = raw_input("WARNING: '%s' exists, overwrite? [Y/n] " % outname)
            else:
                yn = 'y'
                
            if yn not in ('n', 'N'):
                os.unlink(outname)
            else:
                raise RuntimeError("Output file '%s' already exists" % outname)
        ## Open and create a new primary HDU
        hdulist2 = astrofits.open(outname, mode='append')
        primary =   astrofits.PrimaryHDU()
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
        primary.header['HISTORY'] = 'Amplitude scale with %s, revision $Rev$' % os.path.basename(__file__)
        primary.header['HISTORY'] = 'Switched power values from %s' % os.path.basename(os.path.abspath(args.sdm))
        hdulist2.append(primary)
        hdulist2.flush()
        ## Copy the extensions over to the new file
        for hdu in hdulist[1:]:
            if hdu.header['EXTNAME'] == 'UV_DATA':
                ### Updated the UV_DATA table with the dedispersed data
                flux = numpy.where(numpy.isfinite(flux), flux, 0.0)
                flux = flux.view(numpy.float32)
                flux = flux.astype(hdu.data['FLUX'].dtype)
                flux.shape = hdu.data['FLUX'].shape
                hdu.data['FLUX'][...] = flux
                
            hdulist2.append(hdu)
            hdulist2.flush()
        hdulist2.close()
        hdulist.close()
        print("  -> Amplitude scaled FITS IDI file is '%s'" % outname)
        print("  Finished in %.3f s" % (time.time()-t0,))


if __name__ == "__main__":
    numpy.seterr(all='ignore')
    parser = argparse.ArgumentParser(
        description='Amplitude scale FITS-IDI files using the switched power data contained in a VLA SDM', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('sdm', type=str, 
                        help='VLA SDM used for flagging')
    parser.add_argument('filename', type=str, nargs='+', 
                        help='filename to process')
    parser.add_argument('-n', '--norm', type=float, default=1700.0, 
                        help='scale factor normalization factor')
    parser.add_argument('-d', '--drop', action='store_true', 
                        help='drop all existing FLAG tables')
    parser.add_argument('-f', '--force', action='store_true', 
                        help='force overwriting of existing FITS-IDI files')
    args = parser.parse_args()
    main(args)
    