#!/usr/bin/env python

"""
Frequency decimation script for FITS-IDI files containing eLWA data.
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

from flagger import *


def main(args):
    # Parse the command line
    filenames = args.filename
    
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
        
        # Verify we can flag this data
        if uvdata.header['STK_1'] > 0:
            raise RuntimeError("Cannot flag data with STK_1 = %i" % uvdata.header['STK_1'])
        if uvdata.header['NO_STKD'] < 4:
            raise RuntimeError("Cannot flag data with NO_STKD = %i" % uvdata.header['NO_STKD'])
            
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
        ## The actual visibility data
        flux = uvdata.data['FLUX'].astype(numpy.float32)
        weight = uvdata.data['WEIGHT'].astype(numpy.float32)
        
        # Convert the visibilities to something that we can easily work with
        nComp = flux.shape[1] // nBand // nFreq // nStk
        if nComp == 2:
            ## Case 1) - Just real and imaginary data
            flux = flux.view(numpy.complex64)
        else:
            ## Case 2) - Real, imaginary data + weights (drop the weights)
            flux = flux[:,0::nComp] + 1j*flux[:,1::nComp]
        flux.shape = (flux.shape[0], nBand, nFreq, nStk)
        weight.shape = (weight.shape[0], nBand, nFreq, nStk)
        
        # Find unique baselines, times, and sources to work with
        ubls = numpy.unique(bls)
        utimes = numpy.unique(obstimes)
        usrc = numpy.unique(srcs)
        nBL = len(ubls)
        
        # Create a mask of the old flags, if needed
        mask = numpy.zeros(flux.shape, dtype=numpy.bool)
        if not args.drop and fgdata is not None:
            reltimes = obsdates - obsdates[0] + obstimes
            maxtimes = reltimes + inttimes / 2.0 / 86400.0
            mintimes = reltimes - inttimes / 2.0 / 86400.0
            
            bls_ant1 = bls//256
            bls_ant2 = bls%256
            
            for row in fgdata.data:
                ant1, ant2 = row['ANTS']
                tStart, tStop = row['TIMERANG']
                band = row['BANDS']
                try:
                    len(band)
                except TypeError:
                    band = [band,]
                cStart, cStop = row['CHANS']
                if cStop == 0:
                    cStop = -1
                pol = row['PFLAGS'].astype(numpy.bool)
                
                if ant1 == 0 and ant2 == 0:
                    btmask = numpy.where( ( (maxtimes >= tStart) & (mintimes <= tStop) ) )[0]
                elif ant1 == 0 or ant2 == 0:
                    ant1 = max([ant1, ant2])
                    btmask = numpy.where( ( (bls_ant1 == ant1) | (bls_ant2 == ant1) ) \
                                        & ( (maxtimes >= tStart) & (mintimes <= tStop) ) )[0]
                else:
                    btmask = numpy.where( ( (bls_ant1 == ant1) & (bls_ant2 == ant2) ) \
                                        & ( (maxtimes >= tStart) & (mintimes <= tStop) ) )[0]
                for b,v in enumerate(band):
                    if not v:
                        continue
                    mask[btmask,b,cStart-1:cStop,:] |= pol
                        
        # Decimate
        ## Setup
        print("  Found %i channels, each %.3f kHz wide" % (nFreq, (freq[1]-freq[0])/1e3))
        if freq.size % args.decimation != 0:
            to_trim = (freq.size/args.decimation)*args.decimation
            to_drop = freq.size - to_trim
            print("  WARNING: Dropped %i channels (%.1f%%; %.3f kHz)" % (to_drop, 100.0*to_drop/freq.size, to_drop*(freq[1]-freq[0])/1e3))
            freq = freq[:to_trim]
            flux = flux[:,:,:to_trim,:]
            weight = weight[:,:,:to_trim,:]
            mask = mask[:,:,:to_trim,:]
        ## Go
        freq.shape = (freq.shape[0]//args.decimation, args.decimation)
        freq = freq.mean(axis=1)
        flux.shape = (flux.shape[0], flux.shape[1], flux.shape[2]//args.decimation, args.decimation, flux.shape[3])
        flux = flux.mean(axis=3)
        weight.shape = (weight.shape[0], weight.shape[1], weight.shape[2]//args.decimation, args.decimation, weight.shape[3])
        weight = weight.mean(axis=3)
        mask.shape = (mask.shape[0], mask.shape[1], mask.shape[2]//args.decimation, args.decimation, mask.shape[3])
        mask = mask.mean(axis=3).astype(numpy.bool)
        nFreq = freq.size
        print("  Decimated to %i channels, each %.3f kHz wide" % (nFreq, (freq[1]-freq[0])/1e3))
        
        # Convert the masks into a format suitable for writing to a FLAG table
        print("  Building FLAG table")
        ants, times, bands, chans, pols, reas, sevs = [], [], [], [], [], [], []
        ## New Flags
        nBL = len(ubls)
        for i in xrange(nBL):
            blset = numpy.where( bls == ubls[i] )[0]
            ant1, ant2 = (ubls[i]>>8)&0xFF, ubls[i]&0xFF
            if i % 100 == 0 or i+1 == nBL:
                print("    Baseline %i of %i" % (i+1, nBL))
                
            if len(blset) == 0:
                continue
                
            for b,offset in enumerate(fqoffsets):
                maskXX = mask[blset,b,:,0]
                maskYY = mask[blset,b,:,1]
                
                flagsXX, _ = create_flag_groups(obstimes[blset], freq+offset, maskXX)
                flagsYY, _ = create_flag_groups(obstimes[blset], freq+offset, maskYY)
                
                for flag in flagsXX:
                    ants.append( (ant1,ant2) )
                    times.append( (obsdates[blset[flag[0]]]+obstimes[blset[flag[0]]]-obsdates[0], 
                                   obsdates[blset[flag[1]]]+obstimes[blset[flag[1]]]-obsdates[0]) )
                    bands.append( [1 if j == b else 0 for j in xrange(nBand)] )
                    chans.append( (flag[2]+1, flag[3]+1) )
                    pols.append( (1, 0, 1, 1) )
                    reas.append( 'DECIMATEIDI.PY' )
                    sevs.append( -1 )
                for flag in flagsYY:
                    ants.append( (ant1,ant2) )
                    times.append( (obsdates[flag[0],i]+obstimes[flag[0],i]-obsdates[0,0], 
                                   obsdates[flag[1],i]+obstimes[flag[1],i]-obsdates[0,0]) )
                    bands.append( [1 if j == b else 0 for j in xrange(nBand)] )
                    chans.append( (flag[2]+1, flag[3]+1) )
                    pols.append( (0, 1, 1, 1) )
                    reas.append( 'DECIMATEIDI.PY' )
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
        else:
            for hdu in hdulist:
                try: 
                    if hdu.header['EXTNAME'] != 'FLAG': 
                        continue
                except KeyError: 
                    continue
                ### Figure out how to change the channel ranges
                scl = 1.0 / args.decimation
                chans = hdu.data['CHANS']
                chans = chans * scl
                chans = numpy.clip(chans, 1, nFreq)
                hdu.data['CHANS'][...] = chans.astype(hdu.data['CHANS'].dtype)
                hdu.header['HISTORY'] = 'Scaled channel flag value range from [1, %i] to [1, %i]' % (hdu.header['NO_CHAN'], nFreq)
        ## Insert the new table right before UV_DATA 
        hdulist.insert(-1, flags)
        
        # Save
        print("  Saving to disk")
        ## What to call it
        outname = os.path.basename(filename)
        outname, outext = os.path.splitext(outname)
        outname = '%s_decim%s' % (outname, outext)
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
        primary.header['HISTORY'] = 'Decimated by %i with %s, revision $Rev$' % (args.decimation, os.path.basename(__file__))
        hdulist2.append(primary)
        hdulist2.flush()
        ## Copy the extensions over to the new file
        for hdu in hdulist[1:]:
            ### Update the common header values
            hdu.header['NO_CHAN'] = nFreq
            hdu.header['CHAN_BW'] = freq[1]-freq[0]
            
            ### The individual tables that need updating
            if hdu.header['EXTNAME'] == 'FREQUENCY':
                temp = hdu.data['BANDFREQ'] + (freq[1]-freq[0])/2.0
                temp = temp.astype(hdu.data['BANDFREQ'].dtype)
                hdu.data['BANDFREQ'][...] = temp
                temp = hdu.data['CH_WIDTH']*0 + (freq[1]-freq[0])
                temp = temp.astype(hdu.data['CH_WIDTH'].dtype)
                hdu.data['CH_WIDTH'][...] = temp
                temp = hdu.data['TOTAL_BANDWIDTH']*0 + (freq[-1]-freq[0])
                temp = temp.astype(hdu.data['TOTAL_BANDWIDTH'].dtype)
                hdu.data['TOTAL_BANDWIDTH'][...] = temp
                
            elif hdu.header['EXTNAME'] == 'BANDPASS':
                hdu.header['NO_BACH'] = nFreq
                
                columns = []
                for col in hdu.data.columns:
                    temp = hdu.data[col.name]
                    fmt = col.format
                    if col.name in ('BREAL_1', 'BIMAG_1', 'BREAL_2', 'BIMAG_2'):
                        temp = temp.reshape(temp.shape[0], -1)
                        temp.shape = (temp.shape[0], nBand, temp.shape[1]//nBand)
                        temp = temp[:,:,:nFreq]
                        temp.shape = (temp.shape[0], temp.shape[1]*temp.shape[2])
                        temp = temp.astype(hdu.data[col.name].dtype)
                        fmt = '%i%s' % (nBand*nFreq, col.format[-1])
                    columns.append( astrofits.Column(name=col.name, unit=col.unit, format=fmt, array=temp) )
                colDefs = astrofits.ColDefs(columns)
                
                hduprime = astrofits.BinTableHDU.from_columns(colDefs)
                processed = []
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
                
            elif hdu.header['EXTNAME'] == 'UV_DATA':
                hdu.header['MAXIS3'] = nFreq
                hdu.header['CDELT3'] = freq[1]-freq[0]
                hdu.header['CRVAL3'] = freq[0]
                
                columns = []
                for col in hdu.data.columns:
                    temp = hdu.data[col.name]
                    fmt = col.format
                    if col.name in ('DATE', 'TIME'):
                        ## Why?
                        temp = temp.ravel()
                    elif col.name == 'FLUX':
                        temp = flux.view(numpy.float32)
                        temp = temp.astype(hdu.data[col.name].dtype)
                        temp.shape = (temp.shape[0], temp.shape[1]*temp.shape[2]*temp.shape[3])
                        fmt = '%i%s' % (2*nStk*nFreq*nBand, col.format[-1])
                    elif col.name == 'WEIGHT':
                        temp = weight.astype(hdu.data[col.name].dtype)
                        temp.shape = (temp.shape[0], temp.shape[1]*temp.shape[2]*temp.shape[3])
                        fmt = '%i%s' % (nStk*nFreq*nBand, col.format[-1])
                    columns.append( astrofits.Column(name=col.name, unit=col.unit, format=fmt, array=temp) )
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
        print("  -> Decimated FITS IDI file is '%s'" % outname)
        print("  Finished in %.3f s" % (time.time()-t0,))


if __name__ == "__main__":
    numpy.seterr(all='ignore')
    parser = argparse.ArgumentParser(
        description='Decimate the number of spectral channels in FITS-IDI files', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('decimation', type=int, 
                        help='frequency decimation factor')
    parser.add_argument('filename', type=str, nargs='+', 
                        help='filename to process')
    parser.add_argument('-d', '--drop', action='store_true', 
                        help='drop all existing FLAG tables')
    parser.add_argument('-f', '--force', action='store_true', 
                        help='force overwriting of existing FITS-IDI files')
    args = parser.parse_args()
    main(args)
    
