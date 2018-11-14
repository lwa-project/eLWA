#!/usr/bin/env python

"""
DTV pilot carrier flagger for FITS-IDI files containing eLWA data.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import sys
import time
import numpy
import pyfits
import argparse
from datetime import datetime

from lsl.astro import utcjd_to_unix

from flagger import *


def main(args):
    # Parse the command line
    filenames = args.filename
    
    for filename in filenames:
        t0 = time.time()
        print "Working on '%s'" % os.path.basename(filename)
        # Open the FITS IDI file and access the UV_DATA extension
        hdulist = pyfits.open(filename, mode='readonly')
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
        
        # Find unique scans to work on
        blocks = []
        for src in usrc:
            valid = numpy.where( src == srcs )[0]
            
            blocks.append( [valid[0],valid[0]] )
            for v in valid[1:]:
                if v == blocks[-1][1] + 1:
                    blocks[-1][1] = v
                else:
                    blocks.append( [v,v] )
        blocks.sort()
        
        # Build up the mask
        mask = numpy.zeros(flux.shape, dtype=numpy.bool)
        for i,block in enumerate(blocks):
            tS = time.time()
            print '  Working on scan %i of %i' % (i+1, len(blocks))
            match = range(block[0],block[1]+1)
            
            bbls = numpy.unique(bls[match])
            times = obstimes[match] * 86400.0
            scanStart = datetime.utcfromtimestamp( utcjd_to_unix( obsdates[match[ 0]] + obstimes[match[ 0]] ) )
            scanStop  = datetime.utcfromtimestamp( utcjd_to_unix( obsdates[match[-1]] + obstimes[match[-1]] ) )
            print '    Scan spans %s to %s UTC' % (scanStart.strftime('%Y/%m/%d %H:%M:%S'), scanStop.strftime('%Y/%m/%d %H:%M:%S'))
            
            for b,offset in enumerate(fqoffsets):
                print '    IF #%i' % (b+1,)
                crd = uvw[match,:]
                visXX = flux[match,b,:,0]
                visYY = flux[match,b,:,1]
                
                nBL = len(bbls)
                times = times[0::nBL]
                crd.shape = (crd.shape[0]/nBL, nBL, 1, 3)
                visXX.shape = (visXX.shape[0]/nBL, nBL, visXX.shape[1])
                visYY.shape = (visYY.shape[0]/nBL, nBL, visYY.shape[1])
                print '      Scan/IF contains %i times, %i baselines, %i channels' % visXX.shape
                
                maskDTV = numpy.zeros(visXX.shape, dtype=numpy.bool)
                
                ## Mask channels 2, 3, 4, 5, and 6 which are all in the LWA band
                for ledge in (54, 60, 66, 76, 82):
                    ## The pilot is offset 310 kHz above the lower edge of the band
                    pilot = 1e6*(ledge + 0.310)
                    bad = numpy.where( numpy.abs(freq+offset - pilot) <= 25e3 )[0]
                    if len(bad) > 0:
                        maskDTV[:,:,bad] = True
                        
                print '      Saving polarization masks'
                submask = maskDTV
                submask.shape = (len(match), flux.shape[2])
                mask[match,b,:,0] = submask
                submask = maskDTV
                submask.shape = (len(match), flux.shape[2])
                mask[match,b,:,1] = submask
                
                print '      Statistics for this scan/IF'
                print '      -> XX      - %.1f%% flagged' % (100.0*mask[match,b,:,0].sum()/mask[match,b,:,0].size,)
                print '      -> YY      - %.1f%% flagged' % (100.0*mask[match,b,:,1].sum()/mask[match,b,:,0].size,)
                print '      -> Elapsed - %.3f s' % (time.time()-tS,)
                
        # Convert the masks into a format suitable for writing to a FLAG table
        print "  Building FLAG table"
        ants, times, bands, chans, pols, reas, sevs = [], [], [], [], [], [], []
        ## Old flags
        if fgdata is not None:
            for row in fgdata.data:
                ants.append( row['ANTS'] )
                times.append( row['TIMERANG'] )
                bands.append( row['BANDS'] )
                chans.append( row['CHANS'] )
                pols.append( row['PFLAGS'] )
                reas.append( row['REASON'] )
                sevs.append( row['SEVERITY'] )
        ## New Flags
        obsdates.shape = (obsdates.shape[0]/nBL, nBL)
        obstimes.shape = (obstimes.shape[0]/nBL, nBL)
        mask.shape = (mask.shape[0]/nBL, nBL, nBand, nFreq, nStk)
        for i in xrange(nBL):
            ant1, ant2 = (bls[i]>>8)&0xFF, bls[i]&0xFF
            if i % 100 == 0 or i+1 == nBL:
                print "    Baseline %i of %i" % (i+1, nBL)
                
            for b,offset in enumerate(fqoffsets):
                maskXX = mask[:,i,b,:,0]
                maskYY = mask[:,i,b,:,1]
                
                flagsXX, _ = create_flag_groups(obstimes[:,i], freq+offset, maskXX)
                flagsYY, _ = create_flag_groups(obstimes[:,i], freq+offset, maskYY)
                
                for flag in flagsXX:
                    ants.append( (ant1,ant2) )
                    times.append( (obstimes[flag[0],i], obstimes[flag[1],i]) )
                    bands.append( [1 if j == b else 0 for j in xrange(nBand)] )
                    chans.append( (flag[2]+1, flag[3]+1) )
                    pols.append( (1, 0, 1, 1) )
                    reas.append( 'FLAGDTV.PY' )
                    sevs.append( -1 )
                for flag in flagsYY:
                    ants.append( (ant1,ant2) )
                    times.append( (obstimes[flag[0],i], obstimes[flag[1],i]) )
                    bands.append( [1 if j == b else 0 for j in xrange(nBand)] )
                    chans.append( (flag[2]+1, flag[3]+1) )
                    pols.append( (0, 1, 1, 1) )
                    reas.append( 'FLAGDTV.PY' )
                    sevs.append( -1 )
                    
        ## Build the FLAG table
        print '    FITS HDU'
        ### Columns
        nFlags = len(ants)
        c1 = pyfits.Column(name='SOURCE_ID', format='1J',           array=numpy.zeros((nFlags,), dtype=numpy.int32))
        c2 = pyfits.Column(name='ARRAY',     format='1J',           array=numpy.zeros((nFlags,), dtype=numpy.int32))
        c3 = pyfits.Column(name='ANTS',      format='2J',           array=numpy.array(ants, dtype=numpy.int32))
        c4 = pyfits.Column(name='FREQID',    format='1J',           array=numpy.zeros((nFlags,), dtype=numpy.int32))
        c5 = pyfits.Column(name='TIMERANG',  format='2E',           array=numpy.array(times, dtype=numpy.float32))
        c6 = pyfits.Column(name='BANDS',     format='%iJ' % nBand,  array=numpy.array(bands, dtype=numpy.int32).squeeze())
        c7 = pyfits.Column(name='CHANS',     format='2J',           array=numpy.array(chans, dtype=numpy.int32))
        c8 = pyfits.Column(name='PFLAGS',    format='4J',           array=numpy.array(pols, dtype=numpy.int32))
        c9 = pyfits.Column(name='REASON',    format='A40',          array=numpy.array(reas))
        c10 = pyfits.Column(name='SEVERITY', format='1J',           array=numpy.array(sevs, dtype=numpy.int32))
        colDefs = pyfits.ColDefs([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
        ### The table itself
        flags = pyfits.new_table(colDefs)
        ### The header
        flags.header['EXTNAME'] = ('FLAG', 'FITS-IDI table name')
        flags.header['EXTVER'] = (1 if fgdata is None else fgdata.header['EXTVER']+1, 'table instance number') 
        flags.header['TABREV'] = (2, 'table format revision number')
        for key in ('NO_STKD', 'STK_1', 'NO_BAND', 'NO_CHAN', 'REF_FREQ', 'CHAN_BW', 'REF_PIXL', 'OBSCODE', 'ARRNAM', 'RDATE'):
            flags.header[key] = (uvdata.header[key], uvdata.header.comments[key])
        flags.header['HISTORY'] = 'Flagged with %s, revision $Rev$' % os.path.basename(__file__)
        
        # Insert the new table right before UV_DATA
        hdulist.insert(-1, flags)
        
        # Save
        print "  Saving to disk"
        ## What to call it
        outname = os.path.basename(filename)
        outname, outext = os.path.splitext(outname)
        outname = '%s_flagged%s' % (outname, outext)
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
        hdulist2 = pyfits.open(outname, mode='append')
        primary =	pyfits.PrimaryHDU()
        for key in hdulist[0].header:
            primary.header[key] = (hdulist[0].header[key], hdulist[0].header.comments[key])
        hdulist2.append(primary)
        hdulist2.flush()
        ## Copy the extensions over to the new file
        for hdu in hdulist[1:]:
            hdulist2.append(hdu)
            hdulist2.flush()
        hdulist2.close()
        hdulist.close()
        print "  -> Flagged FITS IDI file is '%s'" % outname
        print "  Finished in %.3f s" % (time.time()-t0,)


if __name__ == "__main__":
    numpy.seterr(all='ignore')
    parser = argparse.ArgumentParser(
        description='Flag the DTV pilot carriers in FITS-IDI files', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('filename', type=str, nargs='+', 
                        help='filename to process')
    parser.add_argument('-f', '--force', action='store_true', 
                        help='force overwriting of existing FITS-IDI files')
    args = parser.parse_args()
    main(args)
    