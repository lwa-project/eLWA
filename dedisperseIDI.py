#!/usr/bin/env python

"""
Incoherent dedispersion for FITS-IDI files containing eLWA data.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    raw_input = input
    
import os
import git
import sys
import time
import numpy
from astropy.io import fits as astrofits
import argparse
from datetime import datetime

from lsl.astro import utcjd_to_unix
from lsl.writer.fitsidi import NUMERIC_STOKES
from lsl.misc.dedispersion import delay, incoherent

from flagger import *


def get_source_blocks(hdulist):
    """
    Given a astrofits hdulist, look at the source IDs listed in the UV_DATA table
    and generate a list of contiguous blocks for each source.  This list is 
    returned as a list of blocks where each block is itself a two-element list.
    This two-element list contains the the start and stop rows for the block.
    """
    
    # Get the tables
    uvdata = hdulist['UV_DATA']
    
    # Pull out various bits of information we need to flag the file
    ## Source list
    srcs = uvdata.data['SOURCE']
    ## Time of each integration
    obsdates = uvdata.data['DATE']
    obstimes = uvdata.data['TIME']
    inttimes = uvdata.data['INTTIM']
    
    # Find the source blocks to see if there is something we can use
    # to help the dedispersion
    usrc = numpy.unique(srcs)
    blocks = []
    for src in usrc:
        valid = numpy.where( src == srcs )[0]
        
        blocks.append( [valid[0],valid[0]] )
        for v in valid[1:]:
            if v == blocks[-1][1] + 1 \
                and (obsdates[v] - obsdates[blocks[-1][1]] + obstimes[v] - obstimes[blocks[-1][1]])*86400 < 10*inttimes[v]:
                blocks[-1][1] = v
            else:
                blocks.append( [v,v] )
    blocks.sort()
    
    # Done
    return blocks


def get_flags_as_mask(hdulist, selection=None, version=0):
    """
    Given a astrofits hdulist, build a mask for the visibility data based using 
    the specified version of the FLAG table.  This can also be done for a 
    sub-set of the full visibility data by using the 'selection' keyword to 
    provide a list of visility entries to create the mask for.
    
    .. note::
        Be default the FLAG version used is the most recent version.
    """
    
    # Get the tables
    fgdata = None
    for hdu in hdulist[1:]:
        if hdu.header['EXTNAME'] == 'FLAG':
            if version == 0 or hdu.header['EXTVER'] == version:
                fgdata = hdu
    uvdata = hdulist['UV_DATA']
    
    # Pull out various bits of information we need to flag the file
    ## Frequency and polarization setup
    nBand, nFreq, nStk = uvdata.header['NO_BAND'], uvdata.header['NO_CHAN'], uvdata.header['NO_STKD']
    ## Baseline list
    bls = uvdata.data['BASELINE']
    ## Time of each integration
    obsdates = uvdata.data['DATE']
    obstimes = uvdata.data['TIME']
    inttimes = uvdata.data['INTTIM']
    ## Number of rows in the file
    if selection is None:
        flux_rows = uvdata.data['FLUX'].shape[0]
    else:
        flux_rows = len(selection)
        
    # Create the mask
    mask = numpy.zeros((flux_rows, nBand, nFreq, nStk), dtype=numpy.bool)
    if fgdata is not None:
        reltimes = obsdates - obsdates[0] + obstimes
        maxtimes = reltimes + inttimes / 2.0 / 86400.0
        mintimes = reltimes - inttimes / 2.0 / 86400.0
        if selection is not None:
            bls = bls[selection]
            maxtimes = maxtimes[selection]
            mintimes = mintimes[selection]
            
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
                
    # Done
    return mask


def get_trailing_scan(filename, src_name, needed, drop_mask=False):
    mask = None
    flux = None
    
    nextname, ext = filename.rsplit('_', 1)
    try:
        ext = int(ext, 10)
    except ValueError:
        ext = -100
    ext += 1
    nextname = "%s_%i" % (nextname, ext)
    
    if os.path.exists(nextname):
        # Open up the next file
        hdulist = astrofits.open(nextname, mode='readonly', memmap=True)
        srdata = hdulist['SOURCE']
        uvdata = hdulist['UV_DATA']
        
        # Pull out various bits of information we need to flag the file
        ## Frequency and polarization setup
        nBand, nFreq, nStk = uvdata.header['NO_BAND'], uvdata.header['NO_CHAN'], uvdata.header['NO_STKD']
        ## Baseline list
        bls = uvdata.data['BASELINE']
        ## Integration time
        inttimes = uvdata.data['INTTIM']
        ## Source list
        srcs = uvdata.data['SOURCE']
        
        # Find the source blocks to see if there is something we can use
        # to help the dedispersion
        blocks = get_source_blocks(hdulist)
        block = blocks[0]
        match = range(block[0],block[1]+1)
        
        # Check and see if the first block in 'nextname' matches the last in 'filename'
        next_src_name = srdata.data['SOURCE'][srcs[match[0]]-1]
        if src_name == next_src_name:
            bbls = numpy.unique(bls[match])
            next_needed = inttimes[match[0]]
            next_needed = int(numpy.ceil(needed/next_needed))
            next_needed_rows = next_needed*len(bbls)
            
            match = range(match[0], min([match[0]+next_needed_rows, match[-1]+1]))
            flux = uvdata.data['FLUX'][match].astype(numpy.float32)
            
            # Convert the visibilities to something that we can easily work with
            nComp = flux.shape[1] // nBand // nFreq // nStk
            if nComp == 2:
                ## Case 1) - Just real and imaginary data
                flux = flux.view(numpy.complex64)
            else:
                ## Case 2) - Real, imaginary data + weights (drop the weights)
                flux = flux[:,0::nComp] + 1j*flux[:,1::nComp]
            flux.shape = (flux.shape[0], nBand, nFreq, nStk)
            
            # Create a mask of the old flags, if needed
            mask = get_flags_as_mask(hdulist, selection=match, version=0)
            
        del srdata
        del uvdata
        hdulist.close()
        
    return mask, flux


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
        srdata = hdulist['SOURCE']
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
            
        # NOTE: Assumes that the Stokes parameters increment by -1
        polMapper = {}
        for i in xrange(uvdata.header['NO_STKD']):
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
        
        # Find unique baselines and scans to work on
        ubls = numpy.unique(bls)
        blocks = get_source_blocks(hdulist)
        
        # Create a mask of the old flags, if needed
        old_flag_mask = get_flags_as_mask(hdulist, version=0-args.drop)
        
        # Dedisperse
        mask = numpy.zeros(flux.shape, dtype=numpy.bool)
        for i,block in enumerate(blocks):
            tS = time.time()
            print('  Working on scan %i of %i' % (i+1, len(blocks)))
            match = range(block[0],block[1]+1)
            
            bbls = numpy.unique(bls[match])
            times = obstimes[match] * 86400.0
            ints = inttimes[match]
            scanStart = datetime.utcfromtimestamp( utcjd_to_unix( obsdates[match[ 0]] + obstimes[match[ 0]] ) )
            scanStop  = datetime.utcfromtimestamp( utcjd_to_unix( obsdates[match[-1]] + obstimes[match[-1]] ) )
            print('    Scan spans %s to %s UTC' % (scanStart.strftime('%Y/%m/%d %H:%M:%S'), scanStop.strftime('%Y/%m/%d %H:%M:%S')))
            
            freq_comb = []
            for b,offset in enumerate(fqoffsets):
                freq_comb.append( freq + offset)
            freq_comb = numpy.concatenate(freq_comb)
            
            nBL = len(bbls)
            vis = flux[match,:,:,:]
            ofm = old_flag_mask[match,:,:,:]
            
            ## If this is the last block, check and see if there is anything that 
            ## we can pull out next file in the sequence so that we don't have a
            ## dedispersion gap
            to_trim = -1
            if i == len(blocks)-1:
                src_name = srdata.data['SOURCE'][srcs[match[0]]-1]
                nextmask, nextflux = get_trailing_scan(filename, src_name, 
                                                       max(delay(freq_comb, args.DM)))
                if nextmask is not None and nextflux is not None:
                    to_trim = ofm.shape[0]
                    vis = numpy.concatenate([vis, nextflux])
                    ofm = numpy.concatenate([ofm, nextmask])
                    print('      Appended %i times from the next file in the sequence' % (nextflux.shape[0]//nBL))
                    
            vis.shape = (vis.shape[0]//nBL, nBL, vis.shape[1]*vis.shape[2], vis.shape[3])
            ofm.shape = (ofm.shape[0]//nBL, nBL, ofm.shape[1]*ofm.shape[2], ofm.shape[3])
            print('      Scan contains %i times, %i baselines, %i bands/channels, %i polarizations' % vis.shape)
            
            if vis.shape[0] < 5:
                print('        Too few integrations, skipping')
                vis[:,:,:,:] = numpy.nan
                ofm[:,:,:,:] = True
            else:
                for j in xrange(nBL):
                    for k in xrange(nStk):
                        vis[:,j,:,k] = incoherent(freq_comb, vis[:,j,:,k], ints[0], args.DM, boundary='fill', fill_value=numpy.nan)
                        ofm[:,j,:,k] = incoherent(freq_comb, ofm[:,j,:,k], ints[0], args.DM, boundary='fill', fill_value=True)
            vis.shape = (vis.shape[0]*vis.shape[1], len(fqoffsets), vis.shape[2]//len(fqoffsets), vis.shape[3])
            ofm.shape = (ofm.shape[0]*ofm.shape[1], len(fqoffsets), ofm.shape[2]//len(fqoffsets), ofm.shape[3])
            
            if to_trim != -1:
                print('      Removing the appended times')
                vis = vis[:to_trim,...]
                ofm = ofm[:to_trim,...]
            flux[match,:,:,:] = vis
            
            print('      Saving polarization masks')
            submask = numpy.where(numpy.isfinite(vis), False, True)
            submask.shape = (len(match), flux.shape[1], flux.shape[2], flux.shape[3])
            mask[match,:,:,:] = submask
            
            print('      Statistics for this scan')
            print('      -> %s      - %.1f%% flagged' % (polMapper[0], 100.0*mask[match,:,:,0].sum()/mask[match,:,:,0].size,))
            print('      -> %s      - %.1f%% flagged' % (polMapper[1], 100.0*mask[match,:,:,1].sum()/mask[match,:,:,0].size,))
            print('      -> Elapsed - %.3f s' % (time.time()-tS,))
            
            # Add in the original flag mask
            mask[match,:,:,:] |= ofm
            
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
                    reas.append( 'DEDISPERSEIDI.PY' )
                    sevs.append( -1 )
                for flag in flagsYY:
                    ants.append( (ant1,ant2) )
                    times.append( (obsdates[blset[flag[0]]]+obstimes[blset[flag[0]]]-obsdates[0], 
                                   obsdates[blset[flag[1]]]+obstimes[blset[flag[1]]]-obsdates[0]) )
                    bands.append( [1 if j == b else 0 for j in xrange(nBand)] )
                    chans.append( (flag[2]+1, flag[3]+1) )
                    pols.append( (0, 1, 1, 1) )
                    reas.append( 'DEDISPERSEIDI.PY' )
                    sevs.append( -1 )
                    
        ## Figure out our revision
        try:
            repo = git.Repo(os.path.dirname(os.path.abspath(__file__)))
            try:
                branch = repo.active_branch.name
                hexsha = repo.active_branch.commit.hexsha
            except TypeError:
                branch = '<detached>'
                hexsha = repo.head.commit.hexsha
            shortsha = hexsha[-7:]
            dirty = ' (dirty)' if repo.is_dirty() else ''
        except git.exc.GitError:
            branch = 'unknown'
            hexsha = 'unknown'
            shortsha = 'unknown'
            dirty = ''
            
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
        for key in ('NO_STKD', 'STK_1', 'NO_BAND', 'NO_CHAN', 'REF_FREQ',
                    'CHAN_BW', 'REF_PIXL', 'OBSCODE', 'ARRNAM', 'RDATE'):
            try:
                flags.header[key] = (uvdata.header[key], uvdata.header.comments[key])
            except KeyError:
                pass
        flags.header['HISTORY'] = 'Flagged with %s, revision %s.%s%s' % (os.path.basename(__file__), branch, shortsha, dirty)
        flags.header['HISTORY'] = 'Dedispersed at %.6f pc / cm^3' % args.DM
        
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
        outname = '%s_DM%.4f%s' % (outname, args.DM, outext)
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
        primary.header['HISTORY'] = 'Dedispersed with %s, revision %s.%s%s' % (os.path.basename(__file__), branch, shortsha, dirty)
        primary.header['HISTORY'] = 'Dedispersed at %.6f pc / cm^3' % args.DM
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
        print("  -> Dedispersed FITS IDI file is '%s'" % outname)
        print("  Finished in %.3f s" % (time.time()-t0,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='apply incoherent dedispersion to a collection of FITS-IDI files', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('DM', type=float, 
                        help='dispersion measure to correct for in pc / cm^3')
    parser.add_argument('filename', type=str, nargs='+', 
                        help='filename to process')
    parser.add_argument('-d', '--drop', action='store_true', 
                        help='drop all existing FLAG tables')
    parser.add_argument('-f', '--force', action='store_true', 
                        help='force overwriting of existing FITS-IDI files')
    args = parser.parse_args()
    main(args)
    
