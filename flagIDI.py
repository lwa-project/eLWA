#!/usr/bin/env python

"""
RFI flagger for FITS-IDI files containing eLWA data.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import sys
import time
import numpy
import getopt
import pyfits
from datetime import datetime

from lsl.astro import utcjd_to_unix

from sdm import getFlags, filterFlags
from flagger import *


def usage(exitCode=None):
    print """flagIDI.py - Flag RFI in FITS-IDI files

Usage:
flagIDI.py [OPTIONS] <fits_file> [<fits_file> [...]]

Options:
-h, --help          Display this help information
-s, --sdm           Read in the provided VLA SDM for additional flags
-p, --scf-passes    Number of passes to make through the spurious 
                    correlation sub-routine (default = 0 = disabled,
                    >0 enables)
-f, --force         Force overwriting of existing FITS-IDI files
"""
    
    if exitCode is not None:
        sys.exit(exitCode)
    else:
        return True


def parseConfig(args):
    config = {}
    # Command line flags - default values
    config['sdm'] = None
    config['passes'] = 0
    config['force'] = False
    config['args'] = []
    
    # Read in and process the command line flags
    try:
        opts, args = getopt.getopt(args, "hs:p:f", ["help", "sdm=", "scf-passes=", "force"])
    except getopt.GetoptError, err:
        # Print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage(exitCode=2)
        
    # Work through opts
    for opt, value in opts:
        if opt in ('-h', '--help'):
            usage(exitCode=0)
        elif opt in ('-s', '--sdm'):
            config['sdm'] = value
        elif opt in ('-p', '--scf-passes'):
            config['passes'] = int(value, 10)
        elif opt in ('-f', '--force'):
            config['force'] = True
        else:
            assert False
            
    # Add in arguments
    config['args'] = args
    
    # Validate
    if config['passes'] < 0:
        raise RuntimeError("Invalid number of spurious correlation passes: %i" % config['passes'])
        
    # Return configuration
    return config


def main(args):
    # Parse the command line
    config = parseConfig(args)
    filenames = config['args']
    
    # Parse the SDM, if provided
    sdm_flags = []
    if config['sdm'] is not None:
        flags = getFlags(config['sdm'])
        for flag in flags:
            flag['antennaId'] = flag['antennaId'].replace('EA', 'LWA0')
            sdm_flags.append( flag )
            
    for filename in filenames:
        t0 = time.time()
        print "Working on '%s'" % os.path.basename(filename)
        # Open the FITS IDI file and access the UV_DATA extension
        hdulist = pyfits.open(filename, mode='readonly')
        andata = hdulist['ANTENNA']
        fqdata = hdulist['FREQUENCY']
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
                
                if visXX.shape[0] < 5:
                    print '        Too few integrations, skipping'
                    continue
                    
                antennas = []
                for j in xrange(nBL):
                    ant1, ant2 = (bbls[j]>>8)&0xFF, bbls[j]&0xFF
                    if ant1 not in antennas:
                        antennas.append(ant1)
                    if ant2 not in antennas:
                        antennas.append(ant2)
                        
                print '      Flagging baselines'
                maskXX = mask_bandpass(antennas, times, freq+offset, visXX)
                maskYY = mask_bandpass(antennas, times, freq+offset, visYY)
                
                visXX = numpy.ma.array(visXX, mask=maskXX)
                visYY = numpy.ma.array(visYY, mask=maskYY)
                
                if config['passes'] > 0:
                    print '      Flagging spurious correlations'
                    for p in xrange(config['passes']):
                        print '        Pass #%i' % (p+1,)
                        visXX.mask = mask_spurious(antennas, times, crd, freq+offset, visXX)
                        visYY.mask = mask_spurious(antennas, times, crd, freq+offset, visYY)
                        
                print '      Cleaning masks'
                visXX.mask = cleanup_mask(visXX.mask)
                visYY.mask = cleanup_mask(visYY.mask)
                
                print '      Saving polarization masks'
                submask = visXX.mask
                submask.shape = (len(match), flux.shape[2])
                mask[match,b,:,0] = submask
                submask = visYY.mask
                submask.shape = (len(match), flux.shape[2])
                mask[match,b,:,1] = submask
                
                print '      Statistics for this scan/IF'
                print '      -> XX      - %.1f%% flagged' % (100.0*mask[match,b,:,0].sum()/mask[match,b,:,0].size,)
                print '      -> YY      - %.1f%% flagged' % (100.0*mask[match,b,:,1].sum()/mask[match,b,:,0].size,)
                print '      -> Elapsed - %.3f s' % (time.time()-tS,)
                
        # Convert the masks into a format suitable for writing to a FLAG table
        print "  Building FLAG table"
        obsdates.shape = (obsdates.shape[0]/nBL, nBL)
        obstimes.shape = (obstimes.shape[0]/nBL, nBL)
        mask.shape = (mask.shape[0]/nBL, nBL, nBand, nFreq, nStk)
        ants, times, bands, chans, pols = [], [], [], [], []
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
                for flag in flagsYY:
                    ants.append( (ant1,ant2) )
                    times.append( (obstimes[flag[0],i], obstimes[flag[1],i]) )
                    bands.append( [1 if j == b else 0 for j in xrange(nBand)] )
                    chans.append( (flag[2]+1, flag[3]+1) )
                    pols.append( (0, 1, 1, 1) )
                    
        # Filter the SDM flags for what is relevant for this file
        fileStart = obsdates[ 0,0] + obstimes[ 0,0]
        fileStop  = obsdates[-1,0] + obstimes[-1,0]
        sub_sdm_flags = filterFlags(sdm_flags, fileStart, fileStop)
        
        # Add in the SDM flags
        nSubSDM = len(sub_sdm_flags)
        for i,flag in enumerate(sub_sdm_flags):
            if i % 100 == 0 or i+1 == nSubSDM:
                print "    SDM %i of %i" % (i+1, nSubSDM)
                
            try:
                ant1 = antLookup[flag['antennaId']]
            except KeyError:
                continue
            tStart = max([flag['startTime'] - obsdates[0,0], obstimes[ 0,0]])
            tStop  = min([flag['endTime']   - obsdates[0,0], obstimes[-1,0]])
            ants.append( (ant1,0) )
            times.append( (tStart, tStop) )
            bands.append( [1 for j in xrange(nBand)] )
            chans.append( (0, 0) )
            pols.append( (1, 1, 1, 1) )
            
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
        c9 = pyfits.Column(name='REASON',    format='A40',          array=numpy.array(['FLAGIDI.PY' for i in xrange(nFlags)]))
        c10 = pyfits.Column(name='SEVERITY', format='1J',           array=numpy.zeros((nFlags,), dtype=numpy.int32)-1)
        colDefs = pyfits.ColDefs([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
        ### The table itself
        flags = pyfits.new_table(colDefs)
        ### The header
        flags.header['EXTNAME'] = ('FLAG', 'FITS-IDI table name')
        flags.header['EXTVER'] = (1, 'table instance number') 
        flags.header['TABREV'] = (2, 'table format revision number')
        for key in ('NO_STKD', 'STK_1', 'NO_BAND', 'NO_CHAN', 'REF_FREQ', 'CHAN_BW', 'REF_PIXL', 'OBSCODE', 'ARRNAM', 'RDATE'):
            flags.header[key] = (uvdata.header[key], uvdata.header.comments[key])
        flags.header['HISTORY'] = 'Flagged with %s, revision $Rev$' % os.path.basename(__file__)
        if config['sdm'] is not None:
            flags.header['HISTORY'] = 'SDM flags from %s' % os.path.basename(os.path.abspath(config['sdm']))
        flags.header['HISTORY'] = '%i spurious correlation passes used' % config['passes']
        
        # Clean up the old FLAG tables, if any, and then insert the new table where it needs to be
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
            print "  WARNING: removing old FLAG table - version %i" % ver
        ## Insert the new table right before UV_DATA
        hdulist.insert(-1, flags)
        
        # Save
        print "  Saving to disk"
        ## What to call it
        outname = os.path.basename(filename)
        outname, outext = os.path.splitext(outname)
        outname = '%s_flagged%s' % (outname, outext)
        ## Does it already exist or not
        if os.path.exists(outname):
            if not config['force']:
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
    main(sys.argv[1:])
    
