#!/usr/bin/env python3

"""
A FITS-IDI compatible version of plotFringes2.py geared towards diagnostic
plots.
"""

import os
import sys
import numpy as np
from astropy.io import fits as astrofits
import argparse
from datetime import datetime

from scipy.stats import scoreatpercentile as percentile

from lsl.astro import utcjd_to_unix
from lsl.writer.fitsidi import NUMERIC_STOKES

from matplotlib import pyplot as plt


def main(args):
    # Parse the command line
    ## Polarization
    plot_pols = []
    if args.xx:
        plot_pols.append('XX')
    if args.xy:
        plot_pols.append('XY')
    if args.yx:
        plot_pols.append('YX')
    if args.yy:
        plot_pols.append('YY')
    filename = args.filename
    
    figs = {}
    first = True
    for filename in args.filename:
        print(f"Working on '{os.path.basename(filename)}'")
        # Open the FITS IDI file and access the UV_DATA extension
        hdulist = astrofits.open(filename, mode='readonly')
        andata = hdulist['ANTENNA']
        fqdata = hdulist['FREQUENCY']
        fgdata = None
        for hdu in hdulist[1:]:
                if hdu.header['EXTNAME'] == 'FLAG':
                    fgdata = hdu
        uvdata = hdulist['UV_DATA']
        
        # Pull out various bits of information we need to flag the file
        ## Antenna look-up table
        antLookup = {}
        antLookup_inv = {}
        for an, ai in zip(andata.data['ANNAME'], andata.data['ANTENNA_NO']):
            antLookup[an] = ai
            antLookup_inv[ai] = an
        ## Frequency and polarization setup
        nBand, nFreq, nStk = uvdata.header['NO_BAND'], uvdata.header['NO_CHAN'], uvdata.header['NO_STKD']
        stk0 = uvdata.header['STK_1']
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
        freq = (np.arange(nFreq)-(uvdata.header['CRPIX3']-1))*uvdata.header['CDELT3']
        freq += uvdata.header['CRVAL3']
        ## UVW coordinates
        try:
            u, v, w = uvdata.data['UU'], uvdata.data['VV'], uvdata.data['WW']
        except KeyError:
            u, v, w = uvdata.data['UU---SIN'], uvdata.data['VV---SIN'], uvdata.data['WW---SIN']
        uvw = np.array([u, v, w]).T
        ## The actual visibility data
        flux = uvdata.data['FLUX'].astype(np.float32)
        
        # Convert the visibilities to something that we can easily work with
        nComp = flux.shape[1] // nBand // nFreq // nStk
        if nComp == 2:
            ## Case 1) - Just real and imaginary data
            flux = flux.view(np.complex64)
        else:
            ## Case 2) - Real, imaginary data + weights (drop the weights)
            flux = flux[:,0::nComp] + 1j*flux[:,1::nComp]
        flux.shape = (flux.shape[0], nBand, nFreq, nStk)
        
        # Find unique baselines, times, and sources to work with
        ubls = np.unique(bls)
        utimes = np.unique(obstimes)
        usrc = np.unique(srcs)
        
        # Make sure the reference antenna is in there
        if args.ref_ant is None:
            bl = bls[0]
            i,j = (bl>>8)&0xFF, bl&0xFF
            args.ref_ant = i
        else:
            found = False
            for bl in ubls:
                i,j = (bl>>8)&0xFF, bl&0xFF
                if i == args.ref_ant or j == args.ref_ant:
                    found = True
                    break
                elif antLookup_inv[i] == args.ref_ant:
                    args.ref_ant = i
                    found = True
                    break
                elif antLookup_inv[j] == args.ref_ant:
                    args.ref_ant = j
                    found = True
                    break
            if not found:
                raise RuntimeError("Cannot file reference antenna %s in the data" % args.ref_ant)
                
        # Process the baseline list
        if args.baseline is not None:
            newBaselines = []
            for bl in args.baseline.split(','):
                ## Split and sort out antenna number vs. name
                pair = bl.split('-')
                try:
                    pair[0] = int(pair[0], 10)
                except ValueError:
                    try:
                        pair[0] = antLookup[pair[0]]
                    except KeyError:
                        continue
                try:
                    pair[1] = int(pair[1], 10)
                except ValueError:
                    try:
                        pair[1] = antLookup[pair[1]]
                    except KeyError:
                        continue
                        
                ## Fill the baseline list with the conjugates, if needed
                newBaselines.append(tuple(pair))
                newBaselines.append((pair[1], pair[0]))
                
            ## Update
            args.baseline = newBaselines
            
        # Convert times to real times
        times = utcjd_to_unix(obsdates + obstimes)
        times = np.unique(times)
        
        # Build a mask
        mask = np.zeros(flux.shape, dtype=bool)
        if fgdata is not None and not args.drop:
            reltimes = obsdates - obsdates[0] + obstimes
            maxtimes = reltimes + inttimes / 2.0 / 86400.0
            mintimes = reltimes - inttimes / 2.0 / 86400.0
            
            bls_ant1 = bls//256
            bls_ant2 = bls%256
            
            for row in fgdata.data:
                ant1, ant2 = row['ANTS']
                
                ## Only deal with flags that we need for the plots
                process_flag = False
                if args.include_auto or ant1 != ant2 or ant1 == 0 or ant2 == 0:
                    if ant1 == 0 and ant2 == 0:
                        process_flag = True
                    elif args.baseline is not None:
                        if ant2 == 0 and ant1 in [a0 for a0,a1 in args.baseline]:
                            process_flag = True
                        elif (ant1,ant2) in args.baseline:
                            process_flag = True
                    elif args.ref_ant is not None:
                        if ant1 == args.ref_ant or ant2 == args.ref_ant:
                            process_flag = True
                    else:
                        process_flag = True
                if not process_flag:
                    continue
                    
                tStart, tStop = row['TIMERANG']
                band = row['BANDS']
                try:
                    len(band)
                except TypeError:
                    band = [band,]
                cStart, cStop = row['CHANS']
                if cStop == 0:
                    cStop = -1
                pol = row['PFLAGS'].astype(bool)
                
                if ant1 == 0 and ant2 == 0:
                    btmask = np.where( ( (maxtimes >= tStart) & (mintimes <= tStop) ) )[0]
                elif ant1 == 0 or ant2 == 0:
                    ant1 = max([ant1, ant2])
                    btmask = np.where( ( (bls_ant1 == ant1) | (bls_ant2 == ant1) ) \
                                      & ( (maxtimes >= tStart) & (mintimes <= tStop) ) )[0]
                else:
                    btmask = np.where( ( (bls_ant1 == ant1) & (bls_ant2 == ant2) ) \
                                      & ( (maxtimes >= tStart) & (mintimes <= tStop) ) )[0]
                for b,v in enumerate(band):
                    if not v:
                        continue
                    mask[btmask,b,cStart-1:cStop,:] |= pol
                    
        plot_bls = []
        cross = []
        for i in range(len(ubls)):
            bl = ubls[i]
            ant1, ant2 = (bl>>8)&0xFF, bl&0xFF 
            if args.include_auto or ant1 != ant2:
                if args.baseline is not None:
                    if (ant1,ant2) in args.baseline:
                        plot_bls.append( bl )
                        cross.append( i )
                elif args.ref_ant is not None:
                    if ant1 == args.ref_ant or ant2 == args.ref_ant:
                        plot_bls.append( bl )
                        cross.append( i )
                else:
                    plot_bls.append( bl )
                    cross.append( i )
        nBL = len(cross)
        
        # Decimation, if needed
        if args.decimate > 1:
            if nFreq % args.decimate != 0:
                raise RuntimeError(f"Invalid freqeunce decimation factor:  {nFreq} % {args.decimate} = {nFreq%args.decimate}")

            nFreq //= args.decimate
            freq.shape = (freq.size//args.decimate, args.decimate)
            freq = freq.mean(axis=1)
            
            flux.shape = (flux.shape[0], flux.shape[1], flux.shape[2]//args.decimate, args.decimate, flux.shape[3])
            flux = flux.mean(axis=3)
            
            mask.shape = (mask.shape[0], mask.shape[1], mask.shape[2]//args.decimate, args.decimate, mask.shape[3])
            mask = mask.mean(axis=3)
            
        good = np.arange(freq.size//8, freq.size*7//8)		# Inner 75% of the band

        if first:
            ref_time = obsdates[0] + obstimes[0]
            
        # NOTE: Assumes that the Stokes parameters increment by -1
        namMapper = {}
        for i in range(nStk):
            stk = stk0 - i
            namMapper[i] = NUMERIC_STOKES[stk]
        polMapper = {'XX':0, 'YY':1, 'XY':2, 'YX':3}
        
        for b in range(len(plot_bls)):
            bl = plot_bls[b]
            valid = np.where( bls == bl )[0]
            i,j = (bl>>8)&0xFF, bl&0xFF
            ni,nj = antLookup_inv[i], antLookup_inv[j]
            dTimes = obsdates[valid] + obstimes[valid]
            dTimes -= ref_time      # pylint: disable=possibly-used-before-assignment,used-before-assignment
            dTimes *= 86400.0
            
            for p in plot_pols:
                blName = '%s-%s - %s' % (ni, nj, namMapper[polMapper[p]])
                
                if first or blName not in figs:
                    fig = plt.figure()
                    fig.suptitle(blName)
                    fig.subplots_adjust(hspace=0.001)
                    axA = fig.add_subplot(1, 2, 1)
                    axP = fig.add_subplot(1, 2, 2)
                    figs[blName] = (fig, axA, axP)
                fig, axA, axP = figs[blName]
                
                for band,offset in enumerate(fqoffsets):
                    frq = freq + offset
                    vis = np.ma.array(flux[valid,band,:,polMapper[p]], mask=mask[valid,band,:,polMapper[p]])
                    
                    amp = np.ma.abs(vis)
                    vmin, vmax = percentile(amp, 1), percentile(amp, 99)
                    axA.imshow(amp, extent=(frq[0]/1e6, frq[-1]/1e6, dTimes[0], dTimes[-1]), origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
                    
                    axP.imshow(np.ma.angle(vis), extent=(frq[0]/1e6, frq[-1]/1e6, dTimes[0], dTimes[-1]), origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='nearest')
                    
        first = False

    for blName in figs:
        fig, axA, axP = figs[blName]
        
        fig.suptitle("%s UTC\n%s" % (datetime.utcfromtimestamp(times[0]).strftime("%Y/%m/%d %H:%M"), blName))
        
        axA.axis('auto')
        axA.set_title('Amp.')
        axA.set_xlabel('Frequency [MHz]')
        axA.set_ylabel('Amp. - Elapsed Time [s]')
        
        axP.axis('auto')
        axP.set_title('Phase')
        axP.set_xlabel('Frequency [MHz]')
        
        if args.save_images:
            fig.savefig('fringes-%s.png' % (blName.replace(' ', ''),))
            
    if not args.save_images:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='given a FITS-IDI file, create plots of the visibilities', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, nargs='+',
                        help='filename to process')
    parser.add_argument('-r', '--ref-ant', type=str, 
                        help='limit plots to baselines containing the reference antenna')
    parser.add_argument('-b', '--baseline', type=str, 
                        help="limit plots to the specified baseline in 'ANT-ANT' format")
    parser.add_argument('-o', '--drop', action='store_true', 
                        help='drop FLAG table when displaying')
    parser.add_argument('-a', '--include-auto', action='store_true', 
                         help='display the auto-correlations along with the cross-correlations')
    parser.add_argument('-x', '--xx', action='store_true', 
                        help='plot XX or RR data')
    parser.add_argument('-z', '--xy', action='store_true', 
                        help='plot XY or RL data')
    parser.add_argument('-w', '--yx', action='store_true', 
                        help='plot YX or LR data')
    parser.add_argument('-y', '--yy', action='store_true', 
                        help='plot YY or LL data')
    parser.add_argument('-d', '--decimate', type=int, default=1, 
                        help='frequency decimation factor')
    parser.add_argument('-s', '--save-images', action='store_true',
                        help='save the output images as PNGs rather than displaying them')
    args = parser.parse_args()
    try:
        args.ref_ant = int(args.ref_ant, 10)
    except (TypeError, ValueError):
        pass
    if not args.xx and not args.xy and not args.yx and not args.yy:
        raise RuntimeError("Must specify at least one polarization to plot")
    main(args)
