#!/usr/bin/env python3

"""
Given a collection of .npz files search for course delays and rates.
"""

import os
import sys
import glob
import numpy as np
from astropy.io import fits as astrofits
import argparse
from datetime import datetime

from lsl.statistics import robust
from lsl.misc.mathutils import to_dB
from lsl.astro import utcjd_to_unix
from lsl.writer.fitsidi import NUMERIC_STOKES

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle as Box


def main(args):
    # Parse the command line
    ## Search limits
    args.delay_window = [float(v) for v in args.delay_window.split(',', 1)]
    args.rate_window = [float(v) for v in args.rate_window.split(',', 1)]
    
    figs = {}
    first = True
    for filename in args.filename:
        print(f"Working on '{os.path.basename(filename)}")
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
        freq = []
        for fqoff in fqoffsets:
            freq.append((np.arange(nFreq)-(uvdata.header['CRPIX3']-1))*uvdata.header['CDELT3'])
            freq[-1] += uvdata.header['CRVAL3'] + fqoff
        freq = np.concatenate(freq)
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
        flux.shape = (flux.shape[0], nBand*nFreq, nStk)
        
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
                    pair[0] = antLookup[pair[0]]
                try:
                    pair[1] = int(pair[1], 10)
                except ValueError:
                    pair[1] = antLookup[pair[1]]
                    
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
                if ant1 != ant2 or ant1 == 0 or ant2 == 0:
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
            if ant1 != ant2:
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
        
        iSize = int(round(args.interval/robust.mean(inttimes)))
        iCount = times.size//iSize
        if iCount == 0:
            args.interval = times.size*robust.mean(inttimes)
            iSize = int(round(args.interval/robust.mean(inttimes)))
            iCount = times.size//iSize
            print("WARNING:  Not enough data for requested search interval, changing to {args.interval:.3f} seconds")
        print(f" -> Chunk size is {iSize} intervals ({iSize*robust.mean(inttimes):.3f} seconds)")
        print(f" -> Working with {iCount} chunks of data")
        
        print(f"Number of frequency channels: {len(freq)} (~{freq[1]-freq[0]:.1f} Hz/channel)")

        dTimes = times - times[0]
        if first:
            ref_time = (int(times[0]) / 60) * 60
            
        dMax = 1.0/(freq[1]-freq[0])/4
        dMax = int(dMax*1e6)*1e-6
        if -dMax*1e6 > args.delay_window[0]:
            args.delay_window[0] = -dMax*1e6
        if dMax*1e6 < args.delay_window[1]:
            args.delay_window[1] = dMax*1e6
        rMax = 1.0/robust.mean(inttimes)/4
        rMax = int(rMax*1e2)*1e-2
        if -rMax*1e3 > args.rate_window[0]:
            args.rate_window[0] = -rMax*1e3
        if rMax*1e3 < args.rate_window[1]:
            args.rate_window[1] = rMax*1e3
            
        dres = 0.01
        nDelays = int((args.delay_window[1]-args.delay_window[0])/dres)
        while nDelays < 50:
            dres /= 10
            nDelays = int((args.delay_window[1]-args.delay_window[0])/dres)
        while nDelays > 15000:
            dres *= 10
            nDelays = int((args.delay_window[1]-args.delay_window[0])/dres)
        nDelays += (nDelays + 1) % 2
        
        rres = 10.0
        nRates = int((args.rate_window[1]-args.rate_window[0])/rres)
        while nRates < 50:
            rres /= 10
            nRates = int((args.rate_window[1]-args.rate_window[0])/rres)
        while nRates > 15000:
            rres *= 10
            nRates = int((args.rate_window[1]-args.rate_window[0])/rres)
        nRates += (nRates + 1) % 2
        
        print(f"Searching delays {args.delay_window[0]:.1f} to {args.delay_window[1]:.1f} us in steps of {dres:.2f} us")
        print(f"           rates {args.rate_window[0]:.1f} to {args.rate_window[1]:.1f} mHz in steps of {rres:.2f} mHz")
        print(" ")
        
        delay = np.linspace(args.delay_window[0]*1e-6, args.delay_window[1]*1e-6, nDelays)		# s
        drate = np.linspace(args.rate_window[0]*1e-3,  args.rate_window[1]*1e-3,  nRates )		# Hz
        
        # Find RFI and trim it out.  This is done by computing average visibility 
        # amplitudes (a "spectrum") and running a median filter in frequency to extract
        # the bandpass.  After the spectrum has been bandpassed, 3sigma features are 
        # trimmed.  Additionally, area where the bandpass fall below 10% of its mean
        # value are also masked.
        spec  = np.median(np.abs(flux[:,:,0]), axis=0)
        spec += np.median(np.abs(flux[:,:,1]), axis=0)
        smth = spec*0.0
        winSize = int(250e3/(freq[1]-freq[0]))
        winSize += ((winSize+1)%2)
        for i in range(smth.size):
            mn = max([0, i-winSize//2])
            mx = min([i+winSize, smth.size])
            smth[i] = np.median(spec[mn:mx])
        smth /= robust.mean(smth)
        bp = spec / smth
        good = np.where( (smth > 0.1) & (np.abs(bp-robust.mean(bp)) < 3*robust.std(bp)) )[0]
        nBad = nBand*nFreq - len(good)
        print(f"Masking {nBad} of {nBand*nFreq} channels ({100.0*nBad/nBand/nFreq:.1f}%)")
        
        freq2 = freq*1.0
        freq2.shape += (1,)
        
        dirName = os.path.basename( filename )
        for b in range(len(plot_bls)):
            bl = plot_bls[b]
            valid = np.where( bls == bl )[0]
            i,j = (bl>>8)&0xFF, bl&0xFF
            dTimes = obsdates[valid] + obstimes[valid]
            dTimes = np.array([utcjd_to_unix(v) for v in dTimes])
            
            ## Skip over baselines that are not in the baseline list (if provided)
            if args.baseline is not None:
                if (i,j) not in args.baseline:
                    continue
            ## Skip over baselines that don't include the reference antenna
            elif i != args.ref_ant and j != args.ref_ant:
                continue
                
            ## Check and see if we need to conjugate the visibility, i.e., switch from
            ## baseline (*,ref) to baseline (ref,*)
            doConj = False
            if j == args.ref_ant:
                doConj = True
                
            ## Figure out which polarizations to process
            if args.cross_hands:
                polToUse = ('XX', 'XY', 'YY')
                visToUse = (0, 2, 1)
            else:
                polToUse = ('XX', 'YY')
                visToUse = (0, 1)
                
            blName = (i, j)
            if doConj:
                blName = (j, i)
            blName = '%s-%s' % (antLookup_inv[blName[0]], antLookup_inv[blName[1]])
            
            if first or blName not in figs:
                fig = plt.figure()
                fig.suptitle('%s' % blName)
                fig.subplots_adjust(hspace=0.001)
                axR = fig.add_subplot(2, 1, 1)
                axD = fig.add_subplot(2, 1, 2, sharex=axR)
                figs[blName] = (fig, axR, axD)
            fig, axR, axD = figs[blName]
            
            markers = {'XX':'s', 'YY':'o', 'XY':'v', 'YX':'^'}
            
            for pol,vis in zip(polToUse, visToUse):
                for i in range(iCount):
                    try:
                        subStart, subStop = dTimes[iSize*i], dTimes[iSize*(i+1)-1]
                    except IndexError:
                        continue
                    if (subStop - subStart) > 1.1*args.interval:
                        continue
                        
                    subTime = np.array([dTimes[iSize*i:iSize*(i+1)].mean(),])
                    dTimes2 = dTimes[iSize*i:iSize*(i+1)]*1.0
                    dTimes2 -= dTimes2[0]
                    dTimes2.shape += (1,)
                    subData = flux[valid,...][iSize*i:iSize*(i+1),good,vis]*1.0
                    subPhase = flux[valid,...][iSize*i:iSize*(i+1),good,vis]*1.0
                    if doConj:
                        subData = subData.conj()
                        subPhase = subPhase.conj()
                    subData = np.dot(subData, np.exp(-2j*np.pi*freq2[good,:]*delay))
                    subData /= freq2[good,:].size
                    amp = np.dot(subData.T, np.exp(-2j*np.pi*dTimes2[:subData.shape[0],:]*drate))
                    amp = np.abs(amp / dTimes2.size)
                    
                    subPhase = np.angle(subPhase.mean()) * 180/np.pi
                    subPhase %= 360
                    if subPhase > 180:
                        subPhase -= 360
                        
                    best = np.where( amp == amp.max() )
                    if amp.max() > 0:
                        bsnr = (amp[best]-amp.mean())/amp.std()
                        bdly = delay[best[0]]*1e6
                        brat = drate[best[1]]*1e3
                        
                        c = axR.scatter(subTime-ref_time, brat, c=bsnr, marker=markers[pol],        # pylint: disable=possibly-used-before-assignment,used-before-assignment
                                        cmap='gist_yarg', norm=None, vmin=3, vmax=40)
                        c = axD.scatter(subTime-ref_time, bdly, c=bsnr, marker=markers[pol],        # pylint: disable=possibly-used-before-assignment,used-before-assignment
                                        cmap='gist_yarg', vmin=3, vmax=40)
                        
        first = False
        
    for blName in figs:
        fig, axR, axD = figs[blName]
        
        # Colorbar
        cb = fig.colorbar(c, ax=axR, orientation='horizontal')      # pylint: disable=possibly-used-before-assignment,used-before-assignment
        cb.set_label('SNR')
        # Legend and reference marks
        handles = []
        for pol in polToUse:
            handles.append(Line2D([0,], [0,], linestyle='', marker=markers[pol], color='k', label=pol))
        axR.legend(handles=handles, loc=0)
        oldLim = axR.get_xlim()
        for ax in (axR, axD):
            ax.hlines(0, oldLim[0], oldLim[1], linestyle=':', alpha=0.5)
        axR.set_xlim(oldLim)
        # Set the labels
        axR.set_ylabel('Rate [mHz]')
        axD.set_ylabel('Delay [$\\mu$s]')
        for ax in (axR, axD):
            ax.set_xlabel('Elapsed Time [s since %s]' % datetime.utcfromtimestamp(ref_time).strftime('%Y%b%d %H:%M'))
        # Set the y ranges
        axR.set_ylim((-max([100, max([abs(v) for v in axR.get_ylim()])]), max([100, max([abs(v) for v in axR.get_ylim()])])))
        axD.set_ylim((-max([0.5, max([abs(v) for v in axD.get_ylim()])]), max([0.5, max([abs(v) for v in axD.get_ylim()])])))
        # No-go regions for the delays
        xlim, ylim = axD.get_xlim(), axD.get_ylim()
        axD.add_patch(Box(xy=(xlim[0],ylim[0]), width=xlim[1]-xlim[0], height=-0.5001-ylim[0],
                          fill=True, color='red', alpha=0.2))
        axD.add_patch(Box(xy=(xlim[0],0.5001), width=xlim[1]-xlim[0], height=ylim[1]-0.5001,
                          fill=True, color='red', alpha=0.2))
        axD.set_xlim(xlim)
        axD.set_ylim(ylim)
        
        fig.tight_layout()
        plt.draw()
        
        if args.save_images:
            fig.savefig('sniffer-%s.png' % blName)
            
    if not args.save_images:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='given a collection of .npz files generated by "the next generation of correlator", search for fringes', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('filename', type=str, nargs='+',
                        help='filename to search')
    parser.add_argument('-r', '--ref-ant', type=str, 
                        help='limit plots to baselines containing the reference antenna')
    parser.add_argument('-b', '--baseline', type=str, 
                        help="limit plots to the specified baseline in 'ANT-ANT' format")
    parser.add_argument('-o', '--drop', action='store_true', 
                        help='drop FLAG table when displaying')
    parser.add_argument('-d', '--decimate', type=int, default=1, 
                        help='frequency decimation factor')
    parser.add_argument('-l', '--limit', type=int, default=-1, 
                        help='limit the data loaded to the first N files, -1 = load all')
    parser.add_argument('-c', '--cross-hands', action='store_true', 
                        help='include XY/YX in the plots')
    parser.add_argument('-e', '--delay-window', type=str, default='-inf,inf', 
                        help='delay search window in us; defaults to maximum allowed')
    parser.add_argument('-a', '--rate-window', type=str, default='-inf,inf', 
                        help='rate search window in mHz; defaults to maximum allowed')
    parser.add_argument('-i', '--interval', type=float, default=30.0, 
                        help='fringe search interveral in seconds')
    parser.add_argument('-s', '--save-images', action='store_true',
                        help='save the output images as PNGs rather than displaying them')
    args = parser.parse_args()
    try:
        args.ref_ant = int(args.ref_ant, 10)
    except (TypeError, ValueError):
        pass
    main(args)
