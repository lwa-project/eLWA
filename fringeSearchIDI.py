#!/usr/bin/env python3

"""
A FITS-IDI compatible version of fringeSearch.py to finding course delays and 
rates.

NOTE:  This script does not try to fringe search only a single source.  Rather,
    it searches the file as a whole.
"""

import os
import sys
import numpy as np
from astropy.io import fits as astrofits
import argparse
from datetime import datetime

from lsl.astro import utcjd_to_unix
from lsl.statistics import robust
from lsl.misc.mathutils import to_dB
from lsl.misc import parser as aph

from matplotlib import pyplot as plt


def main(args):
    # Parse the command line
    ## Search limits
    args.delay_window = [float(v) for v in args.delay_window.split(',', 1)]
    args.rate_window = [float(v) for v in args.rate_window.split(',', 1)]
    
    print("Working on '%s'" % os.path.basename(args.filename))
    # Open the FITS IDI file and access the UV_DATA extension
    hdulist = astrofits.open(args.filename, mode='readonly')
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
    antLookup_inv = {}
    for an, ai in zip(andata.data['ANNAME'], andata.data['ANTENNA_NO']):
        antLookup[an] = ai
        antLookup_inv[ai] = an
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
        bl = ubls[0]
        ant1, ant2 = (bl>>8)&0xFF, bl&0xFF 
        args.ref_ant = ant1
    else:
        found = False
        for bl in ubls:
            ant1, ant2 = (bl>>8)&0xFF, bl&0xFF
            if ant1 == args.ref_ant or ant2 == args.ref_ant:
                found = True
                break
            elif antLookup_inv[ant1] == args.ref_ant:
                args.ref_ant = ant1
                found = True
                break
            elif antLookup_inv[ant2] == args.ref_ant:
                args.ref_ant = ant2
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
    
    # Find unique scans to work on, making sure that there are no large gaps
    blocks = []
    for src in usrc:
        valid = np.where( src == srcs )[0]
        
        blocks.append( [valid[0],valid[0]] )
        for v in valid[1:]:
            if v == blocks[-1][1] + 1 \
                and (obsdates[v] - obsdates[blocks[-1][1]] + obstimes[v] - obstimes[blocks[-1][1]])*86400 < 10*inttimes[v]:
                blocks[-1][1] = v
            else:
                blocks.append( [v,v] )
    blocks.sort()
    
    search_bls = []
    cross = []
    for i in range(len(ubls)):
        bl = ubls[i]
        ant1, ant2 = (bl>>8)&0xFF, bl&0xFF 
        if ant1 != ant2:
            search_bls.append( bl )
            cross.append( i )
    nBL = len(cross)
    
    iTimes = np.zeros(times.size-1, dtype=times.dtype)
    for i in range(1, len(times)):
        iTimes[i-1] = times[i] - times[i-1]
    print(" -> Interval: %.3f +/- %.3f seconds (%.3f to %.3f seconds)" % (iTimes.mean(), iTimes.std(), iTimes.min(), iTimes.max()))
    
    print("Number of frequency channels: %i (~%.1f Hz/channel)" % (len(freq), freq[1]-freq[0]))

    dTimes = times - times[0]
    
    dMax = 1.0/(freq[1]-freq[0])/4
    dMax = int(dMax*1e6)*1e-6
    if -dMax*1e6 > args.delay_window[0]:
        args.delay_window[0] = -dMax*1e6
    if dMax*1e6 < args.delay_window[1]:
        args.delay_window[1] = dMax*1e6
    rMax = 1.0/iTimes.mean()/4
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
    
    print("Searching delays %.1f to %.1f us in steps of %.2f us" % (args.delay_window[0], args.delay_window[1], dres))
    print("           rates %.1f to %.1f mHz in steps of %.2f mHz" % (args.rate_window[0], args.rate_window[1], rres))
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
        mx = min([i+winSize//2+1, smth.size])
        smth[i] = np.median(spec[mn:mx])
    smth /= robust.mean(smth)
    bp = spec / smth
    good = np.where( (smth > 0.1) & (np.abs(bp-robust.mean(bp)) < 3*robust.std(bp)) & np.logical_and(freq<=args.hf, freq>=args.lf) )[0]
    nBad = nBand*nFreq - len(good)
    print("Masking %i of %i channels (%.1f%%)" % (nBad, nBand*nFreq, 100.0*nBad/nBand/nFreq))
    if args.plot:
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(freq/1e6, np.log10(spec)*10)
        ax.plot(freq[good]/1e6, np.log10(spec[good])*10)
        ax.set_title('Mean Visibility Amplitude')
        ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('PSD [arb. dB]')
        plt.draw()
    
    freq2 = freq*1.0
    freq2.shape += (1,)
    dTimes2 = dTimes*1.0
    dTimes2.shape += (1,)
    
    # NOTE: Assumed linear data
    polMapper = {'XX':0, 'YY':1, 'XY':2, 'YX':3}
    
    print("%3s  %9s  %2s  %6s  %9s  %11s" % ('#', 'BL', 'Pl', 'S/N', 'Delay', 'Rate'))
    for b in range(len(search_bls)):
        bl = search_bls[b]
        ant1, ant2 = (bl>>8)&0xFF, bl&0xFF
        
        ## Skip over baselines that are not in the baseline list (if provided)
        if args.baseline is not None:
            if (ant1, ant2) not in args.baseline:
                continue
        ## Skip over baselines that don't include the reference antenna
        elif ant1 != args.ref_ant and ant2 != args.ref_ant:
            continue
            
        ## Check and see if we need to conjugate the visibility, i.e., switch from
        ## baseline (*,ref) to baseline (ref,*)
        doConj = False
        if ant2 == args.ref_ant:
            doConj = True
            
        ## Figure out which polarizations to process
        if antLookup_inv[ant1][:3] != 'LWA' and antLookup_inv[ant2][:3] != 'LWA':
            ### Standard VLA-VLA baseline
            polToUse = ('XX', 'YY')
        else:
            ### LWA-LWA or LWA-VLA baseline
            if args.y_only:
                polToUse = ('YX', 'YY')
            else:
                polToUse = ('XX', 'XY', 'YX', 'YY')
                
        if args.plot:
            fig = plt.figure()
            axs = {}
            axs['XX'] = fig.add_subplot(2, 2, 1)
            axs['YY'] = fig.add_subplot(2, 2, 2)
            axs['XY'] = fig.add_subplot(2, 2, 3)
            axs['YX'] = fig.add_subplot(2, 2, 4)
            
        valid = np.where( bls == bl )[0]
        for pol in polToUse:
            subData = flux[valid,:,polMapper[pol]]*1.0
            subData = subData[:,good]
            if doConj:
                subData = subData.conj()
            subData = np.dot(subData, np.exp(-2j*np.pi*freq2[good,:]*delay))
            subData /= freq2[good,:].size
            amp = np.dot(subData.T, np.exp(-2j*np.pi*dTimes2[:subData.shape[0],:]*drate))
            amp = np.abs(amp / dTimes2.size)
            
            blName = (ant1, ant2)
            if doConj:
                blName = (ant2, ant1)
            blName = '%s-%s' % (antLookup_inv[blName[0]], antLookup_inv[blName[1]])
                        
            best = np.where( amp == amp.max() )
            if amp.max() > 0:
                bsnr = (amp[best]-amp.mean())[0]/amp.std()
                bdly = delay[best[0][0]]*1e6
                brat = drate[best[1][0]]*1e3
                print("%3i  %11s  %2s  %6.2f  %6.2f us  %7.2f mHz" % (b, blName, pol, bsnr, bdly, brat))
            else:
                print("%3i  %11s  %2s  %6s  %9s  %11s" % (b, blName, pol, '----', '----', '----'))
                
            if args.plot:
                axs[pol].imshow(amp, origin='lower', interpolation='nearest', 
                            extent=(drate[0]*1e3, drate[-1]*1e3, delay[0]*1e6, delay[-1]*1e6), 
                            cmap='gray_r')
                axs[pol].plot(drate[best[1][0]]*1e3, delay[best[0][0]]*1e6, linestyle='', marker='x', color='r', ms=15, alpha=0.75)
                
        if args.plot:
            fig.suptitle(os.path.basename(args.filename))
            for pol in axs.keys():
                ax = axs[pol]
                ax.axis('auto')
                ax.set_title(pol)
                ax.set_xlabel('Rate [mHz]')
                ax.set_ylabel('Delay [$\\mu$s]')
            fig.suptitle("%s" % blName)
            plt.draw()
            
    if args.plot:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='given a FITS-IDI file, search for fringes', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('filename', type=str, 
                        help='filename to search')
    parser.add_argument('-r', '--ref-ant', type=str, 
                        help='limit plots to baselines containing the reference antenna')
    parser.add_argument('-b', '--baseline', type=str, 
                        help="limit plots to the specified baseline in 'ANT-ANT' format")
    parser.add_argument('-l', '--limit', type=int, default=-1, 
                        help='limit the data loaded to the first N files, -1 = load all')
    parser.add_argument('-y', '--y-only', action='store_true', 
                        help='limit the search on VLA-LWA baselines to the VLA Y pol. only')
    parser.add_argument('-e', '--delay-window', type=str, default='-inf,inf', 
                        help='delay search window in us; defaults to maximum allowed')
    parser.add_argument('-a', '--rate-window', type=str, default='-inf,inf', 
                        help='rate search window in mHz; defaults to maximum allowed')
    parser.add_argument('-p', '--plot', action='store_true', 
                        help='show search plots at the end')
    parser.add_argument('--hf', type=aph.frequency, default='98.0',
                        help='High frequency (in MHz) cutoff to use in fringe searching correlated data. Note: May be useful when high frequency RFI is present')
    parser.add_argument('--lf', type=aph.frequency, default='0.0',
                        help='Low frequency (in MHz) cutoff to use in fringe searching correlated data. Note: May be useful when low frequency RFI is present')
    args = parser.parse_args()
    try:
        args.ref_ant = int(args.ref_ant, 10)
    except (TypeError, ValueError):
        pass
    main(args)
    
