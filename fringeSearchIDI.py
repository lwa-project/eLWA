#!/usr/bin/env python

"""
A FITS-IDI compatible version of fringeSearch.py to finding course delays and 
rates.

NOTE:  This script does not try to fringe search only a single source.  Rather,
    it searches the file as a whole.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import os
import sys
import numpy
from astropy.io import fits as astrofits
import argparse
from datetime import datetime

from lsl.astro import utcjd_to_unix
from lsl.statistics import robust
from lsl.misc.mathutils import to_dB
from lsl.misc import parser as aph

from matplotlib import pyplot as plt


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


def main(args):
    # Parse the command line
    ## Baseline list
    if args.baseline is not None:
        ## Fill the baseline list with the conjugates, if needed
        newBaselines = []
        for pair in args.baseline:
            newBaselines.append( (pair[1],pair[0]) )
        args.baseline.extend(newBaselines)
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
    
    # Find unique baselines, times, and sources to work with
    ubls = numpy.unique(bls)
    utimes = numpy.unique(obstimes)
    usrc = numpy.unique(srcs)
    
    # Convert times to real times
    times = utcjd_to_unix(obsdates + obstimes)
    times = numpy.unique(times)
    
    # Find unique scans to work on, making sure that there are no large gaps
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
    
    # Create a mask of the old flags, and come up with a list of bad channels
    if args.ignore_flags:
        mask = numpy.zeros(nFreq)
    else:
        mask = get_flags_as_mask(hdulist)
        mask = numpy.mean(mask, axis=0)
        mask = numpy.max(mask[0,:,[0,1]], axis=0)
        
    # Make sure the reference antenna is in there
    if args.ref_ant is None:
        bl = ubls[0]
        ant1, ant2 = (bl>>8)&0xFF, bl&0xFF 
        args.ref_ant = ant1
    else:
        found = False
        for bl in ubls:
            ant1, ant2 = (bl>>8)&0xFF, bl&0xFF
            if ant1 == args.ref_ant:
                found = True
                break
        if not found:
            raise RuntimeError("Cannot file reference antenna %i in the data" % args.ref_ant)
            
    search_bls = []
    cross = []
    for i in xrange(len(ubls)):
        bl = ubls[i]
        ant1, ant2 = (bl>>8)&0xFF, bl&0xFF 
        if ant1 != ant2:
            search_bls.append( bl )
            cross.append( i )
    nBL = len(cross)
    
    iTimes = numpy.zeros(times.size-1, dtype=times.dtype)
    for i in xrange(1, len(times)):
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
        
    dres = 1.0
    nDelays = int((args.delay_window[1]-args.delay_window[0])/dres)
    while nDelays < 50:
        dres /= 10
        nDelays = int((args.delay_window[1]-args.delay_window[0])/dres)
    while nDelays > 5000:
        dres *= 10
        nDelays = int((args.delay_window[1]-args.delay_window[0])/dres)
    nDelays += (nDelays + 1) % 2
    
    rres = 10.0
    nRates = int((args.rate_window[1]-args.rate_window[0])/rres)
    while nRates < 50:
        rres /= 10
        nRates = int((args.rate_window[1]-args.rate_window[0])/rres)
    while nRates > 5000:
        rres *= 10
        nRates = int((args.rate_window[1]-args.rate_window[0])/rres)
    nRates += (nRates + 1) % 2
    
    print("Searching delays %.1f to %.1f us in steps of %.2f us" % (args.delay_window[0], args.delay_window[1], dres))
    print("           rates %.1f to %.1f mHz in steps of %.2f mHz" % (args.rate_window[0], args.rate_window[1], rres))
    print(" ")
    
    delay = numpy.linspace(args.delay_window[0]*1e-6, args.delay_window[1]*1e-6, nDelays)		# s
    drate = numpy.linspace(args.rate_window[0]*1e-3,  args.rate_window[1]*1e-3,  nRates )		# Hz
    
    # Find RFI and trim it out.  This is done by computing average visibility 
    # amplitudes (a "spectrum") and running a median filter in frequency to extract
    # the bandpass.  After the spectrum has been bandpassed, 3sigma features are 
    # trimmed.  Additionally, area where the bandpass fall below 10% of its mean
    # value are also masked.
    spec  = numpy.median(numpy.abs(flux[:,0,:,0]), axis=0)
    spec += numpy.median(numpy.abs(flux[:,0,:,1]), axis=0)
    smth = spec*0.0
    winSize = int(250e3/(freq[1]-freq[0]))
    winSize += ((winSize+1)%2)
    for i in xrange(smth.size):
        mn = max([0, i-winSize//2])
        mx = min([i+winSize//2+1, smth.size])
        smth[i] = numpy.median(spec[mn:mx])
    smth /= robust.mean(smth)
    bp = spec / smth
    good = numpy.where( (smth > 0.1) & (numpy.abs(bp-robust.mean(bp)) < 3*robust.std(bp)) & (mask < 0.25) )[0]
    nBad = nFreq - len(good)
    print("Masking %i of %i channels (%.1f%%)" % (nBad, nFreq, 100.0*nBad/nFreq))
    if args.plot:
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(freq/1e6, numpy.log10(spec)*10)
        ax.plot(freq[good]/1e6, numpy.log10(spec[good])*10)
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
    for b in xrange(len(search_bls)):
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
        if ant1 not in (51, 52) and ant2 not in (51, 52):
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
            
        valid = numpy.where( bls == bl )[0]
        for pol in polToUse:
            subData = flux[valid,0,:,polMapper[pol]]*1.0
            subData = subData[:,good]
            if doConj:
                subData = subData.conj()
            subData = numpy.dot(subData, numpy.exp(-2j*numpy.pi*freq2[good,:]*delay))
            subData /= freq2[good,:].size
            amp = numpy.dot(subData.T, numpy.exp(-2j*numpy.pi*dTimes2*drate))
            amp = numpy.abs(amp / dTimes2.size)
            
            blName = (ant1, ant2)
            if doConj:
                blName = (ant2, ant1)
            blName = '%s-%s' % ('EA%02i' % blName[0] if blName[0] < 51 else 'LWA%i' % (blName[0]-50), 
                        'EA%02i' % blName[1] if blName[1] < 51 else 'LWA%i' % (blName[1]-50))
                        
            best = numpy.where( amp == amp.max() )
            if amp.max() > 0:
                bsnr = (amp[best]-amp.mean())[0]/amp.std()
                bdly = delay[best[0][0]]*1e6
                brat = drate[best[1][0]]*1e3
                print("%3i  %9s  %2s  %6.2f  %6.2f us  %7.2f mHz" % (b, blName, pol, bsnr, bdly, brat))
            else:
                print("%3i  %9s  %2s  %6s  %9s  %11s" % (b, blName, pol, '----', '----', '----'))
                
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
    parser.add_argument('-r', '--ref-ant', type=int, 
                        help='limit plots to baselines containing the reference antenna')
    parser.add_argument('-b', '--baseline', type=aph.csv_baseline_list, 
                        help="limit plots to the specified baseline in 'ANT-ANT' format")
    parser.add_argument('-l', '--limit', type=int, default=-1, 
                        help='limit the data loaded to the first N files, -1 = load all')
    parser.add_argument('-y', '--y-only', action='store_true', 
                        help='limit the search on VLA-LWA baselines to the VLA Y pol. only')
    parser.add_argument('-e', '--delay-window', type=str, default='-inf,inf', 
                        help='delay search window in us; defaults to maximum allowed')
    parser.add_argument('-a', '--rate-window', type=str, default='-inf,inf', 
                        help='rate search window in mHz; defaults to maximum allowed')
    parser.add_argument('-i', '--ignore-flags', action='store_true',
                        help='do not use the FLAG table(s) when determining bad channels')
    parser.add_argument('-p', '--plot', action='store_true', 
                        help='show search plots at the end')
    args = parser.parse_args()
    main(args)
    
