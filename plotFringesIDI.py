#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A FITS-IDI compatible version of plotFringes2.py.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import sys
import numpy
import pyfits
import argparse
from datetime import datetime

from scipy.stats import scoreatpercentile as percentile

from lsl.astro import utcjd_to_unix

from matplotlib import pyplot as plt


def main(args):
    # Parse the command line
    # Parse the command line
    ## Baseline list
    if args.baseline is not None:
        args.baseline = [(int(v0,10),int(v1,10)) for v0,v1 in [v.split('-') for v in args.baseline.split(',')]]
        ## Fill the baseline list with the conjugates, if needed
        newBaselines = []
        for pair in args.baseline:
            newBaselines.append( (pair[1],pair[0]) )
        args.baseline.extend(newBaselines)
    ## Polarization
    if args.xx:
        args.polToPlot = 'XX'
    elif args.xy:
        args.polToPlot = 'XY'
    elif args.yx:
        args.polToPlot = 'YX'
    elif args.yy:
        args.polToPlot = 'YY'
    filename = args.filename
    
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
    
    # Convert times to real times
    times = utcjd_to_unix(obsdates + obstimes)
    times = numpy.unique(times)
    
    # Build a mask
    mask = numpy.zeros(flux.shape, dtype=numpy.bool)
    if fgdata is not None and not args.drop:
        reltimes = obsdates - obsdates[0] + obstimes
        maxtimes = reltimes + inttimes / 2.0 / 86400.0
        mintimes = reltimes - inttimes / 2.0 / 86400.0
        
        for row in fgdata.data:
            ant1, ant2 = row['ANTS']
            if ant1 == ant2 and ant1 != 0:
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
            if ant1 == 0 and ant2 == 0:
                btmask = numpy.where( ( (maxtimes >= tStart) & (mintimes <= tStop) ) )[0]
            elif ant1 == 0 or ant2 == 0:
                ant1 = max([ant1, ant2])
                btmask = numpy.where( ( ((bls>>8)&0xFF == ant1) | (bls&0xFF == ant1) ) \
                                      & ( (maxtimes >= tStart) & (mintimes <= tStop) ) )[0]
            else:
                btmask = numpy.where( ( ((bls>>8)&0xFF == ant1) & (bls&0xFF == ant2) ) \
                                      & ( (maxtimes >= tStart) & (mintimes <= tStop) ) )[0]
            for b,v in enumerate(band):
                for p,w in enumerate(row['PFLAGS']):
                    mask[btmask,b,cStart-1:cStop,p] |= bool(v*w)
                    
    plot_bls = []
    cross = []
    for i in xrange(len(ubls)):
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
            raise RuntimeError("Invalid freqeunce decimation factor:  %i %% %i = %i" % (nFreq, args.decimate, nFreq%args.decimate))

        nFreq /= args.decimate
        freq.shape = (freq.size/args.decimate, args.decimate)
        freq = freq.mean(axis=1)
        
        flux.shape = (flux.shape[0], flux.shape[1], flux.shape[2]/args.decimate, args.decimate, flux.shape[3])
        flux = flux.mean(axis=3)
        
        mask.shape = (mask.shape[0], mask.shape[1], mask.shape[2]/args.decimate, args.decimate, mask.shape[3])
        mask = mask.mean(axis=3)
        
    good = numpy.arange(freq.size/8, freq.size*7/8)		# Inner 75% of the band
    
    # NOTE: Assumed linear data
    polMapper = {'XX':0, 'YY':1, 'XY':2, 'YX':3}
    
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()
    
    k = 0
    nRow = int(numpy.sqrt( len(plot_bls) ))
    nCol = int(numpy.ceil(len(plot_bls)*1.0/nRow))
    for b in xrange(len(plot_bls)):
        bl = plot_bls[b]
        valid = numpy.where( bls == bl )[0]
        i,j = (bl>>8)&0xFF, bl&0xFF
        dTimes = obsdates[valid] + obstimes[valid]
        dTimes -= dTimes[0]
        dTimes *= 86400.0
        
        ax1, ax2, ax3, ax4 = None, None, None, None
        for band,offset in enumerate(fqoffsets):
            frq = freq + offset
            vis = numpy.ma.array(flux[valid,band,:,polMapper[args.polToPlot]], mask=mask[valid,band,:,polMapper[args.polToPlot]])
            
            ax1 = fig1.add_subplot(nRow, nCol*nBand, nBand*k+1+band, sharey=ax1)
            ax1.imshow(numpy.ma.angle(vis), extent=(frq[0]/1e6, frq[-1]/1e6, dTimes[0], dTimes[-1]), origin='lower', vmin=-numpy.pi, vmax=numpy.pi, interpolation='nearest')
            ax1.axis('auto')
            ax1.set_xlabel('Frequency [MHz]')
            if band == 0:
                ax1.set_ylabel('Elapsed Time [s]')
            ax1.set_title("%i,%i - %s" % (i,j,args.polToPlot))
            ax1.set_xlim((frq[0]/1e6, frq[-1]/1e6))
            ax1.set_ylim((dTimes[0], dTimes[-1]))
            
            ax2 = fig2.add_subplot(nRow, nCol*nBand, nBand*k+1+band, sharey=ax2)
            amp = numpy.ma.abs(vis)
            vmin, vmax = percentile(amp, 1), percentile(amp, 99)
            ax2.imshow(amp, extent=(frq[0]/1e6, frq[-1]/1e6, dTimes[0], dTimes[-1]), origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
            ax2.axis('auto')
            ax2.set_xlabel('Frequency [MHz]')
            if band == 0:
                ax2.set_ylabel('Elapsed Time [s]')
            ax2.set_title("%i,%i - %s" % (i,j,args.polToPlot))
            ax2.set_xlim((frq[0]/1e6, frq[-1]/1e6))
            ax2.set_ylim((dTimes[0], dTimes[-1]))
                    
            ax3 = fig3.add_subplot(nRow, nCol*nBand, nBand*k+1+band, sharey=ax3)
            ax3.plot(frq/1e6, numpy.ma.abs(vis.mean(axis=0)))
            ax3.set_xlabel('Frequency [MHz]')
            if band == 0:
                ax3.set_ylabel('Mean Vis. Amp. [lin.]')
            ax3.set_title("%i,%i - %s" % (i,j,args.polToPlot))
            ax3.set_xlim((frq[0]/1e6, frq[-1]/1e6))
            
            ax4 = fig4.add_subplot(nRow, nCol*nBand, nBand*k+1+band, sharey=ax4)
            ax4.plot(dTimes, numpy.ma.angle(vis[:,good].mean(axis=1))*180/numpy.pi, linestyle='', marker='+')
            ax4.set_ylim((-180, 180))
            ax4.set_xlabel('Elapsed Time [s]')
            if band == 0:
                ax4.set_ylabel('Mean Vis. Phase [deg]')
            ax4.set_title("%i,%i - %s" % (i,j,args.polToPlot))
            ax4.set_xlim((dTimes[0], dTimes[-1]))
            
            if band > 0:
                for ax in (ax1, ax2, ax3, ax4):
                    plt.setp(ax.get_yticklabels(), visible=False)
            if band < nBand-1:
                for ax in (ax1, ax2, ax3, ax4):
                    xticks = ax.xaxis.get_major_ticks()
                    xticks[-1].label1.set_visible(False)
                    
        k += 1
        
    for f in (fig1, fig2, fig3, fig4):
        f.suptitle("%s to %s UTC" % (datetime.utcfromtimestamp(times[0]).strftime("%Y/%m/%d %H:%M"), datetime.utcfromtimestamp(times[-1]).strftime("%Y/%m/%d %H:%M")))
        if nBand > 1:
            f.subplots_adjust(wspace=0.0)
            
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='given a FITS-IDI file, create plots of the visibilities', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, 
                        help='filename to process')
    parser.add_argument('-r', '--ref-ant', type=int, 
                        help='limit plots to baselines containing the reference antenna')
    parser.add_argument('-b', '--baseline', type=str, 
                        help="limit plots to the specified baseline in 'ANT-ANT' format")
    parser.add_argument('-o', '--drop', action='store_true', 
                        help='drop FLAG table when displaying')
    pgroup = parser.add_mutually_exclusive_group(required=False)
    pgroup.add_argument('-x', '--xx', action='store_true', default=True, 
                        help='plot XX data')
    pgroup.add_argument('-z', '--xy', action='store_true', 
                        help='plot XY data')
    pgroup.add_argument('-w', '--yx', action='store_true', 
                        help='plot YX data')
    pgroup.add_argument('-y', '--yy', action='store_true', 
                        help='plot YY data')
    parser.add_argument('-d', '--decimate', type=int, default=1, 
                        help='frequency decimation factor')
    args = parser.parse_args()
    main(args)
    
