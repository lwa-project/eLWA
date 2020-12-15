#!/usr/bin/env python

"""
Given a collection of .npz files search for course delays and rates.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import os
import sys
import glob
import numpy
from astropy.io import fits as astrofits
import argparse
from datetime import datetime

from lsl.statistics import robust
from lsl.misc.mathutils import to_dB
from lsl.astro import utcjd_to_unix
from lsl.writer.fitsidi import NUMERIC_STOKES
from lsl.misc import parser as aph

from matplotlib import pyplot as plt


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
    
    figs = {}
    first = True
    for filename in args.filename:
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
        
        # Pull out various bits of information we need to flag the file
        ## Antenna look-up table
        antLookup = {}
        for an, ai in zip(andata.data['ANNAME'], andata.data['ANTENNA_NO']):
            antLookup[an] = ai
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
        
        # Build a mask
        mask = numpy.zeros(flux.shape, dtype=numpy.bool)
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
                    
        # Make sure the reference antenna is in there
        if first:
            if args.ref_ant is None:
                bl = bls[0]
                i,j = (bl>>8)&0xFF, bl&0xFF
                args.ref_ant = i
            else:
                found = False
                for bl in bls:
                    i,j = (bl>>8)&0xFF, bl&0xFF
                    if i == args.ref_ant:
                        found = True
                        break
                if not found:
                    raise RuntimeError("Cannot file reference antenna %i in the data" % args.ref_ant)
                    
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

            nFreq //= args.decimate
            freq.shape = (freq.size//args.decimate, args.decimate)
            freq = freq.mean(axis=1)
            
            flux.shape = (flux.shape[0], flux.shape[1], flux.shape[2]//args.decimate, args.decimate, flux.shape[3])
            flux = flux.mean(axis=3)
            
            mask.shape = (mask.shape[0], mask.shape[1], mask.shape[2]//args.decimate, args.decimate, mask.shape[3])
            mask = mask.mean(axis=3)
            
        good = numpy.arange(freq.size//8, freq.size*7//8)		# Inner 75% of the band
        
        iSize = int(round(args.interval/robust.mean(inttimes)))
        print(" -> Chunk size is %i intervals (%.3f seconds)" % (iSize, iSize*robust.mean(inttimes)))
        iCount = times.size//iSize
        print(" -> Working with %i chunks of data" % iCount)
        
        print("Number of frequency channels: %i (~%.1f Hz/channel)" % (len(freq), freq[1]-freq[0]))

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
            mx = min([i+winSize, smth.size])
            smth[i] = numpy.median(spec[mn:mx])
        smth /= robust.mean(smth)
        bp = spec / smth
        good = numpy.where( (smth > 0.1) & (numpy.abs(bp-robust.mean(bp)) < 3*robust.std(bp)) )[0]
        nBad = nFreq - len(good)
        print("Masking %i of %i channels (%.1f%%)" % (nBad, nFreq, 100.0*nBad/nFreq))
        
        freq2 = freq*1.0
        freq2.shape += (1,)
        
        dirName = os.path.basename( filename )
        for b in xrange(len(plot_bls)):
            bl = plot_bls[b]
            valid = numpy.where( bls == bl )[0]
            i,j = (bl>>8)&0xFF, bl&0xFF
            dTimes = obsdates[valid] + obstimes[valid]
            dTimes = numpy.array([utcjd_to_unix(v) for v in dTimes])
            
            ## Skip over baselines that are not in the baseline list (if provided)
            if args.baseline is not None:
                if bl not in args.baseline:
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
            if i not in (51, 52) and j not in (51, 52):
                ### Standard VLA-VLA baseline
                polToUse = ('XX', 'XY', 'YX', 'YY')
                visToUse = (0, 2, 3, 1)
            else:
                ### LWA-LWA or LWA-VLA baseline
                if args.y_only:
                    polToUse = ('YX', 'YY')
                    visToUse = (3, 1)
                else:
                    polToUse = ('XX', 'XY', 'YX', 'YY')
                    visToUse = (0, 2, 3, 1)
                    
            blName = (i, j)
            if doConj:
                blName = (j, i)
            blName = '%s-%s' % ('EA%02i' % blName[0] if blName[0] < 51 else 'LWA%i' % (blName[0]-50), 
                                'EA%02i' % blName[1] if blName[1] < 51 else 'LWA%i' % (blName[1]-50))
                            
            if first or blName not in figs:
                fig = plt.figure()
                fig.suptitle('%s' % blName)
                fig.subplots_adjust(hspace=0.001)
                axR = fig.add_subplot(4, 1, 1)
                axD = fig.add_subplot(4, 1, 2, sharex=axR)
                axP = fig.add_subplot(4, 1, 3, sharex=axR)
                axA = fig.add_subplot(4, 1, 4, sharex=axR)
                figs[blName] = (fig, axR, axD, axP, axA)
            fig, axR, axD, axP, axA = figs[blName]
            
            markers = {'XX':'s', 'YY':'o', 'XY':'v', 'YX':'^'}
            
            for pol,vis in zip(polToUse, visToUse):
                for i in xrange(iCount):
                    subStart, subStop = dTimes[iSize*i], dTimes[iSize*(i+1)-1]
                    if (subStop - subStart) > 1.1*args.interval:
                        continue
                        
                    subTime = dTimes[iSize*i:iSize*(i+1)].mean()
                    dTimes2 = dTimes[iSize*i:iSize*(i+1)]*1.0
                    dTimes2 -= dTimes2[0]
                    dTimes2.shape += (1,)
                    subData = flux[valid,...][iSize*i:iSize*(i+1),0,good,vis]*1.0
                    subPhase = flux[valid,...][iSize*i:iSize*(i+1),0,good,vis]*1.0
                    if doConj:
                        subData = subData.conj()
                        subPhase = subPhase.conj()
                    subData = numpy.dot(subData, numpy.exp(-2j*numpy.pi*freq2[good,:]*delay))
                    subData /= freq2[good,:].size
                    amp = numpy.dot(subData.T, numpy.exp(-2j*numpy.pi*dTimes2*drate))
                    amp = numpy.abs(amp / dTimes2.size)
                    
                    subPhase = numpy.angle(subPhase.mean()) * 180/numpy.pi
                    subPhase %= 360
                    if subPhase > 180:
                        subPhase -= 360
                        
                    best = numpy.where( amp == amp.max() )
                    if amp.max() > 0:
                        bsnr = (amp[best]-amp.mean())[0]/amp.std()
                        bdly = delay[best[0][0]]*1e6
                        brat = drate[best[1][0]]*1e3
                        
                        axR.plot(subTime-ref_time, brat, linestyle='', marker=markers[pol], color='k')
                        axD.plot(subTime-ref_time, bdly, linestyle='', marker=markers[pol], color='k')
                        axP.plot(subTime-ref_time, subPhase, linestyle='', marker=markers[pol], color='k')
                        axA.plot(subTime-ref_time, amp.max()*1e3, linestyle='', marker=markers[pol], color='k', label=pol)
                        
        first = False
        
    for blName in figs:
        fig, axR, axD, axP, axA = figs[blName]
        
        # Legend and reference marks
        handles, labels = axA.get_legend_handles_labels()
        axA.legend(handles[::iCount], labels[::iCount], loc=0)
        oldLim = axR.get_xlim()
        for ax in (axR, axD, axP):
            ax.hlines(0, oldLim[0], oldLim[1], linestyle=':', alpha=0.5)
        axR.set_xlim(oldLim)
        # Turn off redundant x-axis tick labels
        xticklabels = axR.get_xticklabels() + axD.get_xticklabels() + axP.get_xticklabels()
        plt.setp(xticklabels, visible=False)
        for ax in (axR, axD, axP, axA):
            ax.set_xlabel('Elapsed Time [s since %s]' % datetime.utcfromtimestamp(ref_time).strftime('%Y%b%d %H:%M'))
        # Flip the y axis tick labels on every other plot
        for ax in (axR, axP):
            ax.yaxis.set_label_position('right')
            ax.tick_params(axis='y', which='both', labelleft='off', labelright='on')
        # Get the labels
        axR.set_ylabel('Rate [mHz]')
        axD.set_ylabel('Delay [$\\mu$s]')
        axP.set_ylabel('Phase [$^\\circ$]')
        axA.set_ylabel('Amp.$\\times10^3$')
        # Set the y ranges
        axR.set_ylim((-max([abs(v) for v in axR.get_ylim()]), max([abs(v) for v in axR.get_ylim()])))
        axD.set_ylim((-max([abs(v) for v in axD.get_ylim()]), max([abs(v) for v in axD.get_ylim()])))
        ax.set_ylim((-180,180))
        axA.set_ylim((0,axA.get_ylim()[1]))
        
        fig.tight_layout()
        plt.draw()
        
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='given a collection of .npz files generated by "the next generation of correlator", search for fringes', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('filename', type=str, nargs='+',
                        help='filename to search')
    parser.add_argument('-r', '--ref-ant', type=int, 
                        help='limit plots to baselines containing the reference antenna')
    parser.add_argument('-b', '--baseline', type=aph.csv_baseline_list, 
                        help="limit plots to the specified baseline in 'ANT-ANT' format")
    parser.add_argument('-o', '--drop', action='store_true', 
                        help='drop FLAG table when displaying')
    parser.add_argument('-d', '--decimate', type=int, default=1, 
                        help='frequency decimation factor')
    parser.add_argument('-l', '--limit', type=int, default=-1, 
                        help='limit the data loaded to the first N files, -1 = load all')
    parser.add_argument('-y', '--y-only', action='store_true', 
                        help='limit the search on VLA-LWA baselines to the VLA Y pol. only')
    parser.add_argument('-e', '--delay-window', type=str, default='-inf,inf', 
                        help='delay search window in us; defaults to maximum allowed')
    parser.add_argument('-a', '--rate-window', type=str, default='-inf,inf', 
                        help='rate search window in mHz; defaults to maximum allowed')
    parser.add_argument('-i', '--interval', type=float, default=30.0, 
                        help='fringe search interveral in seconds')
    args = parser.parse_args()
    main(args)
    
