#!/usr/bin/env python3

"""
Given a collection of .npz files search for course delays and rates.
"""

import os
import sys
import glob
import numpy as np
import argparse
import tempfile
from datetime import datetime

from lsl.statistics import robust
from lsl.misc.mathutils import to_dB

from utils import read_correlator_configuration

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle as Box


def main(args):
    # Parse the command line
    ## Search limits
    args.delay_window = [float(v) for v in args.delay_window.split(',', 1)]
    args.rate_window = [float(v) for v in args.rate_window.split(',', 1)]
    ## Filenames
    filenames = args.filename
    filenames.sort()
    if args.limit != -1:
        filenames = filenames[:args.limit]
        
    nInt = len(filenames)
    
    dataDict = np.load(filenames[0])
    tInt = dataDict['tInt']
    nBL, nchan = dataDict['vis1XX'].shape
    freq = dataDict['freq1']
    junk0, refSrc, junk1, junk2, junk3, junk4, antennas = read_correlator_configuration(dataDict)
    antLookup = {ant.config_name: ant.stand.id for ant in antennas}
    antLookup_inv = {ant.stand.id: ant.config_name for ant in antennas}
    dataDict.close()
    
    # Make sure the reference antenna is in there
    if args.ref_ant is None:
        args.ref_ant = antennas[0].stand.id
    else:
        found = False
        for ant in antennas:
            if ant.stand.id == args.ref_ant:
                found = True
                break
            elif ant.config_name == args.ref_ant:
                args.ref_ant = ant.stand.id
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
        
    bls = []
    l = 0
    cross = []
    for i in range(0, len(antennas), 2):
        ant1 = antennas[i].stand.id
        for j in range(i, len(antennas), 2):
            ant2 = antennas[j].stand.id
            if ant1 != ant2:
                bls.append( (ant1,ant2) )
                cross.append( l )
            l += 1
    nBL = len(cross)
    
    if args.decimate > 1:
        if nchan % args.decimate != 0:
            raise RuntimeError(f"Invalid freqeunce decimation factor:  {nchan} % {args.decimate} = {nchan%args.decimate}")

        nchan /= args.decimate
        freq.shape = (freq.size/args.decimate, args.decimate)
        freq = freq.mean(axis=1)
        
    times = np.zeros(nInt, dtype=np.float64)
    visXX = np.zeros((nInt,nBL,nchan), dtype=np.complex64)
    if not args.y_only:
        visXY = np.zeros((nInt,nBL,nchan), dtype=np.complex64)
    visYX = np.zeros((nInt,nBL,nchan), dtype=np.complex64)
    visYY = np.zeros((nInt,nBL,nchan), dtype=np.complex64)

    for i,filename in enumerate(filenames):
        dataDict = np.load(filename)

        tStart = dataDict['tStart']
        
        cvisXX = dataDict['vis1XX'][cross,:]
        cvisXY = dataDict['vis1XY'][cross,:]
        cvisYX = dataDict['vis1YX'][cross,:]
        cvisYY = dataDict['vis1YY'][cross,:]
        
        if args.decimate > 1:
            cvisXX.shape = (cvisXX.shape[0], cvisXX.shape[1]//args.decimate, args.decimate)
            cvisXX = cvisXX.mean(axis=2)
            cvisXY.shape = (cvisXY.shape[0], cvisXY.shape[1]//args.decimate, args.decimate)
            cvisXY = cvisXY.mean(axis=2)
            cvisYX.shape = (cvisYX.shape[0], cvisYX.shape[1]//args.decimate, args.decimate)
            cvisYX = cvisYX.mean(axis=2)
            cvisYY.shape = (cvisYY.shape[0], cvisYY.shape[1]//args.decimate, args.decimate)
            cvisYY = cvisYY.mean(axis=2)
            
        visXX[i,:,:] = cvisXX
        if not args.y_only:		
            visXY[i,:,:] = cvisXY
        visYX[i,:,:] = cvisYX
        visYY[i,:,:] = cvisYY	

        times[i] = tStart
        
        dataDict.close()
            
    print("Got %i files from %s to %s (%.1f s)" % (len(filenames), datetime.utcfromtimestamp(times[0]).strftime("%Y/%m/%d %H:%M:%S"), datetime.utcfromtimestamp(times[-1]).strftime("%Y/%m/%d %H:%M:%S"), (times[-1]-times[0])))

    iTimes = np.zeros(nInt-1, dtype=times.dtype)
    for i in range(1, len(times)):
        iTimes[i-1] = times[i] - times[i-1]
    print(f" -> Interval: {robust.mean(iTimes):.3f} +/- {robust.std(iTimes):.3f} seconds ({iTimes.min():.3f} to {iTimes.max():.3f} seconds)")
    iSize = int(round(args.interval/robust.mean(iTimes)))
    iCount = times.size//iSize
    if iCount == 0:
        args.interval = times.size*robust.mean(iTimes)
        iSize = int(round(args.interval/robust.mean(iTimes)))
        iCount = times.size//iSize
        print(f"WARNING:  Not enough data for requested search interval, changing to {args.interval:.3f} seconds")
    print(f" -> Chunk size is {iSize} intervals ({iSize*robust.mean(iTimes):.3f} seconds)")
    print(f" -> Working with {iCount} chunks of data")
    
    print(f"Number of frequency channels: {len(freq)} (~{freq[1]-freq[0]:.1f} Hz/channel)")

    dTimes = times - times[0]
    ref_time = (int(times[0]) / 60) * 60
    
    dMax = 1.0/(freq[1]-freq[0])/4
    dMax = int(dMax*1e6)*1e-6
    if -dMax*1e6 > args.delay_window[0]:
        args.delay_window[0] = -dMax*1e6
    if dMax*1e6 < args.delay_window[1]:
        args.delay_window[1] = dMax*1e6
    rMax = 1.0/robust.mean(iTimes)/4
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
    spec  = np.median(np.abs(visXX.mean(axis=0)), axis=0)
    spec += np.median(np.abs(visYY.mean(axis=0)), axis=0)
    smth = spec*0.0
    winSize = int(250e3/(freq[1]-freq[0]))
    winSize += ((winSize+1)%2)
    for i in range(smth.size):
        mn = max([0, i-winSize//2])
        mx = min([i+winSize, smth.size])
        smth[i] = np.median(spec[mn:mx])
    smth /= robust.mean(smth)
    bp = spec / smth
    try:
        good = np.where( (smth > 0.1) & (np.abs(bp-robust.mean(bp)) < 3*robust.std(bp)) )[0]
    except ValueError:
        # Fall back to NumPy
        good = np.where( (smth > 0.1) & (np.abs(bp-np.mean(bp)) < 3*np.std(bp)) )[0]
    nBad = nchan - len(good)
    print(f"Masking {nBad} of {nchan} channels ({100.0*nBad/nchan:.1f}%)")
    
    freq2 = freq*1.0
    freq2.shape += (1,)
    
    dirName = os.path.basename( os.path.dirname(filenames[0]) )
    for b in range(len(bls)):
        ## Skip over baselines that are not in the baseline list (if provided)
        if args.baseline is not None:
            if bls[b] not in args.baseline:
                continue
        ## Skip over baselines that don't include the reference antenna
        elif bls[b][0] != args.ref_ant and bls[b][1] != args.ref_ant:
            continue
            
        ## Check and see if we need to conjugate the visibility, i.e., switch from
        ## baseline (*,ref) to baseline (ref,*)
        doConj = False
        if bls[b][1] == args.ref_ant:
            doConj = True
            
        ## Figure out which polarizations to process
        if antLookup_inv[bls[b][0]][:3] != 'LWA' and antLookup_inv[bls[b][1]][:3] != 'LWA':
            ### Standard VLA-VLA baseline
            polToUse = ('XX', 'XY', 'YX', 'YY')
            visToUse = (visXX, visXY, visYX, visYY)
        else:
            ### LWA-LWA or LWA-VLA baseline
            if args.y_only:
                polToUse = ('YX', 'YY')
                visToUse = (visYX, visYY)
            else:
                polToUse = ('XX', 'XY', 'YX', 'YY')
                visToUse = (visXX, visXY, visYX, visYY)
                
        blName = bls[b]
        if doConj:
            blName = (bls[b][1],bls[b][0])
        blName = '%s-%s' % (antLookup_inv[blName[0]], antLookup_inv[blName[1]])
        
        fig = plt.figure()
        fig.suptitle(f"{blName} @ {refSrc.name}")
        fig.subplots_adjust(hspace=0.001)
        axR = fig.add_subplot(4, 1, 1)
        axD = fig.add_subplot(4, 1, 2, sharex=axR)
        axP = fig.add_subplot(4, 1, 3, sharex=axR)
        axA = fig.add_subplot(4, 1, 4, sharex=axR)
        markers = {'XX':'s', 'YY':'o', 'XY':'v', 'YX':'^'}
        
        for pol,vis in zip(polToUse, visToUse):
            for i in range(iCount):
                subStart, subStop = times[iSize*i], times[iSize*(i+1)-1]
                if (subStop - subStart) > 1.1*args.interval:
                    continue
                    
                subTime = np.array([times[iSize*i:iSize*(i+1)].mean(),])
                dTimes2 = dTimes[iSize*i:iSize*(i+1)]*1.0
                dTimes2.shape += (1,)
                subData = vis[iSize*i:iSize*(i+1),b,good]*1.0
                subPhase = vis[iSize*i:iSize*(i+1),b,good]*1.0
                if doConj:
                    subData = subData.conj()
                    subPhase = subPhase.conj()
                subData = np.dot(subData, np.exp(-2j*np.pi*freq2[good,:]*delay))
                subData /= freq2[good,:].size
                amp = np.dot(subData.T, np.exp(-2j*np.pi*dTimes2*drate))
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
                    
                    c = axR.scatter(subTime-ref_time, brat, c=bsnr, marker=markers[pol],
                                    cmap='gist_yarg', norm=None, vmin=3, vmax=40)
                    c = axD.scatter(subTime-ref_time, bdly, c=bsnr, marker=markers[pol],
                                    cmap='gist_yarg', norm=None, vmin=3, vmax=40)
                    c = axP.scatter(subTime-ref_time, subPhase, c=bsnr, marker=markers[pol],
                                    cmap='gist_yarg', norm=None, vmin=3, vmax=40)
                    c = axA.scatter(subTime-ref_time, amp.max()*1e3, c=bsnr, marker=markers[pol],
                                    cmap='gist_yarg', norm=None, vmin=3, vmax=40)
                    
        # Colorbar
        cb = fig.colorbar(c, ax=axR, orientation='horizontal')     # pylint: disable=possibly-used-before-assignment,used-before-assignment
        cb.set_label('SNR')
        # Legend and reference marks
        handles = []
        for pol in polToUse:
            handles.append(Line2D([0,], [0,], linestyle='', marker=markers[pol], color='k', label=pol))
        axA.legend(handles=handles, loc=0)
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
        plt.draw()
        
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
    try:
        args.ref_ant = int(args.ref_ant, 10)
    except (TypeError, ValueError):
        pass
    main(args)
