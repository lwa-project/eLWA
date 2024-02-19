#!/usr/bin/env python3

"""
A fancier version of plotFringes.py that makes waterfall-like plots from .npz
files created by the next generation of correlator.
"""

import os
import sys
import glob
import numpy as np
import argparse
import tempfile
from datetime import datetime

from scipy.stats import scoreatpercentile as percentile

from lsl.statistics import robust
from lsl.misc.mathutils import to_dB
from lsl.misc import parser as aph

from utils import read_correlator_configuration

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
    ## Polarization
    if args.xx:
        args.polToPlot = 'XX'
    elif args.xy:
        args.polToPlot = 'XY'
    elif args.yx:
        args.polToPlot = 'YX'
    elif args.yy:
        args.polToPlot = 'YY'
    elif args.stokes_i:
        args.polToPlot = 'I'
    elif args.stokes_v:
        args.polToPlot = 'V'
        
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
    antLookup_inv = {ant.stand.id: ant.config_name for ant in antennas}
    dataDict.close()
    
    # Make sure the reference antenna is in there
    if args.ref_ant is not None:
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
            
    bls = []
    l = 0
    cross = []
    for i in range(0, len(antennas), 2):
        ant1 = antennas[i].stand.id
        for j in range(i, len(antennas), 2):
            ant2 = antennas[j].stand.id
            if args.include_auto or ant1 != ant2:
                if args.baseline is not None:
                    if (ant1,ant2) in args.baseline:
                        bls.append( (ant1,ant2) )
                        cross.append( l )
                elif args.ref_ant is not None:
                    if ant1 == args.ref_ant or ant2 == args.ref_ant:
                        bls.append( (ant1,ant2) )
                        cross.append( l )
                else:
                    bls.append( (ant1,ant2) )
                    cross.append( l )
                    
            l += 1
    nBL = len(cross)
    
    if args.decimate > 1:
        if nchan % args.decimate != 0:
            raise RuntimeError(f"Invalid freqeunce decimation factor:  {nchan} % {args.decimate} = {nchan%args.decimate}")

        nchan //= args.decimate
        freq.shape = (freq.size//args.decimate, args.decimate)
        freq = freq.mean(axis=1)
        
    times = np.zeros(nInt, dtype=np.float64)
    visToPlot = np.zeros((nInt,nBL,nchan), dtype=np.complex64)
    visToMask = np.zeros((nInt,nBL,nchan), dtype=bool)
    
    for i,filename in enumerate(filenames):
        dataDict = np.load(filename)
        
        tStart = dataDict['tStart']
        
        if args.polToPlot == 'I':
            cvis = dataDict['vis1XX'][cross,:] + dataDict['vis1YY'][cross,:]
        elif args.polToPlot == 'V':
            cvis = dataDict['vis1XY'][cross,:] - dataDict['vis1YX'][cross,:]
            cvis /= 1j
        else:
            cvis = dataDict['vis1%s' % args.polToPlot][cross,:]
            
        if args.decimate > 1:
            cvis.shape = (cvis.shape[0], cvis.shape[1]//args.decimate, args.decimate)
            cvis = cvis.mean(axis=2)
            
        visToPlot[i,:,:] = cvis
        
        if not args.drop:
            try:
                delayStepApplied = dataDict['delayStepApplied']
                try:
                    len(delayStepApplied)
                except TypeError:
                    delayStepApplied = [delayStepApplied if ant.stand.id > 50 else False for ant in antennas if ant.pol == 0]
            except KeyError:
                delayStepApplied = [False for ant in antennas if ant.pol == 0]
            delayStepAppliedBL = []
            for j in range(len(delayStepApplied)):
                for k in range(j, len(delayStepApplied)):
                    delayStepAppliedBL.append( delayStepApplied[j] or delayStepApplied[k] )
                    
            visToMask[i,:,:] = [[delayStepAppliedBL[c],] for c in cross]
            
        times[i] = tStart
        
        dataDict.close()
            
    print("Got %i files from %s to %s (%.1f s)" % (len(filenames), datetime.utcfromtimestamp(times[0]).strftime("%Y/%m/%d %H:%M:%S"), datetime.utcfromtimestamp(times[-1]).strftime("%Y/%m/%d %H:%M:%S"), (times[-1]-times[0])))

    iTimes = np.zeros(nInt-1, dtype=times.dtype)
    for i in range(1, len(times)):
        iTimes[i-1] = times[i] - times[i-1]
    print(" -> Interval: {iTimes.mean():.3f} +/- {iTimes.std():.3f} seconds ({iTimes.min():.3f} to {iTimes.max():.3f} seconds)")
    
    print(f"Number of frequency channels: {len(freq)} (~{freq[1]-freq[0]:.1f} Hz/channel)")

    dTimes = times - times[0]
    
    delay = np.linspace(-350e-6, 350e-6, 301)		# s
    drate = np.linspace(-150e-3, 150e-3, 301)		# Hz
    
    good = np.arange(freq.size//8, freq.size*7//8)		# Inner 75% of the band
    
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()
    fig5 = plt.figure()
    
    k = 0
    nRow = int(np.sqrt( len(bls) ))
    nCol = int(np.ceil(len(bls)*1.0/nRow))
    for b in range(len(bls)):
        i,j = bls[b]
        ni,nj = antLookup_inv[i], antLookup_inv[j]
        vis = np.ma.array(visToPlot[:,b,:], mask=visToMask[:,b,:])
        
        ax = fig1.add_subplot(nRow, nCol, k+1)
        ax.imshow(np.ma.angle(vis), extent=(freq[0]/1e6, freq[-1]/1e6, dTimes[0], dTimes[-1]), origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='nearest')
        ax.axis('auto')
        ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('Elapsed Time [s]')
        ax.set_title(f"{ni},{nj} - {args.polToPlot}")
        ax.set_xlim((freq[0]/1e6, freq[-1]/1e6))
        ax.set_ylim((dTimes[0], dTimes[-1]))
        
        ax = fig2.add_subplot(nRow, nCol, k+1)
        amp = np.ma.abs(vis)
        vmin, vmax = percentile(amp, 1), percentile(amp, 99)
        ax.imshow(amp, extent=(freq[0]/1e6, freq[-1]/1e6, dTimes[0], dTimes[-1]), origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.axis('auto')
        ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('Elapsed Time [s]')
        ax.set_title(f"{ni},{nj} - {args.polToPlot}")
        ax.set_xlim((freq[0]/1e6, freq[-1]/1e6))
        ax.set_ylim((dTimes[0], dTimes[-1]))
                
        ax = fig3.add_subplot(nRow, nCol, k+1)
        ax.plot(freq/1e6, np.ma.abs(vis.mean(axis=0)))
        ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('Mean Vis. Amp. [lin.]')
        ax.set_title(f"{ni},{nj} - {args.polToPlot}")
        ax.set_xlim((freq[0]/1e6, freq[-1]/1e6))
        
        ax = fig4.add_subplot(nRow, nCol, k+1)
        ax.plot(np.ma.angle(vis[:,good].mean(axis=1))*180/np.pi, dTimes, linestyle='', marker='+')
        ax.set_xlim((-180, 180))
        ax.set_xlabel('Mean Vis. Phase [deg]')
        ax.set_ylabel('Elapsed Time [s]')
        ax.set_title(f"{ni},{nj} - {args.polToPlot}")
        ax.set_ylim((dTimes[0], dTimes[-1]))
        
        ax = fig5.add_subplot(nRow, nCol, k+1)
        ax.plot(np.ma.abs(vis[:,good].mean(axis=1))*180/np.pi, dTimes, linestyle='', marker='+')
        ax.set_xlabel('Mean Vis. Amp. [lin.]')
        ax.set_ylabel('Elapsed Time [s]')
        ax.set_title(f"{ni},{nj} - {args.polToPlot}")
        ax.set_ylim((dTimes[0], dTimes[-1]))
        
        k += 1
        
    for f in (fig1, fig2, fig3, fig4, fig5):
        f.suptitle("%s to %s UTC" % (datetime.utcfromtimestamp(times[0]).strftime("%Y/%m/%d %H:%M"), datetime.utcfromtimestamp(times[-1]).strftime("%Y/%m/%d %H:%M")))
        
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='given a collection of .npz files generated by "the next generation of correlator", create plots of the visibilities', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, nargs='+', 
                        help='filename to process')
    parser.add_argument('-r', '--ref-ant', type=str, 
                        help='limit plots to baselines containing the reference antenna')
    parser.add_argument('-b', '--baseline', type=aph.csv_baseline_list, 
                        help="limit plots to the specified baseline in 'ANT-ANT' format")
    parser.add_argument('-o', '--drop', action='store_true', 
                        help='drop delay step mask when displaying')
    parser.add_argument('-a', '--include-auto', action='store_true', 
                         help='display the auto-correlations along with the cross-correlations')
    pgroup = parser.add_mutually_exclusive_group(required=False)
    pgroup.add_argument('-x', '--xx', action='store_true', default=True, 
                        help='plot XX data')
    pgroup.add_argument('-z', '--xy', action='store_true', 
                        help='plot XY data')
    pgroup.add_argument('-w', '--yx', action='store_true', 
                        help='plot YX data')
    pgroup.add_argument('-y', '--yy', action='store_true', 
                        help='plot YY data')
    pgroup.add_argument('-i', '--stokes-i', action='store_true', 
                        help='plot Stokes I data')
    pgroup.add_argument('-v', '--stokes-v', action='store_true', 
                        help='plot Stokes V data')
    parser.add_argument('-l', '--limit', type=int, default=-1, 
                        help='limit the data loaded to the first N files, -1 = load all')
    parser.add_argument('-d', '--decimate', type=int, default=1, 
                        help='frequency decimation factor')
    args = parser.parse_args()
    try:
        args.ref_ant = int(args.ref_ant, 10)
    except (TypeError, ValueError):
        pass
    main(args)
