#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A fancier version of plotFringes.py that makes waterfall-like plots from .npz
files created by the next generation of correlator.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import sys
import glob
import numpy
import argparse
import tempfile
from datetime import datetime

from scipy.stats import scoreatpercentile as percentile

from lsl.statistics import robust
from lsl.misc.mathutil import to_dB

from utils import readCorrelatorConfiguration

from matplotlib import pyplot as plt


def main(args):
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
    elif args.stokes_i:
        args.polToPlot = 'I'
    elif args.stokes_v:
        args.polToPlot = 'V'
        
    filenames = args.filename
    filenames.sort()
    if args.limit != -1:
        filenames = filenames[:args.limit]
        
    nInt = len(filenames)
    
    dataDict = numpy.load(filenames[0])
    tInt = dataDict['tInt']
    nBL, nChan = dataDict['vis1XX'].shape
    freq = dataDict['freq1']
    
    cConfig = dataDict['config']
    fh, tempConfig = tempfile.mkstemp(suffix='.txt', prefix='config-')
    fh = open(tempConfig, 'w')
    for line in cConfig:
        fh.write('%s\n' % line)
    fh.close()
    refSrc, junk1, junk2, junk3, junk4, antennas = readCorrelatorConfiguration(tempConfig)
    os.unlink(tempConfig)
    
    dataDict.close()
    
    # Make sure the reference antenna is in there
    if args.ref_ant is not None:
        found = False
        for ant in antennas:
            if ant.stand.id == args.ref_ant:
                found = True
                break
        if not found:
            raise RuntimeError("Cannot file reference antenna %i in the data" % args.ref_ant)
            
    bls = []
    l = 0
    cross = []
    for i in xrange(0, len(antennas), 2):
        ant1 = antennas[i].stand.id
        for j in xrange(i, len(antennas), 2):
            ant2 = antennas[j].stand.id
            if ant1 != ant2:
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
        if nChan % args.decimate != 0:
            raise RuntimeError("Invalid freqeunce decimation factor:  %i %% %i = %i" % (nChan, args.decimate, nChan%args.decimate))

        nChan /= args.decimate
        freq.shape = (freq.size/args.decimate, args.decimate)
        freq = freq.mean(axis=1)
        
    times = numpy.zeros(nInt, dtype=numpy.float64)
    visToPlot = numpy.zeros((nInt,nBL,nChan), dtype=numpy.complex64)
    visToMask = numpy.zeros((nInt,nBL,nChan), dtype=numpy.bool)
    
    for i,filename in enumerate(filenames):
        dataDict = numpy.load(filename)
        
        tStart = dataDict['tStart']
        
        if args.polToPlot == 'I':
            cvis = dataDict['vis1XX'][cross,:] + dataDict['vis1YY'][cross,:]
        elif args.polToPlot == 'V':
            cvis = dataDict['vis1XY'][cross,:] - dataDict['vis1YX'][cross,:]
            cvis /= 1j
        else:
            cvis = dataDict['vis1%s' % args.polToPlot][cross,:]
            
        if args.decimate > 1:
            cvis.shape = (cvis.shape[0], cvis.shape[1]/args.decimate, args.decimate)
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
            for j in xrange(len(delayStepApplied)):
                for k in xrange(j, len(delayStepApplied)):
                    delayStepAppliedBL.append( delayStepApplied[j] or delayStepApplied[k] )
                    
            visToMask[i,:,:] = [[delayStepAppliedBL[c],] for c in cross]
            
        times[i] = tStart
        
        dataDict.close()
            
    print "Got %i files from %s to %s (%.1f s)" % (len(filenames), datetime.utcfromtimestamp(times[0]).strftime("%Y/%m/%d %H:%M:%S"), datetime.utcfromtimestamp(times[-1]).strftime("%Y/%m/%d %H:%M:%S"), (times[-1]-times[0]))

    iTimes = numpy.zeros(nInt-1, dtype=times.dtype)
    for i in xrange(1, len(times)):
        iTimes[i-1] = times[i] - times[i-1]
    print " -> Interval: %.3f +/- %.3f seconds (%.3f to %.3f seconds)" % (iTimes.mean(), iTimes.std(), iTimes.min(), iTimes.max())
    
    print "Number of frequency channels: %i (~%.1f Hz/channel)" % (len(freq), freq[1]-freq[0])

    dTimes = times - times[0]
    
    delay = numpy.linspace(-350e-6, 350e-6, 301)		# s
    drate = numpy.linspace(-150e-3, 150e-3, 301)		# Hz
    
    good = numpy.arange(freq.size/8, freq.size*7/8)		# Inner 75% of the band
    
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()
    
    k = 0
    nRow = int(numpy.sqrt( len(bls) ))
    nCol = int(numpy.ceil(len(bls)*1.0/nRow))
    for b in xrange(len(bls)):
        i,j = bls[b]
        vis = numpy.ma.array(visToPlot[:,b,:], mask=visToMask[:,b,:])
        
        ax = fig1.add_subplot(nRow, nCol, k+1)
        ax.imshow(numpy.ma.angle(vis), extent=(freq[0]/1e6, freq[-1]/1e6, dTimes[0], dTimes[-1]), origin='lower', vmin=-numpy.pi, vmax=numpy.pi, interpolation='nearest')
        ax.axis('auto')
        ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('Elapsed Time [s]')
        ax.set_title("%i,%i - %s" % (i,j,args.polToPlot))
        ax.set_xlim((freq[0]/1e6, freq[-1]/1e6))
        ax.set_ylim((dTimes[0], dTimes[-1]))
        
        ax = fig2.add_subplot(nRow, nCol, k+1)
        amp = numpy.ma.abs(vis)
        vmin, vmax = percentile(amp, 1), percentile(amp, 99)
        ax.imshow(amp, extent=(freq[0]/1e6, freq[-1]/1e6, dTimes[0], dTimes[-1]), origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.axis('auto')
        ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('Elapsed Time [s]')
        ax.set_title("%i,%i - %s" % (i,j,args.polToPlot))
        ax.set_xlim((freq[0]/1e6, freq[-1]/1e6))
        ax.set_ylim((dTimes[0], dTimes[-1]))
                
        ax = fig3.add_subplot(nRow, nCol, k+1)
        ax.plot(freq/1e6, numpy.ma.abs(vis.mean(axis=0)))
        ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('Mean Vis. Amp. [lin.]')
        ax.set_title("%i,%i - %s" % (i,j,args.polToPlot))
        ax.set_xlim((freq[0]/1e6, freq[-1]/1e6))
        
        ax = fig4.add_subplot(nRow, nCol, k+1)
        ax.plot(dTimes, numpy.ma.angle(vis[:,good].mean(axis=1))*180/numpy.pi, linestyle='', marker='+')
        ax.set_ylim((-180, 180))
        ax.set_xlabel('Elapsed Time [s]')
        ax.set_ylabel('Mean Vis. Phase [deg]')
        ax.set_title("%i,%i - %s" % (i,j,args.polToPlot))
        ax.set_xlim((dTimes[0], dTimes[-1]))
        
        k += 1
        
    fig1.suptitle("%s to %s UTC" % (datetime.utcfromtimestamp(times[0]).strftime("%Y/%m/%d %H:%M"), datetime.utcfromtimestamp(times[-1]).strftime("%Y/%m/%d %H:%M")))
    fig2.suptitle("%s to %s UTC" % (datetime.utcfromtimestamp(times[0]).strftime("%Y/%m/%d %H:%M"), datetime.utcfromtimestamp(times[-1]).strftime("%Y/%m/%d %H:%M")))
    fig3.suptitle("%s to %s UTC" % (datetime.utcfromtimestamp(times[0]).strftime("%Y/%m/%d %H:%M"), datetime.utcfromtimestamp(times[-1]).strftime("%Y/%m/%d %H:%M")))
    fig4.suptitle("%s to %s UTC" % (datetime.utcfromtimestamp(times[0]).strftime("%Y/%m/%d %H:%M"), datetime.utcfromtimestamp(times[-1]).strftime("%Y/%m/%d %H:%M")))
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='given a collection of .npz files generated by "the next generation of correlator", create plots of the visibilities', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, nargs='+', 
                        help='filename to process')
    parser.add_argument('-r', '--ref-ant', type=int, 
                        help='limit plots to baselines containing the reference antenna')
    parser.add_argument('-b', '--baseline', type=str, 
                        help="limit plots to the specified baseline in 'ANT-ANT' format")
    parser.add_argument('-o', '--drop', action='store_true', 
                        help='drop delay step mask when displaying')
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
    parser.add_argument('-l', '--limit', type=int, 
                        help='limit the data loaded to the first N files, -1 = load all')
    parser.add_argument('-d', '--decimate', type=int, default=1, 
                        help='frequency decimation factor')
    args = parser.parse_args()
    main(args)
    