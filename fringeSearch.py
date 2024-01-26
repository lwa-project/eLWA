#!/usr/bin/env python3

"""
Given a collection of .npz files search for course delays and rates.
"""

import os
import sys
import glob
import numpy
import argparse
import tempfile

from datetime import datetime

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
    ## Search limits
    args.delay_window = [float(v) for v in args.delay_window.split(',', 1)]
    args.rate_window = [float(v) for v in args.rate_window.split(',', 1)]
    ## Filenames
    filenames = args.filename
    filenames.sort()
    if args.limit != -1:
        filenames = filenames[:args.limit]
        
    nInt = len(filenames)
    
    dataDict = numpy.load(filenames[0])
    tInt = dataDict['tInt']
    nBL, nchan = dataDict['vis1XX'].shape
    freq = dataDict['freq1']
    junk0, refSrc, junk1, junk2, junk3, junk4, antennas = read_correlator_configuration(dataDict)
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
        if not found:
            raise RuntimeError("Cannot file reference antenna %i in the data" % args.ref_ant)
            
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
            raise RuntimeError("Invalid freqeunce decimation factor:  %i %% %i = %i" % (nchan, args.decimate, nchan%args.decimate))

        nchan //= args.decimate
        freq.shape = (freq.size//args.decimate, args.decimate)
        freq = freq.mean(axis=1)
        
    times = numpy.zeros(nInt, dtype=numpy.float64)
    visXX = numpy.zeros((nInt,nBL,nchan), dtype=numpy.complex64)
    if not args.y_only:
        visXY = numpy.zeros((nInt,nBL,nchan), dtype=numpy.complex64)
    visYX = numpy.zeros((nInt,nBL,nchan), dtype=numpy.complex64)
    visYY = numpy.zeros((nInt,nBL,nchan), dtype=numpy.complex64)

    for i,filename in enumerate(filenames):
        dataDict = numpy.load(filename)

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

    iTimes = numpy.zeros(nInt-1, dtype=times.dtype)
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
    spec  = numpy.median(numpy.abs(visXX.mean(axis=0)), axis=0)
    spec += numpy.median(numpy.abs(visYY.mean(axis=0)), axis=0)
    smth = spec*0.0
    winSize = int(250e3/(freq[1]-freq[0]))
    winSize += ((winSize+1)%2)
    for i in range(smth.size):
        mn = max([0, i-winSize//2])
        mx = min([i+winSize//2+1, smth.size])
        smth[i] = numpy.median(spec[mn:mx])
    smth /= robust.mean(smth)
    bp = spec / smth
    good = numpy.where( (smth > 0.1) & (numpy.abs(bp-robust.mean(bp)) < 3*robust.std(bp)) )[0]
    nBad = nchan - len(good)
    print("Masking %i of %i channels (%.1f%%)" % (nBad, nchan, 100.0*nBad/nchan))
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
    
    dirName = os.path.basename( os.path.dirname(filenames[0]) )
    print("%3s  %9s  %2s  %6s  %9s  %11s" % ('#', 'BL', 'Pl', 'S/N', 'Delay', 'Rate'))
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
        if bls[b][0] not in (51, 52, 53) and bls[b][1] not in (51, 52, 53):
            ### Standard VLA-VLA baseline
            polToUse = ('XX', 'YY')
            visToUse = (visXX, visYY)
        else:
            ### LWA-LWA or LWA-VLA baseline
            if args.y_only:
                polToUse = ('YX', 'YY')
                visToUse = (visYX, visYY)
            else:
                polToUse = ('XX', 'XY', 'YX', 'YY')
                visToUse = (visXX, visXY, visYX, visYY)
                
        if args.plot:
            fig = plt.figure()
            axs = {}
            axs['XX'] = fig.add_subplot(2, 2, 1)
            axs['YY'] = fig.add_subplot(2, 2, 2)
            axs['XY'] = fig.add_subplot(2, 2, 3)
            axs['YX'] = fig.add_subplot(2, 2, 4)
            
        for pol,vis in zip(polToUse, visToUse):
            subData = vis[:,b,good]*1.0
            if doConj:
                subData = subData.conj()
            subData = numpy.dot(subData, numpy.exp(-2j*numpy.pi*freq2[good,:]*delay))
            subData /= freq2[good,:].size
            amp = numpy.dot(subData.T, numpy.exp(-2j*numpy.pi*dTimes2*drate))
            amp = numpy.abs(amp / dTimes2.size)
            
            blName = bls[b]
            if doConj:
                blName = (bls[b][1],bls[b][0])
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
            fig.suptitle(dirName)
            for pol in axs.keys():
                ax = axs[pol]
                ax.axis('auto')
                ax.set_title(pol)
                ax.set_xlabel('Rate [mHz]')
                ax.set_ylabel('Delay [$\\mu$s]')
            fig.suptitle("%s @ %s" % (blName, refSrc.name))
            plt.draw()
            
    if args.plot:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='given a collection of .npz files generated by superCorrelator.py, search for fringes', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('filename', type=str, nargs='+', 
                        help='filename to search')
    parser.add_argument('-r', '--ref-ant', type=int, 
                        help='limit plots to baselines containing the reference antenna')
    parser.add_argument('-b', '--baseline', type=aph.csv_baseline_list, 
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
    parser.add_argument('-p', '--plot', action='store_true', 
                        help='show search plots at the end')
    args = parser.parse_args()
    main(args)
    
