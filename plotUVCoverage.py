#!/usr/bin/env python

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import os
import sys
import numpy
import argparse
import tempfile
from datetime import datetime

from lsl.common import stations
from lsl.correlator import uvutils
from lsl.reader import drx, vdif

from utils import *

from matplotlib import pyplot as plt


def main(args):
    # Build up the station
    site = stations.lwa1
    observer = site.get_observer()
    
    # Load in the file file to figure out what to do
    dataDict = numpy.load(args.filename[0])
    tStart = dataDict['tStart'].item()
    tInt = dataDict['tInt']
    freq = dataDict['freq1']
    junk0, refSrc, junk1, junk2, junk3, readers, antennas = read_correlator_configuration(dataDict)
    dataDict.close()
    
    # Prune down to a single polarization
    antennas = [ant for ant in antennas if ant.pol == 0]
    
    # Build up the list of baselines
    blList = uvutils.get_baselines(antennas)
    
    # Loop through the files and do what we need to do
    t = []
    uvw = []
    for filename in args.filename:
        ## Load in the integration
        dataDict = numpy.load(filename)
        
        tStart = dataDict['tStart'].item()
        tInt = dataDict['tInt'].item()
        
        dataDict.close()
        
        ## Update the observation
        observer.date = datetime.utcfromtimestamp(tStart).strftime('%Y/%m/%d %H:%M:%S.%f')
        refSrc.compute(observer)
        HA = (observer.sidereal_time() - refSrc.ra) * 12/numpy.pi
        dec = refSrc.dec * 180/numpy.pi
        
        ## Compute u, v, and w
        t.append( datetime.utcfromtimestamp(tStart) )
        uvw.append( uvutils.compute_uvw(antennas, HA=HA, dec=dec, freq=freq.mean(), site=observer) )
    uvw = numpy.array(uvw) / 1e3
    
    # Compute the baseline lengths
    blLength = numpy.sqrt( (uvw**2).sum(axis=2) )
    print(len(blList), uvw.shape, blLength.shape)
    
    # Report
    print("Phase Center:")
    print("  Name: %s" % refSrc.name)
    print("  RA: %s" % refSrc._ra)
    print("  Dec: %s" % refSrc._dec)
    print("Antennas:")
    print("  Total: %i" % len(readers))
    print("  VDIF: %i" % sum([1 for rdr in readers if rdr is vdif]))
    print("  DRX: %i" % sum([1 for rdr in readers if rdr is drx]))
    print("Baselines:")
    print("  Total: %i" % (uvw.shape[0]*uvw.shape[1]))
    ## Minimum basline length
    m = numpy.argmin(blLength)
    b = m % uvw.shape[1]
    bl0 = blList[b][0].stand.id
    if bl0 < 50:
        bl0 = 'EA%02i' % bl0
    else:
        bl0 = 'LWA%i' % (bl0-50)
    bl1 = blList[b][1].stand.id
    if bl1 < 50:
        bl1 = 'EA%02i' % bl1
    else:
        bl1 = 'LWA%i' % (bl1-50)
    print("  Minimum: %.2f klambda (%s <-> %s)" % (blLength.min(), bl0, bl1))
    print("  Median: %.2f klambda" % numpy.median(blLength))
    ## Maximum baseline length
    m = numpy.argmax(blLength)
    b = m % uvw.shape[1]
    bl0 = blList[b][0].stand.id
    if bl0 < 50:
        bl0 = 'EA%02i' % bl0
    else:
        bl0 = 'LWA%i' % (bl0-50)
    bl1 = blList[b][1].stand.id
    if bl1 < 50:
        bl1 = 'EA%02i' % bl1
    else:
        bl1 = 'LWA%i' % (bl1-50)
    print("  Maximum %.2f klambda (%s <-> %s)" % (blLength.max(), bl0, bl1))
    
    # Plot
    fig = plt.figure()
    ax = fig.gca()
    for i in xrange(uvw.shape[0]):
        l, = ax.plot(uvw[i,:,0], uvw[i,:,1], linestyle='', marker='o', ms=3.0, alpha=0.2)
        ax.plot(-uvw[i,:,0], -uvw[i,:,1], linestyle='', marker='o', ms=3.0, alpha=0.2, color=l.get_color())
    ax.set_xlabel('u [$k\\lambda$]')
    ax.set_ylabel('v [$k\\lambda$]')
    ax.set_title('%s\n%s to %s' % (refSrc.name, min(t).strftime("%m/%d %H:%M:%S"), max(t).strftime("%m/%d %H:%M:%S")))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='plot the uv coverage in a collection of .npz files generated by superCorrelator.py', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('filename', type=str, nargs='+', 
                        help='filename to plot')
    args = parser.parse_args()
    main(args)
    