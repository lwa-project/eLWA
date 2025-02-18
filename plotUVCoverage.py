#!/usr/bin/env python3

import os
import sys
import numpy as np
import argparse
import tempfile
from datetime import datetime

from lsl.common import stations
from lsl.correlator import uvutils
from lsl.reader import drx, drx8, vdif

from utils import *

from matplotlib import pyplot as plt


def main(args):
    # Build up the station
    site = stations.lwa1
    observer = site.get_observer()
    
    # Load in the file file to figure out what to do
    dataDict = np.load(args.filename[0])
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
        dataDict = np.load(filename)
        
        tStart = dataDict['tStart'].item()
        tInt = dataDict['tInt'].item()
        
        dataDict.close()
        
        ## Update the observation
        observer.date = datetime.utcfromtimestamp(tStart).strftime('%Y/%m/%d %H:%M:%S.%f')
        refSrc.compute(observer)
        HA = (observer.sidereal_time() - refSrc.ra) * 12/np.pi
        dec = refSrc.dec * 180/np.pi
        
        ## Compute u, v, and w
        t.append( datetime.utcfromtimestamp(tStart) )
        uvw.append( uvutils.compute_uvw(antennas, HA=HA, dec=dec, freq=freq.mean(), site=observer) )
    uvw = np.array(uvw) / 1e3
    
    # Compute the baseline lengths
    blLength = np.sqrt( (uvw**2).sum(axis=2) )
    print(len(blList), uvw.shape, blLength.shape)
    
    # Report
    print("Phase Center:")
    print("  Name: %s" % refSrc.name)
    print("  RA: %s" % refSrc._ra)
    print("  Dec: %s" % refSrc._dec)
    print("Antennas:")
    print("  Total: %i" % len(readers))
    print("  VDIF: %i" % sum([1 for rdr in readers if rdr is vdif]))
    print("  DRX: %i" % sum([1 for rdr in readers if not rdr is vdif]))
    print("Baselines:")
    print("  Total: %i" % (uvw.shape[0]*uvw.shape[1]))
    ## Minimum basline length
    m = np.argmin(blLength)
    b = m % uvw.shape[1]
    bl0 = blList[b][0].config_name
    bl1 = blList[b][1].config_name
    print(f"  Minimum: {blLength.min():.2f} klambda ({bl0} <-> {bl1})")
    print(f"  Median: {np.median(blLength):.2f} klambda")
    ## Maximum baseline length
    m = np.argmax(blLength)
    b = m % uvw.shape[1]
    bl0 = blList[b][0].config_name
    bl1 = blList[b][1].config_name
    print(f"  Maximum {blLength.max():.2f} klambda ({bl0} <-> {bl1})")
    
    # Plot
    fig = plt.figure()
    ax = fig.gca()
    for i in range(uvw.shape[0]):
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
