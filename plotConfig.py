#!/usr/bin/env python3

"""
Small script to plot up the configuration in a file.
"""

import os
import sys
import numpy as np
import argparse
import tempfile

from lsl.reader import vdif, drx

from utils import *

from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter


def main(args):
    # Parse the correlator configuration
    config, refSrc, filenames, metanames, foffsets, readers, antennas = read_correlator_configuration(args.filename)
        
    # Load in the stand position data and antenna names
    data = []
    names = []
    processed = []
    for ant in antennas:
        if ant.stand.id not in processed:
            data.append( [ant.stand.x, ant.stand.y, ant.stand.z] )
            names.append( ant.config_name )
            processed.append( ant.stand.id )
    data = np.array(data)
    
    # Get the baseline lengths for zenith
    bl = []
    blLength = []
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            bl.append( (i,j) )
            blLength.append( np.sqrt( ((data[i,:]-data[j,:])**2).sum() ) )
    blLength = np.array(blLength) / 1e3
    
    # Report
    print("Filename: %s" % os.path.basename(args.filename))
    print("  Phase Center:")
    print("    Name: %s" % refSrc.name)
    print("    RA: %s" % refSrc._ra)
    print("    Dec: %s" % refSrc._dec)
    print("  Antennas:")
    print("    Total: %i" % len(filenames))
    print("    VDIF: %i" % sum([1 for rdr in readers if rdr is vdif]))
    print("    DRX: %i" % sum([1 for rdr in readers if rdr is drx]))
    print("  Baselines:")
    print("    Total: %i" % len(bl))
    print("    Minimum: %.2f km (%s <-> %s)" % (blLength.min(), names[bl[np.argmin(blLength)][0]], names[bl[np.argmin(blLength)][1]]))
    print("    Median: %.2f km" % np.median(blLength))
    print("    Maximum %.2f km (%s <-> %s)" % (blLength.max(), names[bl[np.argmax(blLength)][0]], names[bl[np.argmax(blLength)][1]]))
    
    # Color-code the stands by their elevation
    color = data[:,2]
    
    # Plot the stands as colored circles
    fig = plt.figure(figsize=(8,8))
    
    ax1 = plt.axes([0.30, 0.30, 0.60, 0.60])
    ax2 = plt.axes([0.30, 0.05, 0.60, 0.15])
    ax3 = plt.axes([0.05, 0.30, 0.15, 0.60])
    ax4 = plt.axes([0.05, 0.05, 0.15, 0.15])
    c = ax1.scatter(data[:,0]/1e3, data[:,1]/1e3, c=color, s=40.0, alpha=0.50)	
    ax1.set_xlabel('$\Delta$X [E-W; km]')
    ax1.set_ylabel('$\Delta$Y [N-S; km]')
    fig.suptitle(f"{os.path.basename(args.filename)} - {refSrc.name}")
    
    ax2.scatter(data[:,0]/1e3, data[:,2], c=color, s=40.0)
    ax2.xaxis.set_major_formatter( NullFormatter() )
    ax2.set_ylabel('$\Delta$Z [m]')
    ax3.scatter(data[:,2], data[:,1]/1e3, c=color, s=40.0)
    ax3.yaxis.set_major_formatter( NullFormatter() )
    ax3.set_xlabel('$\Delta$Z [m]')
    
    # Add in the labels, if requested
    if args.label:
        for i in range(len(names)):
            ax1.annotate(names[i], xy=(data[i,0]/1e3, data[i,1]/1e3))
            
    # Add and elevation colorbar to the right-hand side of the figure
    cb = plt.colorbar(c, cax=ax4, orientation='vertical')
    
    # Set the axis limits
    ax2.set_xlim( ax1.get_xlim() )
    ax3.set_ylim( ax1.get_ylim() )
    
    # Done
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='plot the antenna locations in a configuration or superCorrelator.py .npz file', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('filename', type=str, 
                        help='filename to plot')
    parser.add_argument('-l', '--label', action='store_true', 
                        help='label the stands with their IDs')
    args = parser.parse_args()
    main(args)
