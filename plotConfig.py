#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Small script to plot up the configuration in a file.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import sys
import numpy
import getopt
import tempfile

from lsl.reader import vdif, drx

from utils import *

from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter


def usage(exitCode=None):
    print """plotConfig.py - Plot the antenna locations in a configuration file

Usage:
plotConfig.py [OPTIONS] configFile|npzFile

Options:
-h, --help                  Display this help information
-l, --label                 Label the stands with their IDs
                            (default = no)
"""
    
    if exitCode is not None:
        sys.exit(exitCode)
    else:
        return True


def parseConfig(args):
    config = {}
    # Command line flags - default values
    config['label'] = False
    config['args'] = []
    
    # Read in and process the command line flags
    try:
        opts, args = getopt.getopt(args, "hl", ["help", "label"])
    except getopt.GetoptError, err:
        # Print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage(exitCode=2)
        
    # Work through opts
    for opt, value in opts:
        if opt in ('-h', '--help'):
            usage(exitCode=0)
        elif opt in ('-l', '--label'):
            config['label'] = True
        else:
            assert False
            
    # Add in arguments
    config['args'] = args
    
    # Return configuration
    return config


def main(args):
    # Parse the command line
    config = parseConfig(args)
    filename = config['args'][0]
    
    # Parse the correlator configuration
    try:
        ## .npz file
        dataDict = numpy.load(filename)
        cConfig = dataDict['config']
        fh, tempConfig = tempfile.mkstemp(suffix='.txt', prefix='config-')
        fh = open(tempConfig, 'w')
        for line in cConfig:
            fh.write('%s\n' % line)
        fh.close()
        refSrc, filenames, metanames, foffsets, readers, antennas = readCorrelatorConfiguration(tempConfig)
        os.unlink(tempConfig)
        
    except IOError:
        ## Standard .txt file
        refSrc, filenames, metanames, foffsets, readers, antennas = readCorrelatorConfiguration(filename)
        
    # Load in the stand position data and antenna names
    data = []
    names = []
    processed = []
    for ant in antennas:
        if ant.stand.id not in processed:
            data.append( [ant.stand.x, ant.stand.y, ant.stand.z] )
            if ant.stand.id < 51:
                names.append( 'EA%02i' % ant.stand.id )
            else:
                names.append( 'LWA%i' % (ant.stand.id-50) )
            processed.append( ant.stand.id )
    data = numpy.array(data)
    
    # Get the baseline lengths for zenith
    bl = []
    blLength = []
    for i in xrange(data.shape[0]):
        for j in xrange(i+1, data.shape[0]):
            bl.append( (i,j) )
            blLength.append( numpy.sqrt( ((data[i,:]-data[j,:])**2).sum() ) )
    blLength = numpy.array(blLength) / 1e3
    
    # Report
    print "Filename: %s" % os.path.basename(filename)
    print "  Phase Center:"
    print "    Name: %s" % refSrc.name
    print "    RA: %s" % refSrc._ra
    print "    Dec: %s" % refSrc._dec
    print "  Antennas:"
    print "    Total: %i" % len(filenames)
    print "    VDIF: %i" % sum([1 for rdr in readers if rdr is vdif])
    print "    DRX: %i" % sum([1 for rdr in readers if rdr is drx])
    print "  Baselines:"
    print "    Total: %i" % len(bl)
    print "    Minimum: %.2f km (%s <-> %s)" % (blLength.min(), names[bl[numpy.argmin(blLength)][0]], names[bl[numpy.argmin(blLength)][1]])
    print "    Median: %.2f km" % numpy.median(blLength)
    print "    Maximum %.2f km (%s <-> %s)" % (blLength.max(), names[bl[numpy.argmax(blLength)][0]], names[bl[numpy.argmax(blLength)][1]])
    
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
    fig.suptitle("%s - %s" % (os.path.basename(filename), refSrc.name))
    
    ax2.scatter(data[:,0]/1e3, data[:,2], c=color, s=40.0)
    ax2.xaxis.set_major_formatter( NullFormatter() )
    ax2.set_ylabel('$\Delta$Z [m]')
    ax3.scatter(data[:,2], data[:,1]/1e3, c=color, s=40.0)
    ax3.yaxis.set_major_formatter( NullFormatter() )
    ax3.set_xlabel('$\Delta$Z [m]')
    
    # Add in the labels, if requested
    if config['label']:
        for i in xrange(len(names)):
            ax1.annotate(names[i], xy=(data[i,0]/1e3, data[i,1]/1e3))
            
    # Add and elevation colorbar to the right-hand side of the figure
    cb = plt.colorbar(c, cax=ax4, orientation='vertical')
    
    # Set the axis limits
    ax2.set_xlim( ax1.get_xlim() )
    ax3.set_ylim( ax1.get_ylim() )
    
    # Done
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
    