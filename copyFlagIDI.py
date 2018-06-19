#!/usr/bin/env python

"""
Copy the FLAG tables from one FITS-IDI file to another

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import sys
import time
import numpy
import getopt
import pyfits


def usage(exitCode=None):
    print """copyFlagIDI.py - Copy the FLAG tables from one FITS-IDI file to another

Usage:
copyFlagIDI.py [OPTIONS] <src_fits_file> <dest_fits_file>

Options:
-h, --help          Display this help information
-f, --force         Force overwriting of existing FITS-IDI file
"""
    
    if exitCode is not None:
        sys.exit(exitCode)
    else:
        return True


def parseConfig(args):
    config = {}
    # Command line flags - default values
    config['force'] = False
    config['args'] = []
    
    # Read in and process the command line flags
    try:
        opts, args = getopt.getopt(args, "hf", ["help", "force"])
    except getopt.GetoptError, err:
        # Print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage(exitCode=2)
        
    # Work through opts
    for opt, value in opts:
        if opt in ('-h', '--help'):
            usage(exitCode=0)
        elif opt in ('-f', '--force'):
            config['force'] = True
        else:
            assert False
            
    # Add in arguments
    config['args'] = args
    
    # Validate
    if len(config['args']) != 2:
        raise RuntimeError("Must provide a flag source and destination filename")
        
    # Return configuration
    return config


def main(args):
    # Parse the command line
    config = parseConfig(args)
    srcname = config['args'][0]
    dstname = config['args'][1]
    
    # Open
    t0 = time.time()
    srclist = pyfits.open(srcname, mode='readonly')
    dstlist = pyfits.open(dstname, mode='readonly')
    
    # Clean up the old FLAG tables, if any, and then insert the new table where it needs to be
    ## Find old tables
    toRemove = []
    for hdu in dstlist:
        try:
            if hdu.header['EXTNAME'] == 'FLAG':
                toRemove.append( hdu )
        except KeyError:
            pass
    ## Remove old tables
    for hdu in toRemove:
        ver = hdu.header['EXTVER']
        del dstlist[dstlist.index(hdu)]
        print "  WARNING: removing old FLAG table - version %i" % ver
    ## Find the new table
    toCopy = []
    for hdu in srclist:
        try:
            if hdu.header['EXTNAME'] == 'FLAG':
                toCopy.append( hdu )
        except KeyError:
            pass
    if len(toCopy) == 0:
        raise RuntimeError("No FLAG tables found in '%s'" % os.path.basename(srcname))
    ## Insert the new table right before UV_DATA
    for hdu in toCopy:
        dstlist.insert(-1, hdu)
        dstlist[-2].header['HISTORY'] = 'Copied from \'%s\'' % os.path.basename(srcname)
        
    # Save
    print "  Saving to disk"
    ## What to call it
    outname = os.path.basename(dstname)
    outname, outext = os.path.splitext(outname)
    outname = '%s_flagged%s' % (outname, outext)
    ## Does it already exist or not
    if os.path.exists(outname):
        if not config['force']:
            yn = raw_input("WARNING: '%s' exists, overwrite? [Y/n] " % outname)
        else:
            yn = 'y'
            
        if yn not in ('n', 'N'):
            os.unlink(outname)
        else:
            raise RuntimeError("Output file '%s' already exists" % outname)
    ## Open and create a new primary HDU
    hdulist2 = pyfits.open(outname, mode='append')
    primary =	pyfits.PrimaryHDU()
    for key in dstlist[0].header:
        primary.header[key] = (dstlist[0].header[key], dstlist[0].header.comments[key])
    hdulist2.append(primary)
    hdulist2.flush()
    ## Copy the extensions over to the new file
    for hdu in dstlist[1:]:
        hdulist2.append(hdu)
        hdulist2.flush()
    hdulist2.close()
    dstlist.close()
    srclist.close()
    print "  -> Flagged FITS IDI file is '%s'" % outname
    print "  Finished in %.3f s" % (time.time()-t0,)


if __name__ == "__main__":
    numpy.seterr(all='ignore')
    main(sys.argv[1:])
    