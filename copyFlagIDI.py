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
import pyfits
import argparse


def main(args):
    # Parse the command line
    srcname = args.source_filename
    dstname = args.filename
    
    # Open
    t0 = time.time()
    srclist = pyfits.open(srcname, mode='readonly')
    dstlist = pyfits.open(dstname, mode='readonly')
    
    # Find the FLAG tables to be copied
    toCopy = []
    for hdu in srclist:
        try:
            if hdu.header['EXTNAME'] == 'FLAG':
                toCopy.append( hdu )
        except KeyError:
            pass
    if len(toCopy) == 0:
        raise RuntimeError("No FLAG tables found in '%s'" % os.path.basename(srcname))
        
    # Insert the new tables right before UV_DATA
    for hdu in toCopy:
        dstlist.insert(-1, hdu)
        dstlist[-2].header['HISTORY'] = 'Flagged with %s, revision $Rev$' % os.path.basename(__file__)
        dstlist[-2].header['HISTORY'] = 'Copied from \'%s\'' % os.path.basename(srcname)
        
    # Save
    print "  Saving to disk"
    ## What to call it
    outname = os.path.basename(dstname)
    outname, outext = os.path.splitext(outname)
    outname = '%s_flagged%s' % (outname, outext)
    ## Does it already exist or not
    if os.path.exists(outname):
        if not args.force:
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
    processed = []
    for key in hdulist[0].header:
        if key in ('COMMENT', 'HISTORY'):
            if key not in processed:
                parts = str(hdulist[0].header[key]).split('\n')
                for part in parts:
                    primary.header[key] = part
                processed.append(key)
        else:
            primary.header[key] = (hdulist[0].header[key], hdulist[0].header.comments[key])
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
    parser = argparse.ArgumentParser(
        description='copy the FLAG tables from one FITS-IDI file to another', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('source_filename', type=str, 
                        help='filename to copy the FLAG tables from')
    parser.add_argument('filename', type=str, 
                        help='filename to add the FLAG tables to')
    parser.add_argument('-f', '--force', action='store_true', 
                        help='force overwriting of existing FITS-IDI files')
    args = parser.parse_args()
    main(args)
    