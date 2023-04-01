#!/usr/bin/env python3

"""
Copy the FLAG tables from one FITS-IDI file to another.
"""

import os
import git
import sys
import time
import numpy
from astropy.io import fits as astrofits
import argparse


def main(args):
    # Parse the command line
    srcname = args.source_filename
    dstname = args.filename
    
    # Open
    t0 = time.time()
    srclist = astrofits.open(srcname, mode='readonly')
    dstlist = astrofits.open(dstname, mode='readonly')
    
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
        
    # Figure out our revision
    try:
        repo = git.Repo(os.path.dirname(os.path.abspath(__file__)))
        try:
            branch = repo.active_branch.name
            hexsha = repo.active_branch.commit.hexsha
        except TypeError:
            branch = '<detached>'
            hexsha = repo.head.commit.hexsha
        shortsha = hexsha[-7:]
        dirty = ' (dirty)' if repo.is_dirty() else ''
    except git.exc.GitError:
        branch = 'unknown'
        hexsha = 'unknown'
        shortsha = 'unknown'
        dirty = ''
        
    # Insert the new tables right before UV_DATA
    for hdu in toCopy:
        dstlist.insert(-1, hdu)
        dstlist[-2].header['HISTORY'] = 'Flagged with %s, revision %s.%s%s' % (os.path.basename(__file__), branch, shortsha, dirty)
        dstlist[-2].header['HISTORY'] = 'Copied from \'%s\'' % os.path.basename(srcname)
        ## Check to see if we need to scale the channel masks
        if dstlist[1].header['NO_CHAN'] != dstlist[-2].header['NO_CHAN']:
            ### Figure out how to change the channel ranges
            scl = 1.0 * dstlist[1].header['NO_CHAN'] / dstlist[-2].header['NO_CHAN']
            chans = dstlist[-2].data['CHANS']
            chans = chans * scl
            chans = numpy.clip(chans, 1, dstlist[1].header['NO_CHAN'])
            dstlist[-2].data['CHANS'][...] = chans.astype(dstlist[-2].data['CHANS'].dtype)
            dstlist[-2].header['HISTORY'] = 'Scaled channel flag value range from [1, %i] to [1, %i]' % (dstlist[-2].header['NO_CHAN'], dstlist[1].header['NO_CHAN'])
            
    # Save
    print("  Saving to disk")
    ## What to call it
    outname = os.path.basename(dstname)
    outname, outext = os.path.splitext(outname)
    outname = '%s_flagged%s' % (outname, outext)
    ## Does it already exist or not
    if os.path.exists(outname):
        if not args.force:
            yn = input("WARNING: '%s' exists, overwrite? [Y/n] " % outname)
        else:
            yn = 'y'
            
        if yn not in ('n', 'N'):
            os.unlink(outname)
        else:
            raise RuntimeError("Output file '%s' already exists" % outname)
    ## Open and create a new primary HDU
    hdulist2 = astrofits.open(outname, mode='append')
    primary = astrofits.PrimaryHDU()
    processed = []
    for key in dstlist[0].header:
        if key in ('COMMENT', 'HISTORY'):
            if key not in processed:
                parts = str(dstlist[0].header[key]).split('\n')
                for part in parts:
                    primary.header[key] = part
                processed.append(key)
        else:
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
    print("  -> Flagged FITS IDI file is '%s'" % outname)
    print("  Finished in %.3f s" % (time.time()-t0,))


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
    
