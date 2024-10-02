#!/usr/bin/env python3

"""
Given a collection of .npz files search for course delays and rates.
"""

import os
import sys
import numpy as np
from astropy.io import fits as astrofits
import argparse
from datetime import datetime


def main(args):
    # Go!
    for filename in args.filename:
        print(f"Working on '{os.path.basename(filename)}")
        # Open the FITS IDI file and access the UV_DATA extension
        hdulist = astrofits.open(filename, mode='readonly')
        andata = hdulist['ANTENNA']
        fqdata = hdulist['FREQUENCY']
        fgdata = None
        for hdu in hdulist[1:]:
                if hdu.header['EXTNAME'] == 'FLAG':
                    fgdata = hdu
        uvdata = hdulist['UV_DATA']
        
        # Pull out various bits of information we need to flag the file
        ## Antenna look-up table
        antLookup = {}
        antLookup_inv = {}
        for an, ai in zip(andata.data['ANNAME'], andata.data['ANTENNA_NO']):
            antLookup[an] = ai
            antLookup_inv[ai] = an
        ## Baseline list
        bls = uvdata.data['BASELINE']
        bls_i = (bls >> 8) & 0xFF
        bls_j = bls & 0xFF
        ## Time of each integration
        obsdates = uvdata.data['DATE']
        obstimes = uvdata.data['TIME']
        inttimes = uvdata.data['INTTIM']
        
        # Find unique times to work with
        utimes = np.unique(obstimes)
        
        for ai in sorted(list(antLookup_inv.keys())):
            an = antLookup_inv[ai]
            
            print(f"Antenna {ai}: {an}")
            nint = len(np.where((bls_i == ai) & (bls_j == ai))[0])
            print(f"  Found auto-correlations in {nint} integrations ({nint/len(utimes):.1%})")
            
        hdulist.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='print the antenna table in a FITS-IDI file', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('filename', type=str, nargs='+',
                        help='filename to search')
    args = parser.parse_args()
    main(args)
