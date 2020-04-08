#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import numpy
import pyfits


filename1, filename2 = sys.argv[0:2]
hdulist1 = pyfits.open(filename1)
hdulist2 = pyfits.open(filename2)
for hdu1,hdu2 in zip(hdulist1, hdulist2):
    for key in hdu1.header:
        if key in ('DATE-MAP',):
            continue
        assert(hdu1.header[key] == hdu2.header[key])
