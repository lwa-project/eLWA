#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import numpy
from astropy.io import fits as astrofits


filename1, filename2 = sys.argv[0:2]
hdulist1 = astrofits.open(filename1, ignore_missing_end=True)
hdulist2 = astrofits.open(filename2, ignore_missing_end=True)
for hdu1,hdu2 in zip(hdulist1, hdulist2):
    for key in hdu1.header:
        print(hdu1.header[key], hdu2.header[key])
