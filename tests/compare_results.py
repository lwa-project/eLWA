#!/usr/bin/env python

from __future__ import print_function

import os
import re
import sys
import numpy
import pyfits

_revRE = re.compile('\$Rev.*?\$')

# Open up the files for comparison
filename1, filename2 = sys.argv[1:3]
hdulist1 = pyfits.open(filename1, mode='readonly')
hdulist2 = pyfits.open(filename2, mode='readonly')

# Loop through the HDUs
for hdu1,hdu2 in zip(hdulist1, hdulist2):
    ## Check the header values, modulo the old $Rev$ tag
    print("Checking header for '%s'" % hdu1.name)
    for key in hdu1.header:
        if key in ('DATE-MAP',):
            continue
        h1 = re.sub(_revRE, '', str(hdu1.header[key]))
        h2 = re.sub(_revRE, '', str(hdu2.header[key]))
        same_value = h1 == h2
        print(" %s: %s" % (key, 'OK' if same_value else ("FAILED - '%s' != '%s'" % (h1, h2))))
        assert(same_value)
    print(" ")
    
    ## Check the data values
    if hdu1.name == 'PRIMARY':
        ### The primary header contains no data
        continue
    r = 0
    for row1,row2 in zip(hdu1.data, hdu2.data):
        for f in range(len(row1)):
            try:
                same_value = numpy.allclose(row1[f], row2[f], atol=2e-4)
            except TypeError:
                same_value = numpy.array_equal(row1[f], row2[f])
            if not same_value:
                print("  row %i, field %i (%s): FAILED" % (r, f, hdu1.data.columns[f]))
                print("   - '%s' != '%s'" % (row1[f], row2[f]))
                try:
                    print("   - %s and %s" % (numpy.max(row1[f]-row2[f]), numpy.max(row1[f]-row2[f])/numpy.max(numpy.abs(row2[f]))))
                except TypeError:
                    pass
            assert(same_value)
        r += 1
            
