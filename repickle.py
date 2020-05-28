#!/usr/bin/env python3

"""
Simple script to take a Python protocol 3+ pickle and convert it to a 
protocol 2 pickle that can be read by Python2.
"""

import os
import sys
import pickle

filename = sys.argv[1]
outname = os.path.basename(filename)
outname = os.path.splitext(outname)[0]
outname = outname+'.repickle'

with open(filename, 'rb') as fh:
    data = pickle.load(fh)
with open(outname, 'wb') as fh:
    pickle.dump(data, fh, protocol=2)
