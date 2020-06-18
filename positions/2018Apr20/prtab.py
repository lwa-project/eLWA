#!/usr/bin/env python

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import os
import re
import sys
import numpy


def mindex(array, value):
    found = []
    for i,a in enumerate(array):
        if a == value:
            found.append( i )
    if len(found) == 0:
        array.index(value)
    elif len(found) == 1:
        found = found[0]
    return found


def parse(filename):
    fh = open(filename)
    
    data = {}
    header = []
    reset = False
    inside = False
    for line in fh:
        if len(line) < 3:
            continue
            
        line = line.strip().rstrip()
        if line[:8] == 'COL. NO.':
            if line.startswith('COL. NO.       1'):
                header = []
                data = {}
            reset = True
            continue
            
        if reset and line[:3] == 'ROW':
            fields = re.split(r'   *', line)
            inside = True
            reset = False
            nColumns = len(fields)
            header.extend(fields)
            continue
            
        if inside:
            fields = line.split(None, nColumns-1)
            pfields = []
            for value in fields:
                try:
                    value = int(value, 10)
                    pfields.append( value )
                    continue
                except ValueError:
                    pass
                try:
                    value = float(value)
                    pfields.append( value )
                    continue
                except ValueError:
                    pass
                if value[:4] == 'INDE':
                    value = numpy.nan
                pfields.append( value )
                
            if type(pfields[0]) is not int:
                continue
                
            rowID = pfields[0]
            try:
                data[rowID].extend( pfields )
            except KeyError:
                data[rowID] = pfields
    fh.close()
    
    for rowID in sorted(data.keys()):
        assert( len(data[rowID]) == len(header) )
        
    return header, data


if __name__ == "__main__":
    filename = sys.argv[1]
    header, data = parse(filename)
    print(header)
    print(data[1])
    