#!/usr/bin/env python3

import os
import sys
import ephem

from buildIDI import getSourceName

with open(sys.argv[1], 'r') as fh:
    for line in fh:
        if line.startswith('Pointing at'):
            _, radec = line.rsplit(None, 1)
            ra, dec = radec.split('s', 1)
            for s in ('d', 'h', 'm', 's'):
                ra = ra.replace(s, ':')
                dec = dec.replace(s, ':')
            bdy = ephem.FixedBody()
            bdy._ra = ra
            bdy._dec = dec
            bdy._epoch = ephem.J2000
            print(ra, dec, '->', getSourceName(bdy))
