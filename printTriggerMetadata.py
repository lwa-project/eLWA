#!/usr/bin/env python3

import os
import sys
import ubjson
from datetime import datetime


def main(args):
    for filename in args:
        with open(filename, 'rb') as fh:
            meta = ubjson.load(fh)
            
        print("Filename: %s" % os.path.basename(filename))
        print("  %s" % (meta['meta'].decode(),))
        print("  Peak time: %s" % (datetime.utcfromtimestamp(meta['t0']),))
        print("  Width: %.1f ms" % (meta['width']*1000,))
        print("  DM: %.3f pc/cm^3" % meta['dm'])
        print("  S/N: %.1f" % meta['sn'])


if __name__ == '__main__':
    main(sys.argv[1:])

