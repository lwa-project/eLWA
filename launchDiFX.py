#!/usr/bin/env python3

import os
import sys
import time
import argparse
import subprocess


def main(args):
    # Parse the command line arguments
    inputfile = args[0]
    machinesfile = inputfile.replace('.input', '.machines')
    
    # Check that DIFX_VERSION is set and that we have everything we need in
    # our path
    if os.getenv('DIFX_VERSION', None) is None:
        pritn("WARNING: DIFX_VERSION is not set")
    for cmd in ('mpirun', 'mpifxcorr', 'difxcalc'):
        status = subprocess.check_call(['which', cmd])
        if status != 0:
            raise RuntimeError("Cannot find '%s' in the current $PATH" % cmd)
            
    # Look for the existance of the .calc and .threads file as well as the
    # output.  If either of the first two are missing, we cannot proceede.
    # If the output exists, we cannot proceed.
    calcfile = None
    corefile = None
    outname = None
    with open(inputfile, 'r') as fh:
        for line in fh:
            if line.find('CALC FILENAME:') != -1:
                calcfile = line.split(':', 1)[1]
                calcfile = calcfile.strip().rstrip()
            elif line.find('CORE CONF FILENAME:') != -1:
                corefile = line.split(':', 1)[1]
                corefile = corefile.strip().rstrip()
            elif line.find('OUTPUT FILENAME:') != -1:
                outname = line.split(':', 1)[1]
                outname = outname.strip().rstrip()
    if not os.path.exists(calcfile):
        raise RuntimeError("Cannot find calc file: '%s'" % calcfile)
    if not os.path.exists(corefile):
        raise RuntimeError("Cannot find core file: '%s'" % corefile)
    if os.path.exists(outname):
        raise RuntimeError("Output file already exists: '%s'" % outname)
        
    # Look for the existance of the .im file.  If it doesn't exist, try to
    # create it by running 'difxcalc'.
    imfile = None
    with open(calcfile, 'r') as fh:
        for line in fh:
            if line.find('IM FILENAME:') != -1:
                imfile = line.split(':', 1)[1]
                imfile = imfile.strip().rstrip()
    if not os.path.exists(imfile):
        subprocess.check_call(['difxcalc', calcfile], cwd=os.path.dirname(imfile))
    if not os.path.exists(imfile):
        raise RuntimeError("Cannot find im file: '%s'" % imfile)
        
    # Build up the 'mpifxcorr' command to run.
    cmd = ['mpirun',]
    if os.path.exists(machinesfile):
        cmd.extend(['-hostfile', machinesfile])
    cmd.extend(['mpifxcorr', inputfile])
    
    # Go!
    t0 = time.time()
    subprocess.check_call(cmd)
    print("Finished in %.3f s" % ((time.time()-t0),)


if __name__ == "__main__":
    main(sys.argv[1:])
    
