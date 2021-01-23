#!/usr/bin/env python3

"""
Take a superCorrelator.py configuration file and convert it into a .input and
.calc file that are suitable for DiFX.
"""

import os
import sys
import numpy
import argparse
import subprocess

from astropy.utils import iers
from astropy.time import Time as AstroTime

from lsl import astro
from lsl.reader import drx, vdif

from utils import read_correlator_configuration, is_vlite_vdif
from get_vla_ant_pos import database
from createConfigFile import LWA1_ECEF, VLA_ECEF


def smart_int(s):
    i = 0
    v = None
    while i < len(s):
        try:
            v = int(s[i:], 10)
            break
        except ValueError:
            pass
        i += 1
    if v is None:
        raise ValueError("Cannot convert '%s' to int" % s)
    return v


def clean_lines(filename):
    cleaned = ''
    with open(filename, 'r') as fh:
        for line in fh:
            if line.find(':') == -1:
                cleaned += line
            else:
                key, value = line.rsplit(None, 1)
                if key.find(':') != -1:
                    cleaned += f"{key:20s}{value}\n"
                else:
                    cleaned += f"{key} {value}\n"
                    
    with open(filename, 'w') as fh:
        fh.write(cleaned)


def main(args):
    # Parse the command line
    configname = args.filename
    
    # Open the database connection to NRAO to find the antenna locations
    try:
        db = database('params')
    except Exception as e:
        sys.stderr.write("WARNING: %s" % str(e))
        sys.stderr.flush()
        db = None
        
    # Load in the configuration file and make sure that this is VLITE-only
    config, refSrc, filenames, metanames, foffsets, readers, antennas = read_correlator_configuration(configname)
    
    # Determine the number of antennas, the number of baselines, and what the
    # (non-auto) baselines are
    nAnt = len(filenames)
    nBL = nAnt*(nAnt-1)//2
    bls = []
    for i in range(nAnt):
        for j in range(i, nAnt):
            if j == i:
                continue
            bls.append((i,j))
            
    # Determine the ECEF antenna positions and the start time
    pads = []
    ecef_xyz = []
    for reader,filename in zip(readers, filenames):
        with open(filename, 'rb') as fh:
            if reader is vdif:
                is_vlite = is_vlite_vdif(fh)
            else:
                ## TODO: will we support anything else?
                raise RuntimeError("This only works with VLITE-Fast files for now")
            frame = reader.read_frame(fh)
            
        if reader is vdif:
            antID = frame.id[0] - (0 if is_vlite else 12300)
            tStart =  frame.time
            
            ## Find the antenna location
            pad, edate = db.get_pad('EA%02i' % antID, tStart.datetime)
            x, y, z = db.get_xyz(pad, tStart.datetime)
            xyz = numpy.array([x,y,z])
            xyz += VLA_ECEF
            
            ## Save
            pads.append(pad)
            ecef_xyz.append(xyz)
            
        elif reader is drx:
            ## Assume LWA1 for now
            ecef_xyz.append(LWA1_ECEF)
            
            ## TODO: Will we support anything else?
            raise RuntimeError("This only works with VLITE-Fast files for now")
            
    # Convert the start time into something useful
    mjd, mjdf, mjds = tStart.pulsar_mjd
    dt = tStart.datetime
    
    # Setup the output path/basename
    dirname = args.output_dir
    dirname = os.path.abspath(dirname)
    basename = os.path.basename(configname)
    basename, ext = os.path.splitext(basename)
    try:
        jobid = smart_int(ext)
    except ValueError:
        jobid = 1
        
    # Find the DiFX version
    difx_version = os.getenv('DIFX_VERSION', None)
    if difx_version is None:
        difx_version = 'DiFX-2.6.2'
        print("WARNING: DIFX_VERSION is not set, assuming 'DiFX-2.6.2'")
        
    # .machines file
    difxname = "%s.machines" % (basename,)
    fh = open(difxname, 'w')
    for i in range(nAnt+2+1):
        fh.write("localhost\n")
    fh.close()
    
    # .threads file
    difxname = "%s.threads" % (basename,)
    fh = open(difxname, 'w')
    fh.write("""NUMBER OF CORES:    2
2
2""")
    fh.close()
    
    # .input file
    difxname = "%s.input" % (basename,)
    fh = open(difxname, 'w')
    fh.write(f"""# COMMON SETTINGS ##!
CALC FILENAME:      {dirname}/{basename}.calc
CORE CONF FILENAME: {dirname}/{basename}.threads
EXECUTE TIME (SEC): 3600
START MJD:          {mjd}
START SECONDS:      {int(mjdf*86400)}
ACTIVE DATASTREAMS: {nAnt}
ACTIVE BASELINES:   {nBL}
VIS BUFFER LENGTH:  80
OUTPUT FORMAT:      SWIN
OUTPUT FILENAME:    {dirname}/{basename}.difx
""" )
    
    fh.write("""
# CONFIGURATIONS ###!
NUM CONFIGURATIONS: 1
CONFIG NAME:        pband_default
INT TIME (SEC):     0.010000
SUBINT NANOSECONDS: 10000000
GUARD NANOSECONDS:  400
FRINGE ROTN ORDER:  1
ARRAY STRIDE LENGTH:0
XMAC STRIDE LENGTH: 0
NUM BUFFERED FFTS:  10
WRITE AUTOCORRS:    TRUE
PULSAR BINNING:     FALSE
PHASED ARRAY:       FALSE
""")
    for i in range(nAnt):
        fh.write(f"DATASTREAM {i} INDEX: {i:2d}\n")
    for i in range(nBL):
        fh.write(f"BASELINE {i} INDEX: {i:3d}\n")
        
    fh.write("""
# RULES ############!
NUM RULES:          1
RULE 0 CONFIG NAME: pband_default
""")

    fh.write("""
# FREQ TABLE #######!
FREQ ENTRIES:       1
FREQ (MHZ) 0:       384.00000000000
BW (MHZ) 0:         64.00000000000
SIDEBAND 0:         L
NUM CHANNELS 0:     6400
CHANS TO AVG 0:     1
OVERSAMPLE FAC. 0:  1
DECIMATION FAC. 0:  1
PHASE CALS 0 OUT:   0
""")
    
    fh.write(f"""
# TELESCOPE TABLE ##!
TELESCOPE ENTRIES:  {nAnt}
""")
    for i,ant in enumerate(antennas[0::2]):
        fh.write(f"TELESCOPE NAME {i}:   EA{ant.stand.id:02d}\n")
        fh.write(f"""CLOCK REF MJD {i}:    50000.0000000000
CLOCK POLY ORDER {i}: 1
@ ***** Clock poly coeff N: has units microsec / sec^N ***** @
CLOCK COEFF {i}/0:    {(-ant.cable.clock_offset*1e6):.15e}
CLOCK COEFF {i}/1:    {0:.15e}
""")
    
    fh.write(f"""
# DATASTREAM TABLE #!
DATASTREAM ENTRIES: {nAnt}
DATA BUFFER FACTOR: 32
NUM DATA SEGMENTS:  8""")
    for i in range(nAnt):
        fh.write(f"""
TELESCOPE INDEX:   {i:2d}
TSYS:               0.000000
DATA FORMAT:        INTERLACEDVDIF/0:1
QUANTISATION BITS:  8
DATA FRAME SIZE:    5032
DATA SAMPLING:      REAL
DATA SOURCE:        FILE
FILTERBANK USED:    FALSE
PHASE CAL INT (MHZ):0
NUM RECORDED FREQS: 1
REC FREQ INDEX 0:   0
CLK OFFSET 0 (us):  0.000000
FREQ OFFSET 0 (Hz): 0.000000
NUM REC POLS 0:     2
REC BAND 0 POL:     {"X" if antennas[2*i+0].pol == 0 else "Y"}
REC BAND 0 INDEX:   0
REC BAND 1 POL:     {"X" if antennas[2*i+1].pol == 0 else "Y"}
REC BAND 1 INDEX:   0
NUM ZOOM FREQS:     0""")
    fh.write("\n")

    fh.write(f"""
# BASELINE TABLE ###!
BASELINE ENTRIES:   {nBL}""")
    for i,bl in enumerate(bls):
        fh.write(f"""
D/STREAM A INDEX {i}: {bl[0]}
D/STREAM B INDEX {i}: {bl[1]}
NUM FREQS {i}:        1
POL PRODUCTS {i}/0:   4
D/STREAM A BAND 0:  1
D/STREAM B BAND 0:  1
D/STREAM A BAND 1:  1
D/STREAM B BAND 1:  0
D/STREAM A BAND 2:  0
D/STREAM B BAND 2:  1
D/STREAM A BAND 3:  0
D/STREAM B BAND 3:  0""")
    fh.write("\n")
    
    fh.write("""
# DATA TABLE #######!
""")
    for i in range(nAnt):
        subnames = [filenames[i],]
        while True:
            oldname = subnames[-1]
            nextname, ext = os.path.splitext(oldname)
            base, vfts = nextname.rsplit('_', 1)
            vfts = int(vfts, 10) + 1
            nextname = "%s_%s%s" % (base, vfts, ext)
            if not os.path.exists(nextname):
                break
            subnames.append(nextname)
            
        fh.write(f"D/STREAM {i} FILES: {(len(subnames)):3d}\n")
        for j,filename in enumerate(subnames):
            fh.write(f"FILE {i}/{j}:           {filename}\n")
    fh.write("\n")
    fh.close()
    clean_lines(difxname)
    
    # .calc file
    difxname = "%s.calc" % (basename,)
    fh = open(difxname, 'w')
    fh.write(f"""JOB ID:             {jobid}
JOB START TIME:     {(mjd+mjdf):.6f}
JOB STOP TIME:      {(mjd+mjdf+1.0*len(subnames)/86400.):.6f}
DUTY CYCLE:         1.000000
OBSCODE:            {basename.upper()}
DIFX VERSION:       {difx_version}
DIFX LABEL:         {difx_version}
SUBJOB ID:          0
SUBARRAY ID:        0
START MJD:          {(mjd+mjdf):.6f}
START YEAR:         {dt.year}
START MONTH:        {dt.month}
START DAY:          {dt.day}
START HOUR:         {dt.hour}
START MINUTE:       {dt.minute}
START SECOND:       {dt.second}
SPECTRAL AVG:       1
TAPER FUNCTION:     UNIFORM""")
    
    fh.write(f"""
NUM TELESCOPES:     {nAnt}""")
    for i,ant,pad,xyz in zip(range(nAnt), antennas[::2], pads, ecef_xyz):
        fh.write(f"""
TELESCOPE {i} NAME:   EA{ant.stand.id:02d}
TELESCOPE {i} MOUNT:  AZEL
TELESCOPE {i} OFFSET (m):{0:f}
TELESCOPE {i} X (m):  {xyz[0]:f}
TELESCOPE {i} Y (m):  {xyz[1]:f}
TELESCOPE {i} Z (m):  {xyz[2]:f}
TELESCOPE {i} SHELF:  NONE""")
        
    fh.write(f"""
NUM SOURCES:        1
SOURCE 0 NAME:      {refSrc.name}
SOURCE 0 RA:        {refSrc._ra:.9f}
SOURCE 0 DEC:       {refSrc._dec:.9f}
SOURCE 0 CALCODE:   {refSrc.intent}
SOURCE 0 QUAL:      0""")
    
    fh.write(f"""
NUM SCANS:          1
SCAN 0 IDENTIFIER:  No{jobid:04d}
SCAN 0 START (S):   0
SCAN 0 DUR (S):     {(1.0*len(subnames)):.0f}
SCAN 0 OBS MODE NAME:pband
SCAN 0 UVSHIFT INTERVAL (NS):{(0.01*1e9):.0f}
SCAN 0 AC AVG INTERVAL (NS):{(0.01*1e9):.0f}
SCAN 0 POINTING SRC:0
SCAN 0 NUM PHS CTRS:1
SCAN 0 PHS CTR 0:   0""")
    
    fh.write(f"""
NUM EOPS:           5""")
    eop = iers.IERS_Auto.open()
    for day,emjd in enumerate((int(mjd)-2, int(mjd)-1, int(mjd), int(mjd)+1, int(mjd)+2)):
        eat = AstroTime(emjd, format='mjd', scale='utc')
        tai_utc = astro.leap_secs(eat.jd)
        ut1_utc = eop.ut1_utc(eat)
        pm_xy = eop.pm_xy(eat)
        fh.write(f"""
EOP {day} TIME (mjd):   {emjd}
EOP {day} TAI_UTC (sec):{tai_utc}
EOP {day} UT1_UTC (sec):{ut1_utc.to('s').value}
EOP {day} XPOLE (arcsec):{pm_xy[0].to('arcsec').value}
EOP {day} YPOLE (arcsec):{pm_xy[1].to('arcsec').value}""")
    fh.write(f"""
NUM SPACECRAFT:     0
IM FILENAME:        {dirname}/{basename}.im
""")
    fh.close()
    clean_lines(difxname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='take a superCorrelator.py configuration file and convert it into a .input and .calc file that are suitable for DiFX',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, 
                        help='file or directory name to process')
    parser.add_argument('-o', '--output-dir', type=str, default='.',
                        help='output directory for DiFX config. files')
    args = parser.parse_args()
    main(args)
    
