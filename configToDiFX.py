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

from utils import read_correlator_configuration
from get_vla_ant_pos import database
from createConfigFile import LWA1_ECEF, LWASV_ECEF, VLA_ECEF


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
    for i,reader,filename in zip(range(nAnt), readers, filenames):
        frames = []
        with open(filename, 'rb') as fh:
            if reader is vdif:
                header = vdif.read_guppi_header(fh)
            for j in range(32):
                frames.append(reader.read_frame(fh))
                
            ## Read in the last few frames to find the end time
            fh.seek(os.path.getsize(filename) - 1024*reader.FRAME_SIZE)
            backed = 0
            while backed < 2*reader.FRAME_SIZE:
                try:
                    reader.read_frame(fh)
                    fh.seek(-reader.FRAME_SIZE, 1)
                    break
                except errors.SyncError:
                    backed += 1
                    fh.seek(-reader.FRAME_SIZE-1, 1)
            for j in range(32):
                try:
                    frames.append(reader.read_frame(fh))
                except errors.SyncError:
                    continue
                    
        if reader is vdif:
            antID = frames[0].id[0] - 12300
            tStart = frames[0].time
            tStop = frames[-1].time
            
            ## Find the antenna location
            pad, edate = db.get_pad('EA%02i' % antID, tStart.datetime)
            x, y, z = db.get_xyz(pad, tStart.datetime)
            xyz = numpy.array([x,y,z])
            xyz += VLA_ECEF
            
            ## Save
            pads.append(pad)
            ecef_xyz.append(xyz)
            
            ## TODO: will we support anything else?
            raise RuntimeError("This only works with LWA-only files for now")
            
        elif reader is drx:
            beam = frames[0].id[0]
            tStart = frames[0].time
            tStop = frames[-1].time
            srate = frames[0].sample_rate
            central_freq1 = [frame.central_freq for frame in frames if frame.id[1] == 1][-1]
            central_freq2 = [frame.central_freq for frame in frames if frame.id[1] == 2][-1]
            
            ## Find the antenna location
            if antennas[2*i].stand.id == 51:
                xyz = LWA1_ECEF
            elif antennas[2*i].stand.id == 52:
                xyz = LWASV_ECEF
                
            ## Save
            pads.append(beam)
            ecef_xyz.append(xyz)
            
    # Convert the start time into something useful
    mjd, mjdf, mjds = tStart.pulsar_mjd
    dt = tStart.datetime
    duration_sec = tStop - tStart
    
    # Adjust the channel count
    nchan = config['channels']
    nchan_ns = round(nchan * (1e9/srate), 4)
    while int(nchan_ns) != nchan_ns or 10000000 % int(nchan_ns) != 0:
        nchan += 1
        nchan_ns = round(nchan * (1e9/srate), 4)
        
    # Set the number of tunings
    central_freqs = (central_freq1, central_freq2)
    ntune = len(central_freqs)
    
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
EXECUTE TIME (SEC): {duration_sec:.0f}
START MJD:          {mjd}
START SECONDS:      {int(mjdf*86400)}
ACTIVE DATASTREAMS: {nAnt}
ACTIVE BASELINES:   {nBL}
VIS BUFFER LENGTH:  80
OUTPUT FORMAT:      SWIN
OUTPUT FILENAME:    {dirname}/{basename}.difx
""" )
    
    fh.write(f"""
# CONFIGURATIONS ###!
NUM CONFIGURATIONS: 1
CONFIG NAME:        lwa_default
INT TIME (SEC):     {config['inttime']:.6f}
SUBINT NANOSECONDS: 10000000
GUARD NANOSECONDS:  4000
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
RULE 0 CONFIG NAME: lwa_default
""")

    fh.write(f"""
# FREQ TABLE #######!
FREQ ENTRIES:       {ntune}""")
    for i,central_freq in enumerate(central_freqs):
        fh.write(f"""
FREQ (MHZ) {i}:       {((central_freq-srate/2)/1e6):.11f}
BW (MHZ) {i}:         {(srate/1e6):.11f}
SIDEBAND {i}:         U
NUM CHANNELS {i}:     {nchan}
CHANS TO AVG {i}:     1
OVERSAMPLE FAC. {i}:  1
DECIMATION FAC. {i}:  1
PHASE CALS {i} OUT:   0""")
    fh.write("\n")
    
    fh.write(f"""
# TELESCOPE TABLE ##!
TELESCOPE ENTRIES:  {nAnt}
""")
    for i,ant in enumerate(antennas[0::2]):
        fh.write(f"TELESCOPE NAME {i}:   LWA{ant.stand.id:03d}\n")
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
DATA FORMAT:        INTERLACEDVDIF/{':'.join([str(v) for v in range(2*ntune)])}
QUANTISATION BITS:  4
DATA FRAME SIZE:    7872
DATA SAMPLING:      COMPLEX_DSB
DATA SOURCE:        FILE
FILTERBANK USED:    FALSE
PHASE CAL INT (MHZ):0
NUM RECORDED FREQS: {ntune}""")
        for j in range(ntune):
            fh.write(f"""
REC FREQ INDEX {j}:   {j}
CLK OFFSET {j} (us):  0.000000
FREQ OFFSET {j} (Hz): 0.000000
NUM REC POLS {j}:     2""")
        for j in range(ntune):
            fh.write(f"""
REC BAND {ntune*j+0} POL:     {"X" if antennas[2*i+0].pol == 0 else "Y"}
REC BAND {ntune*j+0} INDEX:   {j}
REC BAND {ntune*j+1} POL:     {"X" if antennas[2*i+1].pol == 0 else "Y"}
REC BAND {ntune*j+1} INDEX:   {j}""")
        fh.write("""
NUM ZOOM FREQS:     0""")
    fh.write("\n")

    fh.write(f"""
# BASELINE TABLE ###!
BASELINE ENTRIES:   {nBL}""")
    for i,bl in enumerate(bls):
        fh.write(f"""
D/STREAM A INDEX {i}: {bl[0]}
D/STREAM B INDEX {i}: {bl[1]}
NUM FREQS {i}:        {ntune}""")
    for j in range(ntune):
        fh.write(f"""
POL PRODUCTS {i}/{j}:   4""")
        for k in range(4):
            l = k // 2
            m = k % 2
            fh.write(f"""
D/STREAM A BAND {k}:  {ntune*j+antennas[2*bl[0]+l].pol}
D/STREAM B BAND {k}:  {ntune*j+antennas[2*bl[1]+m].pol}""")
    fh.write("\n")
    
    fh.write("""
# DATA TABLE #######!
""")
    for i in range(nAnt):
        fh.write(f"D/STREAM {i} FILES: 1\n")
        fh.write(f"FILE {i}/0:           {filenames[i]}.vdif\n")
    fh.write("\n")
    fh.close()
    clean_lines(difxname)
    
    # .calc file
    difxname = "%s.calc" % (basename,)
    fh = open(difxname, 'w')
    fh.write(f"""JOB ID:             {jobid}
JOB START TIME:     {(mjd+mjdf):.6f}
JOB STOP TIME:      {(mjd+mjdf+3600/86400.):.6f}
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
TELESCOPE {i} NAME:   LWA{ant.stand.id:03d}
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
SCAN 0 DUR (S):     {duration_sec:.0f}
SCAN 0 OBS MODE NAME:lwa
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
FLAG FILENAME:      {dirname}/{basename}.flag
""")
    fh.close()
    clean_lines(difxname)
    
    # .flag file
    difxname = "%s.flag" % (basename,)
    fh = open(difxname, 'w')
    fh.write("0")
    fh.close()


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
    
