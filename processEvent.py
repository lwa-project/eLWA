#!/usr/bin/env python

import os
import git
import sys
import copy
import glob
import time
import numpy
import subprocess
from datetime import datetime

import ubjson

from astropy.io import fits as astrofits
from astropy.wcs import WCS as astrowcs
from astropy.coordinates import Angle as AstroAngle

from lsl.misc.dedispersion import delay

from utils import read_correlator_configuration


_BIN_PATH = os.path.dirname(os.path.abspath(__file__))


def _print_command(cmd):
    cmd2 = copy.deepcopy(cmd)
    cmd2[0] = os.path.basename(cmd2[0])
    output = ' '.join(cmd2)
    if len(output) > 75:
        output = output[:75]+'...'
    print("Running '%s'" % output)


def _log_command(fh, cmd):
    # Figure out our revision
    try:
        repo = git.Repo(_BIN_PATH)
        try:
            branch = repo.active_branch.name
            hexsha = repo.active_branch.commit.hexsha
        except TypeError:
            branch = '<detached>'
            hexsha = repo.head.commit.hexsha
        shortsha = hexsha[-7:]
        dirty = ' (dirty)' if repo.is_dirty() else ''
    except git.exc.GitError:
        branch = 'unknown'
        hexsha = 'unknown'
        shortsha = 'unknown'
        dirty = ''
        
    # Write to the logfile
    fh.write('====\n')
    fh.write("%s.%s%s\n" % (branch, shortsha, dirty))
    fh.write("%s\n" % (datetime.utcnow(),))
    fh.write("%s\n" % (' '.join(cmd),))
    fh.write('====\n\n')


def main(args):
    filename = args[0]
    
    # Setup information
    data_path = os.path.dirname(os.path.abspath(filename))
    eventname = os.path.basename(filename)
    eventname = os.path.splitext(eventname)[0]
    timestamp = int(eventname.split('_')[0], 10)
    out_path = eventname
    
    # VDIF names
    vdifsets = []
    vdifnames = []
    for i in range(10):
        file_timestamp = timestamp + i
        newnames = glob.glob(os.path.join(data_path, "*_ea*_%i.vdif" % file_timestamp))
        if len(newnames) == 0:
            break
        vdifsets.append(file_timestamp)
        vdifnames.extend(newnames)
    vdifnames.sort()
    if len(vdifnames) == 0:
        raise RuntimeError("Found no VDIF files associated with this event")
    else:
        print("Found %i VDIF files in %i sets" % (len(vdifnames), len(vdifsets)))
        
    # Calibration names
    calnames = []
    for name in ('CalTab.uvtab', 'Cal.uvtab', 'uvout'):
        newnames = glob.glob(os.path.join(data_path, "*%s*" % name))
        if len(newnames) == 1:
            calnames.append(newnames[0])
    if len(calnames) != 3:
        raise RuntimeError("Found only %i VLITE-Slow calibration files associated with this event" % len(calnames))
    else:
        print("VLITE-Slow calibration files:")
        for calname in calnames:
            print("  %s" % os.path.basename(calname))
            
    # Metadata - for target name, RA, and dec
    metanames = glob.glob(os.path.join(data_path, '*.META'))
    if len(metanames) == 0:
        raise RuntimeError("No VLITE-Slow .META file associated with this event")
    best = 1e9
    finalmeta = None
    for metaname in metanames:
        metastamp = os.path.basename(metaname)
        metastamp = metastamp.split('-', 1)[1]
        metastamp = metastamp.rsplit('-', 1)[0]
        metastamp = datetime.strptime(metastamp, "%Y-%m-%dT%H%M%S")
        metric = metastamp - datetime.utcfromtimestamp(timestamp)
        if abs(metric.total_seconds()) < best:
            best = abs(metric.total_seconds())
            finalmeta = metaname
    metaname = finalmeta
    target, ra, dec = '', '00:00:00', '90:00:00'
    with open(metaname, 'r') as fh:
        for line in fh:
            line = line.strip().rstrip()
            if line.startswith('Source name:'):
                target = line.split(None, 2)[-1]
            elif line.startswith('Delay center RA'):
                ra = line.split(None, 4)[-1]
                ra = ra.replace('h', ':')
                ra = ra.replace('m', ':')
                ra = ra.replace('s', '')
            elif line.startswith('Delay center dec'):
                dec = line.split(None, 4)[-1]
                dec = dec.replace('d', ':')
                dec = dec.replace("'", ':')
                dec = dec.replace('"', '')
    print("Data appears to be for %s at RA %s, dec. %s" % (target, ra, dec))
    
    # Metadata - for event DM and width to get the integration time and
    # channel count
    with open(filename, 'rb') as fh:
        meta = ubjson.load(fh)
    event_dm = meta['dm']
    event_width = meta['width']
    tint = numpy.floor(event_width/0.005)*0.005
    tint = min([0.010, max([0.005, tint])])
    #tint = 0.010
    nchan = 640
    while delay([320e6, 320e6+64e6/nchan], event_dm)[0] > tint/2:
        nchan *= 2
    print("Using %.0f ms integrations and %i channels for event processing" % (tint*1000, nchan))
    
    # Create the output directory for the data
    if os.path.exists(out_path):
        raise RuntimeError("Existing processing directory")
    os.mkdir(out_path)
    
    # Go!
    ## Configuration file
    cmd = [os.path.join(_BIN_PATH, 'createConfigFile.py'), '-o', 'event.config']
    cmd.append("--vlite-target=%s" % target)
    cmd.append("--vlite-ra=%s" % ra)
    cmd.append("--vlite-dec=%s" % dec)
    cmd.extend(vdifnames)
    with open(os.path.join(out_path, 'config.log'), 'w') as lh:
        with open(os.path.join(out_path, 'config.err'), 'w') as eh:
            _print_command(cmd)
            
            _log_command(lh, cmd)
            subprocess.check_call(cmd, stdout=lh, stderr=eh, cwd=out_path)
            
    ## Correlate
    ### Event mode
    cmd = [os.path.join(_BIN_PATH, 'superCorrelator.py'),]
    cmd.extend(['-t', "%.3f" % tint, '-u', "%.3f" % tint])
    cmd.extend(['-l', str(nchan)])
    cmd.extend(['-j', '-g', 'event'])
    cmd.append('event.config')
    with open(os.path.join(out_path, 'correlate.log'), 'w') as lh:
        with open(os.path.join(out_path, 'correlate.err'), 'w') as eh:
            _print_command(cmd)
            
            _log_command(lh, cmd)
            subprocess.check_call(cmd, stdout=lh, stderr=eh, cwd=out_path)
    ### Continuum mode
    cmd = [os.path.join(_BIN_PATH, 'superCorrelator.py'),]
    cmd.extend(['-t', '0.1', '-u', '0.01'])
    cmd.extend(['-l', '640'])
    cmd.extend(['-j', '-g', 'cont'])
    cmd.append('event.config')
    with open(os.path.join(out_path, 'correlate-cont.log'), 'w') as lh:
        with open(os.path.join(out_path, 'correlate-cont.err'), 'w') as eh:
            _print_command(cmd)
            
            _log_command(lh, cmd)
            subprocess.check_call(cmd, stdout=lh, stderr=eh, cwd=out_path)
            
    ## Build
    ### Event mode
    cmd = [os.path.join(_BIN_PATH, 'buildIDI.py'), '-t', 'EVENT']
    cmd.extend(['-r', 'event*.npz'])
    with open(os.path.join(out_path, 'build.log'), 'w') as lh:
        with open(os.path.join(out_path, 'build.err'), 'w') as eh:
            _print_command(cmd)
            
            _log_command(lh, cmd)
            subprocess.check_call(cmd, stdout=lh, stderr=eh, cwd=out_path)
    ### Continuum mode
    cmd = [os.path.join(_BIN_PATH, 'buildIDI.py'), '-t', 'CONT']
    cmd.extend(['-r', 'cont*.npz'])
    with open(os.path.join(out_path, 'build-cont.log'), 'w') as lh:
        with open(os.path.join(out_path, 'build-cont.err'), 'w') as eh:
            _print_command(cmd)
            
            _log_command(lh, cmd)
            subprocess.check_call(cmd, stdout=lh, stderr=eh, cwd=out_path)
            
    ## Calibrate
    ### Event and continuum mode
    cmd = [os.path.join(_BIN_PATH, 'calibrateVLITE.py'), '-f']
    cmd.append(filename)
    cmd.extend(calnames)
    cmd.append('buildIDI_EVENT.FITS_1')
    cmd.append('buildIDI_CONT.FITS_1')
    with open(os.path.join(out_path, 'calibrate.log'), 'w') as lh:
        with open(os.path.join(out_path, 'calibrate.err'), 'w') as eh:
            _print_command(cmd)
            
            _log_command(lh, cmd)
            subprocess.check_call(cmd, stdout=lh, stderr=eh, cwd=out_path)
            
    ## Flag
    ### Event and continuum mode
    cmd = [os.path.join(_BIN_PATH, 'flagIDI.py'), '-f']
    cmd.extend(['-r', '320-321.8,361-384'])
    cmd.append('buildIDI_EVENT_cal.FITS_1')
    cmd.append('buildIDI_CONT_cal.FITS_1')
    with open(os.path.join(out_path, 'flag.log'), 'w') as lh:
        with open(os.path.join(out_path, 'flag.err'), 'w') as eh:
            _print_command(cmd)
            
            _log_command(lh, cmd)
            subprocess.check_call(cmd, stdout=lh, stderr=eh, cwd=out_path)
            
    ## Flag output check for sanity
    with open(os.path.join(out_path, 'flag.log'), 'r') as fh:
        for line in fh:
            if line.find('Working on scan 1 of') != -1:
                _, max_scans = line.rsplit('of', 1)
                max_scans = int(max_scans, 10)
                if max_scans > 1:
                    print("WARNING: Data purports to contain %i scans instead of 1" % max_scans)
                    break
                    
    ## Dedisperse
    ### Event mode only
    cmd = [os.path.join(_BIN_PATH, 'dedisperseIDI.py'), '-f']
    cmd.append(filename)
    cmd.append('buildIDI_EVENT_cal_flagged.FITS_1')
    with open(os.path.join(out_path, 'dedisperse.log'), 'w') as lh:
        with open(os.path.join(out_path, 'dedisperse.err'), 'w') as eh:
            _print_command(cmd)
            
            _log_command(lh, cmd)
            subprocess.check_call(cmd, stdout=lh, stderr=eh, cwd=out_path)
            
    ## Extract
    ### Event mode only
    cmd = [os.path.join(_BIN_PATH, 'extractIDI.py'), '-f']
    cmd.append(filename)
    cmd.append("buildIDI_EVENT_cal_flagged_DM%.4f.FITS_1" % event_dm)
    with open(os.path.join(out_path, 'extract.log'), 'w') as lh:
        with open(os.path.join(out_path, 'extract.err'), 'w') as eh:
            _print_command(cmd)
            
            _log_command(lh, cmd)
            subprocess.check_call(cmd, stdout=lh, stderr=eh, cwd=out_path)
            
    ## Link to a common name
    ## Event mode
    fullname = "buildIDI_EVENT_cal_flagged_DM%.4f_extracted.FITS_1" % event_dm
    linkname = 'buildIDI_EVENT_final.FITS_1'
    subprocess.check_call(['ln', fullname, linkname], cwd=out_path)
    ### Continuum mode
    fullname = 'buildIDI_CONT_cal_flagged.FITS_1'
    linkname = 'buildIDI_CONT_final.FITS_1'
    subprocess.check_call(['ln', fullname, linkname], cwd=out_path)
    
    ## Convert to a measurement set
    cmd = ['/usr/local/casa/bin/casa', '--log2term', '--nologger', '--nogui']
    cmd.extend(['-c', "importfitsidi('buildIDI_EVENT_final.FITS_1', 'event.ms')"])
    with open(os.path.join(out_path, 'ms.log'), 'w') as lh:
        with open(os.path.join(out_path, 'ms.err'), 'w') as eh:
            _print_command(cmd)
            
            _log_command(lh, cmd)
            subprocess.check_call(cmd, stdout=lh, stderr=eh, cwd=out_path)
            
    ## Image parameter determination
    _, _, _, _, _, _, antennas = read_correlator_configuration(os.path.join(out_path, 'event.config'))
    blMax = -1.0
    for ant1 in antennas:
        for ant2 in antennas:
            bl = numpy.sqrt((ant1.stand.x-ant2.stand.x)**2 + (ant1.stand.y-ant2.stand.y)**2)
            if bl > blMax:
                blMax = bl
    pixel_size = round(3600 * 180 / numpy.pi * 0.83 / blMax / 4.5, 2) # asec
    image_size = 3600 * 180 / numpy.pi * 0.94 / 25 / pixel_size
    image_size = int(numpy.ceil(image_size / 100) * 100)
    
    ## Image
    cmd = ['wsclean', '-size', str(image_size), str(image_size), '-scale', "%.2fasec" % pixel_size]
    cmd.extend(['-weight', 'natural', '-niter', '100'])
    cmd.extend(['-channel-range', "%i" % (20*nchan//640), "%i" % (400*nchan//640)])
    cmd.extend(['-no-update-model-required', 'event.ms'])
    with open(os.path.join(out_path, 'wsclean.log'), 'w') as lh:
        with open(os.path.join(out_path, 'wsclean.err'), 'w') as eh:
            _print_command(cmd)
            
            _log_command(lh, cmd)
            subprocess.check_call(cmd, stdout=lh, stderr=eh, cwd=out_path)
            
    ## Report
    for name in ('image', 'dirty'):
        hdulist = astrofits.open(os.path.join(out_path, "wsclean-%s.fits" % name))
        wcs = astrowcs(hdulist[0].header)
        image = hdulist[0].data[0,0,:,:].T
        imax = image.max()
        istd = image.std()
        peak = numpy.where(image == imax)
        px0, py0 = peak[0][0], peak[1][0]
        try:
            sky, _, _ = wcs.pixel_to_world(px0, py0, 0, 0)
            sky_ra = sky.ra.to('hourangle')
            sky_dec = sky.dec
        except AttributeError:
            sky_ra, sky_dec, _, _ = wcs.wcs_pix2world(px0, py0, 0, 0, 0)
            sky_ra %= 360
            sky_ra /= 15
            sky_ra = AstroAngle(str(sky_ra), unit='deg')
            sky_dec = AstroAngle(str(sky_dec), unit='deg')
        hdulist.close()
        print("%s:" % (name.capitalize(),))
        print("  Peak: %f" % imax)
        print("  X: %i" % px0)
        print("  Y: %i" % py0)
        print("  RA: %s" % (str(sky_ra),))
        print("  Dec: %s" % (str(sky_dec),))
        print("  Std. Dev: %f" % istd)
        print("  SNR: %f" % (imax/istd,))
        with open(os.path.join(out_path, "results-%s.log" % name), 'w') as lh:
            lh.write("Peak: %f\n" % imax)
            lh.write("X: %i\n" % px0)
            lh.write("Y: %i\n" % py0)
            lh.write("RA: %s\n" % (str(sky_ra),))
            lh.write("Dec: %s\n" % (str(sky_dec),))
            lh.write("Std. Dev: %f\n" % istd)
            lh.write("SNR: %f\n" % (imax/istd,))
        


if __name__ == '__main__':
    main(sys.argv[1:])
