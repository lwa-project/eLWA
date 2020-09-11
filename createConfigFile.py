#!/usr/bin/env python

"""
Simple script to take in a collection of observation files and build up a 
superCorrelator.py configuration script.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import os
import re
import git
import sys
import glob
import ephem
import numpy
import argparse
from collections import OrderedDict
from datetime import datetime, timedelta

from lsl.reader import drx, vdif, errors
from lsl.common import metabundle, metabundleADP
from lsl.common.mcs import mjdmpm_to_datetime, datetime_to_mjdmpm

import guppi
from utils import *
from get_vla_ant_pos import database


VLA_ECEF = numpy.array((-1601185.4, -5041977.5, 3554875.9)) 


## Derived from the 2018 Feb 28 observations of 3C295 and Virgo A
## with LWA1 and EA03/EA01
LWA1_ECEF = numpy.array((-1602235.14380825, -5042302.73757814, 3553980.03506238))
LWA1_LAT =   34.068956328 * numpy.pi/180
LWA1_LON = -107.628103026 * numpy.pi/180
LWA1_ROT = numpy.array([[ numpy.sin(LWA1_LAT)*numpy.cos(LWA1_LON), numpy.sin(LWA1_LAT)*numpy.sin(LWA1_LON), -numpy.cos(LWA1_LAT)], 
                        [-numpy.sin(LWA1_LON),                     numpy.cos(LWA1_LON),                      0                  ],
                        [ numpy.cos(LWA1_LAT)*numpy.cos(LWA1_LON), numpy.cos(LWA1_LAT)*numpy.sin(LWA1_LON),  numpy.sin(LWA1_LAT)]])

## Derived from the 2018 Feb 23 observations of 3C295 and 3C286
## with LWA1 and LWA-SV.  This also includes the shift detailed
## above for LWA1
LWASV_ECEF = numpy.array((-1531556.98709475, -5045435.8720832, 3579254.27947458))
LWASV_LAT =   34.34841153053564 * numpy.pi/180
LWASV_LON = -106.88582216960029 * numpy.pi/180
LWASV_ROT = numpy.array([[ numpy.sin(LWASV_LAT)*numpy.cos(LWASV_LON), numpy.sin(LWASV_LAT)*numpy.sin(LWASV_LON), -numpy.cos(LWASV_LAT)], 
                         [-numpy.sin(LWASV_LON),                      numpy.cos(LWASV_LON),                       0                   ],
                         [ numpy.cos(LWASV_LAT)*numpy.cos(LWASV_LON), numpy.cos(LWASV_LAT)*numpy.sin(LWASV_LON),  numpy.sin(LWASV_LAT)]])


## Correlator configuration regexs
CORR_CHANNELS = re.compile('corrchannels:(?P<channels>\d+)')
CORR_INTTIME = re.compile('corrinttime:(?P<inttime>\d+(.\d*)?)')
CORR_BASIS = re.compile('corrbasis:(?P<basis>(linear)|(circular)|(stokes))')


## Alternate phase center regexs
ALT_TARGET = re.compile('alttarget(?P<id>\d+):(?P<target>.*);;')
ALT_INTENT = re.compile('altintent(?P<id>\d+):(?P<intent>.*);;')
ALT_RA = re.compile('altra(?P<id>\d+):(?P<ra>\d+(.\d*)?)')
ALT_DEC = re.compile('altdec(?P<id>\d+):(?P<dec>[-+]?\d+(.\d*)?)')


def main(args):
    # Parse the command line
    filenames = args.filename
    
    # Check if the first argument on the command line is a directory.  If so, 
    # use what is in that directory
    if os.path.isdir(filenames[0]):
        filenames = [os.path.join(filenames[0], filename) for filename in os.listdir(filenames[0])]
        filenames.sort()
        
    # Convert the filenames to absolute paths
    filenames = [os.path.abspath(filename) for filename in filenames]
        
    # Open the database connection to NRAO to find the antenna locations
    try:
        db = database('params')
    except Exception as e:
        sys.stderr.write("WARNING: %s" % str(e))
        sys.stderr.flush()
        db = None
        
    # Pass 1 - Get the LWA metadata so we know where we are pointed
    context = {'observer':'Unknown', 'project':'Unknown', 'session':None, 'vlaref':None}
    setup = None
    sources = []
    metadata = {}
    lwasite = {}
    for filename in filenames:
        # Figure out what to do with the file
        ext = os.path.splitext(filename)[1]
        if ext == '.tgz':
            ## LWA Metadata
            try:
                ## Extract the SDF
                if len(sources) == 0:
                    try:
                        sdf = metabundle.get_sdf(filename)
                    except Exception as e:
                        sdf = metabundleADP.get_sdf(filename)
                        
                    context['observer'] = sdf.observer.name
                    context['project'] = sdf.id
                    context['session'] = sdf.sessions[0].id
                    
                    comments = sdf.project_office.sessions[0]
                    mtch = CORR_CHANNELS.search(comments)
                    if mtch is not None:
                        corr_channels = int(mtch.group('channels'), 10)
                    else:
                        corr_channels = None
                    mtch = CORR_INTTIME.search(comments)
                    if mtch is not None:
                        corr_inttime = float(mtch.group('inttime'))
                    else:
                        corr_inttime = None
                    mtch = CORR_BASIS.search(comments)
                    if mtch is not None:
                        corr_basis = mtch.group('basis')
                    else:
                        sys.stderr.write("WARNING: No output correlation polarization basis defined, assuming 'linear'.\n")
                        corr_basis = 'linear'
                    if corr_channels is not None and corr_inttime is not None:
                        setup = {'channels': corr_channels, 'inttime': corr_inttime, 'basis': corr_basis}
                    else:
                        sys.stderr.write("WARNING: No or incomplete correlation configuration defined, setting to be defined at correlation time.\n")
                        
                    for o,obs in enumerate(sdf.sessions[0].observations):
                        if type(obs).__name__ == 'Solar':
                            name = 'Sun'
                            intent = 'target'
                            ra = None
                            dec = None
                        elif type(obs).__name__ == 'Jovian':
                            name = 'Jupiter'
                            intent = 'target'
                            ra = None
                            dec = None
                        else:
                            name = obs.target
                            intent = obs.name
                            ra = ephem.hours(str(obs.ra))
                            dec = ephem.degrees(str(obs.dec))
                        tStart = mjdmpm_to_datetime(obs.mjd, obs.mpm)
                        tStop  = mjdmpm_to_datetime(obs.mjd, obs.mpm+obs.dur)
                        sources.append( {'name':name, 'intent':intent, 'ra2000':ra, 'dec2000':dec, 'start':tStart, 'stop':tStop} )
                        
                        ### Alternate phase centers
                        comments = sdf.project_office.observations[0][o]
                        
                        alts = {}
                        for mtch in ALT_TARGET.finditer(comments):
                            alt_id = int(mtch.group('id'), 10)
                            alt_name = mtch.group('target')
                            try:
                                alts[alt_id]['name'] = alt_name
                            except KeyError:
                                alts[alt_id] = {'name':alt_name, 'intent':'dummy', 'ra':None, 'dec':None}
                        for mtch in ALT_INTENT.finditer(comments):
                            alt_id = int(mtch.group('id'), 10)
                            alt_intent = mtch.group('intent')
                            try:
                                alts[alt_id]['intent'] = alt_intent
                            except KeyError:
                                alts[alt_id] = {'name':None, 'intent':alt_intent, 'ra':None, 'dec':None}
                        for mtch in ALT_RA.finditer(comments):
                            alt_id = int(mtch.group('id'), 10)
                            alt_ra = ephem.hours(mtch.group('ra'))
                            try:
                                alts[alt_id]['ra'] = alt_ra
                            except KeyError:
                                alts[alt_id] = {'name':None, 'intent':'dummy', 'ra':alt_ra, 'dec':None}
                        for mtch in ALT_DEC.finditer(comments):
                            alt_id = int(mtch.group('id'), 10)
                            alt_dec = ephem.degrees(mtch.group('dec'))
                            try:
                                alts[alt_id]['dec'] = alt_dec
                            except KeyError:
                                alts[alt_id] = {'name':None, 'intent':'dummy', 'ra':None, 'dec':alt_dec}
                        for alt_id in sorted(alts.keys()):
                            alt_name, alt_ra, alt_dec = alts[alt_id]
                            if alt_name is None or alt_ra is None or alt_dec is None:
                                sys.stderr.write("WARNING: Incomplete alternate phase center %i, skipping.\n" % alt_id)
                            else:
                                sources.append( {'name':alt_name, 'ra2000':alt_ra, 'dec2000':alt_dec, 'start':tStart, 'stop':tStop} )
                                
                ## Extract the file information so that we can pair things together
                fileInfo = metabundle.get_session_metadata(filename)
                for obsID in fileInfo.keys():
                    metadata[fileInfo[obsID]['tag']] = filename
                    
                ## Figure out LWA1 vs LWA-SV
                try:
                    cs = metabundle.get_command_script(filename)
                    for c in cs:
                        if c['subsystemID'] == 'DP':
                            site = 'LWA1'
                            break
                        elif c['subsystemID'] == 'ADP':
                            site = 'LWA-SV'
                            break
                except (RuntimeError, ValueError):
                    site = 'LWA-SV'
                for obsID in fileInfo.keys():
                    lwasite[fileInfo[obsID]['tag']] = site
                    
            except Exception as e:
                sys.stderr.write("ERROR reading metadata file: %s\n" % str(e))
                sys.stderr.flush()
                
    # Setup what we need to write out a configuration file
    corrConfig = {'context': context, 'setup': setup, 
                  'source': {'name':'', 'ra2000':'', 'dec2000':''}, 
                  'inputs': []}
    
    metadata = {}
    for filename in filenames:
        #print("%s:" % os.path.basename(filename))
        
        # Skip over empty files
        if os.path.getsize(filename) == 0:
            continue
            
        # Open the file
        fh = open(filename, 'rb')
        
        # Figure out what to do with the file
        ext = os.path.splitext(filename)[1]
        if ext == '':
            ## DRX
            try:
                ## Get the site
                try:
                    sitename = lwasite[os.path.basename(filename)]
                except KeyError:
                    sitename = 'LWA1'
                    
                ## Get the location so that we can set site-specific parameters
                if sitename == 'LWA1':
                    xyz = LWA1_ECEF
                    off = args.lwa1_offset
                elif sitename == 'LWA-SV':
                    xyz = LWASV_ECEF
                    off = args.lwasv_offset
                else:
                    raise RuntimeError("Unknown LWA site '%s'" % site)
                    
                ## Move into the LWA1 coordinate system
                ### ECEF to LWA1
                rho = xyz - LWA1_ECEF
                sez = numpy.dot(LWA1_ROT, rho)
                enz = sez[[1,0,2]]
                enz[1] *= -1
                
                ## Read in the first few frames to get the start time
                frames = [drx.read_frame(fh) for i in xrange(1024)]
                streams = []
                freq1, freq2 = 0.0, 0.0
                for frame in frames:
                    beam, tune, pol = frame.id
                    if tune == 1:
                        freq1 = frame.central_freq
                    else:
                        freq2 = frame.central_freq
                    if (beam, tune, pol) not in streams:
                        streams.append( (beam, tune, pol) )
                tStart = frames[0].time.datetime
                tStartAlt = (frames[-1].time - 1023//len(streams)*4096/frames[-1].sample_rate).datetime
                tStartDiff = tStart - tStartAlt
                if abs(tStartDiff) > timedelta(microseconds=10000):
                    sys.stderr.write("WARNING: Stale data found at the start of '%s', ignoring\n" % os.path.basename(filename))
                    sys.stderr.flush()
                    tStart = tStartAlt
                ### ^ Adjustment to the start time to deal with occasional problems
                ###   with stale data in the DR buffers at LWA-SV
                    
                ## Read in the last few frames to find the end time
                fh.seek(os.path.getsize(filename) - 1024*drx.FRAME_SIZE)
                backed = 0
                while backed < 2*drx.FRAME_SIZE:
                    try:
                        drx.read_frame(fh)
                        fh.seek(-drx.FRAME_SIZE, 1)
                        break
                    except errors.SyncError:
                        backed += 1
                        fh.seek(-drx.FRAME_SIZE-1, 1)
                for i in xrange(32):
                    try:
                        frame = drx.read_frame(fh)
                        beam, tune, _ = frame.id
                        if tune == 1:
                            freq1 = frame.central_freq
                        else:
                            freq2 = frame.central_freq
                    except errors.SyncError:
                        continue
                tStop = frame.time.datetime
                
                ## Save
                corrConfig['inputs'].append( {'file': filename, 'type': 'DRX', 
                                              'antenna': sitename, 'pols': 'X, Y', 
                                              'location': (enz[0], enz[1], enz[2]), 
                                              'clockoffset': (off, off), 'fileoffset': 0, 
                                              'beam':beam, 'tstart': tStart, 'tstop': tStop, 'freq':(freq1,freq2)} )
                                        
            except Exception as e:
                sys.stderr.write("ERROR reading DRX file: %s\n" % str(e))
                sys.stderr.flush()
                
        elif ext == '.vdif':
            ## VDIF
            try:
                is_vlite = is_vlite_vdif(fh)
                if is_vlite:
                    ## TODO:  Clean this up
                    header = {'OBSERVER': 'Heimdall',
                              'BASENAME': 'VLITE-FAST_0',
                              'SRC_NAME': args.vlite_target,
                              'RA_STR':   args.vlite_ra,
                              'DEC_STR':  args.vlite_dec,
                              'OBSFREQ':  352e6,
                              'OBSBW':    64e6}
                    
                    ## Load in the metadata
                    try:
                        vlite_meta
                    except NameError:
                        import ubjson
                        vlite_metaname = glob.glob(os.path.join(os.path.dirname(filename), '*.meta'))[0]
                        with open(vlite_metaname, 'rb') as mh:
                            vlite_meta = ubjson.load(mh)
                    header['BASENAME'] = vlite_meta['antprops']['datasetId']
                    
                else:
                    ## Read in the GUPPI header
                    header = vdif.read_guppi_header(fh)
                    
                ## Read in the first frame
                vdif.FRAME_SIZE = vdif.get_frame_size(fh)
                frame = vdif.read_frame(fh)
                antID = frame.id[0] - (0 if is_vlite else 12300)
                tStart =  frame.time.datetime
                nThread = vdif.get_thread_count(fh)
                
                ## Read in the last frame
                nJump = int(os.path.getsize(filename)/vdif.FRAME_SIZE)
                nJump -= 30
                fh.seek(nJump*vdif.FRAME_SIZE, 1)
                mark = fh.tell()
                while True:
                    try:
                        frame = vdif.read_frame(fh)
                        tStop = frame.time.datetime
                    except Exception as e:
                        break
                        
                ## Find the antenna location
                pad, edate = db.get_pad('EA%02i' % antID, tStart)
                x,y,z = db.get_xyz(pad, tStart)
                #print("  Pad: %s" % pad)
                #print("  VLA relative XYZ: %.3f, %.3f, %.3f" % (x,y,z))
                
                ## Move into the LWA1 coordinate system
                ### relative to ECEF
                xyz = numpy.array([x,y,z])
                xyz += VLA_ECEF
                ### ECEF to LWA1
                rho = xyz - LWA1_ECEF
                sez = numpy.dot(LWA1_ROT, rho)
                enz = sez[[1,0,2]]
                enz[1] *= -1
                
                if is_vlite:
                    ## VLITE delays
                    idx = vlite_meta['delays']['vlant'].index(antID)
                    vid = vlite_meta['delays']['va_id'][idx]
                    dX = vlite_meta['delays']['clkoffset'][idx]
                    dY = vlite_meta['delays']['clkoffset'][idx]
                    
                    dX = '%.1fns' % (-dX)
                    dY = '%.1fns' % (-dY)                   
                    off = (dX, dY)
                    
                    ## Update the pad name with the VLITE antenna ID
                    pad = '%s (VLITE antenna V%i)' % (pad, vid)
                else:
                    ## VLA time offset
                    off = (args.vla_offset, args.vla_offset)
                    
                ## Save
                corrConfig['context']['observer'] = header['OBSERVER']
                try:
                    corrConfig['context']['project'] = header['BASENAME'].split('_')[0]
                    corrConfig['context']['session'] = header['BASENAME'].split('_')[1].replace('sb', '')
                except IndexError:
                    corrConfig['context']['project'] = 'Unknown'
                    corrConfig['context']['session'] = None
                corrConfig['context']['vlaref'] = re.sub('\.\d+\.\d+\.[AB][CD]-*', '', header['BASENAME'])
                corrConfig['source']['name'] = header['SRC_NAME']
                corrConfig['source']['intent'] = 'target'
                corrConfig['source']['ra2000'] = header['RA_STR']
                corrConfig['source']['dec2000'] = header['DEC_STR']
                corrConfig['inputs'].append( {'file': filename, 'type': 'VDIF', 
                                              'antenna': 'EA%02i' % antID, 'pols': 'Y, X', 
                                              'location': (enz[0], enz[1], enz[2]),
                                              'clockoffset': (off[1], off[1]), 'fileoffset': 0, 
                                              'pad': pad, 'tstart': tStart, 'tstop': tStop, 'freq':header['OBSFREQ']} )
                
            except Exception as e:
                sys.stderr.write("ERROR reading VDIF file: %s\n" % str(e))
                sys.stderr.flush()
                
        elif ext == '.raw':
            ## GUPPI Raw
            try:
                ## Read in the GUPPI header
                header = vdif.read_guppi_header(fh)
                
                ## Read in the first frame
                guppi.FRAME_SIZE = guppi.get_frame_size(fh)
                frame = guppi.read_frame(fh)
                antID = frame.id[0] - 12300
                tStart =  frame.time.datetime
                nThread = guppi.get_thread_count(fh)
                
                ## Read in the last frame
                nJump = int(os.path.getsize(filename)/guppi.FRAME_SIZE)
                nJump -= 4
                fh.seek(nJump*guppi.FRAME_SIZE, 1)
                mark = fh.tell()
                frame = guppi.read_frame(fh)
                tStop = frame.time.datetime
            
                ## Find the antenna location
                pad, edate = db.get_pad('EA%02i' % antID, tStart)
                x,y,z = db.get_xyz(pad, tStart)
                #print("  Pad: %s" % pad)
                #print("  VLA relative XYZ: %.3f, %.3f, %.3f" % (x,y,z))
                
                ## Move into the LWA1 coordinate system
                ### relative to ECEF
                xyz = numpy.array([x,y,z])
                xyz += VLA_ECEF
                ### ECEF to LWA1
                rho = xyz - LWA1_ECEF
                sez = numpy.dot(LWA1_ROT, rho)
                enz = sez[[1,0,2]]
                enz[1] *= -1
                ### z offset from pad height to elevation bearing
                enz[2] += 11.0
                
                ## VLA time offset
                off = args.vla_offset
                
                ## Save
                corrConfig['context']['observer'] = header['OBSERVER']
                corrConfig['context']['project'] = header['BASENAME'].split('_')[0]
                corrConfig['context']['session'] = header['BASENAME'].split('_')[1].replace('sb', '')
                corrConfig['context']['valref'] = re.sub('\.\d+\.\d+\.[AB][CD]-*', '', header['BASENAME'])
                corrConfig['source']['name'] = header['SRC_NAME']
                corrConfig['source']['intent'] = 'target'
                corrConfig['source']['ra2000'] = header['RA_STR']
                corrConfig['source']['dec2000'] = header['DEC_STR']
                corrConfig['inputs'].append( {'file': filename, 'type': 'GUPPI', 
                                              'antenna': 'EA%02i' % antID, 'pols': 'Y, X', 
                                              'location': (enz[0], enz[1], enz[2]),
                                              'clockoffset': (off, off), 'fileoffset': 0, 
                                              'pad': pad, 'tstart': tStart, 'tstop': tStop, 'freq':header['OBSFREQ']} )
                                        
            except Exception as e:
                sys.stderr.write("ERROR reading GUPPI file: %s\n" % str(e))
                sys.stderr.flush()
                
        elif ext == '.tgz':
            ## LWA Metadata
            try:
                ## Extract the file information so that we can pair things together
                fileInfo = metabundle.get_session_metadata(filename)
                for obsID in fileInfo.keys():
                    metadata[fileInfo[obsID]['tag']] = filename
                    
            except Exception as e:
                sys.stderr.write("ERROR reading metadata file: %s\n" % str(e))
                sys.stderr.flush()
                
        # Done
        fh.close()
        
    # Close out the connection to NRAO
    try:
        db.close()
    except AttributeError:
        pass
        
    # Choose a VDIF reference file, if there is one, and mark whether or 
    # not DRX files were found
    vdifRefFile = None
    isDRX = False
    for cinp in corrConfig['inputs']:
        if cinp['type'] in ('VDIF', 'GUPPI'):
            if vdifRefFile is None:
                vdifRefFile = cinp
        elif cinp['type'] == 'DRX':
                isDRX = True
            
    # Set a state variable so that we can generate a warning about missing
    # DRX files
    drxFound = False
    
    # Purge DRX files that don't make sense
    toPurge = []
    drxFound = False
    lwasvFound = False
    for cinp in corrConfig['inputs']:
        ### Sort out multiple DRX files - this only works if we have only one LWA station
        if cinp['type'] == 'DRX':
            if vdifRefFile is not None:
                l0, l1 = cinp['tstart'], cinp['tstop']
                v0, v1 = vdifRefFile['tstart'], vdifRefFile['tstop']
                ve = (v1 - v0).total_seconds()
                overlapWithVDIF = (v0>=l0 and v0<l1) or (l0>=v0 and l0<v1)
                lvo = (min([v1,l1]) - max([v0,l0])).total_seconds()
                if not overlapWithVDIF or lvo < 0.25*ve:
                    toPurge.append( cinp )
                drxFound = True
            if cinp['antenna'] == 'LWA-SV':
                lwasvFound = True
    for cinp in toPurge:
        del corrConfig['inputs'][corrConfig['inputs'].index(cinp)]
        
    # Sort the inputs based on the antenna name - this puts LWA1 first, 
    # LWA-SV second, and the VLA at the end in 'EA' antenna order, i.e., 
    # EA01, EA02, etc.
    corrConfig['inputs'].sort(key=lambda x: 0 if x['antenna'] == 'LWA1' else (1 if x['antenna'] == 'LWA-SV' else int(x['antenna'][2:], 10)))
    
    # VDIF/DRX warning check/report
    if vdifRefFile is not None and isDRX and not drxFound:
        sys.stderr.write("WARNING: DRX files provided but none overlapped with VDIF data")
        
    # Duplicate antenna check
    antCounts = {}
    for cinp in corrConfig['inputs']:
        try:
            antCounts[cinp['antenna']] += 1
        except KeyError:
            antCounts[cinp['antenna']] = 1
    for ant in antCounts.keys():
        if antCounts[ant] != 1:
            sys.stderr.write("WARNING: Antenna '%s' is defined %i times" % (ant, antCounts[ant]))
            
    # Update the file offsets to get things lined up better
    tMax = max([cinp['tstart'] for cinp in corrConfig['inputs']])
    for cinp in corrConfig['inputs']:
        diff = tMax - cinp['tstart']
        offset = diff.days*86400 + diff.seconds + diff.microseconds/1e6
        cinp['fileoffset'] = max([0, offset])
        
    # Reconcile the source lists for when we have eLWA data.  This is needed so
    # that we use the source information contained in the VDIF files rather than
    # the stub information contained in the SDFs
    if len(sources) <= 1:
        if corrConfig['source']['name'] != '':
            ## Update the source information with what comes from the VLA
            try:
                sources[0] = corrConfig['source']
            except IndexError:
                sources.append( corrConfig['source'] )
    # Update the dwell time using the minimum on-source time for all inputs if 
    # there is only one source, i.e., for full eLWA runs
    if len(sources) == 1:
        sources[0]['start'] = max([cinp['tstart'] for cinp in corrConfig['inputs']])
        sources[0]['stop'] = min([cinp['tstop'] for cinp in corrConfig['inputs']])
        
    # Render the configuration
    startRef = sources[0]['start']
    s = 0
    for source in sources:
        startOffset = source['start'] - startRef
        startOffset = startOffset.total_seconds()
        
        dur = source['stop'] - source['start']
        dur = dur.total_seconds()
        
        ## Skip over dummy scans and scans that start after the files end
        if source['intent'] in (None, 'dummy'):
            continue
        if source['start'] > max([cinp['tstop'] for cinp in corrConfig['inputs']]):
            print("Skipping scan of %s which starts at %s, %.3f s after the data end" % (source['name'], source['start'], (source['start'] - max([cinp['tstop'] for cinp in corrConfig['inputs']])).total_seconds()))
            continue
            
        ## Small correction for the first scan to compensate for stale data at LWA-SV
        if lwasvFound and s == 0:
            startOffset += 10.0
            dur -= 10.0
            
        ## Setup
        if args.output is None:
            fh = sys.stdout
        else:
            outname = args.output
            if len(sources) > 1:
                outname += str(s+1)
            fh = open(outname, 'w')
            
        try:
            repo = git.Repo(os.path.dirname(os.path.abspath(__file__)))
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
            
        ## Preamble
        fh.write("# Created\n")
        fh.write("#  on %s\n" % datetime.now())
        fh.write("#  using %s, revision %s.%s%s\n" % (os.path.basename(__file__), branch, shortsha, dirty))
        fh.write("\n")
        ## Observation context
        fh.write("Context\n")
        fh.write("  Observer  %s\n" % corrConfig['context']['observer'])
        fh.write("  Project   %s\n" % corrConfig['context']['project'])
        if corrConfig['context']['session'] is not None:
            fh.write("  Session   %s\n" % corrConfig['context']['session'])
        if corrConfig['context']['vlaref'] is not None:
            fh.write("  VLARef    %s\n" % corrConfig['context']['vlaref'])
        fh.write("EndContext\n")
        fh.write("\n")
        ## Configuration, if present
        if corrConfig['setup'] is not None:
            fh.write("Configuration\n")
            fh.write("  Channels     %i\n" % corrConfig['setup']['channels'])
            fh.write("  IntTime      %.3f\n" % corrConfig['setup']['inttime'])
            fh.write("  PolBasis     %s\n" % corrConfig['setup']['basis'])
            fh.write("EndConfiguration\n")
            fh.write("\n")
        ## Source
        fh.write("Source\n")
        fh.write("# Observation start is %s\n" % source['start'])
        fh.write("# Duration is %s\n" % (source['stop'] - source['start'],))
        fh.write("  Name     %s\n" % source['name'])
        fh.write("  Intent   %s\n" % source['intent'].lower())
        if source['name'] not in ('Sun', 'Jupiter'):
            fh.write("  RA2000   %s\n" % source['ra2000'])
            fh.write("  Dec2000  %s\n" % source['dec2000'])
        fh.write("  Duration %.3f\n" % dur)
        fh.write("SourceDone\n")
        fh.write("\n")
        ## Input files
        for cinp in corrConfig['inputs']:
            fh.write("Input\n")
            fh.write("# Start time is %s\n" % cinp['tstart'])
            fh.write("# Stop time is %s\n" % cinp['tstop'])
            try:
                fh.write("# Beam is %i\n" % cinp['beam'])
            except KeyError:
                pass
            try:
                fh.write("# VLA pad is %s\n" % cinp['pad'])
            except KeyError:
                pass
            try:
                fh.write("# Frequency tuning 1 is %.3f Hz\n" % cinp['freq'][0])
                fh.write("# Frequency tuning 2 is %.3f Hz\n" % cinp['freq'][1])
            except TypeError:
                fh.write("# Frequency tuning is %.3f Hz\n" % cinp['freq'])
            fh.write("  File         %s\n" % cinp['file'])
            try:
                metaname = metadata[os.path.basename(cinp['file'])]
                fh.write("  MetaData     %s\n" % metaname)
            except KeyError:
                if cinp['type'] == 'DRX':
                    sys.stderr.write("WARNING: No metadata found for '%s', source %i\n" % (os.path.basename(cinp['file']), s+1))
                    sys.stderr.flush()
                pass
            fh.write("  Type         %s\n" % cinp['type'])
            fh.write("  Antenna      %s\n" % cinp['antenna'])
            fh.write("  Pols         %s\n" % cinp['pols'])
            fh.write("  Location     %.6f, %.6f, %.6f\n" % cinp['location'])
            fh.write("  ClockOffset  %s, %s\n" % cinp['clockoffset'])
            fh.write("  FileOffset   %.3f\n" % (startOffset + cinp['fileoffset'],))
            fh.write("InputDone\n")
            fh.write("\n")
        if fh != sys.stdout:
            fh.close()
            
        # Increment the source/file counter
        s += 1


if __name__ == "__main__":
    # Helper function for validating the time offset options
    def time_string(value):
        try:
            parse_time_string(value)
        except Exception as e:
            msg = "%r does not appear to be a time string"
            raise argparse.ArgumentTypeError(msg)
        return value.strip()
        
    # Cleanup the command line since argparse has a problem with the negative 
    # clock offsets
    for i in xrange(len(sys.argv)):
        if sys.argv[i][0] == '-':
            try:
                time_string(sys.argv[i])
                ## If we made it this far it looks like we have a time.  
                ## Protect it with a leading space
                sys.argv[i] = " %s" % sys.argv[i]
            except:
                pass
                
    parser = argparse.ArgumentParser(
        description='given a collection of LWA/VLA data files, or a directory containing LWA/VLA/eLWA data files, generate a configuration file for superCorrelator.py',
        epilog="If 'filename' is a directory, only the first argument is used to find data files.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, nargs='+', 
                        help='file or directory name to process')
    parser.add_argument('-l', '--lwa1-offset', type=time_string, default='0.0', 
                        help='LWA1 clock offset')
    parser.add_argument('-s', '--lwasv-offset', type=time_string, default='0.0', 
                        help='LWA-SV clock offset')
    parser.add_argument('-v', '--vla-offset', type=time_string, default='0.0',
                        help='VLA clock offset')
    parser.add_argument('-o', '--output', type=str, 
                        help='write the configuration to the specified file')
    parser.add_argument('-t', '--vlite-target', type=str, default='Unknown',
                        help='VLITE target name')
    parser.add_argument('-r', '--vlite-ra', type=str, default='00:00:00.00',
                        help='VLITE target RA (rad or HH:MM:SS.SS, J2000)')
    parser.add_argument('-d', '--vlite-dec', type=str, default='+90:00:00.00',
                        help='VLITE target dec. (rad or sDD:MM:SS.S, J2000)')
    args = parser.parse_args()
    try:
        args.vlite_ra = str(ephem.hours(float(args.vlite_ra)))
    except ValueError:
        pass
    try:
        args.vlite_dec = str(ephem.degrees(float(args.vlite_dec)))
    except ValueError:
        pass
    main(args)
    
