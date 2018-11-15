#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to take in a collection of observation files and build up a 
superCorrelator.py configuration script.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import sys
import ephem
import numpy
import argparse
from datetime import datetime, timedelta

from lsl.reader import drx, vdif, errors
from lsl.common import metabundle, metabundleADP
from lsl.common.mcs import mjdmpm2datetime

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
                        sdf = metabundle.getSessionDefinition(filename)
                    except Exception as e:
                        sdf = metabundleADP.getSessionDefinition(filename)
                    for obs in sdf.sessions[0].observations:
                        if type(obs).__name__ == 'Solar':
                            name = 'Sun'
                            ra = None
                            dec = None
                        elif type(obs).__name__ == 'Jovian':
                            name = 'Jupiter'
                            ra = None
                            dec = None
                        else:
                            name = obs.target
                            ra = ephem.hours(str(obs.ra))
                            dec = ephem.hours(str(obs.dec))
                        tStart = mjdmpm2datetime(obs.mjd, obs.mpm)
                        tStop  = mjdmpm2datetime(obs.mjd, obs.mpm+obs.dur)
                        sources.append( {'name':name, 'ra2000':ra, 'dec2000':dec, 'start':tStart, 'stop':tStop} )
                        
                ## Extract the file information so that we can pair things together
                fileInfo = metabundle.getSessionMetaData(filename)
                for obsID in fileInfo.keys():
                    metadata[fileInfo[obsID]['tag']] = filename
                    
                ## Figure out LWA1 vs LWA-SV
                try:
                    cs = metabundle.getCommandScript(filename)
                    for c in cs:
                        if c['subsystemID'] == 'DP':
                            site = 'LWA1'
                            break
                        elif c['subsystemID'] == 'ADP':
                            site = 'LWA-SV'
                            break
                except ValueError:
                    site = 'LWA-SV'
                for obsID in fileInfo.keys():
                    lwasite[fileInfo[obsID]['tag']] = site
                    
            except Exception as e:
                sys.stderr.write("ERROR reading metadata file: %s\n" % str(e))
                sys.stderr.flush()
                
    # Setup what we need to write out a configuration file
    corrConfig = {'source': {'name':'', 'ra2000':'', 'dec2000':''}, 
                  'inputs': []}
    
    metadata = {}
    for filename in filenames:
        #print "%s:" % os.path.basename(filename)
        
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
                frames = [drx.readFrame(fh) for i in xrange(1024)]
                streams = []
                freq1, freq2 = 0.0, 0.0
                for frame in frames:
                    beam, tune, pol = frame.parseID()
                    if tune == 1:
                        freq1 = frame.getCentralFreq()
                    else:
                        freq2 = frame.getCentralFreq()
                    if (beam, tune, pol) not in streams:
                        streams.append( (beam, tune, pol) )
                tStart = datetime.utcfromtimestamp(frames[0].getTime())
                tStartAlt = datetime.utcfromtimestamp(frames[-1].getTime() \
                                                      - 1023/len(streams)*4096/frames[-1].getSampleRate())
                tStartDiff = tStart - tStartAlt
                if abs(tStartDiff) > timedelta(microseconds=10000):
                    sys.stderr.write("WARNING: Stale data found at the start of '%s', ignoring\n" % os.path.basename(filename))
                    sys.stderr.flush()
                    tStart = tStartAlt
                ### ^ Adjustment to the start time to deal with occasional problems
                ###   with stale data in the DR buffers at LWA-SV
                    
                ## Read in the last few frames to find the end time
                fh.seek(os.path.getsize(filename) - 1024*drx.FrameSize)
                backed = 0
                while backed < 2*drx.FrameSize:
                    try:
                        drx.readFrame(fh)
                        fh.seek(-drx.FrameSize, 1)
                        break
                    except errors.syncError:
                        backed += 1
                        fh.seek(-drx.FrameSize-1, 1)
                for i in xrange(32):
                    try:
                        frame = drx.readFrame(fh)
                        beam, tune, _ = frame.parseID()
                        if tune == 1:
                            freq1 = frame.getCentralFreq()
                        else:
                            freq2 = frame.getCentralFreq()
                    except errors.syncError:
                        continue
                tStop = datetime.utcfromtimestamp(frame.getTime())
                
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
                ## Read in the GUPPI header
                header = readGUPPIHeader(fh)
                
                ## Read in the first frame
                vdif.FrameSize = vdif.getFrameSize(fh)
                frame = vdif.readFrame(fh)
                antID = frame.parseID()[0] - 12300
                tStart =  datetime.utcfromtimestamp(frame.getTime())
                nThread = vdif.getThreadCount(fh)
                
                ## Read in the last frame
                nJump = int(os.path.getsize(filename)/vdif.FrameSize)
                nJump -= 4
                fh.seek(nJump*vdif.FrameSize, 1)
                mark = fh.tell()
                frame = vdif.readFrame(fh)
                tStop = datetime.utcfromtimestamp(frame.getTime())
            
                ## Find the antenna location
                pad, edate = db.get_pad('EA%02i' % antID, tStart)
                x,y,z = db.get_xyz(pad, tStart)
                #print "  Pad: %s" % pad
                #print "  VLA relative XYZ: %.3f, %.3f, %.3f" % (x,y,z)
                
                ## Move into the LWA1 coordinate system
                ### relative to ECEF
                xyz = numpy.array([x,y,z])
                xyz += VLA_ECEF
                ### ECEF to LWA1
                rho = xyz - LWA1_ECEF
                sez = numpy.dot(LWA1_ROT, rho)
                enz = sez[[1,0,2]]
                enz[1] *= -1
                                
                ## Save
                corrConfig['source']['name'] = header['SRC_NAME']
                corrConfig['source']['ra2000'] = header['RA_STR']
                corrConfig['source']['dec2000'] = header['DEC_STR']
                corrConfig['inputs'].append( {'file': filename, 'type': 'VDIF', 
                                              'antenna': 'EA%02i' % antID, 'pols': 'Y, X', 
                                              'location': (enz[0], enz[1], enz[2]),
                                              'clockoffset': (0.0, 0.0), 'fileoffset': 0, 
                                              'pad': pad, 'tstart': tStart, 'tstop': tStop, 'freq':header['OBSFREQ']} )
                                        
            except Exception as e:
                sys.stderr.write("ERROR reading VDIF file: %s\n" % str(e))
                sys.stderr.flush()
                
        elif ext == '.raw':
            ## GUPPI Raw
            try:
                ## Read in the GUPPI header
                header = readGUPPIHeader(fh)
                
                ## Read in the first frame
                guppi.FrameSize = guppi.getFrameSize(fh)
                frame = guppi.readFrame(fh)
                antID = frame.parseID()[0] - 12300
                tStart =  datetime.utcfromtimestamp(frame.getTime())
                nThread = guppi.getThreadCount(fh)
                
                ## Read in the last frame
                nJump = int(os.path.getsize(filename)/guppi.FrameSize)
                nJump -= 4
                fh.seek(nJump*guppi.FrameSize, 1)
                mark = fh.tell()
                frame = guppi.readFrame(fh)
                tStop = datetime.utcfromtimestamp(frame.getTime())
            
                ## Find the antenna location
                pad, edate = db.get_pad('EA%02i' % antID, tStart)
                x,y,z = db.get_xyz(pad, tStart)
                #print "  Pad: %s" % pad
                #print "  VLA relative XYZ: %.3f, %.3f, %.3f" % (x,y,z)
                
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
                
                ## Save
                corrConfig['source']['name'] = header['SRC_NAME']
                corrConfig['source']['ra2000'] = header['RA_STR']
                corrConfig['source']['dec2000'] = header['DEC_STR']
                corrConfig['inputs'].append( {'file': filename, 'type': 'GUPPI', 
                                              'antenna': 'EA%02i' % antID, 'pols': 'Y, X', 
                                              'location': (enz[0], enz[1], enz[2]),
                                              'clockoffset': (0.0, 0.0), 'fileoffset': 0, 
                                              'pad': pad, 'tstart': tStart, 'tstop': tStop, 'freq':header['OBSFREQ']} )
                                        
            except Exception as e:
                sys.stderr.write("ERROR reading GUPPI file: %s\n" % str(e))
                sys.stderr.flush()
                
        elif ext == '.tgz':
            ## LWA Metadata
            try:
                ## Extract the file information so that we can pair things together
                fileInfo = metabundle.getSessionMetaData(filename)
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
    for input in corrConfig['inputs']:
        if input['type'] in ('VDIF', 'GUPPI'):
            if vdifRefFile is None:
                vdifRefFile = input
        elif input['type'] == 'DRX':
                isDRX = True
            
    # Set a state variable so that we can generate a warning about missing
    # DRX files
    drxFound = False
    
    # Purge DRX files that don't make sense
    toPurge = []
    drxFound = False
    lwasvFound = False
    for input in corrConfig['inputs']:
        ### Sort out multiple DRX files - this only works if we have only one LWA station
        if input['type'] == 'DRX':
            if vdifRefFile is not None:
                l0, l1 = input['tstart'], input['tstop']
                v0, v1 = vdifRefFile['tstart'], vdifRefFile['tstop']
                ve = (v1 - v0).total_seconds()
                overlapWithVDIF = (v0>=l0 and v0<l1) or (l0>=v0 and l0<v1)
                lvo = (min([v1,l1]) - max([v0,l0])).total_seconds()
                if not overlapWithVDIF or lvo < 0.25*ve:
                    toPurge.append( input )
                drxFound = True
            if input['antenna'] == 'LWA-SV':
                lwasvFound = True
    for input in toPurge:
        del corrConfig['inputs'][corrConfig['inputs'].index(input)]
        
    # Sort the inputs based on the antenna name - this puts LWA1 first, 
    # LWA-SV second, and the VLA at the end in 'EA' antenna order, i.e., 
    # EA01, EA02, etc.
    corrConfig['inputs'].sort(key=lambda x: 0 if x['antenna'] == 'LWA1' else (1 if x['antenna'] == 'LWA-SV' else x['antenna']))
    
    # VDIF/DRX warning check/report
    if vdifRefFile is not None and isDRX and not drxFound:
        sys.stderr.write("WARNING: DRX files provided but none overlapped with VDIF data")
        
    # Update the file offsets to get things lined up better
    tMax = max([input['tstart'] for input in corrConfig['inputs']])
    for input in corrConfig['inputs']:
        diff = tMax - input['tstart']
        offset = diff.days*86400 + diff.seconds + diff.microseconds/1e6
        input['fileoffset'] = max([0, offset])
        
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
        sources[0]['start'] = max([input['tstart'] for input in corrConfig['inputs']])
        sources[0]['stop'] = min([input['tstop'] for input in corrConfig['inputs']])
        
    # Render the configuration
    startRef = sources[0]['start']
    for s,source in enumerate(sources):
        startOffset = source['start'] - startRef
        startOffset = startOffset.total_seconds()
        
        dur = source['stop'] - source['start']
        dur = dur.total_seconds()
        
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
            
        ## Preamble
        fh.write("# Created\n")
        fh.write("#  on %s\n" % datetime.now())
        fh.write("#  using %s, revision $Rev$\n" % os.path.basename(__file__))
        fh.write("\n")
        ## Source
        fh.write("Source\n")
        fh.write("# Observation start is %s\n" % source['start'])
        fh.write("# Duration is %s\n" % (source['stop'] - source['start'],))
        fh.write("  Name     %s\n" % source['name'])
        if source['name'] not in ('Sun', 'Jupiter'):
            fh.write("  RA2000   %s\n" % source['ra2000'])
            fh.write("  Dec2000  %s\n" % source['dec2000'])
        fh.write("  Duration %.3f\n" % dur)
        fh.write("SourceDone\n")
        fh.write("\n")
        ## Input files
        for input in corrConfig['inputs']:
            fh.write("Input\n")
            fh.write("# Start time is %s\n" % input['tstart'])
            fh.write("# Stop time is %s\n" % input['tstop'])
            try:
                fh.write("# Beam is %i\n" % input['beam'])
            except KeyError:
                pass
            try:
                fh.write("# VLA pad is %s\n" % input['pad'])
            except KeyError:
                pass
            try:
                fh.write("# Frequency tuning 1 is %.3f Hz\n" % input['freq'][0])
                fh.write("# Frequency tuning 2 is %.3f Hz\n" % input['freq'][1])
            except TypeError:
                fh.write("# Frequency tuning is %.3f Hz\n" % input['freq'])
            fh.write("  File         %s\n" % input['file'])
            try:
                metaname = metadata[os.path.basename(input['file'])]
                fh.write("  MetaData     %s\n" % metaname)
            except KeyError:
                if input['type'] == 'DRX':
                    sys.stderr.write("WARNING: No metadata found for '%s', source %i\n" % (os.path.basename(input['file']), s+1))
                    sys.stderr.flush()
                pass
            fh.write("  Type         %s\n" % input['type'])
            fh.write("  Antenna      %s\n" % input['antenna'])
            fh.write("  Pols         %s\n" % input['pols'])
            fh.write("  Location     %.6f, %.6f, %.6f\n" % input['location'])
            fh.write("  ClockOffset  %s, %s\n" % input['clockoffset'])
            fh.write("  FileOffset   %.3f\n" % (startOffset + input['fileoffset'],))
            fh.write("InputDone\n")
            fh.write("\n")
        if fh != sys.stdout:
            fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='given a collection of LWA/VLA data files, or a directory containing LWA/VLA/eLWA data files, generate a configuration file for superCorrelator.py',
        epilog="If 'filename' is a directory, only the first argument is used to find data files.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, nargs='+', 
                        help='file or directory name to process')
    parser.add_argument('-l', '--lwa1-offset', type=str, default=0.0, 
                        help='LWA1 clock offset')
    parser.add_argument('-s', '--lwasv-offset', type=str, default=0.0, 
                        help='LWA-SV clock offset')
    parser.add_argument('-o', '--output', type=str, 
                        help='write the configuration to the specified file')
    args = parser.parse_args()
    main(args)
    