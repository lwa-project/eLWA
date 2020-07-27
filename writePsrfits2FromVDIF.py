#!/usr/bin/env python

"""
Given a DRX file, create one of more PSRFITS file(s).
"""

from __future__ import print_function, division

import os
import sys
import numpy
import ephem
import ctypes
import argparse

import psrfits_utils.psrfits_utils as pfu

import lsl.reader.vdif as vdif
import lsl.reader.errors as errors
from lsl.reader.buffer import VDIFFrameBuffer
import lsl.astro as astro
import lsl.common.progress as progress
from lsl.common.dp import fS
from lsl.statistics import kurtosis
from lsl.misc import parser as aph

from utils import is_vlite_vdif

try:
    from _psr import *
except ImportError:
    # Here we go...
    import subprocess
    ## Downlaod everything we need to build _psr.so
    for filename in ('dedispersion.c', 'fft.c', 'helper.c', 'kurtosis.c', 'psr.c', 'psr.h', 
                     'py3_compat.h', 'quantize.c', 'reduce.c', 'utils.c', 'Makefile'):
        subprocess.check_call(['wget', 'https://raw.githubusercontent.com/lwa-project/pulsar/master/%s' % filename, 
                                       '-O', filename])
    # Build _psr.so
    subprocess.check_call(['make', 'clean'])
    subprocess.check_call(['make'])


def main(args):
    # FFT length
    LFFT = args.nchan
    
    fh = open(args.filename, "rb")
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
    else:
        header = vdif.read_guppi_header(fh)
    vdif.FRAME_SIZE = vdif.get_frame_size(fh)
    nFramesFile = os.path.getsize(args.filename) // vdif.FRAME_SIZE
    
    # Extract the source parameters
    source = header['SRC_NAME']
    ra = header['RA_STR']
    dec = header['DEC_STR']
    
    # Find the good data (non-zero decimation)
    while True:
        try:
            junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
            srate = junkFrame.sample_rate
            break
        except ZeroDivisionError:
            pass
    fh.seek(-vdif.FRAME_SIZE, 1)
    
    # Load in basic information about the data
    junkFrame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
    fh.seek(-vdif.FRAME_SIZE, 1)
    ## What's in the data?
    beam,pol = junkFrame.id
    srate = junkFrame.sample_rate
    central_freq1 = junkFrame.central_freq
    nSampsFrame = junkFrame.payload.data.size
    beams = 1
    tunepols = (2,)
    tunepol = tunepols[0]
    
    ## Date
    beginDate = junkFrame.time
    beginTime = beginDate.datetime
    mjd = beginDate.mjd
    mjd_day = int(mjd)
    mjd_sec = (mjd-mjd_day)*86400
    if args.output is None:
        args.output = "vdif_%05d_%s" % (mjd_day, source.replace(' ', ''))
        
    # File summary
    print("Input Filename: %s" % args.filename)
    print("Date of First Frame: %s (MJD=%f)" % (str(beginDate),mjd))
    print("Beams: %i" % beams)
    print("Tune/Pols: %i" % tunepols)
    print("Tunings: %.1f Hz" % central_freq1)
    print("Sample Rate: %i Hz" % srate)
    print("Sample Time: %f s" % (2*LFFT/srate,))
    print("Frames: %i (%.3f s)" % (nFramesFile, nSampsFrame*nFramesFile / srate / tunepol))
    print("---")
    print("Using FFTW Wisdom? %s" % useWisdom)
    
    # Create the output PSRFITS file(s)
    pfu_out = []
    nsblk = args.nsblk
    if (not args.no_summing):
        polNames = 'I'
        nPols = 1
        reduceEngine = CombineToIntensity
    elif args.stokes:
        polNames = 'IQUV'
        nPols = 4
        reduceEngine = CombineToStokes
    #elif args.circular:
    #    polNames = 'LLRR'
    #    nPols = 2
    #    reduceEngine = CombineToCircular
    else:
        polNames = 'XXYY'
        nPols = 2
        reduceEngine = CombineToLinear
        
    if args.four_bit_data:
        OptimizeDataLevels = OptimizeDataLevels4Bit
    else:
        OptimizeDataLevels = OptimizeDataLevels8Bit
        
    ## Basic structure and bounds
    pfo = pfu.psrfits()
    pfo.basefilename = "%s_b%i" % (args.output, beam)
    pfo.filenum = 0
    pfo.tot_rows = pfo.N = pfo.T = pfo.status = pfo.multifile = 0
    pfo.rows_per_file = 32768
    
    ## Frequency, bandwidth, and channels
    pfo.hdr.fctr=central_freq1/1e6
    pfo.hdr.BW = srate/2/1e6
    pfo.hdr.nchan = LFFT
    pfo.hdr.df = srate/2/1e6/LFFT
    pfo.hdr.dt = 2*LFFT / srate
    
    ## Metadata about the observation/observatory/pulsar
    pfo.hdr.observer = "wP2FromVDIF.py"
    pfo.hdr.source = source
    pfo.hdr.fd_hand = 1
    pfo.hdr.nbits = 4 if args.four_bit_data else 8
    pfo.hdr.nsblk = nsblk
    pfo.hdr.ds_freq_fact = 1
    pfo.hdr.ds_time_fact = 1
    pfo.hdr.npol = nPols
    pfo.hdr.summed_polns = 1 if (not args.no_summing) else 0
    pfo.hdr.obs_mode = "SEARCH"
    pfo.hdr.telescope = "LWA"
    pfo.hdr.frontend = "LWA"
    pfo.hdr.backend = "VDIF"
    pfo.hdr.project_id = "Pulsar"
    pfo.hdr.ra_str = ra
    pfo.hdr.dec_str = dec
    pfo.hdr.poln_type = "LIN" #if not args.circular else "CIRC"
    pfo.hdr.poln_order = polNames
    pfo.hdr.date_obs = str(beginTime.strftime("%Y-%m-%dT%H:%M:%S"))     
    pfo.hdr.MJD_epoch = pfu.get_ld(mjd)
    
    ## Setup the subintegration structure
    pfo.sub.tsubint = pfo.hdr.dt*pfo.hdr.nsblk
    pfo.sub.bytes_per_subint = pfo.hdr.nchan*pfo.hdr.npol*pfo.hdr.nsblk*pfo.hdr.nbits//8
    pfo.sub.dat_freqs   = pfu.malloc_doublep(pfo.hdr.nchan*8)				# 8-bytes per double @ LFFT channels
    pfo.sub.dat_weights = pfu.malloc_floatp(pfo.hdr.nchan*4)				# 4-bytes per float @ LFFT channels
    pfo.sub.dat_offsets = pfu.malloc_floatp(pfo.hdr.nchan*pfo.hdr.npol*4)		# 4-bytes per float @ LFFT channels per pol.
    pfo.sub.dat_scales  = pfu.malloc_floatp(pfo.hdr.nchan*pfo.hdr.npol*4)		# 4-bytes per float @ LFFT channels per pol.
    if args.four_bit_data:
        pfo.sub.data = pfu.malloc_ucharp(pfo.hdr.nchan*pfo.hdr.npol*pfo.hdr.nsblk)	# 1-byte per unsigned char @ (LFFT channels x pols. x nsblk sub-integrations) samples
        pfo.sub.rawdata = pfu.malloc_ucharp(pfo.hdr.nchan*pfo.hdr.npol*pfo.hdr.nsblk//2)	# 4-bits per nibble @ (LFFT channels x pols. x nsblk sub-integrations) samples
    else:
        pfo.sub.rawdata = pfu.malloc_ucharp(pfo.hdr.nchan*pfo.hdr.npol*pfo.hdr.nsblk)	# 1-byte per unsigned char @ (LFFT channels x pols. x nsblk sub-integrations) samples
        
    ## Create and save it for later use
    pfu.psrfits_create(pfo)
    
    freqBaseMHz = numpy.fft.fftshift( numpy.fft.fftfreq(LFFT, d=2.0/srate) ) / 1e6
    # Define the frequencies available in the file (in MHz)
    pfu.convert2_double_array(pfo.sub.dat_freqs, freqBaseMHz + pfo.hdr.fctr, LFFT)
    
    # Define which part of the spectra are good (1) or bad (0).  All channels
    # are good except for the two outermost.
    pfu.convert2_float_array(pfo.sub.dat_weights, numpy.ones(LFFT),  LFFT)
    pfu.set_float_value(pfo.sub.dat_weights, 0,      0)
    pfu.set_float_value(pfo.sub.dat_weights, LFFT-1, 0)
    
    # Define the data scaling (default is a scale of one and an offset of zero)
    pfu.convert2_float_array(pfo.sub.dat_offsets, numpy.zeros(LFFT*nPols), LFFT*nPols)
    pfu.convert2_float_array(pfo.sub.dat_scales,  numpy.ones(LFFT*nPols),  LFFT*nPols)
    
    # Speed things along, the data need to be processed in units of 'nsblk'.  
    # Find out how many frames per tuning/polarization that corresponds to.
    chunkSize = nsblk*2*LFFT//nSampsFrame
    
    # Calculate the SK limites for weighting
    if (not args.no_sk_flagging):
        skLimits = kurtosis.get_limits(4.0, 1.0*nsblk)
        
        GenerateMask = lambda x: ComputeSKMask(x, skLimits[0], skLimits[1])
    else:
        def GenerateMask(x):
            flag = numpy.ones((2, LFFT), dtype=numpy.float32)
            flag[:,0] = 0.0
            flag[:,-1] = 0.0
            return flag
            
    # Create the progress bar so that we can keep up with the conversion.
    pbar = progress.ProgressBarPlus(max=nFramesFile//(2*chunkSize), span=55)
    
    # Go!
    done = False
    siCount = 0
    rawData = numpy.zeros((2, nSampsFrame*chunkSize), dtype=numpy.complex64)
    
    vdifBuffer = VDIFFrameBuffer(threads=[0,1])
    
    while True:
        ## Read in the data
        rawData *= 0.0
        count = [0 for i in xrange(rawData.shape[0])]
        
        i = 0
        while i < chunkSize:
            try:
                frame = vdif.read_frame(fh, central_freq=header['OBSFREQ'], sample_rate=header['OBSBW']*2.0)
                vdifBuffer.append(frame)
            except errors.EOFError:
                try:
                    ## VLITE files are limited to 1 s so there may be multiple files
                    ## to read in.  Try to find the next one.
                    if not is_vlite:
                        assert(False)
                    oldname = fh.name
                    nextname, ext = os.path.splitext(oldname)
                    base, vfts = nextname.rsplit('_', 1)
                    vfts = int(vfts, 10) + 1
                    nextname = "%s_%s%s" % (base, vfts, ext)
                    assert(os.path.exists(nextname))
                    fh.close()
                    fh = open(nextname, 'rb')
                    print("Switched from %s to %s" % (os.path.basename(oldname), vfts))
                except (ValueError, AssertionError) as e:
                    done = True
                    break
            except errors.SyncError:
                fh.seek(vdif.FRAME_SIZE, 1)
                continue
                
            frames = vdifBuffer.get()
            if frames is None:
                continue
                
            for frame in frames:
                beam,pol = frame.id
                tune = 1
                aStand = 2*(tune-1) + pol
                
                rawData[aStand, count[aStand]*nSampsFrame:(count[aStand]+1)*nSampsFrame] = frame.payload.data
                count[aStand] += 1
            i += 1
            
        siCount += 1
        
        ## Are we done yet?
        if done:
            break
            
        ## FFT
        rawSpectra = PulsarEngineRaw(rawData, 2*LFFT)
        
        ## Prune off the negative frequencies
        rawSpectra = numpy.ascontiguousarray(rawSpectra[:,LFFT:,:])
        if is_vlite:
            ### Flip
            rawSpectra = rawSpectra[:,::-1,:]
            
        ## S-K flagging
        flag = GenerateMask(rawSpectra)
        if is_vlite:
            ### Flag MUOS
            flag[:,5*LFFT//8:] = 1
        weight1 = numpy.where( flag[:2,:].sum(axis=0) == 0, 0, 1 ).astype(numpy.float32)
        ff1 = 1.0*(LFFT - weight1.sum()) / LFFT
        
        ## Detect power
        data = reduceEngine(rawSpectra)
        
        ## Optimal data scaling
        bzero, bscale, bdata = OptimizeDataLevels(data, LFFT)
        
        ## Polarization mangling
        bzero1 = bzero[:nPols,:].T.ravel()
        bscale1 = bscale[:nPols,:].T.ravel()
        bdata1 = bdata[:nPols,:].T.ravel()
        
        ## Write the spectra to the PSRFITS files
        ### Time
        pfo.sub.offs = (pfo.tot_rows)*pfo.hdr.nsblk*pfo.hdr.dt+pfo.hdr.nsblk*pfo.hdr.dt/2.
        
        ### Data
        ptr, junk = bdata1.__array_interface__['data']
        if args.four_bit_data:
            ctypes.memmove(int(pfo.sub.data), ptr, pfo.hdr.nchan*nPols*pfo.hdr.nsblk)
        else:
            ctypes.memmove(int(pfo.sub.rawdata), ptr, pfo.hdr.nchan*nPols*pfo.hdr.nsblk)
            
        ### Zero point
        ptr, junk = bzero1.__array_interface__['data']
        ctypes.memmove(int(pfo.sub.dat_offsets), ptr, pfo.hdr.nchan*nPols*4)
        
        ### Scale factor
        ptr, junk = bscale1.__array_interface__['data']
        ctypes.memmove(int(pfo.sub.dat_scales), ptr, pfo.hdr.nchan*nPols*4)
        
        ### SK
        ptr, junk = weight1.__array_interface__['data']
        ctypes.memmove(int(pfo.sub.dat_weights), ptr, pfo.hdr.nchan*4)
        
        ### Save
        pfu.psrfits_write_subint(pfo)
        
        ### Update the progress bar and remaining time estimate
        pbar.inc()
        sys.stdout.write('%5.1f%% %s\r' % (ff1*100, pbar.show()))
        sys.stdout.flush()
        
    # Update the progress bar with the total time used
    sys.stdout.write('              %s\n' % pbar.show())
    sys.stdout.flush()
    
    # Final report on the buffer
    print(vdifBuffer.status())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='read in YUPPI VDIF files and create one or more PSRFITS file(s)', 
        epilog='NOTE:  If a source name is provided and the RA or declination is not, the script will attempt to determine these values.', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, 
                        help='filename to process')
    parser.add_argument('-o', '--output', type=str, 
                        help='output file basename')
    parser.add_argument('-c', '--nchan', type=aph.positive_int, default=4096, 
                        help='FFT length')
    parser.add_argument('-b', '--nsblk', type=aph.positive_int, default=5000, 
                        help='number of spetra per sub-block')
    parser.add_argument('-p', '--no-sk-flagging', action='store_true', 
                        help='disable on-the-fly SK flagging of RFI')
    parser.add_argument('-n', '--no-summing', action='store_true', 
                        help='do not sum linear polarizations')
    parser.add_argument('-4', '--four-bit-data', action='store_true', 
                        help='save the spectra in 4-bit mode instead of 8-bit mode')
    parser.add_argument('-t', '--vlite-target', type=str, default='Unknown',
                        help='VLITE target name')
    parser.add_argument('-r', '--vlite-ra', type=str, default='00:00:00.00',
                        help='VLITE target RA (rad or HH:MM:SS.SS, J2000)')
    parser.add_argument('-d', '--vlite-dec', type=str, default='+90:00:00.00',
                        help='VLITE target dec. (rad or sDD:MM:SS.S, J2000)')
    args = parser.parse_args()
    main(args)
    
