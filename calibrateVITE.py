#!/usr/bin/env python

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    raw_input = input
    
import os
import git
import sys
import time
import numpy
import ubjson
from astropy.io import fits as astrofits
import argparse
from datetime import datetime

from scipy.interpolate import interp1d, interp2d

from lsl.astro import utcjd_to_unix
from lsl.writer.fitsidi import NUMERIC_STOKES
from lsl.misc.dedispersion import delay as ddelay


def _select(data, key2, key3):
    x, y = [], []
    for key1 in data.keys():
        try:
            value = data[key1][key2][key3]
            x.append(key1)
            y.append(value)
        except KeyError:
            pass
    return x, y


def load_event_window(filename):
    with open(filename, 'rb') as mh:
        meta = ubjson.load(mh)
        
    dm = meta['dm']
    peak = meta['peak_time']
    width = meta['width']
    t1, t0 = ddelay([320e6, 384e6], dm)
    t0 += peak - width
    t1 += peak + width
    
    return t0/86400.0, t1/86400.0


def load_antenna_mapper(filename):
    with open(filename, 'rb') as mh:
        meta = ubjson.load(mh)
        
    mapper = {}
    for i in range(len(meta['delays']['vlant'])):
        mapper[meta['delays']['vlant'][i]] = meta['delays']['va_id'][i]
    return mapper


def load_antenna_unmapper(filename):
    with open(filename, 'rb') as mh:
        meta = ubjson.load(mh)
        
    unmapper = {}
    for i in range(len(meta['delays']['vlant'])):
        unmapper[meta['delays']['va_id'][i]] = meta['delays']['vlant'][i]
    return unmapper


def load_caltab_fg(filename, start=-numpy.inf, stop=numpy.inf, margin=60.0):
    hdulist = astrofits.open(filename)
    ref_date = hdulist[1].header['TZERO4']
    nFreq = hdulist[1].header['TDIM13']
    nFreq = nFreq.split(',')[2]
    nFreq = float(nFreq)
    
    flags = {'ants':[], 'times':[], 'chans':[], 'pflags':[], 'reason':[]}
    count = 0
    for hdu in hdulist[1:]:
        if hdu.header['EXTNAME'] == 'AIPS FG':
            ## The actual flag table - only keep the second one
            count += 1
            if count != 2:
                continue
                
            for row in hdu.data:
                ### Make sure it is from a relevant time
                t = row['TIME RANGE'] + ref_date
                if t[1] < (start - margin/86400.0) or t[0] > (stop + margin/86400.0):
                    continue
                    
                a1, a2 = row['ANTS']
                c1, c2 = row['CHANS']/nFreq
                p1, p2, p3, p4 = row['PFLAGS']
                r = row['REASON']
                
                flags['ants'].append([a1,a2])
                flags['times'].append([t[0],t[1]])
                flags['chans'].append([c1,c2])
                flags['pflags'].append([p1,p2,p3,p4])                
                flags['reason'].append(r)
                
    return flags


def load_caltab_cl(filename, start=-numpy.inf, stop=numpy.inf, margin=60.0):
    hdulist = astrofits.open(filename)
    ref_date = hdulist[1].header['TZERO4']
    
    gains = {}
    count = 0
    for hdu in hdulist[1:]:
        if hdu.header['EXTNAME'] == 'AIPS CL':
            ## The actual calibration information - only keep the third (last) one
            count += 1
            if count != 3:
                continue
                
            for row in hdu.data:
                ### Make sure it is from a relevant time
                t, tint = row['TIME'], row['TIME INTERVAL']
                t += ref_date
                if (t - tint/2) < (start - margin/86400.0) or (t + tint/2) > (stop + margin/86400.0):
                    continue
                    
                ### Delays and complex gain
                a = row['ANTENNA NO.']-1
                d1, r1, i1 = row['DELAY 1'], row['REAL1'], row['IMAG1']
                d2, r2, i2 = row['DELAY 2'], row['REAL2'], row['IMAG2']
                g1 = r1 + 1j*i1
                g2 = r2 + 1j*i2
                #if a == 1:
                #    print(a, d1, r1, i1, d2, r2, i2)
                
                ### Save the timestamp
                try:
                    gains[t]
                except KeyError:
                    gains[t] = {}
                    
                ### Save the delays/complex gains
                try:
                    gains[t][a]['d1'] += d1
                    gains[t][a]['g1'] *= g1
                    gains[t][a]['d2'] += d2
                    gains[t][a]['g2'] *= g2
                except KeyError:
                    gains[t][a] = {'d1': d1, 'g1': g1, 'd2': d2, 'g2': g2}
    hdulist.close()
    
    # Build up the delay and gain interpolators
    times = numpy.array(list(gains.keys()))
    names = []
    for t in times:
        for name in gains[t].keys():
            if name not in names:
                names.append(name)
    delays1, delays2 = {}, {}
    gains1, gains2 = {}, {}
    for a in names:
        try:
            delays1[a] = interp1d(*(_select(gains, a, 'd1')), bounds_error=False, fill_value=0.0, kind='quadratic')
            delays2[a] = interp1d(*(_select(gains, a, 'd2')), bounds_error=False, fill_value=0.0, kind='quadratic')
            gains1[a]  = interp1d(*(_select(gains, a, 'g1')), bounds_error=False, fill_value=numpy.complex64(1.0), kind='linear')
            gains2[a]  = interp1d(*(_select(gains, a, 'g2')), bounds_error=False, fill_value=numpy.complex64(1.0), kind='linear')
        except KeyError:
            print("CalTab: %s not found, setting delays to 0 and gains to 1" % a)
            delays1[a] = interp1d(times, [0.0 for t in times])
            delays2[a] = interp1d(times, [0.0 for t in times])
            gains1[a]  = interp1d(times, [numpy.complex64(1.0) for t in times])
            gains2[a]  = interp1d(times, [numpy.complex64(1.0) for t in times])
            
    return delays1, delays2, gains1, gains2


def load_caltab_bp(filename, start=-numpy.inf, stop=numpy.inf, margin=60.0):
    hdulist = astrofits.open(filename)
    ref_date = hdulist[1].header['TZERO4']
    ref_freq = hdulist[1].header['3CRVL13']
    
    ref_width = 0.0
    for hdu in hdulist[1:]:
        if hdu.header['EXTNAME'] == 'AIPS FQ':
            ref_width = hdu.data[0]['CH WIDTH']
            
    gains = {}
    count = 0
    for hdu in hdulist[1:]:
        if hdu.header['EXTNAME'] == 'AIPS BP':
            ## The actual calibration information - only keep the first one
            count += 1
            if count != 1:
                continue
                
            for row in hdu.data:
                ### Make sure it is from a relevant time
                t, tint = row['TIME'], row['INTERVAL']
                t += ref_date
                if (t - tint/2) < (start - margin/86400.0) or (t + tint/2) > (stop + margin/86400.0):
                    continue
                    
                ### Complex gain
                a = row['ANTENNA']-1
                r1, i1 = row['REAL 1'], row['IMAG 1']
                r2, i2 = row['REAL 2'], row['IMAG 2']
                g1 = r1 + 1j*i1
                g2 = r2 + 1j*i2
                g1 = numpy.where(numpy.isfinite(g1.real), g1, 1+0j)
                g2 = numpy.where(numpy.isfinite(g2.real), g2, 1+0j)
                
                ### Save the timestamp
                try:
                    gains[t]
                except KeyError:
                    gains[t] = {}
                    
                ### Save the complex gains
                gains[t][a] = {'g1': g1, 'g2': g2}
    hdulist.close()
    
    # Build up the gain interpolators
    times = numpy.array(list(gains.keys()))
    freqs = numpy.arange(len(g1))*ref_width + ref_freq
    names = []
    for t in times:
        for name in gains[t].keys():
            if name not in names:
                names.append(name)
    gains1, gains2 = {}, {}
    for a in names:
        try:
            v = _select(gains, a, 'g1')
            vt, va = numpy.array(v[0]), numpy.array(v[1])
            rgains  = interp2d(vt, freqs, va.T.real, bounds_error=False, fill_value=1.0)
            igains  = interp2d(vt, freqs, va.T.imag, bounds_error=False, fill_value=0.0)
            gains1[a] = lambda x, y: (rgains(x,y) + 1j*igains(x,y)).ravel()
            v = _select(gains, a, 'g2')
            vt, va = numpy.array(v[0]), numpy.array(v[1])
            rgains  = interp2d(vt, freqs, va.T.real, bounds_error=False, fill_value=1.0)
            igains  = interp2d(vt, freqs, va.T.imag, bounds_error=False, fill_value=0.0)
            gains2[a] = lambda x, y: (rgains(x,y) + 1j*igains(x,y)).ravel()
        except KeyError:
            print("CalTab: %s not found, setting bandpass to 1" % a)
            rgains  = interp2d(times, freqs, numpy.ones((len(times), len(freqs)), dtype=numpy.float32).T)
            igains  = interp2d(times, freqs, numpy.ones((len(times), len(freqs)), dtype=numpy.float32).T)
            gains1[a] = lambda x, y: (rgains(x,y) + 1j*igains(x,y)).ravel()
            rgains  = interp2d(times, freqs, numpy.ones((len(times), len(freqs)), dtype=numpy.float32).T)
            igains  = interp2d(times, freqs, numpy.ones((len(times), len(freqs)), dtype=numpy.float32).T)
            gains2[a] = lambda x, y: (rgains(x,y) + 1j*igains(x,y)).ravel()
            
    return gains1, gains2


def load_uvout_uv(filename, start=-numpy.inf, stop=numpy.inf, margin=120.0):
    hdulist = astrofits.open(filename)
    ref_date = hdulist[0].header['PZERO4']

    coords = {}
    hdu = hdulist[0]
    for row in hdu.data:
        ### Make sure it is from a relevant time
        t, tint = row['DATE'] + row['_DATE'], row['INTTIM']/86400.0
        #t += ref_date
        if (t - tint/2) < (start - margin/86400.0) or (t + tint/2) > (stop + margin/86400.0):
            continue

        ### Baselines and coordinates
        bl = int(row['BASELINE'])
        u, v, w = row['UU---SIN'], row['VV---SIN'], row['WW---SIN']

        for t in (t,):
            ### Save the timestamp
            try:
                coords[t]
            except KeyError:
                coords[t] = {}
                
            ### Save the delays/complex gains
            try:
                coords[t][bl]['u'] = u
                coords[t][bl]['v'] = v
                coords[t][bl]['w'] = w
            except KeyError:
                coords[t][bl] = {'u': u, 'v': v, 'w': w}
    hdulist.close()
    
    # Build up the delay and gain interpolators
    times = numpy.array(list(coords.keys()))
    names = []
    for t in times:
        for name in coords[t].keys():
            if name not in names:
                names.append(name)
    u, v, w = {}, {}, {}
    for bl in names:
        try:
            u[bl] = interp1d(*(_select(coords, bl, 'u')), bounds_error=False, fill_value=0.0, kind='quadratic')
            v[bl] = interp1d(*(_select(coords, bl, 'v')), bounds_error=False, fill_value=0.0, kind='quadratic')
            w[bl] = interp1d(*(_select(coords, bl, 'w')), bounds_error=False, fill_value=0.0, kind='quadratic')
        except KeyError:
            pass
            
    return u, v, w


def load_uvout_sn(filename, start=-numpy.inf, stop=numpy.inf, margin=120.0):
    hdulist = astrofits.open(filename)
    ref_date = hdulist[0].header['PZERO4']
    
    gains = {}
    count = 0
    for hdu in hdulist[1:]:
        if hdu.header['EXTNAME'] == 'AIPS SN':
            ### The actual calibration information - only keep the last one
            count += 1
            if count != 2:
                continue

            for row in hdu.data:
                ### Make sure it is from a relevant time
                t, tint = row['TIME'], row['TIME INTERVAL']
                t += ref_date
                if (t - tint/2) < (start - margin/86400.0) or (t + tint/2) > (stop + margin/86400.0):
                    continue
                    
                ### Delays and complex gain
                a = row['ANTENNA NO.']-1
                d1, r1, i1 = row['DELAY 1'], row['REAL1'], row['IMAG1']
                d2, r2, i2 = row['DELAY 2'], row['REAL2'], row['IMAG2']
                g1 = r1 + 1j*i1
                g2 = r2 + 1j*i2
                
                ### Save the timestamp
                try:
                    gains[t]
                except KeyError:
                    gains[t] = {}
                    
                ### Save the delays/complex gains
                try:
                    gains[t][a]['d1'] += d1
                    gains[t][a]['g1'] *= g1
                    gains[t][a]['d2'] += d2
                    gains[t][a]['g2'] *= g2
                except KeyError:
                    gains[t][a] = {'d1': d1, 'g1': g1, 'd2': d2, 'g2': g2}
    hdulist.close()
    
    # Build up the delay and gain interpolators
    times = numpy.array(list(gains.keys()))
    names = []
    for t in times:
        for name in gains[t].keys():
            if name not in names:
                names.append(name)
    delays1, delays2 = {}, {}
    gains1, gains2 = {}, {}
    for a in names:
        try:
            delays1[a] = interp1d(*(_select(gains, a, 'd1')), bounds_error=False, fill_value=0.0, kind='quadratic')
            delays2[a] = interp1d(*(_select(gains, a, 'd2')), bounds_error=False, fill_value=0.0, kind='quadratic')
            gains1[a]  = interp1d(*(_select(gains, a, 'g1')), bounds_error=False, fill_value=numpy.complex64(1.0), kind='linear')
            gains2[a]  = interp1d(*(_select(gains, a, 'g2')), bounds_error=False, fill_value=numpy.complex64(1.0), kind='linear')
        except KeyError:
            print("uvout: %s not found, setting delays to 0 and gains to 1" % a)
            delays1[a] = interp1d(times, [0.0 for t in times])
            delays2[a] = interp1d(times, [0.0 for t in times])
            gains1[a]  = interp1d(times, [numpy.complex64(1.0) for t in times])
            gains2[a]  = interp1d(times, [numpy.complex64(1.0) for t in times])
            
    return delays1, delays2, gains1, gains2


def get_source_blocks(hdulist):
    """
    Given a astrofits hdulist, look at the source IDs listed in the UV_DATA table
    and generate a list of contiguous blocks for each source.  This list is 
    returned as a list of blocks where each block is itself a two-element list.
    This two-element list contains the the start and stop rows for the block.
    """
    
    # Get the tables
    uvdata = hdulist['UV_DATA']
    
    # Pull out various bits of information we need to flag the file
    ## Source list
    srcs = uvdata.data['SOURCE']
    ## Time of each integration
    obsdates = uvdata.data['DATE']
    obstimes = uvdata.data['TIME']
    inttimes = uvdata.data['INTTIM']
    
    # Find the source blocks to see if there is something we can use
    # to help the dedispersion
    usrc = numpy.unique(srcs)
    blocks = []
    for src in usrc:
        valid = numpy.where( src == srcs )[0]
        
        blocks.append( [valid[0],valid[0]] )
        for v in valid[1:]:
            if v == blocks[-1][1] + 1 \
                and (obsdates[v] - obsdates[blocks[-1][1]] + obstimes[v] - obstimes[blocks[-1][1]])*86400 < 10*inttimes[v]:
                blocks[-1][1] = v
            else:
                blocks.append( [v,v] )
    blocks.sort()
    
    # Done
    return blocks


def main(args):
    # Parse the command line
    filenames = args.filename
    if args.uvout.lower() == "none":
        args.apply_uvout = False
        
    # Load the VLITE-Fast metadata
    t0 = time.time()
    event_window = load_event_window(args.metadata)
    mapper = load_antenna_mapper(args.metadata)
    unmapper = load_antenna_unmapper(args.metadata)
    
    # Load in the VLITE-Slow calibration information
    flags = {'ants':[]}
    #flags = load_caltab_fg(args.caltab)
    delays_cl_p0, delays_cl_p1, gains_cl_p0, gains_cl_p1 = load_caltab_cl(args.caltab)
    delays_cl = {0: delays_cl_p0, 1: delays_cl_p1}
    gains_cl = {0: gains_cl_p0, 1: gains_cl_p1}
    gains_bp_p0, gains_bp_p1 = load_caltab_bp(args.caltab)
    gains_bp = {0: gains_bp_p0, 1: gains_bp_p1}
    if args.apply_uvout:
        #uvw = load_uvout_uv(args.uvout)
        delays_sn_p0, delays_sn_p1, gains_sn_p0, gains_sn_p1 = load_uvout_sn(args.uvout)
        delays_sn = {0: delays_sn_p0, 1: delays_sn_p1}
        gains_sn = {0: gains_sn_p0, 1: gains_sn_p1}
    print("Loaded metadata and calibration information in %.3f s" % (time.time()-t0,))
    print("")
    
    for filename in filenames:
        t0 = time.time()
        print("Working on '%s'" % os.path.basename(filename))
        # Open the FITS IDI file and access the UV_DATA extension
        hdulist = astrofits.open(filename, mode='readonly')
        andata = hdulist['ANTENNA']
        fqdata = hdulist['FREQUENCY']
        srdata = hdulist['SOURCE']
        fgdata = None
        for hdu in hdulist[1:]:
            if hdu.header['EXTNAME'] == 'FLAG':
                fgdata = hdu
        uvdata = hdulist['UV_DATA']
        
        # Verify we can flag this data
        if uvdata.header['STK_1'] > 0:
            raise RuntimeError("Cannot flag data with STK_1 = %i" % uvdata.header['STK_1'])
        if uvdata.header['NO_STKD'] < 4:
            raise RuntimeError("Cannot flag data with NO_STKD = %i" % uvdata.header['NO_STKD'])
            
        # NOTE: Assumes that the Stokes parameters increment by -1
        polMapper = {}
        for i in xrange(uvdata.header['NO_STKD']):
            stk = uvdata.header['STK_1'] - i
            polMapper[i] = NUMERIC_STOKES[stk]
            
        # Pull out various bits of information we need to flag the file
        ## Antenna look-up table
        antLookup = {}
        for an, ai in zip(andata.data['ANNAME'], andata.data['ANTENNA_NO']):
            antLookup[an] = ai
        ## Frequency and polarization setup
        nBand, nFreq, nStk = uvdata.header['NO_BAND'], uvdata.header['NO_CHAN'], uvdata.header['NO_STKD']
        ## Baseline list
        bls = uvdata.data['BASELINE']
        uu = uvdata.data['UU']
        vv = uvdata.data['VV']
        ww = uvdata.data['WW']
        ## Time of each integration
        obsdates = uvdata.data['DATE']
        obstimes = uvdata.data['TIME']
        inttimes = uvdata.data['INTTIM']
        ## Source list
        srcs = uvdata.data['SOURCE']
        ## Band information
        fqoffsets = fqdata.data['BANDFREQ'].ravel()
        ## Frequency channels
        freq = (numpy.arange(nFreq)-(uvdata.header['CRPIX3']-1))*uvdata.header['CDELT3']
        freq += uvdata.header['CRVAL3']
        ## The actual visibility data
        flux = uvdata.data['FLUX'].astype(numpy.float32)
        
        # Convert the visibilities to something that we can easily work with
        nComp = flux.shape[1] // nBand // nFreq // nStk
        if nComp == 2:
            ## Case 1) - Just real and imaginary data
            flux = flux.view(numpy.complex64)
        else:
            ## Case 2) - Real, imaginary data + weights (drop the weights)
            flux = flux[:,0::nComp] + 1j*flux[:,1::nComp]
        flux.shape = (flux.shape[0], nBand, nFreq, nStk)
        
        # Find unique baselines and scans to work on
        ubls = numpy.unique(bls)
        blocks = get_source_blocks(hdulist)
        
        # Calibrate
        for i,block in enumerate(blocks):
            tS = time.time()
            print('  Working on scan %i of %i' % (i+1, len(blocks)))
            match = range(block[0],block[1]+1)
            
            bbls = numpy.unique(bls[match])
            times = obstimes[match] * 86400.0
            ints = inttimes[match]
            scanStart = datetime.utcfromtimestamp( utcjd_to_unix( obsdates[match[ 0]] + obstimes[match[ 0]] ) )
            scanStop  = datetime.utcfromtimestamp( utcjd_to_unix( obsdates[match[-1]] + obstimes[match[-1]] ) )
            print('    Scan spans %s to %s UTC' % (scanStart.strftime('%Y/%m/%d %H:%M:%S'), scanStop.strftime('%Y/%m/%d %H:%M:%S')))
            
            freq_comb = []
            for b,offset in enumerate(fqoffsets):
                freq_comb.append( freq + offset)
            freq_comb = numpy.concatenate(freq_comb)
            
            nBL = len(bbls)
            vis = flux[match,:,:,:]
            vis.shape = (vis.shape[0]//nBL, nBL, vis.shape[1]*vis.shape[2], vis.shape[3])
            print('      Scan contains %i times, %i baselines, %i bands/channels, %i polarizations' % vis.shape)
            
            vis.shape = (vis.shape[0]*nBL, vis.shape[2], vis.shape[3])
            for j in xrange(vis.shape[0]):
                a1, a2 = (bls[match[j]] >> 8), (bls[match[j]]) & 0xFF
                bl2 = (mapper[a1]+1)*256 + (mapper[a2] + 1)
                bl3 = mapper[a2]*256 + mapper[a1]
                #try:
                #    dw = ww[match[j]] - uvw[2][bl2](t)
                #    print(a1, a2, '@', dw, '&', delays_cl[0][mapper[a1]](t), delays_cl[1][mapper[a1]](t))
                #except KeyError:
                #    dw = 0.0
                #    
                #try:
                #    uu[match[j]] = uvw[0][bl2](t)
                #    vv[match[j]] = uvw[1][bl2](t)
                #    ww[match[j]] = uvw[2][bl2](t)
                #except KeyError:
                #    uu[match[j]] = vv[match[j]] = ww[match[j]] = 0.0
                
                t = obsdates[match[j]] + obstimes[match[j]]
                ## CalTab
                try:
                    # Antenna 1 - delays and gains
                    d1p0, d1p1 = delays_cl[0][mapper[a1]](t), delays_cl[1][mapper[a1]](t)
                    g1p0, g1p1 = gains_cl[0][mapper[a1]](t),  gains_cl[1][mapper[a1]](t)
                except KeyError:
                    print('skip', 'd1p*/g1p*', a2)
                    d1p0, d1p1 = 0.0, 0.0
                    g1p0, g1p1 = numpy.complex64(1.0), numpy.complex64(1.0)
                try:
                    # Antenna 2 - delays and gains
                    d2p0, d2p1 = delays_cl[0][mapper[a2]](t), delays_cl[1][mapper[a2]](t)
                    g2p0, g2p1 = gains_cl[0][mapper[a2]](t),  gains_cl[1][mapper[a2]](t)
                except KeyError:
                    print('skip', 'd2p*/g2p*', a2)
                    d2p0, d2p1 = 0.0, 0.0
                    g2p0, g2p1 = numpy.complex64(1.0), numpy.complex64(1.0)
                try:
                    # Antenna 1 - bandpass
                    b1p0, b1p1 = gains_bp[0][mapper[a1]](t, freq_comb),  gains_bp[1][mapper[a1]](t, freq_comb)
                except KeyError:
                    print('skip', 'b1p*', a1)
                    b1p0, b1p1 = numpy.complex64(1.0), numpy.complex64(1.0)
                try:
                    # Antenna 2 - bandpass
                    b2p0, b2p1 = gains_bp[0][mapper[a2]](t, freq_comb),  gains_bp[1][mapper[a2]](t, freq_comb)
                except KeyError:
                    print('skip', 'b2p*', a2)
                    b2p0, b2p1 = numpy.complex64(1.0), numpy.complex64(1.0)
                    
                ## uvout
                if args.apply_uvout:
                    try:
                        # Antenna 1 - supplemental delays and gains
                        sd1p0, sd1p1 = delays_sn[0][mapper[a1]](t), delays_sn[1][mapper[a1]](t)
                        sg1p0, sg1p1 = gains_sn[0][mapper[a1]](t),  gains_sn[1][mapper[a1]](t)  
                    except KeyError:
                        print('skip', 'sd1p*/sg1p*', a1)
                        sd1p0, sd1p1 = 0.0, 0.0
                        sg1p0, sg1p1 = numpy.complex64(1.0), numpy.complex64(1.0)
                    try:
                        # Antenna 2 - supplemental delays and gains
                        sd2p0, sd2p1 = delays_sn[0][mapper[a2]](t), delays_sn[1][mapper[a2]](t)            
                        sg2p0, sg2p1 = gains_sn[0][mapper[a2]](t),  gains_sn[1][mapper[a2]](t)
                    except KeyError:
                        print('skip', 'sd2p*/sg2p*', a2)
                        sd2p0, sd2p1 = 0.0, 0.0
                        sg2p0, sg2p1 = numpy.complex64(1.0), numpy.complex64(1.0)
                        
                # XX
                k = 0
                cgain = numpy.exp(2j*numpy.pi*freq_comb*(d1p0-d2p0)) / (g1p0*g2p0.conj()) / (b1p0*b2p0.conj())
                # cgain *= numpy.exp(2j*numpy.pi*freq_comb*-dw)
                vis[j,:,k] *= cgain
                if args.apply_uvout:
                    cgain = numpy.exp(2j*numpy.pi*freq_comb*(sd1p0-sd2p0)) / (sg1p0*sg2p0.conj())
                    vis[j,:,k] *= cgain
                    
                # YY
                k = 1
                cgain = numpy.exp(2j*numpy.pi*freq_comb*(d1p1-d2p1)) / (g1p1*g2p1.conj()) / (b1p1*b2p1.conj())
                # cgain *= numpy.exp(2j*numpy.pi*freq_comb*-dw)
                vis[j,:,k] *= cgain
                if args.apply_uvout:
                    cgain = numpy.exp(2j*numpy.pi*freq_comb*(sd1p1-sd2p1)) / (sg1p1*sg2p1.conj())
                    vis[j,:,k] *= cgain
                    
                # XY
                k = 2
                cgain = numpy.exp(2j*numpy.pi*freq_comb*(d1p0-d2p1)) / (g1p0*g2p1.conj()) / (b1p0*b2p1.conj())
                # cgain *= numpy.exp(2j*numpy.pi*freq_comb*-dw)
                vis[j,:,k] *= cgain
                if args.apply_uvout:
                    cgain = numpy.exp(2j*numpy.pi*freq_comb*(sd1p0-sd2p1)) / (sg1p0*sg2p1.conj())
                    vis[j,:,k] *= cgain
                    
                # YX
                k = 3
                cgain = numpy.exp(2j*numpy.pi*freq_comb*(d1p1-d2p0)) / (g1p1*g2p0.conj()) / (b1p1*b2p0.conj())
                # cgain *= numpy.exp(2j*numpy.pi*freq_comb*-dw)
                vis[j,:,k] *= cgain
                if args.apply_uvout:
                    cgain = numpy.exp(2j*numpy.pi*freq_comb*(sd1p1-sd2p0)) / (sg1p1*sg2p0.conj())
                    vis[j,:,k] *= cgain
                    
            vis.shape = (vis.shape[0], len(fqoffsets), vis.shape[1]//len(fqoffsets), vis.shape[2])
            flux[match,:,:,:] = vis
            
            print('      Statistics for this scan')
            print('      -> Elapsed - %.3f s' % (time.time()-tS,))
            
        ## Figure out our revision
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
            
        ## Build the FLAG table
        print('    FITS HDU')
        ### Filter and remap
        flags2 = {'ants':[], 'times':[], 'chans':[], 'pflags':[], 'reason':[]}
        nFlags = len(flags['ants'])
        for f in xrange(nFlags):
            if flags['times'][f][1] < obsdates[0] + obstimes[0]:
                continue
            elif flags['times'][f][0] > obsdates[-1] + obstimes[-1]:
                continue
                
            if flags['ants'][f][0] != 0 and flags['ants'][f][0] not in unmapper.keys():
                continue
            if flags['ants'][f][1] != 0 and flags['ants'][f][1] not in unmapper.keys():
                continue
                
            try:
                flags['ants'][f][0] = unmapper[flags['ants'][f][0]]
                flags['ants'][f][1] = unmapper[flags['ants'][f][1]]
            except KeyError:
                pass
            flags['times'][f][0] -= obsdates[0]
            flags['times'][f][1] -= obsdates[0]
            flags['chans'][f][0] = int(flags['chans'][f][0]*nFreq)
            flags['chans'][f][1] = int(flags['chans'][f][1]*nFreq + 1)
            
            flags2['ants'].append(flags['ants'][f])
            flags2['times'].append(flags['times'][f])
            flags2['chans'].append(flags['chans'][f])
            flags2['pflags'].append(flags['pflags'][f])
            flags2['reason'].append(flags['reason'][f])
        ### Mask the event, if requested
        if args.blank_event:
            flags2['ants'].append([0,0])
            flags2['times'].append(event_window)
            flags2['chans'].append([0,0])
            flags2['pflags'].append([1,1,1,1])
            flags2['reason'].append('EVENT')
        flags = flags2
        nFlags = len(flags['ants'])
        ### Columns
        c1 = astrofits.Column(name='SOURCE_ID', format='1J',           array=numpy.zeros((nFlags,), dtype=numpy.int32))
        c2 = astrofits.Column(name='ARRAY',     format='1J',           array=numpy.zeros((nFlags,), dtype=numpy.int32))
        c3 = astrofits.Column(name='ANTS',      format='2J',           array=numpy.array(flags['ants'], dtype=numpy.int32))
        c4 = astrofits.Column(name='FREQID',    format='1J',           array=numpy.zeros((nFlags,), dtype=numpy.int32))
        c5 = astrofits.Column(name='TIMERANG',  format='2E',           array=numpy.array(flags['times'], dtype=numpy.float32))
        c6 = astrofits.Column(name='BANDS',     format='%iJ' % nBand,  array=numpy.zeros((nFlags,), dtype=numpy.int32).squeeze())
        c7 = astrofits.Column(name='CHANS',     format='2J',           array=numpy.array(flags['chans'], dtype=numpy.int32))
        c8 = astrofits.Column(name='PFLAGS',    format='4J',           array=numpy.array(flags['pflags'], dtype=numpy.int32))
        c9 = astrofits.Column(name='REASON',    format='A40',          array=numpy.array(flags['reason']))
        c10 = astrofits.Column(name='SEVERITY', format='1J',           array=numpy.ones((nFlags,), dtype=numpy.int32))
        colDefs = astrofits.ColDefs([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
        ### The table itself
        flags = astrofits.BinTableHDU.from_columns(colDefs)
        ### The header
        flags.header['EXTNAME'] = ('FLAG', 'FITS-IDI table name')
        flags.header['EXTVER'] = (1 if fgdata is None else fgdata.header['EXTVER']+1, 'table instance number') 
        flags.header['TABREV'] = (2, 'table format revision number')
        for key in ('NO_STKD', 'STK_1', 'NO_BAND', 'NO_CHAN', 'REF_FREQ', 'CHAN_BW', 'REF_PIXL', 'OBSCODE', 'ARRNAM', 'RDATE'):
            flags.header[key] = (uvdata.header[key], uvdata.header.comments[key])
        flags.header['HISTORY'] = 'Flagged with %s, revision %s.%s%s' % (os.path.basename(__file__), branch, shortsha, dirty)
        flags.header['HISTORY'] = 'Flag source: %s' % args.caltab
        
        # Clean up the old FLAG tables, if any, and then insert the new table where it needs to be 
        if args.drop:
            ## Reset the EXTVER on the new FLAG table
            flags.header['EXTVER'] = (1, 'table instance number')
            ## Find old tables 
            toRemove = [] 
            for hdu in hdulist: 
                try: 
                    if hdu.header['EXTNAME'] == 'FLAG': 
                        toRemove.append( hdu ) 
                except KeyError: 
                    pass 
            ## Remove old tables 
            for hdu in toRemove: 
                ver = hdu.header['EXTVER'] 
                del hdulist[hdulist.index(hdu)] 
                print("  WARNING: removing old FLAG table - version %i" % ver )
        ### Insert the new table right before UV_DATA 
        #hdulist.insert(-1, flags)
        
        # Save
        print("  Saving to disk")
        ## What to call it
        outname = os.path.basename(filename)
        outname, outext = os.path.splitext(outname)
        outname = '%s_cal%s' % (outname, outext)
        ## Does it already exist or not
        if os.path.exists(outname):
            if not args.force:
                yn = raw_input("WARNING: '%s' exists, overwrite? [Y/n] " % outname)
            else:
                yn = 'y'
                
            if yn not in ('n', 'N'):
                os.unlink(outname)
            else:
                raise RuntimeError("Output file '%s' already exists" % outname)
        ## Open and create a new primary HDU
        hdulist2 = astrofits.open(outname, mode='append')
        primary =   astrofits.PrimaryHDU()
        processed = []
        for key in hdulist[0].header:
            if key in ('COMMENT', 'HISTORY'):
                if key not in processed:
                    parts = str(hdulist[0].header[key]).split('\n')
                    for part in parts:
                        primary.header[key] = part
                    processed.append(key)
            else:
                primary.header[key] = (hdulist[0].header[key], hdulist[0].header.comments[key])
        primary.header['HISTORY'] = 'Calibrated with %s, revision %s.%s%s' % (os.path.basename(__file__), branch, shortsha, dirty)
        primary.header['HISTORY'] = 'Metadata file: %s' % args.metadata
        primary.header['HISTORY'] = 'CalTab file: %s' % args.caltab
        if args.apply_uvout:
            primary.header['HISTORY'] = 'uvout file: %s' % args.uvout
        hdulist2.append(primary)
        hdulist2.flush()
        ## Copy the extensions over to the new file
        for hdu in hdulist[1:]:
            if hdu.header['EXTNAME'] == 'UV_DATA':
                ### Updated the UV_DATA table with the calibrated data
                #hdu.data['UU'][...] = uu
                #hdu.data['VV'][...] = vv
                #hdu.data['WW'][...] = ww
                flux = numpy.where(numpy.isfinite(flux), flux, 0.0)
                flux = flux.view(numpy.float32)
                flux = flux.astype(hdu.data['FLUX'].dtype)
                flux.shape = hdu.data['FLUX'].shape
                hdu.data['FLUX'][...] = flux
                
            hdulist2.append(hdu)
            hdulist2.flush()
        hdulist2.close()
        hdulist.close()
        print("  -> Calibrated FITS IDI file is '%s'" % outname)
        print("  Finished in %.3f s" % (time.time()-t0,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='apply VLITE-Slow calibration inforation to a  FITS-IDI files', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('metadata', type=str, 
                        help='VLITE-Fast ubjson metadata file')
    parser.add_argument('caltab', type=str,
                        help='VLITE-Slow CalTab UVFITS file')
    parser.add_argument('uvout', type=str,
                        help='VLITE-Slow uvout UVFITS file')
    parser.add_argument('filename', type=str, nargs='+',
                        help='FITS-IDI file')
    parser.add_argument('-b', '--blank-event', action='store_true',
                        help='blank the event using the FLAG table')
    parser.add_argument('-n', '--no-uvout', dest='apply_uvout', action='store_false',
                        help='do not apply the SN tables from the uvout file')
    parser.add_argument('-d', '--drop', action='store_true', 
                        help='drop all existing FLAG tables')
    parser.add_argument('-f', '--force', action='store_true', 
                        help='force overwriting of existing FITS-IDI files')
    args = parser.parse_args()
    main(args)
    
