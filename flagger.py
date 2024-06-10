"""
RFI flagging module for use with eLWA data.
"""

import sys
import time
import numpy as np
from io import StringIO
    
from lsl.common.stations import lwa1
from lsl.statistics import robust
from lsl.correlator import uvutils


def flag_bandpass_freq(freq, data, width=250e3, clip=3.0, grow=True, freq_range=None):
    """
    Given an array of frequencies and a 2-D (time by frequency) data set, 
    flag channels that appear to deviate from the median bandpass.  Returns
    a two-element tuple of the median bandpass and the channels to flag.
    Setting the 'grow' keyword to True enlarges each contiguous channel 
    mask by one channel on either side to mask edge effects.
    """
    
    # Ready the frequency range flagger
    if freq_range is None:
        freq_range = [np.inf, -np.inf]
        
    # Create the median bandpass and setup the median smoothed bandpass model
    if data.dtype.kind == 'c':
        spec = np.abs(data)
    else:
        spec = data
    spec = np.median(spec, axis=0)
    smth = spec*0.0
    
    # Calculate the median window size - the target is determined from the 
    # 'width' keyword, which is assumed to be in Hz
    winSize = int(1.0*width/(freq[1]-freq[0]))
    winSize += ((winSize+1)%2)
    
    # Compute the smoothed bandpass model
    for i in range(smth.size):
        mn = max([0, i-winSize//2])
        mx = min([i+winSize//2+1, smth.size])
        smth[i] = np.median(spec[mn:mx])
    try:
        scl = robust.mean(smth)
        smth /= robust.mean(smth)
    except ValueError:
        scl = np.mean(smth)
        smth /= np.mean(smth)
        
    # Apply the model and find deviant channels
    bp = spec / smth
    try:
        dm = robust.mean(bp)
        ds = robust.std(bp)
    except ValueError:
        dm = np.mean(bp)
        ds = np.std(bp)
    fmask = np.zeros(freq.size, dtype=bool)
    fmask[np.where( (np.abs(bp-dm) > clip*ds) | (smth < 0.1) )] = True
    if isinstance(freq_range[0], (tuple, list)):
        for section in freq_range:
            fmask[np.where( ((freq >= section[0]) & (freq <= section[1])) )] = True
    else:
        fmask[np.where( ((freq >= freq_range[0]) & (freq <= freq_range[1])) )] = True
    bad = np.where(fmask == True)[0]
    
    # Make sure we have flagged appropriately and revert the flags as needed.  We
    # specifically need this when we have flagged everything because the bandpass 
    # is very smooth, i.e., LWA-SV and some of the LWA1-LWA-SV data
    if len(bad) == bp.size and ds < 1e-6 and spec.mean() > 1e-6:
        dm = np.mean(bp)
        ds = np.std(bp)
        fmask = np.zeros(freq.size, dtype=bool)
        fmask[np.where( (np.abs(bp-dm) > clip*ds) | (smth < 0.1) )] = True
        if isinstance(freq_range[0], (tuple, list)):
            for section in freq_range:
                fmask[np.where( ((freq >= section[0]) & (freq <= section[1])) )] = True
        else:
            fmask[np.where( ((freq >= freq_range[0]) & (freq <= freq_range[1])) )] = True
        bad = np.where(fmask == True)[0]
        
    if grow:
        try:
            # If we need to grow the mask, find contiguous flagged regions
            windows = [[bad[0],bad[0]],]
            for b in bad[1:]:
                if b == windows[-1][1] + 1:
                    windows[-1][1] = b
                else:
                    windows.append( [b,b] )
                    
            # For each flagged region, pad the beginning and the end
            for window in windows:
                start, stop = window
                start -= 1
                stop += 1
                if start > 0 and start not in bad:
                    bad = np.append(bad, start)
                if stop < bp.size and stop not in bad:
                    bad = np.append(bad, stop)
        except IndexError:
            pass
            
    # Done
    return spec, bad


def flag_bandpass_time(times, data, width=30.0, clip=3.0, time_range=None):
    """
    Given an array of times and a 2-D (time by frequency) data set, flag
    times that appear to deviate from the overall drift in power.  Returns
    a two-element tuple of the median power drift and the times to flag.
    """
    
    # Ready the frequency range flagger
    if time_range is None:
        time_range = [np.inf, -np.inf]
        
    # Create the median drift and setup the median smoothed drift model
    if data.dtype.kind == 'c':
        drift = np.abs(data)
    else:
        drift = data
    drift = np.median(drift, axis=1)
    smth = drift*0.0
    
    # Calculate the median window size - the target is determined from the 
    # 'width' keyword, which is assumed to be in s
    winSize = int(1.0*width/(times[1]-times[0]))
    winSize += ((winSize+1)%2)
    
    # Compute the smoothed drift model
    for i in range(smth.size):
        mn = max([0, i-winSize//2])
        mx = min([i+winSize//2+1, smth.size])
        smth[i] = np.median(drift[mn:mx])
    try:
        scl = robust.mean(smth)
        smth /= robust.mean(smth)
    except ValueError:
        scl = np.mean(smth)
        smth /= np.mean(smth)
        
    # Apply the model and find deviant times
    bp = drift / smth
    try:
        dm = robust.mean(bp)
        ds = robust.std(bp)
    except ValueError:
        dm = np.mean(bp)
        ds = np.std(bp)
    bad = np.where( (np.abs(bp-dm) > clip*ds) \
                   | ((times >= time_range[0]) & (times <= time_range[1])) )[0]
    
    # Done
    return drift, bad


def mask_bandpass(antennas, times, freq, data, width_time=30.0, width_freq=250e3, clip=3.0, grow=True, freq_range=None, time_range=None, verbose=False):
    """
    Given a list of antennas, an array of times, and array of frequencies, 
    and a 3-D (time by baseline by frequency) data set, flag RFI and return 
    a bool mask suitable for creating a masked array.  This function:
    1) Calls flag_bandpass_freq() and flag_bandpass_time() to create an
        initial mask, 
    2) Uses the median bandpass and power drift from (1) to flatten data, 
        and
    3) Flags any remaining deviant points in the flattened data.
    """
    
    # Load up the lists of baselines
    try:
        blList = uvutils.get_baselines([ant for ant in antennas if ant.pol == 0], include_auto=True)
    except AttributeError:
        blList = uvutils.get_baselines(antennas, include_auto=True)
        
    # Get the initial power and mask
    power = np.abs(data)
    try:
        mask = data.mask
    except AttributeError:
        mask = np.zeros(data.shape, dtype=bool)
        
    # Loop over baselines
    for i,bl in enumerate(blList):
        try:
            ant1, ant2 = bl[0].stand.id, bl[1].stand.id
        except AttributeError:
            ant1, ant2 = bl
            
        ##
        ## Part 0 - Sanity check
        ##
        subpower = power[:,i,:]
        if subpower.sum() == 0.0:
            mask[:,i,:] = True
            if verbose:
                print(f"Flagging {100.0*subpower.mask.sum()/subpower.mask.size:6.1f}% on baseline {ant1:2d}, {ant2:2d}")
            continue
            
        ##
        ## Part 1 - Initial flagging with flag_bandpass_freq() and flag_bandpass_time()
        ##
        bp, flagsF = flag_bandpass_freq(freq, subpower, width=width_freq, clip=clip, grow=grow,
                                        freq_range=freq_range)
        drift, flagsT = flag_bandpass_time(times, subpower, width=width_time, clip=clip,
                                           time_range=time_range)
        
        ## Build up a np.ma version of the data using the flags we just found
        try:
            subpower.mask
            subpower = np.ma.array(subpower.data, mask=mask[:,i,:])
        except AttributeError:
            subpower = np.ma.array(subpower, mask=mask[:,i,:])
        subpower.mask[flagsT,:] = True
        subpower.mask[:,flagsF] = True
        
        ##
        ## Part 2 - Flatten the data with we we just found
        ##
        for j in range(subpower.shape[0]):
            subpower.data[j,:] /= bp
            subpower.data[j,:] /= drift[j]
            
        ##
        ## Part 3 - Flag deviant points
        ##
        try:
            dm = robust.mean(subpower)
            ds = robust.std(subpower)
        except ValueError:
            dm = np.mean(subpower)
            ds = np.std(subpower)
        bad = np.where( (np.abs(subpower-dm) > clip*ds) )
        subpower.mask[bad] = True
        
        ## Report, if requested
        if verbose:
            print(f"Flagging {100.0*subpower.mask.sum()/subpower.mask.size:6.1f}% on baseline {ant1:2d}, {ant2:2d}")
            
        ## Update the global mask
        mask[:,i,:] = subpower.mask
        
    # Done
    return mask


def mask_spurious(antennas, times, uvw, freq, data, clip=3.0, nearest=15, includeLWA=False, verbose=False):
    """
    Given a list of antenna, an array of times, an array of uvw coordinates, 
    an array of frequencies, and a 3-D (times by baselines by frequencies) 
    data set, look for and flag baselines with spurious correlations.  Returns
    a bool mask suitable for creating a masked array.
    """
    
    # Build the exclusion list
    exclude = ()
    if not includeLWA:
        antLookup = {ant.config_name: ant.stand.id for ant in antennas if ant.pol == 0}
        for name in ('LWA1', 'LWASV', 'LWANA', 'OVROLWA'):
            try:
                exclude.append( antLookup[name] )
            except KeyError:
                pass
                
    # Load up the lists of baselines
    try:
        blList = uvutils.get_baselines([ant for ant in antennas if ant.pol == 0], include_auto=True)
    except AttributeError:
        blList = uvutils.get_baselines(antennas, include_auto=True)
        
    # Get the initial power and mask
    power = np.abs(data)
    try:
        mask = data.mask
    except AttributeError:
        mask = np.zeros(data.shape, dtype=bool)
        
    # Setup StringIO so that we can deal with the annoying
    # 'Warning: converting a masked element to nan.' messages.
    # This is a *little* dangerous since it can also hide 
    # exceptions but I guess that is the price.
    sys.stderr = StringIO()
    
    # Loop through the baselines to find out what is an auto-correlations 
    # and what is not.  If it is an auto-correlation, save the median power
    # so that we can compare the cross-correlations appropriately.  If it is
    # a cross-correlations, save the baseline index so that we can use it 
    # later.
    cross, auto = [], {}
    for i,bl in enumerate(blList):
        try:
            ant1, ant2 = bl[0].stand.id, bl[1].stand.id
        except AttributeError:
            ant1, ant2 = bl
            
        if ant1 == ant2:
            auto[bl[0]] = np.ma.median(power[:,i,:])
        elif ant1 not in exclude and ant2 not in exclude:
            cross.append(i)
            
    # Average the power over frequency
    power = np.ma.mean(power, axis=2)

    # Average the uvw coordinates over time and frequency
    uvw = uvw.mean(axis=0).mean(axis=1)
    
    # Loop through the baselines to find baselines that seem too strong based 
    # on their neighbors
    for i,bl in enumerate(blList):
        ## Is it an auto-correlation?  If so, ship it
        try:
            ant1, ant2 = bl[0].stand.id, bl[1].stand.id
        except AttributeError:
            ant1, ant2 = bl
        if ant1 == ant2:
            continue
        if ant1 in exclude or ant2 in exclude:
            continue
            
        ## Compute the distance to all other baselines and select the
        ## nearest 'nearest' non-auto-correlation points.
        dist = (uvw[cross,0]-uvw[i,0])**2 + (uvw[cross,1]-uvw[i,1])**2
        closest = [cross[j] for j in np.argsort(dist)[1:nearest+1]]
        
        ## Compute the relative gain corrections from the auto-correlations
        cgain = [np.sqrt(auto[blList[j][0]]*auto[blList[j][1]]) for j in closest]
        bgain = np.sqrt(auto[bl[0]]*auto[bl[1]])
        
        ## Flag deviant times
        try:
            dm = robust.mean(power[:,closest] / cgain)
            ds = robust.std(power[:,closest] / cgain)
        except ValueError:
            dm = np.mean(power[:,closest] / cgain)
            ds = np.std(power[:,closest] / cgain)
        bad = np.where( np.abs(power[:,i] / bgain - dm) > clip*ds )[0]
        mask[bad,i,:] = True
        
        ## Report, if requested
        if len(bad) > 0 and verbose:
            print(f"Flagging {len(bad):3d} integrations on baseline {ant1:2d}, {ant2:2d}")
            
    # Cleanup the StringIO instance
    sys.stderr.close()
    sys.stderr = sys.__stderr__
    
    # Done
    return mask


def cleanup_mask(mask, max_frac=0.75):
    """
    Given a 3-D (times by baseline by frequency) np.ma array mask, look 
    for dimensions with high flagging.  Completely mask those with more than
    'max_frac' flagged.
    """
    
    nt, nb, nc = mask.shape
    # Time
    for i in range(nt):
        frac = 1.0*mask[i,:,:].sum() / nb / nc
        if frac > max_frac:
            mask[i,:,:] = True
    # Baseline
    for i in range(nb):
        frac = 1.0*mask[:,i,:].sum() / nt / nc
        if frac > max_frac:
            mask[:,i,:] = True
    # Frequency
    for i in range(nc):
        frac = 1.0*mask[:,:,i].sum() / nt / nb
        if frac > max_frac:
            mask[:,:,i] = True
            
    # Done
    return mask


def summarize_mask(antennas, times, freq, mask):
    """
    Print a simple text-based report of the flagging specified in the 
    provided mask.
    """
    
    # Load up the lists of baselines
    try:
        blList = uvutils.get_baselines([ant for ant in antennas if ant.pol == 0], include_auto=True)
    except AttributeError:
        blList = uvutils.get_baselines(antennas, include_auto=True)
        
    # Build an antennas list to keep track of flagging
    antennaFracs = {}
    
    # Loop over baselines
    print("Baseline Statistics:")
    for i,bl in enumerate(blList):
        try:
            ant1, ant2 = bl[0].stand.id, bl[1].stand.id
        except AttributeError:
            ant1, ant2 = bl
            
        ## Pull out this baselines mask
        submask = mask[:,i,:]
        frac = 100.0*submask.sum() / submask.size
        
        ## Save on a per-antenna basis for a global report
        try:
            antennaFracs[ant1].append(frac)
        except KeyError:
            antennaFracs[ant1] = [frac,]
        if ant1 != ant2:
            try:
                antennaFracs[ant2].append(frac)
            except KeyError:
                antennaFracs[ant2] = [frac,]
                
        ## Baseline report
        print(f"  {i+1:3d}) Flagged {frac:.1f}% on baseline {ant1:2d}, {ant2:2d}")
        
    frac = 100.0*mask.sum() / mask.size
    print("Global Statistics:")
    print(f"  Flagged {frac:.1f}% globally")
    print("  Antenna Breakdown:")
    for ant in sorted(antennaFracs.keys()):
        fracs = antennaFracs[ant]
        frac = 1.0 * sum(fracs) / len(fracs)
        print(f"    {ant:2d} flagged {frac:.1f}% on average")


def create_flag_groups(times, freq, mask):
    """
    Given a 2-D (times by frequency) data set, create rectangular flagging 
    regions suitable for use in a FLAG table.  Returns a two-element tuple
    containing:
    1) a list of element based regions
        -> (idx0 start, idx0 stop, idx1 start, idx1 stop) and
    2) a list of physical based regions
        -> (time start, time stop, frequency start, frequency stop).
    """
    
    # Pass 0 - Check to see if the mask is full
    full = mask.sum() // mask.size
    if full:
        flagsD = [(0, mask.shape[0]-1, 0, mask.shape[1]-1),]
        flagsP = [(times.min(), times.max(), freq.min(), freq.max()),]
        return flagsD, flagsP
        
    flagsD = []
    flagsP = []
    claimed = np.zeros(mask.shape, dtype=bool)
    group = np.where( mask )
    for l in range(len(group[0])):
        i, j = group[0][l], group[1][l]
        if claimed[i,j]:
            continue
            
        ## Some things are large, check that first
        if i == 0:
            dx, dy = mask.shape[0], 1
            sub = mask[i:i+dx,j:j+dy]
            if sub.sum() == sub.size:
                while sub.sum() == sub.size and j+dy <= mask.shape[1]:
                    dy += 1
                    sub = mask[i:i+dx,j:j+dy]
                dy -= 1
                
                claimed[i:i+dx,j:j+dy] = True
                flagsD.append( (i, i+dx-1, j, j+dy-1) )
                flagsP.append( (times[i], times[i+dx-1], freq[j], freq[j+dy-1]) )
                continue
                
        elif j == 0:
            dx, dy = 1, mask.shape[1]
            sub = mask[i:i+dx,j:j+dy]
            if sub.sum() == sub.size:
                while sub.sum() == sub.size and i+dx <= mask.shape[0]:
                    dx += 1
                    sub = mask[i:i+dx,j:j+dy]
                dx -= 1
                
                claimed[i:i+dx,j:j+dy] = True
                flagsD.append( (i, i+dx-1, j, j+dy-1) )
                flagsP.append( (times[i], times[i+dx-1], freq[j], freq[j+dy-1]) )
                continue
                
        ## Grow along the first dimension
        dx, dy = 1, 1
        sub = mask[i:i+dx,j:j+dy]
        while sub.sum() == sub.size and i+dx <= mask.shape[0]:
            dx += 1
            sub = mask[i:i+dx,j:j+dy]
        dx -= 1
        
        ## Grow along the second dimension
        sub = mask[i:i+dx,j:j+dy]
        while sub.sum() == sub.size and j+dy <= mask.shape[1]:
            dy += 1
            sub = mask[i:i+dx,j:j+dy]
        dy -= 1
        
        ## Claim as processed
        claimed[i:i+dx,j:j+dy] = True
        flagsD.append( (i, i+dx-1, j, j+dy-1) )
        flagsP.append( (times[i], times[i+dx-1], freq[j], freq[j+dy-1]) )
        
    return flagsD, flagsP
