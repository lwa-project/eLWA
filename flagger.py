"""
RFI flagging module for use with eLWA data.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import sys
import time
import numpy
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
    
from lsl.common.stations import lwa1
from lsl.statistics import robust
from lsl.correlator import uvutil


def flag_bandpass_freq(freq, data, width=250e3, clip=3.0, grow=True):
    """
    Given an array of frequencies and a 2-D (time by frequency) data set, 
    flag channels that appear to deviate from the median bandpass.  Returns
    a two-element tuple of the median bandpass and the channels to flag.
    Setting the 'grow' keyword to True enlarges each contiguous channel 
    mask by one channel on either side to mask edge effects.
    """
    
    # Create the median bandpass and setup the median smoothed bandpass model
    if data.dtype.kind == 'c':
        spec = numpy.abs(data)
    else:
        spec = data
    spec = numpy.median(spec, axis=0)
    smth = spec*0.0
    
    # Calculate the median window size - the target is determined from the 
    # 'width' keyword, which is assumed to be in Hz
    winSize = int(1.0*width/(freq[1]-freq[0]))
    winSize += ((winSize+1)%2)
    
    # Compute the smoothed bandpass model
    for i in xrange(smth.size):
        mn = max([0, i-winSize/2])
        mx = min([i+winSize/2+1, smth.size])
        smth[i] = numpy.median(spec[mn:mx])
    try:
        scl = robust.mean(smth)
        smth /= robust.mean(smth)
    except ValueError:
        scl = numpy.mean(smth)
        smth /= numpy.mean(smth)
        
    # Apply the model and find deviant channels
    bp = spec / smth
    try:
        dm = robust.mean(bp)
        ds = robust.std(bp)
    except ValueError:
        dm = numpy.mean(bp)
        ds = numpy.std(bp)
    bad = numpy.where( (numpy.abs(bp-dm) > clip*ds) | (smth < 0.1) )[0]
    
    # Make sure we have flagged appropriately and revert the flags as needed.  We
    # specifically need this when we have flagged everything because the bandpass 
    # is very smooth, i.e., LWA-SV and some of the LWA1-LWA-SV data
    if len(bad) == bp.size and ds < 1e-6 and spec.mean() > 1e-6:
        dm = numpy.mean(bp)
        ds = numpy.std(bp)
        bad = numpy.where( (numpy.abs(bp-dm) > clip*ds) | (smth < 0.1) )[0]
        
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
                    bad = numpy.append(bad, start)
                if stop < bp.size and stop not in bad:
                    bad = numpy.append(bad, stop)
        except IndexError:
            pass
            
    # Done
    return spec, bad


def flag_bandpass_time(times, data, width=30.0, clip=3.0):
    """
    Given an array of times and a 2-D (time by frequency) data set, flag
    times that appear to deviate from the overall drift in power.  Returns
    a two-element tuple of the median power drift and the times to flag.
    """
    
    # Create the median drift and setup the median smoothed drift model
    if data.dtype.kind == 'c':
        drift = numpy.abs(data)
    else:
        drift = data
    drift = numpy.median(drift, axis=1)
    smth = drift*0.0
    
    # Calculate the median window size - the target is determined from the 
    # 'width' keyword, which is assumed to be in s
    winSize = int(1.0*width/(times[1]-times[0]))
    winSize += ((winSize+1)%2)
    
    # Compute the smoothed drift model
    for i in xrange(smth.size):
        mn = max([0, i-winSize/2])
        mx = min([i+winSize/2+1, smth.size])
        smth[i] = numpy.median(drift[mn:mx])
    try:
        scl = robust.mean(smth)
        smth /= robust.mean(smth)
    except ValueError:
        scl = numpy.mean(smth)
        smth /= numpy.mean(smth)
        
    # Apply the model and find deviant times
    bp = drift / smth
    try:
        dm = robust.mean(bp)
        ds = robust.std(bp)
    except ValueError:
        dm = numpy.mean(bp)
        ds = numpy.std(bp)
    bad = numpy.where( (numpy.abs(bp-dm) > clip*ds) )[0]
    
    # Done
    return drift, bad


def mask_bandpass(antennas, times, freq, data, width_time=30.0, width_freq=250e3, clip=3.0, grow=True, verbose=False):
    """
    Given a list of antennas, an array of times, and array of frequencies, 
    and a 3-D (time by baseline by frequency) data set, flag RFI and return 
    a numpy.bool mask suitable for creating a masked array.  This function:
    1) Calls flag_bandpass_freq() and flag_bandpass_time() to create an
        initial mask, 
    2) Uses the median bandpass and power drift from (1) to flatten data, 
        and
    3) Flags any remaining deviant points in the flattened data.
    """
    
    # Load up the lists of baselines
    try:
        blList = uvutil.get_baselines([ant for ant in antennas if ant.pol == 0], include_auto=True)
    except AttributeError:
        blList = uvutil.get_baselines(antennas, include_auto=True)
        
    # Get the initial power and mask
    power = numpy.abs(data)
    try:
        mask = data.mask
    except AttributeError:
        mask = numpy.zeros(data.shape, dtype=numpy.bool)
        
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
                print("Flagging %6.1f%% on baseline %2i, %2i" % (100.0*subpower.mask.sum()/subpower.mask.size, ant1, ant2))
            continue
            
        ##
        ## Part 1 - Initial flagging with flag_bandpass_freq() and flag_bandpass_time()
        ##
        bp, flagsF = flag_bandpass_freq(freq, subpower, width=width_freq, clip=clip, grow=grow)
        drift, flagsT = flag_bandpass_time(times, subpower, width=width_time, clip=clip)
        
        ## Build up a numpy.ma version of the data using the flags we just found
        try:
            subpower.mask
            subpower = numpy.ma.array(subpower.data, mask=mask[:,i,:])
        except AttributeError:
            subpower = numpy.ma.array(subpower, mask=mask[:,i,:])
        subpower.mask[flagsT,:] = True
        subpower.mask[:,flagsF] = True
        
        ##
        ## Part 2 - Flatten the data with we we just found
        ##
        for j in xrange(subpower.shape[0]):
            subpower.data[j,:] /= bp
            subpower.data[j,:] /= drift[j]
            
        ##
        ## Part 3 - Flag deviant points
        ##
        try:
            dm = robust.mean(subpower)
            ds = robust.std(subpower)
        except ValueError:
            dm = numpy.mean(subpower)
            ds = numpy.std(subpower)
        bad = numpy.where( (numpy.abs(subpower-dm) > clip*ds) )
        subpower.mask[bad] = True
        
        ## Report, if requested
        if verbose:
            print("Flagging %6.1f%% on baseline %2i, %2i" % (100.0*subpower.mask.sum()/subpower.mask.size, ant1, ant2))
            
        ## Update the global mask
        mask[:,i,:] = subpower.mask
        
    # Done
    return mask


def mask_spurious(antennas, times, uvw, freq, data, clip=3.0, nearest=15, includeLWA=False, verbose=False):
    """
    Given a list of antenna, an array of times, an array of uvw coordinates, 
    an array of frequencies, and a 3-D (times by baselines by frequencies) 
    data set, look for and flag baselines with spurious correlations.  Returns
    a numpy.bool mask suitable for creating a masked array.
    """
    
    # Build the exclusion list
    exclude = ()
    if not includeLWA:
        exclude = (51, 52)
        
    # Load up the lists of baselines
    try:
        blList = uvutil.get_baselines([ant for ant in antennas if ant.pol == 0], include_auto=True)
    except AttributeError:
        blList = uvutil.get_baselines(antennas, include_auto=True)
        
    # Get the initial power and mask
    power = numpy.abs(data)
    try:
        mask = data.mask
    except AttributeError:
        mask = numpy.zeros(data.shape, dtype=numpy.bool)
        
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
            auto[bl[0]] = numpy.ma.median(power[:,i,:])
        elif ant1 not in exclude and ant2 not in exclude:
            cross.append(i)
            
    # Average the power over frequency
    power = numpy.ma.mean(power, axis=2)

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
        closest = [cross[j] for j in numpy.argsort(dist)[1:nearest+1]]
        
        ## Compute the relative gain corrections from the auto-correlations
        cgain = [numpy.sqrt(auto[blList[j][0]]*auto[blList[j][1]]) for j in closest]
        bgain = numpy.sqrt(auto[bl[0]]*auto[bl[1]])
        
        ## Flag deviant times
        try:
            dm = robust.mean(power[:,closest] / cgain)
            ds = robust.std(power[:,closest] / cgain)
        except ValueError:
            dm = numpy.mean(power[:,closest] / cgain)
            ds = numpy.std(power[:,closest] / cgain)
        bad = numpy.where( numpy.abs(power[:,i] / bgain - dm) > clip*ds )[0]
        mask[bad,i,:] = True
        
        ## Report, if requested
        if len(bad) > 0 and verbose:
            print("Flagging %3i integrations on baseline %2i, %2i" % (len(bad), ant1, ant2))
            
    # Cleanup the StringIO instance
    sys.stderr.close()
    sys.stderr = sys.__stderr__
    
    # Done
    return mask


def cleanup_mask(mask, max_frac=0.75):
    """
    Given a 3-D (times by baseline by frequency) numpy.ma array mask, look 
    for dimensions with high flagging.  Completely mask those with more than
    'max_frac' flagged.
    """
    
    nt, nb, nc = mask.shape
    # Time
    for i in xrange(nt):
        frac = 1.0*mask[i,:,:].sum() / nb / nc
        if frac > max_frac:
            mask[i,:,:] = True
    # Baseline
    for i in xrange(nb):
        frac = 1.0*mask[:,i,:].sum() / nt / nc
        if frac > max_frac:
            mask[:,i,:] = True
    # Frequency
    for i in xrange(nc):
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
        blList = uvutil.get_baselines([ant for ant in antennas if ant.pol == 0], include_auto=True)
    except AttributeError:
        blList = uvutil.get_baselines(antennas, include_auto=True)
        
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
        print("  %3i) Flagged %.1f%% on baseline %2i, %2i" % (i+1, frac, ant1, ant2))
        
    frac = 100.0*mask.sum() / mask.size
    print("Global Statistics:")
    print("  Flagged %.1f%% globally" % frac)
    print("  Antenna Breakdown:")
    for ant in sorted(antennaFracs.keys()):
        fracs = antennaFracs[ant]
        frac = 1.0 * sum(fracs) / len(fracs)
        print("    %2i flagged %.1f%% on average" % (ant, frac))


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
    full = mask.sum() / mask.size
    if full:
        flagsD = [(0, mask.shape[0]-1, 0, mask.shape[1]-1),]
        flagsP = [(times.min(), times.max(), freq.min(), freq.max()),]
        return flagsD, flagsP
        
    flagsD = []
    flagsP = []
    claimed = numpy.zeros(mask.shape, dtype=numpy.bool)
    group = numpy.where( mask )
    for l in xrange(len(group[0])):
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
