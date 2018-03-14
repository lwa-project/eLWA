"""
RFI flagging module for use with eLWA data.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import time
import numpy

from lsl.common.stations import lwa1
from lsl.statistics import robust
from lsl.correlator import uvUtils


def flag_bandpass_freq(freq, data, clip=3.0, grow=True):
	"""
	Given an array of frequencies and a 2-D (time by frequency) data set, 
	flag channels that appear to deviate from the median bandpass.  Returns
	a two-element tuple of the median bandpass and the channels to flag.
	Setting	the 'grow' keyword to True enlarges each contiguous channel 
	mask by one channel on either side to mask edge effects.
	"""
	
	# Create the median bandpass and setup the median smoothed bandpass model
	spec = numpy.abs(data)
	spec = numpy.median(spec, axis=0)
	smth = spec*0.0
	
	# Calculate the median window size - the target is 250 kHz in frequency
	winSize = int(250e3/(freq[1]-freq[0]))
	winSize += ((winSize+1)%2)
	
	# Compute the smoothed bandpass model
	for i in xrange(smth.size):
		mn = max([0, i-winSize/2])
		mx = min([i+winSize, smth.size])
		smth[i] = numpy.median(spec[mn:mx])
	smth /= robust.mean(smth)
	
	# Apply the model and find deviant channels
	bp = spec / smth
	try:
		dm = robust.mean(bp)
		ds = robust.std(bp)
	except ValueError:
		dm = numpy.mean(bp)
		ds = robust.std(bp)
	bad = numpy.where( (numpy.abs(bp-dm) > clip*ds) )[0]
	
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


def flag_bandpass_time(times, data, clip=3.0):
	"""
	Given an array of times and a 2-D (time by frequency) data set, flag
	times that appear to deviate from the overall drift in power.  Returns
	a two-element tuple of the median power drift and the times to flag.
	"""
	
	# Create the median drift and setup the median smoothed drift model
	drift = numpy.abs(data)
	drift = numpy.median(drift, axis=1)
	smth = drift*0.0
	
	# Calculate the median window size - the target is 30 seconds in time
	winSize = int(30.0/(times[1]-times[0]))
	winSize += ((winSize+1)%2)
	
	# Compute the smoothed drift model
	for i in xrange(smth.size):
		mn = max([0, i-winSize/2])
		mx = min([i+winSize, smth.size])
		smth[i] = numpy.median(drift[mn:mx])
	smth /= robust.mean(smth)
	
	# Apply the model and find deviant times
	bp = drift / smth
	try:
		dm = robust.mean(bp)
		ds = robust.std(bp)
	except ValueError:
		dm = numpy.mean(bp)
		ds = robust.std(bp)
	bad = numpy.where( (numpy.abs(bp-dm) > clip*ds) )[0]
	
	# Done
	return drift, bad


def mask_bandpass(times, freq, data, clip=3.0, grow=True):
	"""
	Given an array of times, and array of frequencies, and a 2-D (time by 
	frequency) data set, flag RFI and return a numpy.bool mask suitable for
	creating a masked array.  This function:
	 1) Calls flag_bandpass_freq() and flag_bandpass_time() to create an
	    initial mask, 
	 2) Uses the median bandpass and power drift from (1) to flatten data, 
	    and
	 3) Flags any remaining deviant points in the flattened data.
	"""
	
	#
	# Part 1 - Initial flagging with flag_bandpass_freq() and flag_bandpass_time()
	#
	bp, flagsF = flag_bandpass_freq(freq, data, clip=clip, grow=grow)
	drift, flagsT = flag_bandpass_time(times, data, clip=clip)
	
	# Build up a numpy.ma version of the data using the flags we just found
	data2 = numpy.ma.array(numpy.abs(data), mask=numpy.zeros(data.shape, dtype=numpy.bool))
	data2.mask[flagsT,:] = True
	data2.mask[:,flagsF] = True
	
	#
	# Part 2 - Flatten the data with we we just found
	#
	for i in xrange(data2.shape[0]):
		data2[i,:] /= bp
		data2[i,:] /= drift[i]
		
	#
	# Part 3 - Flag deviant points
	#
	try:
		dm = robust.mean(data2)
		ds = robust.std(data2)
	except ValueError:
		dm = numpy.mean(data2)
		ds = numpy.std(data2)
	bad = numpy.where( (numpy.abs(data2-dm) > clip*ds) )
	
	# Update the mask with what we have just learned
	mask = data2.mask
	mask[bad] = True
	
	# Done
	return mask


def mask_spurious(antennas, times, uvw, freq, data, clip=3.0, nearest=20, verbose=False):
	"""
	Given a list of antenna, an array of times, an array of uvw coordinates, 
	an array of frequencies, and a 3-D (times by baselines by frequencies) 
	data set, look for and flag baselines with spurious correlations.  Returns
	a numpy.bool mask suitable for creating a masked array.
	"""
	
	# Load up the lists of baselines
	try:
		blList = uvUtils.getBaselines([ant for ant in antennas if ant.pol == 0], IncludeAuto=True)
	except AttributeError:
		blList = uvUtils.getBaselines(antennas, IncludeAuto=True)
		
	# Get the initial power and mask
	try:
		power = numpy.abs(data.data)
		mask = data.mask*1
	except AttributeError:
		power = numpy.abs(data)
		mask = numpy.zeros(data.shape, dtype=numpy.bool)
		
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
			auto[bl[0]] = numpy.median(power[:,i,:])
		else:
			cross.append(i)
			
	# Average the power over frequency
	power = power.mean(axis=2)
	
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
		if len(bad) > 0 and verbose:
			print "Flagging %i integrations on baseline %i, %i" % (len(bad), ant1, ant2)
			
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
	while claimed.sum() < mask.sum():
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
