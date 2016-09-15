import os
import sys
import ephem
import numpy

from lsl.common.constants import c as vLight
from lsl.common import dp as dp_common
from lsl.common.constants import *
from lsl.correlator import _core, uvUtils
from lsl.correlator.fx import pol2pol, noWindow


def MRF(signals, antennas, LFFT=64, Overlap=1, IncludeAuto=False, verbose=False, window=noWindow, SampleRate=None, CentralFreq=0.0, Pol='XX', GainCorrect=False, ReturnBaselines=False, ClipLevel=0, phaseCenter='z'):
	# Decode the polarization product into something that we can use to figure 
	# out which antennas to use for the cross-correlation
	pol1, pol2 = pol2pol(Pol)
	
	antennas1 = [a for a in antennas if a.pol == pol1]
	signalsIndex1 = [i for (i, a) in enumerate(antennas) if a.pol == pol1]
	
	nStands = len(antennas1)
	
	# Figure out if we are working with complex (I/Q) data or only real.  This
	# will determine how the FFTs are done since the real data mirrors the pos-
	# itive and negative Fourier frequencies.
	if signals.dtype.kind == 'c':
		lFactor = 1
		doFFTShift = True
		CentralFreq = float(CentralFreq)
	else:
		lFactor = 2
		doFFTShift = False
		
	if SampleRate is None:
		SampleRate = dp_common.fS
	freq = numpy.fft.fftfreq(lFactor*LFFT, d=1.0/SampleRate)
	if doFFTShift:
		freq += CentralFreq
		freq = numpy.fft.fftshift(freq)
	freq = freq[:LFFT]
	
	# Get the location of the phase center in radians and create a 
	# pointing vector
	if phaseCenter == 'z':
		azPC = 0.0
		elPC = numpy.pi/2.0
	else:
		if isinstance(phaseCenter, ephem.Body):
			azPC = phaseCenter.az * 1.0
			elPC = phaseCenter.alt * 1.0
		else:
			azPC = phaseCenter[0]*numpy.pi/180.0
			elPC = phaseCenter[1]*numpy.pi/180.0
			
	source = numpy.array([numpy.cos(elPC)*numpy.sin(azPC), 
					  numpy.cos(elPC)*numpy.cos(azPC), 
					  numpy.sin(elPC)])
					  
	# Define the cable/signal delay caches to help correlate along and compute 
	# the delays that we need to apply to align the signals
	dlyRef = len(freq)/2
	delays1 = numpy.zeros((nStands,LFFT))
	for i in list(range(nStands)):
		xyz1 = numpy.array([antennas1[i].stand.x, antennas1[i].stand.y, antennas1[i].stand.z])
		
		delays1[i,:] = antennas1[i].cable.delay(freq) - numpy.dot(source, xyz1) / vLight + 40e-6
	minDelay = delays1[:,dlyRef].min()
	if minDelay < 0:
		print 'II', antennas1[0].stand.id, antennas1[0].pol, minDelay*1e6, minDelay*SampleRate, signals.shape
	#delays1 *= 0.0
	
	# F - defaults to running parallel in C via OpenMP
	if window is noWindow:
		# Data without a window function provided
		if signals.dtype.kind == 'c':
			FEngine = _core.FEngineC2
		else:
			FEngine = _core.FEngineR2
		signalsF1, validF1 = FEngine(signals[signalsIndex1,:], freq, delays1, LFFT=LFFT, Overlap=Overlap, SampleRate=SampleRate, ClipLevel=ClipLevel)
		
	else:
		# Data with a window function provided
		if signals.dtype.kind == 'c':
			FEngine = _core.FEngineC3
		else:
			FEngine = _core.FEngineR3
		signalsF1, validF1 = FEngine(signals[signalsIndex1,:], freq, delays1, LFFT=LFFT, Overlap=Overlap, SampleRate=SampleRate, ClipLevel=ClipLevel, window=window)
		
	return freq, signalsF1/numpy.sqrt(LFFT), validF1, delays1


def MRX(signalsF1, validF1, signalsF2, validF2):
	output = _core.XEngine2(signalsF1, signalsF2, validF1, validF2)
	return output
