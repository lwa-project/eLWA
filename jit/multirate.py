# -*- coding: utf-8 -*-

"""
Module that provide the multi-rate F-engine needed to correlate data at 
different sample rates.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import ephem
import numpy

from lsl.common.constants import c as vLight
from lsl.common import dp as dp_common
from lsl.common.constants import *
from lsl.correlator import _core
from lsl.correlator.fx import pol2pol, noWindow

from jit import justInTimeOptimizer

__version__ = '0.2'
__revision__ = '$Rev$'
__all__ = ['getOptimalDelayPadding', 'MRF', 'MRX', 'MRX3', 
           '__version__', '__revision__', '__all__']


jitopt = justInTimeOptimizer()


def getOptimalDelayPadding(antennaSet1, antennaSet2, LFFT=64, SampleRate=None, CentralFreq=0.0, Pol='XX', phaseCenter='z'):
    # Decode the polarization product into something that we can use to figure 
    # out which antennas to use for the cross-correlation
    if Pol == '*':
        antennas1 = antennaSet1
        antennas2 = antennaSet2
        
    else:
        pol1, pol2 = pol2pol(Pol)
        
        antennas1 = [a for a in antennaSet1 if a.pol == pol1]
        antennas2 = [a for a in antennaSet2 if a.pol == pol1]
        
    # Combine the two sets and proceede
    antennas1.extend(antennas2)
    nStands = len(antennas1)
    
    # Create a reasonable mock setup for computing the delays
    if SampleRate is None:
        SampleRate = dp_common.fS
    freq = numpy.fft.fftfreq(LFFT, d=1.0/SampleRate)
    freq += float(CentralFreq)
    freq = numpy.fft.fftshift(freq)
    
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
        
        delays1[i,:] = antennas1[i].cable.delay(freq) - numpy.dot(source, xyz1) / vLight
    minDelay = delays1[:,dlyRef].min()
    
    # Round to the next lowest 5 us, negate, and return
    minDelay = numpy.floor( minDelay / 5e-6) * 5e-6
    return -minDelay


def MRF(signals, antennas, LFFT=64, Overlap=1, IncludeAuto=False, verbose=False, window=noWindow, SampleRate=None, CentralFreq=0.0, Pol='XX', GainCorrect=False, ReturnBaselines=False, ClipLevel=0, phaseCenter='z', delayPadding=40e-6):
    """
    Multi-rate F engine based on the lsl.correlator.fx.FXMaster() function.
    """
    
    # Decode the polarization product into something that we can use to figure 
    # out which antennas to use for the cross-correlation
    if Pol == '*':
        antennas1 = antennas
        signalsIndex1 = [i for (i, a) in enumerate(antennas)]
        
    else:
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
    freq = numpy.fft.fftfreq(lFactor*LFFT, d=1.0/SampleRate) + CentralFreq
    if doFFTShift:
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
        
        delays1[i,:] = antennas1[i].cable.delay(freq) - numpy.dot(source, xyz1) / vLight + delayPadding
    minDelay = delays1[:,dlyRef].min()
    if minDelay < 0:
        raise RuntimeError('Minimum data stream delay is negative: %.3f us' % (minDelay*1e6,))
        
    # Optimize
    if len(signalsIndex1) != signals.shape[0]:
        FEngine = jitopt.getFunction(_core.FEngineC2, signals[signalsIndex1,:], freq, delays1, LFFT=LFFT, Overlap=Overlap, SampleRate=SampleRate, ClipLevel=ClipLevel)
    else:
        FEngine = jitopt.getFunction(_core.FEngineC2, signals, freq, delays1, LFFT=LFFT, Overlap=Overlap, SampleRate=SampleRate, ClipLevel=ClipLevel)
    
    # F - defaults to running parallel in C via OpenMP
    if len(signalsIndex1) != signals.shape[0]:
        signalsF1, validF1 = FEngine(signals[signalsIndex1,:], freq, delays1, SampleRate=SampleRate)
    else:
        signalsF1, validF1 = FEngine(signals, freq, delays1, SampleRate=SampleRate)
    
    return freq, signalsF1, validF1, delays1


def MRX(signalsF1, validF1, signalsF2, validF2):
    """
    X-engine for the outputs of MRF().
    """
    
    # Optimize
    #XEngine = jitopt.getFunction('XEngine2', signalsF1, signalsF2, validF1, validF2)
    XEngine = _core.XEngine2

    output = XEngine(signalsF1, signalsF2, validF1, validF2)
    return output


def MRX3(signalsF1, validF1, signalsF2, validF2):
    """
    X-engine for the outputs of MRF().
    """
    
    # Optimize
    XEngine = jitopt.getFunction('XEngine3', signalsF1, signalsF2, validF1, validF2)

    output = XEngine(signalsF1, signalsF2, validF1, validF2)
    return output
