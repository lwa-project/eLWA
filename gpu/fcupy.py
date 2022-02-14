"""
Module that provides GPU-based F-engines built using cupy.
"""

# Python2 compatibility
from __future__ import print_function, division

import ephem
import cupy as cp
import numpy as np

from cupyx.scipy.fft import fft as cufft

from astropy.constants import c as vLight
from astropy.coordinates import AltAz as AstroAltAz

vLight = vLight.to('m/s').value

from lsl.common import dp as dp_common
from lsl.correlator import _core
from lsl.correlator.fx import pol_to_pols, null_window

from .cache import get_from_shape, get_from_ndarray, copy_using_cache

__version__ = '0.1'
__all__ = ['fengine', 'frequency_correct']


_FENGINE = cp.RawModule(code=r"""
#define M_PI 3.1415926535897931e+0

inline __host__ __device__ void operator/=(float2 &a, float b) {
  a.x /= b;
  a.y /= b;
}

extern "C" __global__
void delay_and_trim_r(const float* signals,
                      const int* sample_delays,
                      int nStand,
                      int nSamp,
                      int nChan,
                      int nWin,
                      float2* output,
                      unsigned char* valid) {
  int a = blockIdx.x;
  int c = blockIdx.y*blockDim.x + threadIdx.x;
  int w = blockIdx.z*blockDim.y + threadIdx.y;
  
  int s = w*nChan + c;
  int d = *(sample_delays + a);

  if( c >= nChan || w >= nWin ) {
    // Nothing'
  } else {
    float2 temp = make_float2(0, 0);
    if( s + d < nSamp ) {
      temp.x = *(signals + a*nSamp + s + d);
    } else {
      *(valid + a*nWin + w) = 0;
    }
    *(output + a*nWin*nChan + w*nChan + c) = temp;
  }
}

extern "C" __global__
void delay_and_trim_c(const float2* signals,
                      const int* sample_delays,
                      int nStand,
                      int nSamp,
                      int nChan,
                      int nWin,
                      float2* output,
                      unsigned char* valid) {
  int a = blockIdx.x;
  int c = blockIdx.y*blockDim.x + threadIdx.x;
  int w = blockIdx.z*blockDim.y + threadIdx.y;
  
  int s = w*nChan + c;
  int d = *(sample_delays + a);

  if( c >= nChan || w >= nWin ) {
    // Nothing'
  } else {
    if( s + d < nSamp ) {
      *(output + a*nWin*nChan + w*nChan + c) = *(signals + a*nSamp + s + d);
    } else {
      *(output + a*nWin*nChan + w*nChan + c) = make_float2(0, 0);
      *(valid + a*nWin + w) = 0;
    }
  }
}

extern "C" __global__
void postf_rotate_r(const float2* signals,
                    const double* frequency,
                    const int* sample_delays,
                    const double* frac_delays,
                    double sample_rate,
                    int nStand,
                    int nChan,
                    int nWin,
                    float2* output) {
  int a = blockIdx.x;
  int c = blockIdx.y*blockDim.x + threadIdx.x;
  int w = blockIdx.z*blockDim.y + threadIdx.y;
  if( c >= nChan || w >= nWin ) {
    // Nothin'
  } else {
    float arg;
    float2 phase;
    arg = 2*M_PI * *(frequency + c) * *(frac_delays + a*nChan + c) \
          + 2*M_PI * *(frequency + 0) / sample_rate * *(sample_delays + a);
    phase.x = cos(arg);
    phase.y = sin(arg);
    phase /= sqrt(2.0*nChan);
    
    float2 temp, temp2;
    temp = *(signals + a*nWin*2*nChan + w*2*nChan + c);
    temp2.x = temp.x*phase.x - temp.y*phase.y;
    temp2.y = temp.y*phase.x + temp.x*phase.y;
    
    *(output + a*nChan*nWin + c*nWin + w) = temp2;
  }
}

extern "C" __global__
void postf_rotate_c(const float2* signals,
                    const double* frequency,
                    const int* sample_delays,
                    const double* frac_delays,
                    double sample_rate,
                    int nStand,
                    int nChan,
                    int nWin,
                    float2* output) {
  int a = blockIdx.x;
  int c = blockIdx.y*blockDim.x + threadIdx.x;
  int w = blockIdx.z*blockDim.y + threadIdx.y;
  if( c >= nChan || w >= nWin ) {
    // Nothin'
  } else {
    float arg;
    float2 phase;
    arg = 2*M_PI * *(frequency + c) * *(frac_delays + a*nChan + c) \
          + 2*M_PI * *(frequency + nChan/2) / sample_rate * *(sample_delays + a);
    phase.x = cos(arg);
    phase.y = sin(arg);
    phase /= sqrt((float) nChan);
    
    float2 temp, temp2;
    temp = *(signals + a*nWin*nChan + w*nChan + c);
    temp2.x = temp.x*phase.x - temp.y*phase.y;
    temp2.y = temp.y*phase.x + temp.x*phase.y;
    
    *(output + a*nChan*nWin + c*nWin + w) = temp2;
  }
}""")

_DELAY_AND_TRIM_R = _FENGINE.get_function('delay_and_trim_r')
_DELAY_AND_TRIM_C = _FENGINE.get_function('delay_and_trim_c')
_POSTF_ROTATE_R = _FENGINE.get_function('postf_rotate_r')
_POSTF_ROTATE_C = _FENGINE.get_function('postf_rotate_c')


def _fengine(signals, frequency, delays, LFFT=64, sample_rate=196e6, blockDim=(4,16)):
    is_complex = signals.dtype == np.complex64
    sample_delays = np.int32(np.round(delays[:,[delays.shape[1]//2,]]*sample_rate))
    frac_delays = delays - sample_delays/sample_rate
    
    signals = copy_using_cache(signals)
    frequency = copy_using_cache(frequency)
    sample_delays = copy_using_cache(sample_delays)
    frac_delays = copy_using_cache(frac_delays)
    
    nAnt, nSamp = signals.shape
    nChan = LFFT*(1 if is_complex else 2)
    nWin = nSamp // nChan
    norm = np.sqrt(nChan)
    
    output = get_from_shape((nAnt,nWin,nChan), dtype=np.complex64)
    validF = get_from_shape((nAnt,nWin), dtype=np.uint8)
    validF[...] = np.uint8(1)
    
    nct, nwt = blockDim
    ncb = int(np.ceil(nChan/nct))
    nwb = int(np.ceil(nWin/nwt))
    nab = nAnt
    
    delay_func = _DELAY_AND_TRIM_C if is_complex else _DELAY_AND_TRIM_R
    delay_func((nab,ncb,nwb), (nct,nwt),
               (signals, sample_delays,
                cp.int32(nAnt), cp.int32(nSamp),
                cp.int32(nChan), cp.int32(nWin),
                output, validF))
    
    if is_complex:
        cufft(output, n=nChan, axis=2, overwrite_x=True)
        signalsF = cp.fft.fftshift(output, axes=2)
    else:
        signalsF = cp.fft.fft(output, n=nChan, axis=2)
        nChan //= 2
        
    outputF = get_from_shape((nAnt,nChan,nWin), dtype=np.complex64)
    
    nct, nwt = blockDim
    ncb = int(np.ceil(nChan/nct))
    nwb = int(np.ceil(nWin/nwt))
    nab = nAnt
    
    rotate_func = _POSTF_ROTATE_C if is_complex else _POSTF_ROTATE_R
    rotate_func((nab,ncb,nwb), (nct,nwt),
                (signalsF, frequency, sample_delays, frac_delays, cp.float64(sample_rate),
                 cp.int32(nAnt), cp.int32(nChan), cp.int32(nWin),
                 outputF))
    
    return outputF, validF


def fengine(signals, antennas, LFFT=64, overlap=1, include_auto=False, verbose=False, window=null_window, sample_rate=None, central_freq=0.0, Pol='XX', gain_correct=False, return_baselines=False, clip_level=0, phase_center='z', delayPadding=40e-6, blockDim=(4,16)):
    """
    Multi-rate F engine based on the lsl.correlator.fx.FXMaster() function.
    """
    
    # This currently only supports a sub-set of all options
    assert(overlap == 1)
    assert(window == null_window)
    assert(clip_level == 0)
    
    # Decode the polarization product into something that we can use to figure 
    # out which antennas to use for the cross-correlation
    if Pol == '*':
        antennas1 = antennas
        signalsIndex1 = [i for (i, a) in enumerate(antennas)]
        
    else:
        pol1, pol2 = pol_to_pols(Pol)
        
        antennas1 = [a for a in antennas if a.pol == pol1]
        signalsIndex1 = [i for (i, a) in enumerate(antennas) if a.pol == pol1]
    
    nStands = len(antennas1)
    
    # Figure out if we are working with complex (I/Q) data or only real.  This
    # will determine how the FFTs are done since the real data mirrors the pos-
    # itive and negative Fourier frequencies.
    if signals.dtype.kind == 'c':
        lFactor = 1
        doFFTShift = True
        central_freq = float(central_freq)
    else:
        lFactor = 2
        doFFTShift = False
        
    if sample_rate is None:
        sample_rate = dp_common.fS
    freq = np.fft.fftfreq(lFactor*LFFT, d=1.0/sample_rate) + central_freq
    if doFFTShift:
        freq = np.fft.fftshift(freq)
    freq = freq[:LFFT]
    
    # Get the location of the phase center in radians and create a 
    # pointing vector
    if phase_center == 'z':
        azPC = 0.0
        elPC = np.pi/2.0
    else:
        if isinstance(phase_center, ephem.Body):
            azPC = phase_center.az * 1.0
            elPC = phase_center.alt * 1.0
        elif isinstance(phase_center, AstroAltAz):
            azPC = phase_center.az.radian
            elPC = phase_center.alt.radian
        else:
            azPC = phase_center[0]*np.pi/180.0
            elPC = phase_center[1]*np.pi/180.0
            
    source = np.array([np.cos(elPC)*np.sin(azPC), 
                    np.cos(elPC)*np.cos(azPC), 
                    np.sin(elPC)])
                    
    # Define the cable/signal delay caches to help correlate along and compute 
    # the delays that we need to apply to align the signals
    dlyRef = len(freq)//2
    delays1 = np.zeros((nStands,LFFT))
    for i in list(range(nStands)):
        try:
            xyz1 = np.array([antennas1[i].apparent_stand.x, antennas1[i].apparent_stand.y, antennas1[i].apparent_stand.z])
        except AttributeError:
            xyz1 = np.array([antennas1[i].stand.x, antennas1[i].stand.y, antennas1[i].stand.z])
            
        delays1[i,:] = antennas1[i].cable.delay(freq) - np.dot(source, xyz1) / vLight + delayPadding
    minDelay = delays1[:,dlyRef].min()
    if minDelay < 0:
        raise RuntimeError('Minimum data stream delay is negative: %.3f us' % (minDelay*1e6,))
        
    # F - defaults to running parallel in C via OpenMP
    if len(signalsIndex1) != signals.shape[0]:
        signalsF1, validF1 = _fengine(signals[signalsIndex1,:], freq, delays1, LFFT=LFFT, sample_rate=sample_rate, blockDim=blockDim)
    else:
        signalsF1, validF1 = _fengine(signals, freq, delays1, LFFT=LFFT, sample_rate=sample_rate, blockDim=blockDim)
        
    return freq, signalsF1, validF1, delays1


_FREQ_FRINGE_ROTATE = cp.RawKernel(r"""
#define M_PI 3.1415926535897931e+0

extern "C" __global__
void freq_fringe_rotate(const double *time,
                       float2 *signals,
                       double freqOffset,
                       int nStand,
                       int nChan,
                       int nWin) {
  int a = blockIdx.x;
  int c = blockIdx.y*blockDim.x + threadIdx.x;
  int w = blockIdx.z*blockDim.y + threadIdx.y;
  
  if( c >= nChan || w >= nWin) {
    // Nothin'
  } else {
    double arg = -2*M_PI*freqOffset * *(time + w);
    double2 phase;
    phase.x = cos(arg);
    phase.y = sin(arg);
    
    float2 temp, temp2;
    temp = *(signals + a*nChan*nWin + c*nWin + w);
    temp2.x = temp.x*phase.x - temp.y*phase.y;
    temp2.y = temp.y*phase.x + temp.x*phase.y;
    
    *(signals + a*nChan*nWin + c*nWin + w) = temp2;
  }
}              
""", 'freq_fringe_rotate')


def frequency_correct(t, signals, freqOffset, blockDim=(4,16)):
    nStand, nChan, nWin = signals.shape
    
    t = copy_using_cache(t[::nChan])
    signals = copy_using_cache(signals)
    
    nct, nwt = blockDim
    ncb = ncb = int(np.ceil(nChan/nct))
    nwb = int(np.ceil(nWin/nwt))
    nab = nStand
    
    _FREQ_FRINGE_ROTATE((nab,ncb,nwb), (nct,nwt),
                        (t, signals, cp.float64(freqOffset),
                         cp.int32(nStand), cp.int32(nChan), cp.int32(nWin)))
    return signals
