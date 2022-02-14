"""
Module that provides miscellaneous GPU-based correlation functions built using
cupy.
"""

# Python2 compatibility
from __future__ import print_function, division

import cupy as cp
import numpy as np

from .cache import get_from_shape, get_from_ndarray, copy_using_cache


__version__ = '0.1'
__all__ = ['inplace_freq_roll', 'pol_fill']


_INPLACE_FREQ_ROLL = cp.RawKernel(r"""
extern "C" __global__
void inplace_freq_roll(float2* signals,
                       int offset,
                       int nStand,
                       int nChan,
                       int nWin) {
  int s = blockIdx.x*blockDim.x + threadIdx.x;
  int w = blockIdx.y*blockDim.y + threadIdx.y;
  
  if( s >= nStand || w >= nWin) {
    // Nothin'
  } else {
    int c;
    float2 temp;
    for(c=-offset; c<nChan; c++) {
      temp = *(signals + s*nChan*nWin + c*nWin + w);
      *(signals + s*nChan*nWin + (c+offset)*nWin + w) = temp;
    }
  }
}
""", 'inplace_freq_roll')


def inplace_freq_roll(signals, offset, blockDim=(4,16)):
    nStand, nChan, nWin = signals.shape
    assert(offset <= 0)
    
    signals = copy_using_cache(signals)
    if offset == 0:
        return signals
        
    nat, nwt = blockDim
    nab = int(np.ceil(nStand/nat))
    nwb = int(np.ceil(nWin/nwt))
    
    _INPLACE_FREQ_ROLL((nab,nwb), (nat,nwt),
                       (signals, cp.int32(offset),
                        cp.int32(nStand), cp.int32(nChan), cp.int32(nWin)))
    
    return signals


_POL_FILL = cp.RawKernel(r"""
extern "C" __global__
void pol_fill(const float2 *signals,
              const unsigned char *valid,
              const unsigned char *swap,
              int nStand,
              int nChanAct,
              int nWinAct,
              int nChan,
              int nWin,
              int offset,
              float2 *outputX,
              unsigned char *validX,
              float2 *outputY,
              unsigned char *validY) {
  int c = blockIdx.x*blockDim.x + threadIdx.x;
  int w = blockIdx.y*blockDim.y + threadIdx.y;
  
  if( c >= nChan || w >= nWin) {
    // Nothin'
  } else {
    int s;
    float2 tempA, tempB, tempT;
    unsigned char validA, validB, validT;
    for(s=0; s<nStand/2; s++) {
      tempA = *(signals + (2*s+0)*nChanAct*nWinAct + c*nWinAct + w);
      tempB = *(signals + (2*s+1)*nChanAct*nWinAct + c*nWinAct + w);
      validA = *(valid + (2*s+0)*nWinAct + w);
      validB = *(valid + (2*s+1)*nWinAct + w);
      
      if( *(swap + s) > 0 ) {
        tempT = tempA;
        tempA = tempB;
        tempB = tempT;
        
        validT = validA;
        validA = validB;
        validB = validT;
      }
      
      *(outputX + (s+offset)*nChan*nWin + c*nWin + w) = tempA;
      *(outputY + (s+offset)*nChan*nWin + c*nWin + w) = tempB;
      *(validX + (s+offset)*nWin + w) = validA;
      *(validY + (s+offset)*nWin + w) = validB;
    }
  }
}
""", 'pol_fill') 


def pol_fill(signals, valid, swap, offset, outputX, validX, outputY, validY, blockDim=(4,16)):
    nStand, nChanAct, nWinAct = signals.shape
    _, nChan, nWin = outputX.shape
    assert(offset >= 0)
    
    signals = copy_using_cache(signals)
    valid = copy_using_cache(valid)
    swap = copy_using_cache(swap)
    outputX = copy_using_cache(outputX, tag='pfX')
    validX = copy_using_cache(validX, tag='pfX')
    outputY = copy_using_cache(outputY, tag='pfY')
    validY = copy_using_cache(validY, tag='pfY')
    
    nct, nwt = blockDim
    ncb = int(np.ceil(nChan/nct))
    nwb = int(np.ceil(nWin/nwt))
    
    _POL_FILL((ncb,nwb), (nct,nwt),
              (signals, valid, swap, 
               cp.int32(nStand), cp.int32(nChanAct), cp.int32(nWinAct),
               cp.int32(nChan), cp.int32(nWin), cp.int32(offset),
               outputX, validX, outputY, validY))
                
    return outputX, validX, outputY, validY
