"""
Module that provides GPU-based X-engines built using cupy.
"""

import cupy as cp
import numpy as np

from .cache import get_from_shape, get_from_ndarray, copy_using_cache


__version__ = '0.1'
__all__ = ['bin_accumulate_vis']


_BIN_ACCUMULATE = cp.RawKernel(r"""
inline __host__ __device__ void operator*=(float2 &a, float b) {
  a.x *= b;
  a.y *= b;
}

inline __host__ __device__ void operator+=(float2 &a, float2 b) {
  a.x += b.x;
  a.y += b.y;
}

extern "C" __global__
void bin_accumulate(const float2 *input,
                    const unsigned char *mask,
                    int nPol,
                    int nBL,
                    int nChan,
                    int reset,
                    float scale,
                    float2 *output) {
  int p = blockIdx.x;
  int b = blockIdx.y*blockDim.x + threadIdx.x;
  int c = blockIdx.z*blockDim.y + threadIdx.y;
  
  if( b >= nBL || c >= nChan ) {
    // Nothin'
  } else {
    float2 temp;
    if( reset > 0 ) {
      *(output + p*nBL*nChan + b*nChan + c) = make_float2(0, 0);
    }
    if( *(mask + c) == 1 ) {
      temp = *(input +p*nBL*nChan + b*nChan + c);
      temp *= scale;
      *(output + p*nBL*nChan + b*nChan + c) += temp;
    }
  }
}
""", 'bin_accumulate')

def bin_accumulate_vis(chan_mask, vis_accum, vis, scale=1, reset=False, blockDim=(4,16)):
    nPol, nBL, nChan = vis.shape
    assert(chan_mask.size == nChan)
    
    chan_mask = copy_using_cache(chan_mask)
    vis_accum = copy_using_cache(vis_accum)
    vis = copy_using_cache(vis)
    
    nbt, nct = blockDim
    nbb = int(np.ceil(nBL/nbt))
    ncb = int(np.ceil(nChan/nct))
    npb = nPol
    
    _BIN_ACCUMULATE((npb, nbb,ncb), (nbt, nct),
                    (vis, chan_mask, cp.int32(nPol), cp.int32(nBL), cp.int32(nChan),
                     cp.int32(reset), cp.float32(scale), vis_accum))
    return vis_accum
