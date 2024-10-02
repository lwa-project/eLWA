"""
Module that provides GPU-based Jones matrix rotations built using cupy.
"""

import cupy as cp
import numpy as np

from .cache import get_from_shape, get_from_ndarray, copy_using_cache


__version__ = '0.1'
__all__ = ['apply_matrix']


_APPLY = cp.RawKernel(r"""
inline __host__ __device__ float2 operator*(float a, float2 b) {
  return make_float2(a * b.x, a * b.y);
}

inline __host__ __device__ float2 operator+(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

extern "C" __global__
void apply_c(float2 *signals,
             const float4 *matrix,
             int nStand,
             int nSamp) {
  int t = blockIdx.x*blockDim.x + threadIdx.x;
  
  if( t >= nSamp ) {
    // Nothin'
  } else {
    int a;
    float2 tempX, tempY;
    for(a=0; a<nStand; a+=2) {
      tempX = *(signals + (a+0)*nSamp + t);
      tempY = *(signals + (a+1)*nSamp + t);
      *(signals + (a+0)*nSamp + t) = matrix->x * tempX + matrix->y * tempY;
      *(signals + (a+1)*nSamp + t) = matrix->z * tempX + matrix->w * tempY;      
    }
  }
}""", 'apply_c')


def apply_matrix(data, matrix, blockDim=(16,)):
    """
    Given a 2-D data streams (inputs by time) and a 2-D Jones matrix, apply 
    the matrix to the data.
    """
    
    nStand, nSamp = data.shape
    
    data = copy_using_cache(data)
    matrix = copy_using_cache(matrix.astype(np.float32), tag='jam')
    
    nst = blockDim[0]
    nsb = int(np.ceil(nSamp/nst))
    
    _APPLY((nsb,), (nst,),
           (data, matrix,
            cp.int32(nStand), cp.int32(nSamp)))
    
    return data
