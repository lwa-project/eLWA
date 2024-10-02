"""
Module that provides GPU-based X-engines built using cupy.
"""

import cupy as cp
import numpy as np

from .cache import get_from_shape, get_from_ndarray, copy_using_cache


__version__ = '0.2'
__all__ = ['xengine', 'combined_xengine_full', 'xengine_full', 'accumulate_vis']


_XENGINE = cp.RawModule(code=r"""
inline __host__ __device__ void operator*=(float2 &a, unsigned char b) {
  a.x *= (float) b;
  a.y *= (float) b;
}

inline __host__ __device__ float2 operator*(float2 a, unsigned char b) {
  return make_float2(a.x * (float) b, a.y * (float) b);
}

inline __host__ __device__ void operator/=(float2 &a, float b) {
  a.x /= b;
  a.y /= b;
}

inline __host__ __device__ float2 operator/(float2 a, float b) {
  return make_float2(a.x / b, a.y / b);
}

inline __host__ __device__ float2 conjy(float2 a) {
  return make_float2(a.x, -a.y);
}

inline __host__ __device__ float4 pack_pair(float2 a, float2 b) {
  return make_float4(a.x, a.y, b.x, b.y);
}

inline __host__ __device__ void unpack_pair(float4 a, float2 &b, float2 &c) {
  b.x = a.x;
  b.y = a.y;
  c.x = a.z;
  c.y = a.w;
}

extern "C" __global__
void xengine2(const float2 *signals1,
              const float2 *signals2,
              const unsigned char *sigValid1,
              const unsigned char *sigValid2,
              int nStand,
              int nBL,
              int nChan,
              int nFFT,
              float2 *output) {
  int bl = blockIdx.x*blockDim.x + threadIdx.x;
  int chan = blockIdx.y*blockDim.y + threadIdx.y;
  if( bl >= nBL || chan >= nChan ) {
    // Nothin'
  } else {
    int i, j, k;
    i = (int) (-0.5*sqrt(4.0*nStand*nStand + 4.0*nStand - 8.0*bl + 1.0) + nStand + 0.5);
    j = bl - i*(2*(nStand-1) + 1 - i)/2;
    
    float2 temp1, temp2, tempO;
    unsigned char valid;
    
    valid  = *(sigValid1 + i*nFFT + k);
    valid &= *(sigValid2 + j*nFFT + k);
    
    int count = 0;
    tempO.x = tempO.y = 0.0;
    for(k=0; k<nFFT; k++) {
      temp1 = *(signals1 + i*nChan*nFFT + chan*nFFT + k);
      temp2 = *(signals2 + j*nChan*nFFT + chan*nFFT + k);
      
      tempO.x += valid*(temp2.x*temp1.x + temp2.y*temp1.y);
      tempO.y += valid*(temp2.x*temp1.y - temp2.y*temp1.x);
      count += valid;
    }
    
    tempO /= count;
    
    *(output + bl*nChan + chan) = tempO;
  }
}

extern "C" __global__
void interleave(const float2 *signalsX,
                const float2 *signalsY,
                const unsigned char *sigValidX,
                const unsigned char *sigValidY,
                int nStand,
                int nChan,
                int nFFT,
                float4 *output,
                uchar2 *valid) {
  int stand = blockIdx.x;
  int chan = blockIdx.y*blockDim.x + threadIdx.x;
  int fft = blockIdx.z*blockDim.y + threadIdx.y;
  
  float2 tempX, tempY;
  unsigned char validX, validY;
  float4 tempO;
  uchar2 tempV;
  
  if(chan >= nChan || fft >= nFFT) {
    // Nothin'
  } else {
    tempX = *(signalsX + stand*nChan*nFFT + chan*nFFT + fft);
    tempY = *(signalsY + stand*nChan*nFFT + chan*nFFT + fft);
    
    validX = *(sigValidX + stand*nFFT + fft);
    validY = *(sigValidY + stand*nFFT + fft);
    
    tempX *= validX;
    tempY *= validY;
    
    tempO = make_float4(tempX.x, tempX.y, tempY.x, tempY.y);
    tempV = make_uchar2(validX, validY);
    
    *(output + stand*nChan*nFFT + chan*nFFT + fft) = tempO;
    *(valid + stand*nFFT + fft) = tempV;
  }
}

extern "C" __global__
void xengine3(const float4 *signals,
              const uchar2 *valid,
              int nStand,
              int nBL,
              int nChan,
              int nFFT,
              float2 *output) {
  int bl = blockIdx.x*blockDim.x + threadIdx.x;
  int chan = blockIdx.y*blockDim.y + threadIdx.y;
  if( bl >= nBL || chan >= nChan ) {
    // Nothin'
  } else {
    int i, j, k;
    i = (int) (-0.5*sqrt(4.0*nStand*nStand + 4.0*nStand - 8.0*bl + 1.0) + nStand + 0.5);
    j = bl - i*(2*(nStand-1) + 1 - i)/2;
    
    float4 temp1, temp2;
    uchar2 valid1, valid2;
    float2 temp1X, temp1Y, temp2X, temp2Y;
    float2 tempXX, tempXY, tempYX, tempYY;
    unsigned char valid1X, valid1Y, valid2X, valid2Y;
    unsigned char validXX, validXY, validYX, validYY;
    unsigned short countXX, countXY, countYX, countYY;
    
    tempXX = tempXY = tempYX = tempYY = make_float2(0.0, 0.0);
    countXX = countXY = countYX = countYY = 0.0;
    for(k=0; k<nFFT; k++) {
      valid1 = *(valid + i*nFFT + k);
      valid2 = *(valid + j*nFFT + k);
      valid1X = valid1.x;
      valid1Y = valid1.y;
      valid2X = valid2.x;
      valid2Y = valid2.y;
      
      validXX = valid1X & valid2X;
      validXY = valid1X & valid2Y;
      validYX = valid1Y & valid2X;
      validYY = valid1Y & valid2Y;
      
      temp1 = *(signals + i*nChan*nFFT + chan*nFFT + k);
      temp2 = *(signals + j*nChan*nFFT + chan*nFFT + k);
      
      unpack_pair(temp1, temp1X, temp1Y);
      unpack_pair(temp2, temp2X, temp2Y);
      
      tempXX.x += (temp2X.x*temp1X.x + temp2X.y*temp1X.y);
      tempXX.y += (temp2X.x*temp1X.y - temp2X.y*temp1X.x);
      countXX += validXX;
      
      tempXY.x += (temp2Y.x*temp1X.x + temp2Y.y*temp1X.y);
      tempXY.y += (temp2Y.x*temp1X.y - temp2Y.y*temp1X.x);
      countXY += validXY;
      
      tempYX.x += (temp2X.x*temp1Y.x + temp2X.y*temp1Y.y);
      tempYX.y += (temp2X.x*temp1Y.y - temp2X.y*temp1Y.x);
      countYX += validYX;
      
      tempYY.x += (temp2Y.x*temp1Y.x + temp2Y.y*temp1Y.y);
      tempYY.y += (temp2Y.x*temp1Y.y - temp2Y.y*temp1Y.x);
      countYY += validYY;
    }
    
    tempXX /= countXX;
    tempXY /= countXY;
    tempYX /= countYX;
    tempYY /= countYY;
    
    *(output + 0*nBL*nChan + bl*nChan + chan) = tempXX;
    *(output + 1*nBL*nChan + bl*nChan + chan) = tempXY;
    *(output + 2*nBL*nChan + bl*nChan + chan) = tempYX;
    *(output + 3*nBL*nChan + bl*nChan + chan) = tempYY;
  }
}
""")

_XENGINE2 = _XENGINE.get_function('xengine2')
_INTERLEAVE = _XENGINE.get_function('interleave')
_XENGINE3 = _XENGINE.get_function('xengine3')


def xengine(signalsF1, validF1, signalsF2, validF2, blockDim=(4,16)):
    """
    X-engine for the outputs of fengine().
    """
    
    nStand, nChan, nWin = signalsF1.shape
    nBL = nStand*(nStand+1) // 2
    
    signalsF1 = cp.asarray(signalsF1)
    signalsF2 = cp.asarray(signalsF2)
    validF1 = cp.asarray(validF1)
    validF2 = cp.asarray(validF2)
    
    output = get_from_shape((1,nBL,nChan), dtype=np.complex64)
    
    nbt, nct = blockDim
    nbb = int(np.ceil(nBL/nbt))
    ncb = int(np.ceil(nChan/nct))
    
    _XENGINE2((nbb,ncb), (nbt, nct),
              (signalsF1, signalsF2, validF1, validF2,
               cp.int32(nStand), cp.int32(nBL), cp.int32(nChan), cp.int32(nWin),
               output))
    
    return output


def combined_xengine_full(signalsFX, validFX, signalsFY, validFY, blockDim=(4,16)):
    """
    X-engine for the outputs of fengine() where all four polarization products
    are in a single array.
    """
    
    nStand, nChan, nWin = signalsFX.shape
    nBL = nStand*(nStand+1) // 2
    
    signalsFX = copy_using_cache(signalsFX)
    signalsFY = copy_using_cache(signalsFY)
    validFX = copy_using_cache(validFX)
    validFY = copy_using_cache(validFY)
    
    combined = get_from_shape((2*nStand,nChan,nWin), dtype=np.complex64)
    valid = get_from_shape((2*nStand,nWin), dtype=np.uint8)
    
    nct, nwt = blockDim
    ncb = int(np.ceil(nChan/nct))
    nwb = int(np.ceil(nWin/nwt))
        
    _INTERLEAVE((nStand,ncb,nwb), (nct,nwt),
                (signalsFX, signalsFY, validFX, validFY,
                 cp.int32(nStand), cp.int32(nChan), cp.int32(nWin),
                 combined, valid))
    
    output = get_from_shape((4,nBL,nChan), dtype=np.complex64, tag='fvis')
    
    nbt, nct = blockDim
    nbb = int(np.ceil(nBL/nbt))
    ncb = int(np.ceil(nChan/nct))
    
    _XENGINE3((nbb,ncb), (nbt, nct),
              (combined, valid,
               cp.int32(nStand), cp.int32(nBL), cp.int32(nChan), cp.int32(nWin),
                output))
    
    return output


def xengine_full(signalsFX, validFX, signalsFY, validFY, blockDim=(4,16)):
    """
    X-engine for the outputs of fengine().
    """
    
    output = combined_xengine_full(signalsFX, validFX, signalsFY, validFY, blockDim=blockDim)
    return output[0,:,:], output[1,:,:], output[2,:,:], output[3,:,:]


_ACCUMULATE = cp.RawKernel(r"""
inline __host__ __device__ void operator*=(float2 &a, float b) {
  a.x *= b;
  a.y *= b;
}

inline __host__ __device__ void operator+=(float2 &a, float2 b) {
  a.x += b.x;
  a.y += b.y;
}

extern "C" __global__
void accumulate(const float2 *input,
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
    float2 temp = *(input +p*nBL*nChan + b*nChan + c);
    temp *= scale;
    if( reset > 0 ) {
      *(output + p*nBL*nChan + b*nChan + c) = make_float2(0, 0);
    }
    *(output + p*nBL*nChan + b*nChan + c) += temp;
  }
}
""", 'accumulate')

def accumulate_vis(vis_accum, vis, scale=1, reset=False, blockDim=(4,16)):
    nPol, nBL, nChan = vis.shape
    
    vis_accum = copy_using_cache(vis_accum)
    vis = copy_using_cache(vis)
    
    nbt, nct = blockDim
    nbb = int(np.ceil(nBL/nbt))
    ncb = int(np.ceil(nChan/nct))
    npb = nPol
    
    _ACCUMULATE((npb, nbb,ncb), (nbt, nct),
                (vis, cp.int32(nPol), cp.int32(nBL), cp.int32(nChan),
                 cp.int32(reset), cp.float32(scale), vis_accum))
    return vis_accum
