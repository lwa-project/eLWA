"""
Module that provides GPU-based X-engines built using cupy.
"""

# Python2 compatibility
from __future__ import print_function, division

import cupy
import time
import numpy

__version__ = '0.1'
__all__ = ['select_gpu', 'get_memory_usage_limit', 'set_memory_usage_limit',
           'xengine', 'xengine_full']


_XENGINE2 = cupy.RawKernel(r"""
inline __host__ __device__ void operator/=(float2 &a, int b) {
  a.x /= b;
  a.y /= b;
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
}""", 'xengine2')

_XENGINE3 = cupy.RawModule(code=r"""
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

_INTERLEAVE = _XENGINE3.get_function('interleave')
_XENGINE3 = _XENGINE3.get_function('xengine3')


class _MemoryCache(object):
    def __init__(self, device=None, max_len=10):
        if device is not None:
            dev = cupy.cuda.Device(device)
        else:
            dev = cupy.cuda.Device()
            device = dev.id
        dev.use()
        self._device = device
        
        self._pool = cupy.get_default_memory_pool()
        self._cache = {}
        self._created = {}
        self._max_len = int(max_len)
        
    def __getitem__(self, key):
        try:
            value = self._cache[key]
        except KeyError:
            try:
                value = cupy.empty(shape=tuple([k for k in key[:-1]]), dtype=key[-1])
            except cupy.cuda.memory.OutOfMemoryError:
                self.free()
                value = cupy.empty(shape=tuple([k for k in key[:-1]]), dtype=key[-1])
            self.__setitem__(key, value)
        self._created[key] = time.time()
        return value
        
    def __setitem__(self, key, value):
        if len(self._cache) > self._max_len:
            oldest = 0
            oldest_key = None
            for k in self._cache:
                age = time.time() - self._created[k]
                if age > oldest:
                    oldest = age
                    oldest_key = k
            del self._cache[oldest_key]
            del self._created[oldest_key]
            self.free()
            
        self._cache[key] = value
        self._created[key] = time.time()
        
    def get_limit(self):
        return self._pool.get_limit()
        
    def set_limit(self, size_bytes):
        self._pool.set_limit(size=int(size_bytes))
        
    def free(self):
        self._pool.free_all_blocks()
        
    def select_gpu(self, device=0):
        if self._device == device:
            return
            
        size_bytes = self.get_limit()
        self._cache.clear()
        self._created.clear()
        self.free()
        
        cupy.cuda.Device(device).use()
        self._pool = cupy.get_default_memory_pool()
        self._cache = {}
        self._created = {}
        self._device = device
        self.set_limit(size_bytes)


_CACHE = _MemoryCache()


def select_gpu(device=0):
    _CACHE.select_gpu(device)


def get_memory_usage_limit():
    return _CACHE.get_limit()


def set_memory_usage_limit(size_bytes):
    _CACHE.set_limit(size_bytes)


def xengine(signalsF1, validF1, signalsF2, validF2, blockDim=(4,16)):
    """
    X-engine for the outputs of fengine().
    """
    
    nStand, nChan, nWin = signalsF1.shape
    nBL = nStand*(nStand+1) // 2
    
    with cupy.cuda.Stream():
        try:
            signalsF1 = cupy.asarray(signalsF1)
            signalsF2 = cupy.asarray(signalsF2)
            validF1 = cupy.asarray(validF1)
            validF2 = cupy.asarray(validF2)
        except cupy.cuda.memory.OutOfMemoryError:
            _CACHE.free()
            signalsF1 = cupy.asarray(signalsF1)
            signalsF2 = cupy.asarray(signalsF2)
            validF1 = cupy.asarray(validF1)
            validF2 = cupy.asarray(validF2)
            
        try:
            output = _CACHE[(1,nBL,nChan,numpy.complex64)]
        except KeyError:
            output = cupy.empty((nBL,nChan), dtype=numpy.complex64)
            _CACHE[(1,nBL,nChan,numpy.complex64)] = output
            
        nbt, nct = blockDim
        nbb = int(numpy.ceil(nBL/nbt))
        ncb = int(numpy.ceil(nChan/nct))
        
        _XENGINE2((nbb,ncb), (nbt, nct),
                  (signalsF1, signalsF2, validF1, validF2,
                   cupy.int32(nStand), cupy.int32(nBL), cupy.int32(nChan), cupy.int32(nWin),
                   output))
        
        output_cpu = cupy.asnumpy(output)
    return output_cpu


def xengine_full(signalsFX, validFX, signalsFY, validFY, blockDim=(4,16)):
    """
    X-engine for the outputs of fengine().
    """
    
    nStand, nChan, nWin = signalsFX.shape
    nBL = nStand*(nStand+1) // 2
    
    with cupy.cuda.Stream():
        try:
            signalsFX = cupy.asarray(signalsFX)
            signalsFY = cupy.asarray(signalsFY)
            validFX = cupy.asarray(validFX)
            validFY = cupy.asarray(validFY)
        except cupy.cuda.memory.OutOfMemoryError:
            _CACHE.free()
            signalsFX = cupy.asarray(signalsFX)
            signalsFY = cupy.asarray(signalsFY)
            validFX = cupy.asarray(validFX)
            validFY = cupy.asarray(validFY)
            
        try:
            combined = _CACHE[(2*nStand,nChan,nWin,numpy.complex64)]
            valid = _CACHE[(2*nStand,nWin,numpy.uint8)]
        except KeyError:
            combined = cupy.empty((2*nStand,nChan,nWin), dtype=numpy.complex64)
            valid = cupy.empty((2*nStand,nWin), dtype=numpy.uint8)
            _CACHE[(2*nStand,nChan,nWin,numpy.complex64)] = combined
            _CACHE[(2*nStand,nWin,numpy.uint8)] = valid
            
        nct, nwt = blockDim
        ncb = int(numpy.ceil(nChan/nct))
        nwb = int(numpy.ceil(nWin/nwt))
            
        _INTERLEAVE((nStand,ncb,nwb), (nct,nwt),
                    (signalsFX, signalsFY, validFX, validFY,
                     cupy.int32(nStand), cupy.int32(nChan), cupy.int32(nWin),
                     combined, valid))
        
        try:
            output = _CACHE[(4,nBL,nChan,numpy.complex64)]
        except KeyError:
            output = cupy.empty((4,nBL,nChan), dtype=numpy.complex64)
            _CACHE[(4,nBL,nChan,numpy.complex64)] = output
            
        nbt, nct = blockDim
        nbb = int(numpy.ceil(nBL/nbt))
        ncb = int(numpy.ceil(nChan/nct))
        
        _XENGINE3((nbb,ncb), (nbt, nct),
                  (combined, valid,
                   cupy.int32(nStand), cupy.int32(nBL), cupy.int32(nChan), cupy.int32(nWin),
                    output))
        
        output_cpu = cupy.asnumpy(output)
    return output_cpu[0,:,:], output_cpu[1,:,:], output_cpu[2,:,:], output_cpu[3,:,:]
