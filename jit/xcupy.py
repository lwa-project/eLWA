"""
Module that provides GPU-based X-engines built using cupy.
"""

# Python2 compatibility
from __future__ import print_function, division

import cupy
import time
import numpy


_XENGINE2 = cupy.RawKernel(r"""
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
  int bl = blockIdx.x;
  int chan = blockIdx.y*512 + threadIdx.x;
  if( chan < nChan ) {
    int i, j, k;
    i = (int) (-0.5*sqrt(4.0*nStand*nStand + 4.0*nStand - 8.0*bl + 1.0) + nStand + 0.5);
    j = bl - i*(2*(nStand-1) + 1 - i)/2;
    
    float2 temp1, temp2, tempO, tempO;
    unsigned char valid1, valid2, valid;
    int count = 0;
    tempOR = tempOI = 0.0;
    for(k=0; k<nFFT; k++) {
      temp1 = *(signals1 + i*nChan*nFFT + chan*nFFT + k);
      temp2 = *(signals2 + j*nChan*nFFT + chan*nFFT + k);
      
      valid1 = *(sigValid1 + i*nFFT + k);
      valid2 = *(sigValid2 + j*nFFT + k);
      
      valid = valid1 & valid2;
      tempO.x += valid*(temp2.x*temp1.x + temp2.y*temp1.y);
      tempO.y += valid*(temp2.x*temp1.y - temp2.y*temp1.x);
      count += valid;
    }
    
    tempO.x /= count;
    tempO.y /= count;
    
    *(output + bl*nChan + chan) = tempO;
  }
}""", 'xengine2')

_XENGINE3 = cupy.RawKernel(r"""
inline __host__ __device__ void operator/=(float2 &a, int b) {
  a.x /= b;
  a.y /= b;
}

extern "C" __global__
void xengine3(const float2 *signalsX,
              const float2 *signalsY,
              const unsigned char *sigValidX,
              const unsigned char *sigValidY,
              int nStand,
              int nBL,
              int nChan,
              int nFFT,
              float2 *output) {
  int bl = blockIdx.x;
  int chan = blockIdx.y*512 + threadIdx.x;
  if( chan < nChan ) {
    int i, j, k;
    i = (int) (-0.5*sqrt(4.0*nStand*nStand + 4.0*nStand - 8.0*bl + 1.0) + nStand + 0.5);
    j = bl - i*(2*(nStand-1) + 1 - i)/2;
    
    float2 temp1X, temp1Y, temp2X, temp2Y;
    float2 tempXX, tempXY, tempYX, tempYY;
    unsigned char valid1X, valid1Y, valid2X, valid2Y, valid;
    int countXX, countXY, countYX, countYY;
    tempXX = tempXY = tempYX = tempYY = make_float2(0.0, 0.0);
    countXX = countXY = countYX = countYY = 0;
    for(k=0; k<nFFT; k++) {
      temp1X = *(signalsX + i*nChan*nFFT + chan*nFFT + k);
      temp1Y = *(signalsY + i*nChan*nFFT + chan*nFFT + k);
      temp2X = *(signalsX + j*nChan*nFFT + chan*nFFT + k);
      temp2Y = *(signalsY + j*nChan*nFFT + chan*nFFT + k);
      
      valid1X = *(sigValidX + i*nFFT + k);
      valid1Y = *(sigValidY + i*nFFT + k);
      valid2X = *(sigValidX + j*nFFT + k);
      valid2Y = *(sigValidY + j*nFFT + k);
      
      valid = valid1X & valid2X;
      tempXX.x += valid*(temp2X.x*temp1X.x + temp2X.y*temp1X.y);
      tempXX.y += valid*(temp2X.x*temp1X.y - temp2X.y*temp1X.x);
      countXX += valid;
      
      valid = valid1X & valid2Y;
      tempXY.x += valid*(temp2Y.x*temp1X.x + temp2Y.y*temp1X.y);
      tempXY.y += valid*(temp2Y.x*temp1X.y - temp2Y.y*temp1X.x);
      countXY += valid;
      
      valid = valid1Y & valid2X;
      tempYX.x += valid*(temp2X.x*temp1Y.x + temp2X.y*temp1Y.y);
      tempYX.y += valid*(temp2X.x*temp1Y.y - temp2X.y*temp1Y.x);
      countYX += valid;
      
      valid = valid1Y & valid2Y;
      tempYY.x += valid*(temp2Y.x*temp1Y.x + temp2Y.y*temp1Y.y);
      tempYY.y += valid*(temp2Y.x*temp1Y.y - temp2Y.y*temp1Y.x);
      countYY += valid;
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
}""", 'xengine3')


class _MemoryCache(object):
    def __init__(self, device=None, max_len=10):
        if device is not None:
            cupy.cuda.Device(device).use()
            self._device = device
            
        self._pool = cupy.get_default_memory_pool()
        self._cache = {}
        self._created = {}
        self._max_len = int(max_len)
        
    def __getitem__(self, key):
        value = self._cache[key]
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
        self._pool.set_limit(size=size_bytes)
        
    def free(self):
        self._pool.free_all_blocks()
        
    def select_gpu(self, device=0):
        try:
            if self._device == device:
                return
        except AttributeError:
            pass
            
        size_bytes = self.get_limit()
        self._cache.clear()
        self._created.clear()
        self.free()
        
        cupy.cuda.Device(device).use()
        self._pool = cupy.get_default_memory_pool()
        self._cache = {}
        self._created = {}
        self._device = device


_CACHE = _MemoryCache()


def select_gpu(device=0):
    _CACHE.select_gpu(device)


def get_memory_usage_limit():
    return _CACHE.get_limit()


def set_memory_usage_limit(size_bytes):
    _CACHE.set_limit(size_bytes)


set_memory_usage_limit(2*1024**3)
print("Loaded GPU X-engine support with %.2f GB" % (get_memory_usage_limit()/1024.0**3,))


def xengine(signalsF1, validF1, signalsF2, validF2):
    """
    X-engine for the outputs of fengine().
    """
    
    nStand, nChan, nWin = signalsF1.shape
    nBL = nStand*(nStand+1) // 2
    
    with cupy.cuda.Stream():
        try:
            signalsF1 = cupy.asarray(signalsF1).view(numpy.float32)
            signalsF2 = cupy.asarray(signalsF2).view(numpy.float32)
            validF1 = cupy.asarray(validF1)
            validF2 = cupy.asarray(validF2)
        except cupy.cuda.memory.OutOfMemoryError:
            _CACHE.free()
            signalsF1 = cupy.asarray(signalsF1).view(numpy.float32)
            signalsF2 = cupy.asarray(signalsF2).view(numpy.float32)
            validF1 = cupy.asarray(validF1)
            validF2 = cupy.asarray(validF2)
            
        try:
            output = _CACHE[(1,nBL,nChan)]
        except KeyError:
            output = cupy.empty((nBL,nChan), dtype=numpy.complex64)
            _CACHE[(1,nBL,nChan)] = output
        _XENGINE2((nBL,int(numpy.ceil(nChan/512))), (min([512, nChan]),), (signalsF1, signalsF2, validF1, validF2,
                                     cupy.int32(nStand), cupy.int32(nBL), cupy.int32(nChan), cupy.int32(nWin),
                                     output.view(numpy.float32)))
        output_cpu = cupy.asnumpy(output)
    return output_cpu


def xengine_full(signalsFX, validFX, signalsFY, validFY):
    """
    X-engine for the outputs of fengine().
    """
    
    nStand, nChan, nWin = signalsFX.shape
    nBL = nStand*(nStand+1) // 2
    
    with cupy.cuda.Stream():
        try:
            signalsFX = cupy.asarray(signalsFX).view(numpy.float32)
            signalsFY = cupy.asarray(signalsFY).view(numpy.float32)
            validFX = cupy.asarray(validFX)
            validFY = cupy.asarray(validFY)
        except cupy.cuda.memory.OutOfMemoryError:
            _CACHE.free()
            signalsFX = cupy.asarray(signalsFX).view(numpy.float32)
            signalsFY = cupy.asarray(signalsFY).view(numpy.float32)
            validFX = cupy.asarray(validFX)
            validFY = cupy.asarray(validFY)
            
        try:
            output = _CACHE[(4,nBL,nChan)]
        except KeyError:
            output = cupy.empty((4,nBL,nChan), dtype=numpy.complex64)
            _CACHE[(4,nBL,nChan)] = output
        _XENGINE3((nBL,int(numpy.ceil(nChan/512))), (min([512, nChan]),), (signalsFX, signalsFY, validFX, validFY,
                                     cupy.int32(nStand), cupy.int32(nBL), cupy.int32(nChan), cupy.int32(nWin),
                                     output.view(numpy.float32)))
        output_cpu = cupy.asnumpy(output)
    return output_cpu[0,:,:], output_cpu[1,:,:], output_cpu[2,:,:], output_cpu[3,:,:]
