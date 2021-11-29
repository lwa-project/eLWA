from __future__ import print_function, division

import cupy
import time
import numpy


_XENGINE2 = cupy.RawKernel(r"""
extern "C" __global__
void xengine2(const float *signals1,
              const float *signals2,
              const char *sigValid1,
              const char *sigValid2,
              int nStand,
              int nBL,
              int nChan,
              int nFFT,
              float *output) {
  int bl = blockIdx.x;
  int chan = blockIdx.y*512 + threadIdx.x;
  if( chan < nChan ) {
    int i, j, k;
    i = (int) (-0.5*sqrt(4.0*nStand*nStand + 4.0*nStand - 8.0*bl + 1.0) + nStand + 0.5);
    j = bl - i*(2*(nStand-1) + 1 - i)/2;
    
    float temp1R, temp1I, temp2R, temp2I, tempOR, tempOI;
    char valid1, valid2, valid;
    int count = 0;
    tempOR = tempOI = 0.0;
    for(k=0; k<nFFT; k++) {
      temp1R = *(signals1 + i*nChan*nFFT*2 + chan*nFFT*2 + 2*k + 0);
      temp1I = *(signals1 + i*nChan*nFFT*2 + chan*nFFT*2 + 2*k + 1);
      temp2R = *(signals2 + j*nChan*nFFT*2 + chan*nFFT*2 + 2*k + 0);
      temp2I = *(signals2 + j*nChan*nFFT*2 + chan*nFFT*2 + 2*k + 1);
      
      valid1 = *(sigValid1 + i*nFFT + k);
      valid2 = *(sigValid2 + j*nFFT + k);
      valid = valid1*valid2;
      
      tempOR += valid*(temp2R*temp1R + temp2I*temp1I);
      tempOI += valid*(temp2R*temp1I - temp2I*temp1R);
      count += valid;
    }
    
    *(output + bl*nChan*2 + chan*2 + 0) = tempOR / count;
    *(output + bl*nChan*2 + chan*2 + 1) = tempOI / count;
  }
}""", 'xengine2')

_XENGINE3 = cupy.RawKernel(r"""
extern "C" __global__
void xengine3(const float *signalsX,
              const float *signalsY,
              const char *sigValidX,
              const char *sigValidY,
              int nStand,
              int nBL,
              int nChan,
              int nFFT,
              float *output) {
  int bl = blockIdx.x;
  int chan = blockIdx.y*512 + threadIdx.x;
  if( chan < nChan ) {
    int i, j, k;
    i = (int) (-0.5*sqrt(4.0*nStand*nStand + 4.0*nStand - 8.0*bl + 1.0) + nStand + 0.5);
    j = bl - i*(2*(nStand-1) + 1 - i)/2;
    
    float temp1XR, temp1XI, temp1YR, temp1YI;
    float temp2XR, temp2XI, temp2YR, temp2YI;
    float tempXXR, tempXXI, tempXYR, tempXYI, tempYXR, tempYXI, tempYYR, tempYYI;
    char valid1X, valid1Y, valid2X, valid2Y;
    int countXX, countXY, countYX, countYY;
    tempXXR = tempXXI = tempXYR = tempXYI = tempYXR = tempYXI = tempYYR = tempYYI = 0.0;
    countXX = countXY = countYX = countYY = 0;
    for(k=0; k<nFFT; k++) {
      temp1XR = *(signalsX + i*nChan*nFFT*2 + chan*nFFT*2 + 2*k + 0);
      temp1XI = *(signalsX + i*nChan*nFFT*2 + chan*nFFT*2 + 2*k + 1);
      temp1YR = *(signalsY + i*nChan*nFFT*2 + chan*nFFT*2 + 2*k + 0);
      temp1YI = *(signalsY + i*nChan*nFFT*2 + chan*nFFT*2 + 2*k + 1);
      temp2XR = *(signalsX + j*nChan*nFFT*2 + chan*nFFT*2 + 2*k + 0);
      temp2XI = *(signalsX + j*nChan*nFFT*2 + chan*nFFT*2 + 2*k + 1);
      temp2YR = *(signalsY + j*nChan*nFFT*2 + chan*nFFT*2 + 2*k + 0);
      temp2YI = *(signalsY + j*nChan*nFFT*2 + chan*nFFT*2 + 2*k + 1);
      
      valid1X = *(sigValidX + i*nFFT + k);
      valid1Y = *(sigValidY + i*nFFT + k);
      valid2X = *(sigValidX + j*nFFT + k);
      valid2Y = *(sigValidY + j*nFFT + k);
      
      tempXXR += valid1X*valid2X*(temp2XR*temp1XR + temp2XI*temp1XI);
      tempXXI += valid1X*valid2X*(temp2XR*temp1XI - temp2XI*temp1XR);
      countXX += valid1X*valid2X;
      
      tempXYR += valid1X*valid2Y*(temp2YR*temp1XR + temp2YI*temp1XI);
      tempXYI += valid1X*valid2Y*(temp2YR*temp1XI - temp2YI*temp1XR);
      countXY += valid1X*valid2Y;
      
      tempYXR += valid1Y*valid2X*(temp2XR*temp1YR + temp2XI*temp1YI);
      tempYXI += valid1Y*valid2X*(temp2XR*temp1YI - temp2XI*temp1YR);
      countYX += valid1Y*valid2X;
      
      tempYYR += valid1Y*valid2Y*(temp2YR*temp1YR + temp2YI*temp1YI);
      tempYYI += valid1Y*valid2Y*(temp2YR*temp1YI - temp2YI*temp1YR);
      countYY += valid1Y*valid2Y;
    }
    
    *(output + 0*nBL*nChan*2 + bl*nChan*2 + chan*2 + 0) = tempXXR / countXX;
    *(output + 0*nBL*nChan*2 + bl*nChan*2 + chan*2 + 1) = tempXXI / countXX;
    
    *(output + 1*nBL*nChan*2 + bl*nChan*2 + chan*2 + 0) = tempXYR / countXY;
    *(output + 1*nBL*nChan*2 + bl*nChan*2 + chan*2 + 1) = tempXYI / countXY;
    
    *(output + 2*nBL*nChan*2 + bl*nChan*2 + chan*2 + 0) = tempYXR / countYX;
    *(output + 2*nBL*nChan*2 + bl*nChan*2 + chan*2 + 1) = tempYXI / countYX;
    
    *(output + 3*nBL*nChan*2 + bl*nChan*2 + chan*2 + 0) = tempYYR / countYY;
    *(output + 3*nBL*nChan*2 + bl*nChan*2 + chan*2 + 1) = tempYYI / countYY;
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
