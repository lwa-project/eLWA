import cupy as cp
import numpy as np


__version__ = '0.1'
__all__ = ['select_gpu', 'get_memory_limit', 'set_memory_limit',
           'get_from_shape', 'get_from_ndarray', 'copy_using_cache']


class _ReusableCache(object):
    """
    _ReusableCache wraps a cupy memory pool to make it easier to make an
    allocation on the GPU and resuse it.  The access is done through
    """
    
    def __init__(self, device=None):
        if device is not None:
            dev = cp.cuda.Device(device)
        else:
            dev = cp.cuda.Device()
            device = dev.id
        dev.use()
        self._device = device
        
        self._pool = cp.get_default_memory_pool()
        self._cache = {}
        
    def get_from_shape(self, shape, dtype=np.float32, tag=''):
        """
        Given a shape and a dtype, return the cache entry for that shape/type of
        array.  If the array does not already exist in the cache attempt to
        create it.  Return a cp.ndarray on success or raises a
        cp.cuda.memory.OutOfMemoryError on failure.
        """
        
        array_key = shape + (dtype,tag)
        try:
            cached_array = self._cache[array_key]
        except KeyError:
            try:
                cached_array = cp.empty(shape=shape, dtype=dtype)
            except cp.cuda.memory.OutOfMemoryError:
                self._pool.free_all_blocks()
                cached_array = cp.empty(shape=shape, dtype=dtype)
            self._cache[array_key] = cached_array
        return cached_array
        
    def get_from_ndarray(self, ndarray, tag=''):
        """
        Given a numpy.ndarray, return the cache entry for that shape/type of
        array.  If the array does not already exist in the cache attempt to
        create it.  Return a cp.ndarray on success or raises a
        cp.cuda.memory.OutOfMemoryError on failure.
        """
        
        return get_from_shape(ndarray.shape, dtype=ndarray.dtype, tag=tag)
        
    def get_memory_limit(self):
        """
        Get the pool memory usage limit in bytes.
        """
        
        return self._pool.get_limit()
        
    def set_memory_limit(self, size_bytes):
        """
        Set the pool memory usage limit in bytes.
        """
        
        self._pool.set_limit(size=int(size_bytes))
        
    def select_gpu(self, device=0):
        """
        Select which GPU device this cache should live on.  If the device is
        changed the cache is flushed.
        """
        
        if self._device == device:
            return
            
        size_bytes = self.get_memory_limit()
        self._cache.clear()
        self._pool.free_all_blocks()
        
        cp.cuda.Device(device).use()
        self._pool = cp.get_default_memory_pool()
        self._cache = {}
        self._device = device
        self.set_memory_limit(size_bytes)


_REUSABLE_CACHE = _ReusableCache()


def select_gpu(device=0):
    """
    Select which GPU device the cache lives on.  If the device is
    changed the cache is flushed.
    """
    
    _REUSABLE_CACHE.select_gpu(device)


def get_memory_limit():
    """
    Get the pool memory usage limit in bytes.
    """
    
    return _REUSABLE_CACHE.get_memory_limit()


def set_memory_limit(size_bytes):
    """
    Set the pool memory usage limit in bytes.
    """
    
    _REUSABLE_CACHE.set_memory_limit(size_bytes)


def get_from_shape(shape, dtype=np.float32, tag=''):
    """
    Given a shape and a dtype, return the cache entry for that shape/type of
    array.  If the array does not already exist in the cache attempt to
    create it.  Return a cp.ndarray on success or raises a
    cp.cuda.memory.OutOfMemoryError on failure.
    """
    
    return _REUSABLE_CACHE.get_from_shape(shape, dtype=dtype, tag=tag)


def get_from_ndarray(ndarray, tag=''):
    """
    Given a numpy.ndarray, return the cache entry for that shape/type of
    array.  If the array does not already exist in the cache attempt to
    create it.  Return a cp.ndarray on success or raises a
    cp.cuda.memory.OutOfMemoryError on failure.
    """
    
    return _REUSABLE_CACHE.get_from_ndarray(ndarray, tag=tag)


def copy_using_cache(ndarray, tag=''):
    """
    Given a numpy.ndarray try to copy that array to the GPU using memory already
    in the cache.  Return a cp.ndarray on success or raise a
    cp.cuda.memory.OutOfMemoryError on failure.
    """
    
    # Check if it's already on the GPU
    if isinstance(ndarray, cp.ndarray):
        return ndarray
        
    # Get the current stream
    stream = cp.cuda.get_current_stream()
    
    # Create the GPU array and queue the copy
    cparray = _REUSABLE_CACHE.get_from_ndarray(ndarray, tag=tag)
    cp.cuda.runtime.memcpyAsync(cparray.data.ptr,
                                ndarray.ctypes.data,
                                ndarray.size*ndarray.dtype.itemsize,
                                cp.cuda.runtime.memcpyHostToDevice,
                                stream.ptr)
    return cparray
