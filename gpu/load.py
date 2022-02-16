import cupy as cp
import numpy as np

from .misc import get_from_ndarray


__version__ = '0.1'
__all__ = ['load_raw_subint']


def load_raw_subint(ndarray, tag=''):
    """
    Given a 2D numpy.ndarray try to copy that array to the GPU using memory
    already in the cache.  Return a cp.ndarray on success or raise a
    cp.cuda.memory.OutOfMemoryError on failure.
    
    .. note:: This function differs from copy_using_cache in that it does not
              first create a contiguous copy of the input array.  Instead it
              copies the input array row-by-row.
    """
    
    # Check if it's already on the GPU
    if isinstance(ndarray, cp.ndarray):
        return ndarray
       
    # Validate that our input is 2D and contiguous on the fastest dimension
    assert(len(ndarray.shape) == 2)
    assert(ndarray.strides[1] == ndarray.dtype.itemsize)
    
    # Get the row size
    nbytes_row = ndarray.shape[1]*ndarray.dtype.itemsize
    
    # Get the current stream
    stream = cp.cuda.get_current_stream()
    
    # Create the GPU array and queue the copy - one row at a time
    cparray = get_from_ndarray(ndarray, tag=tag)
    for i in range(ndarray.shape[0]):
        dst = cparray.data.ptr+ i*cparray.strides[0]
        src = ndarray.ctypes.data + i*ndarray.strides[0]
        cp.cuda.runtime.memcpyAsync(dst, src, nbytes_row,
                                    cp.cuda.runtime.memcpyHostToDevice,
                                    stream.ptr)
    return cparray
