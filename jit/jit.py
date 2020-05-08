"""
Module for creating optimized data processing code when it is needed.
"""

# Python2 compatibility
from __future__ import print_function, division, absolute_import
    
import os
import sys
import glob
import time
import numpy
import importlib
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import platform
import warnings
import subprocess
from distutils import sysconfig
from distutils import ccompiler

from jinja2 import Environment, FileSystemLoader, Template

from lsl.correlator.fx import null_window


__version__ = '0.1'
__all__ = ['JustInTimeOptimizer',]


# Setup
_CACHE_DIR = os.path.dirname( os.path.abspath(__file__) )


class JustInTimeOptimizer(object):
    # Mappings
    ## Real or complex
    _call_mapping = {'int8': 'real', 
                     'int16': 'real', 
                     'int32': 'real', 
                     'int64': 'real', 
                     'float32': 'real', 
                     'float64': 'real', 
                     'complex64': 'cplx', 
                     'complex128': 'cplx'}
    ## NumPy NPY_??? type
    _numpy_mapping = {'int8': 'INT8', 
                      'int16': 'INT16', 
                      'int32': 'INT32', 
                      'int64': 'INT64', 
                      'float32': 'FLOAT32', 
                      'float64': 'FLOAT32', 
                      'complex64': 'COMPLEX64', 
                      'complex128': 'COMPLEX128'}
    ## C type for accessing the NumPy Array
    _ctype_mapping = {'int8': 'signed char', 
                      'int16': 'short int', 
                      'int32': 'int', 
                      'int64': 'long int', 
                      'float32': 'float', 
                      'float64': 'double', 
                      'complex64': 'float complex', 
                      'complex128': 'double complex'}
                     
    def __init__(self, cache_dir=None, verbose=True):
        # Setup the Python version tag
        self._tag = "py%i%i" % (sys.version_info.major, sys.version_info.minor)
        try:
            self._tag = self._tag+sys.abiflags.replace('.', '')
        except AttributeError:
            pass
            
        # Setup the module cache and fill it
        if cache_dir is None:
            cache_dir = os.path.dirname(__file__)
        self.cache_dir = os.path.abspath(cache_dir)
        self._cache = {}
        self._load_cache_dir(verbose=verbose)
        
        # Setup the compiler
        cc = self.get_compiler()
        cflags, ldflags = self.get_flags()
        self.cc = cc
        self.cflags = cflags
        self.ldflags = ldflags
        
        # Setup the template cache and fill it
        self._templates = {}
        self._load_templates()
        
    def _load_cache_dir(self, verbose=False):
        """
        Populate the cache with with valid .so modules that have been found.
        """
        
        if verbose:
            print("JIT cache directory: %s" % self.cache_dir)
            
        # Make sure the cache directory is in the path as well
        if self.cache_dir not in sys.path:
            sys.path.append(self.cache_dir)
            
        # Come up with a 'reference time' that we can use to see what may be outdated
        refFiles = glob.glob(os.path.join(os.path.dirname(__file__), '*.tmpl'))
        refFiles.append(__file__)
        refTime = max([os.stat(refFile)[8] for refFile in refFiles])
        
        # Find the modules and load the valid ones
        for soFile in glob.glob(os.path.join(self.cache_dir, '*.so')):
            soTime = os.stat(soFile)[8]
            module = os.path.splitext(os.path.basename(soFile))[0]
            if module.find(self._tag) == -1:
                continue
            if soTime < refTime:
                ## This file is too old, clean it out
                if verbose:
                    print(" -> Purged %s as outdated" % module)
                for ext in ('.c', '.o', '.so'):
                    try:
                        os.unlink(os.path.join(self.cache_dir, '%s%s' % (module, ext)))
                    except OSError:
                        pass
                        
            else:
                ## This file is OK, load it and cache it
                if verbose:
                    print(" -> Loaded %s" % module)
                loadedModule = importlib.import_module(module)
                self._cache[module] = loadedModule
                
    def get_compiler(self):
        """
        Return the compiler to use for the JIT code generation.
        """
        
        compiler = ccompiler.new_compiler()
        sysconfig.get_config_vars()
        sysconfig.customize_compiler(compiler)
        cc = compiler.compiler
        return cc[0]
        
    def get_flags(self, cc=None):
        """
        Return a two-element tuple of CFLAGS and LDFLAGS for the compiler to use for
        JIT code generation.
        """
        
        if cc is None:
            cc = self.get_compiler()
        cflags, ldflags = [], []
        
        # Python
        try:
            pyconfig = subprocess.check_output(['which', sys.executable+'-config']).rstrip()
        except subprocess.CalledProcessError:
            pyconfig = 'python-config'
        cflags.extend( subprocess.check_output([pyconfig, '--cflags']).split() )
        ldflags.extend( subprocess.check_output([pyconfig, '--ldflags']).split() )
        
        # Native architecture
        #cflags.append( '-march=native' )
        #ldflags.append( '-march=native' )
        
        # fPIC since it seems to be needed
        cflags.append( '-fPIC' )
        ldflags.extend( ['-shared', '-fPIC'] )
        
        # NumPy
        cflags.append( '-I%s' % numpy.get_include() )
        cflags.append( '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION' )
        ldflags.append( '-lm' )
        
        # ATLAS
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        from numpy.distutils.system_info import get_info
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=DeprecationWarning)
            atlas_info = get_info('atlas_blas', notfound_action=2)
        atlas_version = ([v[3:-3] for k,v in atlas_info.get('define_macros',[])
                        if k == 'ATLAS_INFO']+[None])[0]
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        try:
            cflags.extend( ['-I%s' % idir for idir in atlas_info['include_dirs']] )
        except KeyError:
            pass
        try:
            ldflags.extend( ['-L%s' % ldir for ldir in atlas_info['library_dirs']] )
        except KeyError:
            pass
        try:
            ldflags.extend( ['-l%s' % lib for lib in atlas_info['libraries']] )
        except KeyError:
            pass
            
        # FFTW3
        try:
            subprocess.check_output(['pkg-config', 'fftw3f', '--exists'])
            cflags.extend( subprocess.check_output(['pkg-config', 'fftw3f', '--cflags']).split() )
            ldflags.extend( subprocess.check_output(['pkg-config', 'fftw3f', '--libs']).split() )
        except subprocess.CalledProcessError:
            cflags.extend( [] )
            ldflags.extend( ['-lfftw3f', '-lm'] )
               
        # OpenMP
        with open('openmp_test.c', 'w') as fh:
            fh.write(r"""#include <omp.h>
#include <stdio.h>
int main(void) {
#pragma omp parallel
printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
return 0;
}
            """)
        try:
            call = [cc,]
            call.extend(cflags)
            call.extend(['-fopenmp', 'openmp_test.c', '-o', 'openmp_test', '-lgomp'])
            output = subprocess.check_output(call, stderr=subprocess.STDOUT)
            cflags.append( '-fopenmp' )
            ldflags.append( '-lgomp' )
            os.unlink('openmp_test')
        except subprocess.CalledProcessError:
            pass
        finally:
            os.unlink('openmp_test.c')
            
        # Remove duplicates
        cflags = list(set(cflags))
        ldflags = list(set(ldflags))
        
        return cflags, ldflags
        
    def _load_templates(self):
        """
        Load in the various templates we might need.
        """
        
        # Setup the jinja2 environment
        env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
        
        tmplFiles = glob.glob(os.path.join(os.path.dirname(__file__), '*.tmpl'))
        for tmplFile in tmplFiles:
            base = os.path.splitext(os.path.basename(tmplFile))[0]
            self._templates[base] = env.get_template(os.path.basename(tmplFile))
            
    def _compile(self, srcName, objName, verbose=True):
        """
        Simple compile function.
        """
        
        call = [self.cc, '-c']
        call.extend(self.cflags)
        call.extend(['-o', objName, srcName])
        
        if verbose:
            return subprocess.check_output(call, stderr=subprocess.STDOUT)
        else:
            return subprocess.check_output(call)
            
    def _link(self, objName, modName, verbose=True):
        """
        Simple linker function.
        """
        
        call = [self.cc, '-o', modName]
        if type(objName) is list:
            call.extend(objName)
        else:
            call.append(objName)
        call.extend(self.ldflags)
        call.extend(self.cflags)
        
        if verbose:
            return subprocess.check_output(call, stderr=subprocess.STDOUT)
        else:
            return subprocess.check_output(call)
            
    def get_module(self, dtype, nStand, nSamps, nChan, nOverlap, ClipLevel, window=null_window):
        """
        Generate an optimized version of the various time-domain functions 
        for the given parameters, update the cache, and return the module.
        """
        
        # Figure out if we are in window mode or not
        useWindow = False
        if window is not null_window:
            useWindow = True
            
        # Sort out the data types we need
        try:
            funcTemplate = self._call_mapping[dtype]
            dtypeN = self._numpy_mapping[dtype]
            dtypeC = self._ctype_mapping[dtype]
        except KeyError:
            raise RuntimeError("Unknown data type: %s" % dtype)
            
        # Build up the file names we need
        module = '%s_%i_%i_%i_%i_%i_%s' % (dtype, nStand, nSamps, nChan, nOverlap, ClipLevel, self._tag)
        srcname = os.path.join(self.cache_dir, '%s.c' % module)
        objname = os.path.join(self.cache_dir, '%s.o' % module)
        soname = os.path.join(self.cache_dir, '%s.so' % module)
        
        # Is it cached?
        loadedModule = None
        try:
            loadedModule = self._cache[module]
            ## Yes!
        except KeyError:
            ## No, additional work is needed
            ### Get the number of FFT windows and the number of baselines
            if funcTemplate == 'real':
                nFFT = nSamps // (2*nChan//nOverlap) - 2*nChan//(2*nChan//nOverlap) + 1
            else:
                nFFT = nSamps // (nChan//nOverlap) - nChan//(nChan//nOverlap) + 1
            nBL = nStand*(nStand+1)//2
            
            ### Generate the code
            config = {'module':module, 'dtype':dtype, 'dtypeN':dtypeN, 'dtypeC':dtypeC, 
                      'nStand':'%iL'%nStand, 'nSamps':'%iL'%nSamps, 'nChan':'%iL'%nChan, 'nOverlap':'%iL'%nOverlap, 
                      'nFFT':'%iL'%nFFT, 'nBL':'%iL'%nBL, 'ClipLevel':ClipLevel, 'useWindow':useWindow}
            with open(os.path.join(self.cache_dir, srcname), 'w') as fh:
                fh.write( self._templates['head'].render(**config) )
                fh.write( self._templates[funcTemplate].render(**config) )
                fh.write( self._templates['post'].render(**config) )
                
            ### Compile, link, and cleanup
            self._compile(srcname, objname)
            self._link(objname, soname)
            try:
                os.unlink(objname)
            except OSError:
                pass
                
            ## Load and cache
            loadedModule = importlib.import_module(module)
            self._cache[module] = loadedModule
            
        # Done
        return loadedModule
        
    def get_function(self, func, *args, **kwds):
        """
        Given a base LSL function and a call signature, return optimzed version of the function for that call.
        """
        
        # Figure out what we need to do
        try:
            ftype = func.__name__
        except AttributeError:
            ftype = func
        if ftype[:4] == 'FPSD':
            ftype = 'spec'
        elif ftype[:7] == 'FEngine':
            ftype = 'FEngine'
        elif ftype[:9] == 'PFBEngine':
            ftype = 'PFBEngine'
        elif ftype[:8] == 'XEngine2':
            ftype == 'XEngine2'
        elif ftype[:8] == 'XEngine3':
            ftype == 'XEngine3'
        else:
            ## We can't do anything for this one
            return func
            
        # Figure out how to optimize it
        dtype = args[0].dtype.type.__name__
        if ftype in ('XEngine2', 'XEngine3'):
            if len(args[0].shape) == 3:
                nStand = args[0].shape[0]
                nSamps = args[0].shape[1]*args[0].shape[2]
                nChan = args[0].shape[1]
            else:
                nStand = args[0].shape[0]
                nSamps = args[0].shape[1]
                nChan = kwds['LFFT']
        else:
            nStand = args[0].shape[0]
            nSamps = args[0].shape[1]
            nChan = kwds['LFFT']
        try:
            nOverlap = kwds['overlap']
        except KeyError:
            nOverlap = 1
        try:
            ClipLevel = kwds['clip_level']
        except KeyError:
            ClipLevel = 0
        try:
            window = kwds['window']
        except KeyError:
            window = null_window
            
        # Get the optimized module
        mod = self.get_module(dtype, nStand, nSamps, nChan, nOverlap, ClipLevel, window)
        
        # Do we need to measure it?
        try:
            doMeasure = kwds['measure']
        except KeyError:
            doMeasure = False
        if doMeasure:
            ## Yes
            if ftype == 'spec':
                t0 = time.time()
                for i in range(10):
                    getattr(mod, 'specS')(*args)
                tS = time.time()-t0
                
                t0 = time.time()
                for i in range(10):
                    getattr(mod, 'specF')(*args)
                tF = time.time()-t0
                
                t0 = time.time()
                for i in range(10):
                    getattr(mod, 'specL')(*args)
                tL = time.time()-t0
                
                tBest = min([tS, tF, tL])
                if tS == tBest:
                    ftype = 'specS'
                elif tF == tBest:
                    ftype = 'specF'
                else:
                    ftype = 'specL'
                    
        else:
            ## No, use heuristics
            if ftype == 'spec':
                if nStand <= 2:
                    ftype = 'specF'
                elif nStand <= 5:
                    ftype = 'specL'
                else:
                    ftype = 'specS'
                    
        # Return the call
        return getattr(mod, ftype)
