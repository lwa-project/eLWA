# -*- coding: utf-8 -*-

"""
Module for creating optimized data processing code when it is needed.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import sys
import glob
import time
import numpy
import StringIO
import platform
import warnings
import subprocess
from distutils import sysconfig
from distutils import ccompiler

from jinja2 import Environment, FileSystemLoader, Template

from lsl.correlator.fx import noWindow


__version__ = '0.1'
__revision__ = '$Rev$'
__all__ = ['justInTimeOptimizer', '__version__', '__revision__', '__all__']


# Setup
_cacheDir = os.path.dirname( os.path.abspath(__file__) )


class justInTimeOptimizer(object):
	# Mappings
	## Real or complex
	_callMapping = {'int8': 'real', 
				 'int16': 'real', 
				 'int32': 'real', 
				 'int64': 'real', 
				 'float32': 'real', 
				 'float64': 'real', 
				 'complex64': 'cplx', 
				 'complex128': 'cplx'}
	## NumPy NPY_??? type
	_numpyMapping = {'int8': 'INT8', 
				 'int16': 'INT16', 
				 'int32': 'INT32', 
				 'int64': 'INT64', 
				 'float32': 'FLOAT32', 
				 'float64': 'FLOAT32', 
				 'complex64': 'COMPLEX64', 
				 'complex128': 'COMPLEX128'}
	## C type for accessing the NumPy Array
	_ctypeMapping = {'int8': 'signed char', 
				 'int16': 'short int', 
				 'int32': 'int', 
				 'int64': 'long int', 
				 'float32': 'float', 
				 'float64': 'double', 
				 'complex64': 'float complex', 
				 'complex128': 'double complex'}
				 
	def __init__(self, cacheDir=None, verbose=True):
		# Setup the module cache and fill it
		if cacheDir is None:
			cacheDir = os.path.dirname(__file__)
		self.cacheDir = os.path.abspath(cacheDir)
		self._cache = {}
		self._loadCacheDir(verbose=verbose)
		
		# Setup the compiler
		cc = self.getCompiler()
		cflags, ldflags = self.getFlags()
		self.cc = cc
		self.cflags = cflags
		self.ldflags = ldflags
		
		# Setup the template cache and fill it
		self._templates = {}
		self._loadTemplates()
		
	def _loadCacheDir(self, verbose=False):
		"""
		Populate the cache with with valid .so modules that have been found.
		"""
		
		if verbose:
			print "JIT cache directory: %s" % self.cacheDir
			
		# Make sure the cache directory is in the path as well
		if self.cacheDir not in sys.path:
			sys.path.append(self.cacheDir)
			
		# Come up with a 'reference time' that we can use to see what may be outdated
		refFiles = glob.glob(os.path.join(os.path.dirname(__file__), '*.tmpl'))
		refFiles.append(__file__)
		refTime = max([os.stat(refFile)[8] for refFile in refFiles])
		
		# Find the modules and load the valid ones
		for soFile in glob.glob(os.path.join(self.cacheDir, '*.so')):
			soTime = os.stat(soFile)[8]
			module = os.path.splitext(os.path.basename(soFile))[0]
			if soTime < refTime:
				## This file is too old, clean it out
				if verbose:
					print " -> Purged %s as outdated" % module
				for ext in ('.c', '.o', '.so'):
					try:
						os.unlink(os.path.join(self.cacheDir, '%s%s' % (module, ext)))
					except OSError:
						pass
						
			else:
				## This file is OK, load it and cache it
				if verbose:
					print " -> Loaded %s" % module
				exec("import %s as loadedModule" % module)
				exec("self._cache['%s'] = loadedModule" % module)
				
	def getCompiler(self):
		"""
		Return the compiler to use for the JIT code generation.
		"""
		
		compiler = ccompiler.new_compiler()
		sysconfig.get_config_vars()
		sysconfig.customize_compiler(compiler)
		cc = compiler.compiler
		return cc[0]
		
	def getFlags(self, cc=None):
		"""
		Return a two-element tuple of CFLAGS and LDFLAGS for the compiler to use for
		JIT code generation.
		"""
		
		if cc is None:
			cc = self.getCompiler()
		cflags, ldflags = [], []
		
		# Python
		cflags.extend( subprocess.check_output(['python-config', '--cflags']).split() )
		ldflags.extend( subprocess.check_output(['python-config', '--ldflags']).split() )
		
		# Native architecture
		cflags.append( '-march=native' )
		ldflags.append( '-march=native' )
		
		# fPIC since it seems to be needed
		cflags.append( '-fPIC' )
		ldflags.extend( ['-shared', '-fPIC'] )
		
		# NumPy
		cflags.append( '-I%s' % numpy.get_include() )
		cflags.append( '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION' )
		ldflags.append( '-lm' )
		
		# ATLAS
		sys.stdout = StringIO.StringIO()
		sys.stderr = StringIO.StringIO()
		from numpy.distutils.system_info import get_info
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore",category=DeprecationWarning)
			atlas_info = get_info('atlas_blas', notfound_action=2)
		atlas_version = ([v[3:-3] for k,v in atlas_info.get('define_macros',[])
						if k == 'ATLAS_INFO']+[None])[0]
		sys.stdout = sys.__stdout__
		sys.stderr = sys.__stderr__
		cflags.extend( ['-I%s' % idir for idir in atlas_info['include_dirs']] )
		ldflags.extend( ['-L%s' % ldir for ldir in atlas_info['library_dirs']] )
		ldflags.extend( ['-l%s' % lib for lib in atlas_info['libraries']] )
		
		# FFTW3
		try:
			subprocess.check_output(['pkg-config', 'fftw3f', '--exists'])
			cflags.extend( subprocess.check_output(['pkg-config', 'fftw3f', '--cflags']).split() )
			ldflags.extend( subprocess.check_output(['pkg-config', 'fftw3f', '--libs']).split() )
		except subprocess.CalledProcessError:
			if platform.system() != 'FreeBSD':
				cflags.extend( [] )
				ldflags.extend( ['-lfftw3f', '-lm'] )
			else:
				cflags.extend( ['-I/usr/local/include',] )
				ldflags.extend( ['-L/usr/local/lib', '-lfftw3f', '-lm'] )
				
		# OpenMP
		fh = open('openmp_test.c', 'w')
		fh.write(r"""#include <omp.h>
		#include <stdio.h>
		int main() {
		#pragma omp parallel
		printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
		return 0;
		}
		""")
		fh.close()
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
		
	def _loadTemplates(self):
		"""
		Load in the various templates we might need.
		"""
		
		# Setup the jinja2 environment
		env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
		
		tmplFiles = glob.glob(os.path.join(os.path.dirname(__file__), '*.tmpl'))
		for tmplFile in tmplFiles:
			base = os.path.splitext(os.path.basename(tmplFile))[0]
			self._templates[base] = env.get_template(os.path.basename(tmplFile))
			
	def _compile(self, srcName, objName, verbose=False):
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
			
	def _link(self, objName, modName, verbose=False):
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
			
	def getModule(self, dtype, nStand, nSamps, nChan, nOverlap, ClipLevel, window=noWindow):
		"""
		Generate an optimized version of the various time-domain functions 
		for the given parameters, update the cache, and return the module.
		"""
		
		# Figure out if we are in window mode or not
		useWindow = False
		if window is not noWindow:
			useWindow = True
			
		# Sort out the data types we need
		try:
			funcTemplate = self._callMapping[dtype]
			dtypeN = self._numpyMapping[dtype]
			dtypeC = self._ctypeMapping[dtype]
		except KeyError:
			raise RuntimeError("Unknown data type: %s" % dtype)
			
		# Build up the file names we need
		module = '%s_%i_%i_%i_%i_%i' % (dtype, nStand, nSamps, nChan, nOverlap, ClipLevel)
		srcname = os.path.join(self.cacheDir, '%s.c' % module)
		objname = os.path.join(self.cacheDir, '%s.o' % module)
		soname = os.path.join(self.cacheDir, '%s.so' % module)
		
		# Is it cached?
		try:
			exec("loadedModule = self._cache['%s']" % module)
			## Yes!
		except KeyError:
			## No, additional work is needed
			### Get the number of FFT windows and the number of baselines
			if funcTemplate == 'real':
				nFFT = nSamps / (2*nChan/nOverlap) - 2*nChan/(2*nChan/nOverlap) + 1
			else:
				nFFT = nSamps / (nChan/nOverlap) - nChan/(nChan/nOverlap) + 1
			nBL = nStand*(nStand+1)/2
			
			### Generate the code
			config = {'module':module, 'dtype':dtype, 'dtypeN':dtypeN, 'dtypeC':dtypeC, 
					'nStand':'%iL'%nStand, 'nSamps':'%iL'%nSamps, 'nChan':'%iL'%nChan, 'nOverlap':'%iL'%nOverlap, 
					'nFFT':'%iL'%nFFT, 'nBL':'%iL'%nBL, 'ClipLevel':ClipLevel, 'useWindow':useWindow}
			fh = open(os.path.join(self.cacheDir, srcname), 'w')
			fh.write( self._templates['head'].render(**config) )
			fh.write( self._templates[funcTemplate].render(**config) )
			fh.write( self._templates['post'].render(**config) )
			fh.close()
			
			### Compile, link, and cleanup
			self._compile(srcname, objname)
			self._link(objname, soname)
			try:
				os.unlink(objname)
			except OSError:
				pass
				
			## Load and cache
			exec("import %s as loadedModule" % module)
			exec("self._cache['%s'] = loadedModule" % module)
			
		# Done
		return loadedModule
		
	def getFunction(self, func, *args, **kwds):
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
			nOverlap = kwds['Overlap']
		except KeyError:
			nOverlap = 1
		try:
			ClipLevel = kwds['ClipLevel']
		except KeyError:
			ClipLevel = 0
		try:
			window = kwds['window']
		except KeyError:
			window = noWindow
			
		# Get the optimized module
		mod = self.getModule(dtype, nStand, nSamps, nChan, nOverlap, ClipLevel, window)
		
		# Do we need to measure it?
		try:
			doMeasure = kwds['measure']
		except KeyError:
			doMeasure = False
		if doMeasure:
			## Yes
			if ftype == 'spec':
				t0 = time.time()
				for i in xrange(10):
					getattr(mod, 'specS')(*args)
				tS = time.time()-t0
				
				t0 = time.time()
				for i in xrange(10):
					getattr(mod, 'specF')(*args)
				tF = time.time()-t0
				
				t0 = time.time()
				for i in xrange(10):
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
