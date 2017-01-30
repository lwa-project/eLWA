# -*- coding: utf-8 -*-

"""
Module for working with polarization to get LWA and VLA into the same frame.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import sys
import time
import ephem
import numpy

from lsl.common.paths import data as dataPath

__version__ = '0.1'
__revision__ = '$Rev$'
__all__ = ['getParallacticAngle', 'getMatrixLWA1', 'getMatrixVLA', 'applyMatrix', 
		 '__version__', '__revision__', '__all__']


def getParallacticAngle(lat, dec, HA):
	"""
	Given an observation latitude, a source declination, and a source hour 
	angle (all in radians), return the parllactic angle (also in radians).
	"""
	
	num = numpy.cos(lat)*numpy.sin(HA)
	den = numpy.sin(lat)*numpy.cos(dec) - numpy.cos(lat)*numpy.sin(dec)*numpy.cos(HA)
	pang = numpy.arctan2(num, den)
	
	return pang


beamDict = numpy.load(os.path.join(dataPath, 'lwa1-dipole-emp.npz'))
for pol,beamCoeff in zip(('X', 'Y'), (beamDict['fitX'], beamDict['fitY'])):
	alphaE = numpy.polyval(beamCoeff[0,0,:], 74e6)
	betaE =  numpy.polyval(beamCoeff[0,1,:], 74e6)
	gammaE = numpy.polyval(beamCoeff[0,2,:], 74e6)
	deltaE = numpy.polyval(beamCoeff[0,3,:], 74e6)
	alphaH = numpy.polyval(beamCoeff[1,0,:], 74e6)
	betaH =  numpy.polyval(beamCoeff[1,1,:], 74e6)
	gammaH = numpy.polyval(beamCoeff[1,2,:], 74e6)
	deltaH = numpy.polyval(beamCoeff[1,3,:], 74e6)
	print "Beam Coeffs. X: a=%.2f, b=%.2f, g=%.2f, d=%.2f" % (alphaH, betaH, gammaH, deltaH)
	print "Beam Coeffs. Y: a=%.2f, b=%.2f, g=%.2f, d=%.2f" % (alphaE, betaE, gammaE, deltaE)
	
	def BeamPattern(az, alt):
		zaR = numpy.pi/2 - alt*numpy.pi / 180.0 
		azR = az*numpy.pi / 180.0

		pE = (1-(2*zaR/numpy.pi)**alphaE)*numpy.cos(zaR)**betaE + gammaE*(2*zaR/numpy.pi)*numpy.cos(zaR)**deltaE
		pH = (1-(2*zaR/numpy.pi)**alphaH)*numpy.cos(zaR)**betaH + gammaH*(2*zaR/numpy.pi)*numpy.cos(zaR)**deltaH

		return numpy.sqrt((pE*numpy.cos(azR))**2 + (pH*numpy.sin(azR))**2)
		
	if pol == 'X':
		print "here"
		beamFuncX = BeamPattern
	else:
		beamFuncY = BeamPattern


def getLWA1AntennaGain(site, src):
	# Basic location/source parameters
	az, el = src.az, src.alt
	az *= 180/numpy.pi
	el *= 180/numpy.pi
	
	# Compute
	bx, by = beamFuncX(az, el), beamFuncY(az, el)
	
	# Done
	return bx, by


def getMatrixLWA1(site, src, inverse=False):
	"""
	Given an ephem.Observer instances and an ephem.Body instance, get the 
	Jones matrix for an LWA station for the direction towards the source.
	
	From:
		https://casa.nrao.edu/aips2_docs/notes/185/node6.html
	"""
	
	# Basic location/source parameters
	HA = site.sidereal_time() - src.ra
	dec = src.dec
	lat = site.lat
	
	# Beam pattern
	bx, by = getLWA1AntennaGain(site, src)
	bx, by = numpy.sqrt(bx), numpy.sqrt(by)
	
	# Matrix elements
	cosGA =  numpy.cos(HA)*numpy.sin(dec)*numpy.cos(lat) - numpy.cos(dec)*numpy.sin(lat)
	sinGA = -numpy.sin(HA)*numpy.cos(lat)
	cosGB =  numpy.cos(HA)
	sinGB = -numpy.sin(HA)*numpy.sin(dec)
	
	# The matrix
	matrix = numpy.array([[cosGA, -sinGA], 
					  [sinGB,  cosGB]])
	if inverse:
		matrix = numpy.linalg.inv(matrix)
		for i in xrange(2):
			norm = numpy.sqrt((matrix[i,:]**2).sum())
			matrix[i,:] /= norm
			
	# The beam correction
	if inverse:
		matrix[:,0] *= bx
		matrix[:,1] *= by
	else:
		matrix[:,0] /= bx
		matrix[:,1] /= by
		
	# Done
	return matrix


def getMatrixVLA(site, src, inverse=False, feedRotation=ephem.degrees('0:00:00')):
	"""
	Given an ephem.Observer instances and an ephem.Body instance, get the 
	Jones matrix for the VLA for the direction towards the source.
	
	From:
		https://casa.nrao.edu/aips2_docs/notes/185/node6.html
	"""
	
	# Basic location/source parameters
	HA = site.sidereal_time() - src.ra
	dec = src.dec
	lat = site.lat
	
	# Get the gamma angles - A and B are the same here
	gammaA = getParallacticAngle(lat, dec, HA) + feedRotation
	gammaB = gammaA
	
	# Matrix elements
	cosGA = numpy.cos(gammaA)
	sinGA = numpy.sin(gammaA)
	cosGB = numpy.cos(gammaB)
	sinGB = numpy.sin(gammaB)
	
	# The matrix
	matrix = numpy.array([[cosGA, -sinGA], 
					  [sinGB,  cosGB]])
	if inverse:
		matrix = numpy.linalg.inv(matrix)
		for i in xrange(2):
			norm = numpy.sqrt((matrix[i,:]**2).sum())
			matrix[i,:] /= norm
			
	# Done
	return matrix


def applyMatrix(data, matrix):
	"""
	Given a 2-D data streams (inputs by time) and a 2-D Jones matrix, apply 
	the matrix to the data.
	"""
	
	# Get the input dimentions
	nStand, nSamps = data.shape
	
	# Apply
	for i in xrange(nStand/2):
		s0 = 2*i + 0
		s1 = 2*i + 2
		data[s0:s1,:] = numpy.dot(matrix, data[s0:s1,:])
		
	# Done
	return data
