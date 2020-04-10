# -*- coding: utf-8 -*-

"""
Module for working with polarization to get LWA and VLA into the same frame.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import os
import sys
import time
import ephem
import numpy

from lsl.common.paths import data as dataPath

__version__ = '0.3'
__all__ = ['get_lwa_antenna_gain', 'get_matrix_lwa', 'get_matrix_vla', 
           'apply_matrix']


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
    #print "Beam Coeffs. X: a=%.2f, b=%.2f, g=%.2f, d=%.2f" % (alphaH, betaH, gammaH, deltaH)
    #print "Beam Coeffs. Y: a=%.2f, b=%.2f, g=%.2f, d=%.2f" % (alphaE, betaE, gammaE, deltaE)
    
    def BeamPattern(az, alt):
        zaR = numpy.pi/2 - alt*numpy.pi / 180.0 
        azR = az*numpy.pi / 180.0

        pE = (1-(2*zaR/numpy.pi)**alphaE)*numpy.cos(zaR)**betaE + gammaE*(2*zaR/numpy.pi)*numpy.cos(zaR)**deltaE
        pH = (1-(2*zaR/numpy.pi)**alphaH)*numpy.cos(zaR)**betaH + gammaH*(2*zaR/numpy.pi)*numpy.cos(zaR)**deltaH

        return numpy.sqrt((pE*numpy.cos(azR))**2 + (pH*numpy.sin(azR))**2)
        
    if pol == 'X':
        beamFuncX = BeamPattern
    else:
        beamFuncY = BeamPattern


def get_lwa_antenna_gain(site, src):
    # Basic location/source parameters
    az, el = src.az, src.alt
    az *= 180/numpy.pi
    el *= 180/numpy.pi
    
    # Compute
    bx, by = beamFuncX(az, el), beamFuncY(az, el)
    
    # Done
    return bx, by


def get_matrix_lwa(site, src, inverse=False):
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
    alpha = numpy.pi/2	# The LWA dipoles are vertical, i.e., pointed towards zenith
    
    # Matrix elements
    GA = numpy.arctan2(-numpy.sin(HA)*numpy.cos(lat-alpha), numpy.cos(HA)*numpy.sin(dec)*numpy.cos(lat-alpha) - numpy.cos(dec)*numpy.sin(lat-alpha))
    GB = numpy.arctan2(-numpy.sin(HA)*numpy.sin(dec), numpy.cos(HA))
    cosGA = numpy.cos(GA)
    sinGA = numpy.sin(GA)
    cosGB = numpy.cos(GB) 
    sinGB = numpy.sin(GB)
    
    # The matrix
    if not inverse:
        matrix = numpy.array([[cosGA, -sinGA], 
                            [sinGB,  cosGB]])
    else:
        matrix = numpy.array([[ cosGB, sinGA],
                            [-sinGB, cosGA]])
        matrix /= cosGA*cosGB + sinGA*sinGB
        
    # Done
    return matrix


def get_matrix_vla(site, src, inverse=False, feedRotation=ephem.degrees('0:00:00.0')):
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
    
    # Matrix elements
    G = numpy.arctan2(numpy.cos(lat)*numpy.sin(HA), numpy.cos(dec)*numpy.sin(lat) - numpy.sin(dec)*numpy.cos(lat)*numpy.cos(HA))
    GA = G + feedRotation
    GB = G + feedRotation
    cosGA = numpy.cos(GA)
    sinGA = numpy.sin(GA)
    cosGB = numpy.cos(GB)
    sinGB = numpy.sin(GB)
    
    # The matrix
    if not inverse:
        matrix = numpy.array([[cosGA, -sinGA], 
                            [sinGB,  cosGB]])
    else:
        matrix = numpy.array([[ cosGB, sinGA],
                            [-sinGB, cosGA]])
        matrix /= cosGA*cosGB + sinGA*sinGB
        
    # Done
    return matrix


def apply_matrix(data, matrix):
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
