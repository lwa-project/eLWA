"""
Module for working with polarization to get LWA and VLA into the same frame.
"""

import os
import sys
import time
import ephem
import numpy as np
from functools import lru_cache

from lsl.sim.beam import beam_response

__version__ = '0.4'
__all__ = ['get_lwa_antenna_gain', 'get_matrix_lwa', 'get_matrix_vla', 
           'apply_matrix']



def get_lwa_antenna_gain(site, src, freq=74e6):
    # Basic location/source parameters
    az, el = src.az, src.alt
    az *= 180/np.pi
    el *= 180/np.pi
    
    az, el = np.array([az]), np.array([el])
    
    # Compute
    bx = beam_response('empirical', 'XX', az, el, frequency=freq, degrees=True)[0]
    by = beam_response('empirical', 'YY', az, el, frequency=freq, degrees=True)[0]
    
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
    alpha = np.pi/2	# The LWA dipoles are vertical, i.e., pointed towards zenith
    
    # Matrix elements
    GA = np.arctan2(-np.sin(HA)*np.cos(lat-alpha), np.cos(HA)*np.sin(dec)*np.cos(lat-alpha) - np.cos(dec)*np.sin(lat-alpha))
    GB = np.arctan2(-np.sin(HA)*np.sin(dec), np.cos(HA))
    cosGA = np.cos(GA)
    sinGA = np.sin(GA)
    cosGB = np.cos(GB) 
    sinGB = np.sin(GB)
    
    # The matrix
    if not inverse:
        matrix = np.array([[cosGA, -sinGA], 
                           [sinGB,  cosGB]])
    else:
        matrix = np.array([[ cosGB, sinGA],
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
    G = np.arctan2(np.cos(lat)*np.sin(HA), np.cos(dec)*np.sin(lat) - np.sin(dec)*np.cos(lat)*np.cos(HA))
    GA = G + feedRotation
    GB = G + feedRotation
    cosGA = np.cos(GA)
    sinGA = np.sin(GA)
    cosGB = np.cos(GB)
    sinGB = np.sin(GB)
    
    # The matrix
    if not inverse:
        matrix = np.array([[cosGA, -sinGA], 
                           [sinGB,  cosGB]])
    else:
        matrix = np.array([[ cosGB, sinGA],
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
    for i in range(nStand//2):
        s0 = 2*i + 0
        s1 = 2*i + 2
        data[s0:s1,:] = np.dot(matrix, data[s0:s1,:])
        
    # Done
    return data
