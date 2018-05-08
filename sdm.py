# -*- coding: utf-8 -*-

"""
Module for working with VLA SDM files for flagging purposes

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
from xml.etree import ElementTree

from lsl.astro import MJD_OFFSET


__version__ = '0.1'
__revision__ = '$Rev'
__all__ = ['vla_to_utcmjd', 'vla_to_utcjd', 'getAntennas', 'getFlags', 'filterFlags',
           '__version__', '__revision__', '__all__']


def vla_to_utcmjd(timetag):
	mjd = timetag / 1e9 / 86400.0
	return mjd


def vla_to_utcjd(timetag):
	return vla_to_utcmjd(timetag) + MJD_OFFSET


def _parse_compound(text):
	rows, cols, data = text.split(None, 2)
	rows, cols = int(rows, 10), int(cols, 10)
	if rows == 1 and cols == 1:
		return data
	else:
		data = data.split(None, rows*cols-1)
		output = []
		for i,d in enumerate(data):
			if i % cols == 0:
				output.append( [] )
			ouput[-1].append( d )
		return output


def getAntennas(filename):
	if os.path.isdir(filename):
		filename = os.path.join(filename, 'Antenna.xml')
		
	tree = ElementTree.parse(filename)
	root = tree.getroot()
	
	ants = {}
	for row in root.iter('row'):
		entry = {}
		for field in row:
			name, value = field.tag, field.text
			entry[name] = value
			
		ants[entry['antennaId']] = entry['name'].upper()
		
	return ants


def getFlags(filename):
	if os.path.isdir(filename):
		filename = os.path.join(filename, 'Flag.xml')
		
	antname = os.path.join(os.path.dirname(filename), 'Antenna.xml')
	ants = getAntennas(antname)
	
	tree = ElementTree.parse(filename)
	root = tree.getroot()
	
	flags = []
	for row in root.iter('row'):
		entry = {}
		for field in row:
			name, value = field.tag, field.text
			if name[-4:] == 'Time':
				value = vla_to_utcjd( int(value, 10) )
			elif name == 'antennaId':
				value = _parse_compound(value)
				value = ants[value]
			entry[name] = value
			
		if entry['reason'] in ('SUBREFLECTOR_ERROR', 'FOCUS_ERROR'):
			continue
			
		flags.append( entry )
		
	return flags


def filterFlags(flags, startJD, stopJD):
	f = lambda x: False if x['endTime'] < startJD or x['startTime'] > stopJD else True
	return filter(f, flags)
