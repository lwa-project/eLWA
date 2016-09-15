# -*- coding: utf-8 -*-

"""
Module to help with manipulating HDF5 beam data files.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import h5py
import numpy
from datetime import datetime

from lsl.common.sdf import parseSDF
from lsl.common import dp, mcs, metabundle
from lsl.reader.drx import filterCodes


__version__ = "0.5"
__revision__ = "$Rev$"
__all__ = ['createNewFile', 'fillMinimum', 'fillFromMetabundle', 'fillFromSDF', 'getObservationSet', 'createDataSets', 'getDataSet', 
		 '__version__', '__revision__', '__all__']


def _valuetoDelay(value):
	try:
		return mcs.MCSDtodelay(value)
	except:
		value = ((value & 0xFF) << 8) | ((value >>8) & 0xFF)
		return dp.DPDtodelay(value)


def _valuetoGain(value):
	try:
		return mcs.MCSGtogain(value)
	except:
		value = ((value & 0xFF) << 8) | ((value >>8) & 0xFF)
		return dp.DPGtogain(value)


def createNewFile(filename):
	"""
	Create a new HDF5 and return the handle for it.  This sets up all of 
	the required attributes and groups and fills them with dummy values.
	
	Returns an open h5py.File instance.
	"""
	
	# Create the file
	f = h5py.File(filename, 'w')
	
	# Observer and Project Info.
	f.attrs['ObserverID'] = 0
	f.attrs['ObserverName'] = ''
	f.attrs['ProjectID'] = ''
	f.attrs['SessionID'] = 0
	
	# File creation time
	f.attrs['FileCreation'] = datetime.utcnow().strftime("UTC %Y/%m/%d %H:%M:%S")
	f.attrs['FileGenerator'] = ''
	
	# Input file info.
	f.attrs['InputData'] = ''
	f.attrs['InputMetadata'] = ''
	
	return f


def fillMinimum(f, obsID, beam, srate, srateUnits='samples/s'):
	"""
	Minimum metadata filling for a particular observation.
	"""
	
	# Get the group or create it if it doesn't exist
	obs = f.get('/Observation%i' % obsID, None)
	if obs is None:
		obs = f.create_group('/Observation%i' % obsID)
		
	# Target info.
	obs.attrs['TargetName'] = ''
	obs.attrs['RA'] = -99.0
	obs.attrs['RA_Units'] = 'hours'
	obs.attrs['Dec'] = -99.0
	obs.attrs['Dec_Units'] = 'degrees'
	obs.attrs['Epoch'] = 2000.0
	obs.attrs['TrackingMode'] = 'Unknown'
	
	# Observation info
	obs.attrs['ARX_Filter'] = -1.0
	obs.attrs['ARX_Gain1'] = -1.0
	obs.attrs['ARX_Gain2'] = -1.0
	obs.attrs['ARX_GainS'] = -1.0
	obs.attrs['Beam'] = beam
	obs.attrs['DRX_Gain'] = -1.0
	obs.attrs['sampleRate'] = srate
	obs.attrs['sampleRate_Units'] = srateUnits
	obs.attrs['tInt'] = -1.0
	obs.attrs['tInt_Units'] = 's'
	obs.attrs['LFFT'] = -1
	obs.attrs['nChan'] = -1
	obs.attrs['RBW'] = -1.0
	obs.attrs['RBW_Units'] = 'Hz'
	
	return True


def fillFromMetabundle(f, tarball):
	"""
	Fill in a HDF5 file based off an input metadata file.
	"""
	
	# Pull out what we need from the tarball
	sdf = metabundle.getSessionDefinition(tarball)
	cds = metabundle.getCommandScript(tarball)
	
	# Observer and Project Info.
	f.attrs['ObserverID'] = sdf.observer.id
	f.attrs['ObserverName'] = sdf.observer.name
	f.attrs['ProjectID'] = sdf.id
	f.attrs['SessionID'] = sdf.sessions[0].id
	
	# Input file info.
	f.attrs['InputMetadata'] = os.path.basename(tarball)
	
	# ARX configuration summary
	try:
		arx = metabundle.getASPConfigurationSummary(tarball)
	except:
		arx = {'filter': -1, 'at1': -1, 'at2': -1, 'atsplit': -1}
		
	for i,obsS in enumerate(sdf.sessions[0].observations):
		# Detailed observation information
		obsD = metabundle.getObservationSpec(tarball, selectObs=i+1)
		
		# Get the group or create it if it doesn't exist
		grp = f.get('/Observation%i' % (i+1,), None)
		if grp is None:
			grp = f.create_group('/Observation%i' % (i+1))
			
		# Target info.
		grp.attrs['ObservationName'] = obsS.name
		grp.attrs['TargetName'] = obsS.target
		grp.attrs['RA'] = obsD['RA']
		grp.attrs['RA_Units'] = 'hours'
		grp.attrs['Dec'] = obsD['Dec']
		grp.attrs['Dec_Units'] = 'degrees'
		grp.attrs['Epoch'] = 2000.0
		grp.attrs['TrackingMode'] = mcs.mode2string(obsD['Mode'])
		
		# Observation info
		grp.attrs['ARX_Filter'] = arx['filter']
		grp.attrs['ARX_Gain1'] = arx['at1']
		grp.attrs['ARX_Gain2'] = arx['at2']
		grp.attrs['ARX_GainS'] = arx['atsplit']
		grp.attrs['Beam'] = obsD['drxBeam']
		grp.attrs['DRX_Gain'] = obsD['drxGain']
		grp.attrs['sampleRate'] = float(filterCodes[obsD['BW']])
		grp.attrs['sampleRate_Units'] = 'samples/s'
		
		# Deal with stepped mode
		if mcs.mode2string(obsD['Mode']) == 'STEPPED':
			stps = grp.create_group('Pointing')
			stps.attrs['StepType'] = 'RA/Dec' if obsD['StepRADec'] else 'Az/Alt'
			stps.attrs['col0'] = 'StartTime'
			stps.attrs['col0_Unit'] = 's'
			stps.attrs['col1'] = 'RA' if obsD['StepRADec'] else 'Azimuth'
			stps.attrs['col1_Unit'] = 'h' if obsD['StepRADec'] else 'd'
			stps.attrs['col2'] = 'Dec' if obsD['StepRADec'] else 'Elevation'
			stps.attrs['col2_Unit'] = 'd'
			stps.attrs['col3'] = 'Tuning1'
			stps.attrs['col3_Unit'] = 'Hz'
			stps.attrs['col4'] = 'Tuning2'
			stps.attrs['col4_Unit'] = 'Hz'
		
			# Extract the data for the steps
			data = numpy.zeros((len(obsD['steps']), 5))
			t = obsD['MJD']*86400.0 + obsD['MPM']/1000.0 - 3506716800.0
			for i,s in enumerate(obsD['steps']):
				data[i,0] = t
				data[i,1] = s.OBS_STP_C1
				data[i,2] = s.OBS_STP_C2
				data[i,3] = dp.word2freq(s.OBS_STP_FREQ1)
				data[i,4] = dp.word2freq(s.OBS_STP_FREQ2)
			
				## Update the start time for the next step
				t += s.OBS_STP_T / 1000.0
				
			# Save it
			stps['Steps'] = data
				
			# Deal with specified delays and gains if needed
			if obsD['steps'][0].OBS_STP_B == 3:
				cbfg = grp.create_group('CustomBeamforming')
				dlys = cbfg.create_dataset('Delays', (len(obsD['steps']), 520+1), 'f4')
				dlys.attrs['col0'] = 'StartTime'
				dlys.attrs['col0_Unit'] = 's'
				for j in xrange(520):
					dlys.attrs['col%i' % (j+1)] = 'DP Digitizer %i' % (j+1)
					dlys.attrs['col%i_Unit' % (j+1)] = 'ns'
					
				# Extract the delays
				dataD = numpy.zeros((len(obsD['steps']), 520+1))
				t = obsD['MJD']*86400.0 + obsD['MPM']/1000.0 - 3506716800.0
				for i,s in enumerate(obsD['steps']):
					dataD[i,0] = t
					for j in xrange(520):
						dataD[i,1+j] = _valuetoDelay(s.delay[j])
						
				# Save the delays
				dlys[:,:] = dataD
				
				gais = cbfg.create_dataset('Gains', (len(obsD['steps']), 260*2*2+1), 'f4')
				gais.attrs['col0'] = 'StartTime'
				gais.attrs['col0_Unit'] = 's'
				m = 1
				for j in xrange(260):
					for k in xrange(2):
						for l in xrange(2):
							gais.attrs['col%i' % m] = 'DP Stand %i %s contribution to beam %s' % (j+1, 'X' if k == 0 else 'Y', 'X' if l == 0 else 'Y')
							gais.attrs['col%i_Unit' % m] = 'None'
							m += 1
							
				# Extract the gains
				dataG = numpy.zeros((len(obsD['steps']), 260*2*2+1))
				for i,s in enumerate(obsD['steps']):
					dataG[i,0] = t
					for j in xrange(260):
						dataG[i,1+4*j+0] = _valuetoGain(s.gain[j][0][0])
						dataG[i,1+4*j+1] = _valuetoGain(s.gain[j][0][1])
						dataG[i,1+4*j+2] = _valuetoGain(s.gain[j][1][0])
						dataG[i,1+4*j+3] = _valuetoGain(s.gain[j][1][1])
						
				# Save the gains
				gais[:,:] = dataG
				
	return True


def fillFromSDF(f, sdfFilename):
	"""
	Fill in a HDF5 file based off an input session definition file.
	"""
	
	# Pull out what we need from the tarball
	sdf = parseSDF(sdfFilename)
	
	# Observer and Project Info.
	f.attrs['ObserverID'] = sdf.observer.id
	f.attrs['ObserverName'] = sdf.observer.name
	f.attrs['ProjectID'] = sdf.id
	f.attrs['SessionID'] = sdf.sessions[0].id
	
	# Input file info.
	f.attrs['InputMetadata'] = os.path.basename(sdfFilename)
	
	# ARX configuration summary
	arx = {'filter': -1, 'at1': -1, 'at2': -1, 'atsplit': -1}
	arx['filter'] = numpy.median( sdf.sessions[0].observations[0].aspFlt )
	arx['at1'] = numpy.median( sdf.sessions[0].observations[0].aspAT1 )
	arx['at2'] = numpy.median( sdf.sessions[0].observations[0].aspAT2 )
	arx['atsplit'] = numpy.median( sdf.sessions[0].observations[0].aspATS )
	
	for i,obsS in enumerate(sdf.sessions[0].observations):
		# Get the group or create it if it doesn't exist
		grp = f.get('/Observation%i' % (i+1,), None)
		if grp is None:
			grp = f.create_group('/Observation%i' % (i+1))
			
		# Target info.
		grp.attrs['ObservationName'] = obsS.name
		grp.attrs['TargetName'] = obsS.target
		grp.attrs['RA'] = obsS.ra
		grp.attrs['RA_Units'] = 'hours'
		grp.attrs['Dec'] = obsS.dec
		grp.attrs['Dec_Units'] = 'degrees'
		grp.attrs['Epoch'] = 2000.0
		grp.attrs['TrackingMode'] = obsS.mode
		
		# Observation info
		grp.attrs['ARX_Filter'] = arx['filter']
		grp.attrs['ARX_Gain1'] = arx['at1']
		grp.attrs['ARX_Gain2'] = arx['at2']
		grp.attrs['ARX_GainS'] = arx['atsplit']
		grp.attrs['Beam'] = sdf.sessions[0].drxBeam
		grp.attrs['DRX_Gain'] = obsS.gain
		grp.attrs['sampleRate'] = float(filterCodes[obsS.filter])
		grp.attrs['sampleRate_Units'] = 'samples/s'
		
		# Deal with stepped mode
		if obsS.mode == 'STEPPED':
			stps = grp.create_group('Pointing')
			stps.attrs['StepType'] = 'RA/Dec' if obsS.RADec else 'Az/Alt'
			stps.attrs['col0'] = 'StartTime'
			stps.attrs['col0_Unit'] = 's'
			stps.attrs['col1'] = 'RA' if obsS.RADec else 'Azimuth'
			stps.attrs['col1_Unit'] = 'h' if obsS.RADec else 'd'
			stps.attrs['col2'] = 'Dec' if obsS.RADec else 'Elevation'
			stps.attrs['col2_Unit'] = 'd'
			stps.attrs['col3'] = 'Tuning1'
			stps.attrs['col3_Unit'] = 'Hz'
			stps.attrs['col4'] = 'Tuning2'
			stps.attrs['col4_Unit'] = 'Hz'
			
			# Extract the data for the steps
			data = numpy.zeros((len(obsS.steps), 5))
			t = obsS.mjd*86400.0 + obsS.mpm/1000.0 - 3506716800.0
			for i,s in enumerate(obsS.steps):
				data[i,0] = t
				data[i,1] = s.c1
				data[i,2] = s.c2
				data[i,3] = dp.word2freq(s.freq1)
				data[i,4] = dp.word2freq(s.freq2)
				
				## Update the start time for the next step
				t += s.dur / 1000.0
				
			# Save it
			stps['Steps'] = data
			
			# Deal with specified delays and gains if needed
			if obsS.steps[0].delays is not None and obsS.steps[0].gains is not None:
				cbfg = grp.create_group('CustomBeamforming')
				dlys = cbfg.create_dataset('Delays', (len(obsS.steps), 520+1), 'f4')
				dlys.attrs['col0'] = 'StartTime'
				dlys.attrs['col0_Unit'] = 's'
				for j in xrange(520):
					dlys.attrs['col%i' % (j+1)] = 'DP Digitizer %i' % (j+1)
					dlys.attrs['col%i_Unit' % (j+1)] = 'ns'
					
				# Extract the delays
				dataD = numpy.zeros((len(obsS.steps), 520+1))
				t = obsS.mjd*86400.0 + obsS.mpm/1000.0 - 3506716800.0
				for i,s in enumerate(obsS.steps):
					dataD[i,0] = t
					for j in xrange(520):
						dataD[i,1+j] = _valuetoDelay(s.delays[j])
						
				# Save the delays
				dlys[:,:] = dataD
				
				gais = cbfg.create_dataset('Gains', (len(obsS.steps), 260*2*2+1), 'f4')
				gais.attrs['col0'] = 'StartTime'
				gais.attrs['col0_Unit'] = 's'
				m = 1
				for j in xrange(260):
					for k in xrange(2):
						for l in xrange(2):
							gais.attrs['col%i' % m] = 'DP Stand %i %s contribution to beam %s' % (j+1, 'X' if k == 0 else 'Y', 'X' if l == 0 else 'Y')
							gais.attrs['col%i_Unit' % m] = 'None'
							m += 1
							
				# Extract the gains
				dataG = numpy.zeros((len(obsS.steps), 260*2*2+1))
				for i,s in enumerate(obsS.steps):
					dataG[i,0] = t
					for j in xrange(260):
						dataG[i,1+4*j+0] = _valuetoGain(s.gains[j][0][0])
						dataG[i,1+4*j+1] = _valuetoGain(s.gains[j][0][1])
						dataG[i,1+4*j+2] = _valuetoGain(s.gains[j][1][0])
						dataG[i,1+4*j+3] = _valuetoGain(s.gains[j][1][1])
						
				# Save the gains
				gais[:,:] = dataG
				
	return True


def getObservationSet(f, observation):
	"""
	Return a reference to the specified observation.
	"""
	
	# Get the observation
	obs = f.get('/Observation%i' % observation, None)
	if obs is None:
		raise RuntimeError('No such observation: %i' % observation)
		
	return obs


def createDataSets(f, observation, tuning, frequency, chunks, dataProducts=['XX', 'YY']):
	"""
	Fill in a tuning group with the right set of dummy data sets and 
	attributes.
	"""
	
	# Get the observation
	obs = f.get('/Observation%i' % observation, None)
	if obs is None:
		obs = f.create_group('/Observation%i' % observation)
		
		# Target info.
		obs.attrs['TargetName'] = ''
		obs.attrs['RA'] = -99.0
		obs.attrs['RA_Units'] = 'hours'
		obs.attrs['Dec'] = -99.0
		obs.attrs['Dec_Units'] = 'degrees'
		obs.attrs['Epoch'] = 2000.0
		obs.attrs['TrackingMode'] = 'Unknown'
		
		# Observation info
		obs.attrs['ARX_Filter'] = -1.0
		obs.attrs['ARX_Gain1'] = -1.0
		obs.attrs['ARX_Gain2'] = -1.0
		obs.attrs['ARX_GainS'] = -1.0
		obs.attrs['Beam'] = -1.0
		obs.attrs['DRX_Gain'] = -1.0
		obs.attrs['sampleRate'] = -1.0
		obs.attrs['sampleRate_Units'] = 'samples/s'
		obs.attrs['tInt'] = -1.0
		obs.attrs['tInt_Units'] = 's'
		obs.attrs['LFFT'] = -1
		obs.attrs['nChan'] = -1
		obs.attrs['RBW'] = -1.0
		obs.attrs['RBW_Units'] = 'Hz'
		
	# Get the group or create it if it doesn't exist
	grp = obs.get('Tuning%i' % tuning, None)
	if grp is None:
		grp = obs.create_group('Tuning%i' % tuning)
		
	grp['freq'] = frequency
	grp['freq'].attrs['Units'] = 'Hz'
	for p in dataProducts:
		d = grp.create_dataset(p, (chunks, frequency.size), 'f4')
		d.attrs['axis0'] = 'time'
		d.attrs['axis1'] = 'frequency'
	d = grp.create_dataset('Saturation', (chunks, 2), 'i8')
	d.attrs['axis0'] = 'time'
	d.attrs['axis1'] = 'polarization'
	
	return True


def getDataSet(f, observation, tuning, dataProduct):
	"""
	Return a reference to the specified data set.
	"""
	
	# Get the observation
	obs = f.get('/Observation%i' % observation, None)
	if obs is None:
		raise RuntimeError('No such observation: %i' % observation)
		
	# Get the groups
	grp = obs.get('Tuning%i' % tuning, None)
	if grp is None:
		raise RuntimeError("Unknown tuning: %i" % tuning)
		
	# Get the data set
	try:
		d = grp[dataProduct]
	except:
		raise RuntimeError("Unknown data product for Observation %i, Tuning %i: %s" % (observation, tuning, dataProduct))
		
	return d
