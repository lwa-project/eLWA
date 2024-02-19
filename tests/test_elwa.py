"""
Unit tests for the a small eLWA correlation job.
"""

import unittest
import os
import re
import sys
import glob
import numpy
from astropy.io import fits as astrofits
import subprocess

_RAW = 'eLWA_test_small_raw.tar.gz'
_REF = 'eLWA_test_ref.tar.gz'


class elwa_tests(unittest.TestCase):
    def setUp(self):
        """Make sure we have the comparison files in place."""
        
        # get_vla_ant_pos.py
        if not os.path.exists('../get_vla_ant_pos.py'):
            with open('../get_vla_ant_pos.py', 'w') as fh:
                fh.write("""import numpy
class database(object):
    def __init__(self, *args, **kwds):
        self._ready = True
    def get_pad(self,ant,date):
        return 'W40', None
    def get_xyz(self,ant,date):
        return numpy.array((-6777.0613, -360.7018, -3550.9465), dtype=numpy.float64)
    def close(self):
        return True""")
            
        # Raw data
        if not os.path.exists(_RAW):
            subprocess.check_call(['curl',
                                   'https://fornax.phys.unm.edu/lwa/data/%s' % _RAW,
                                   '-o', _RAW])
            subprocess.check_call(['tar', 'xzf', _RAW])
            
        # Other variables
        self._FILES = ['0*', '*.vdif', 'LT004_*.tgz']
        self._BASENAME = 'elwa'
        
    def test_0_create(self):
        """Build the correlator configuration file."""
        
        files = []
        for regex in self._FILES:
            files.extend(glob.glob(regex))
        files.sort()
        
        cmd = [sys.executable, '../createConfigFile.py', '-o', '%s.config' % self._BASENAME]
        cmd.extend(files)
        with open('%s-create.log' % self._BASENAME, 'w') as logfile:
            try:
                status = subprocess.check_call(cmd, stdout=logfile)
            except subprocess.CalledProcessError:
                status = 1
        if status == 1:
            with open('%s-create.log' % self._BASENAME, 'r') as logfile:
                print(logfile.read())
        self.assertEqual(status, 0)
        
    def test_1_correlate(self):
        """Run the correlator on eLWA data."""
        
        cmd = [sys.executable, '../superCorrelator.py', '-t', '1', '-l', '4096', 
               '-g', self._BASENAME, '%s.config' % self._BASENAME]
        with open('%s-correlate.log' % self._BASENAME, 'w') as logfile:
            try:
                status = subprocess.check_call(cmd, stdout=logfile)
            except subprocess.CalledProcessError:
                status = 1
        if status == 1:
            with open('%s.config' % self._BASENAME, 'r') as configfile:
                print(configfile.read())
            with open('%s-correlate.log' % self._BASENAME, 'r') as logfile:
                print(logfile.read())
        self.assertEqual(status, 0)
        
    def test_2_build(self):
        """Build a FITS-IDI file for the eLWA data."""
        
        files = glob.glob('%s-*.npz' % self._BASENAME)
        cmd = [sys.executable, '../buildIDI.py', '-f', '-t', self._BASENAME]
        cmd.extend(files)
        with open('%s-build.log' % self._BASENAME, 'w') as logfile:
            try:
                status = subprocess.check_call(cmd, stdout=logfile)
            except subprocess.CalledProcessError:
                status = 1
        if status == 1:
            with open('%s-build.log' % self._BASENAME, 'r') as logfile:
                print(logfile.read())
        self.assertEqual(status, 0)
        
    def test_3_flag_rfi(self):
        """Flag interference in the FITS-IDI file."""
        
        cmd = [sys.executable, '../flagIDI.py', '-f', 'buildIDI_%s.FITS_1' % self._BASENAME]
        with open('%s-flag.log' % self._BASENAME, 'w') as logfile:
            try:
                status = subprocess.check_call(cmd, stdout=logfile)
            except subprocess.CalledProcessError:
                status = 1
        if status == 1:
            with open('%s-flag.log' % self._BASENAME, 'r') as logfile:
                print(logfile.read())
        self.assertEqual(status, 0)
        
    def test_4_fringe_search(self):
        """Seach for fringes in the FITS-IDI file."""
        
        cmd = [sys.executable, '../fringeSearchIDI.py', '-r', '51', 'buildIDI_%s_flagged.FITS_1' % self._BASENAME]
        with open('%s-fringe.log' % self._BASENAME, 'w') as logfile:
            try:
                status = subprocess.check_call(cmd, stdout=logfile)
            except subprocess.CalledProcessError:
                status = 1
        if status == 1:
            with open('%s-fringe.log' % self._BASENAME, 'r') as logfile:
                print(logfile.read())
        self.assertEqual(status, 0)
        
        with open('%s-fringe.log' % self._BASENAME, 'r') as logfile:
            for line in logfile:
                if line.find('LWA1-') != -1 and (line.find('XX') != -1 or line.find('YY') != -1):
                    fields = line.split(None)
                    snr = float(fields[3])
                    self.assertTrue(snr >= 19)


class elwa_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the eLWA correlation tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(elwa_tests)) 


if __name__ == '__main__':
    unittest.main()
    
