"""
Unit tests for the a small LWA correlation job.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import unittest
unittest.TestLoader.sortTestMethodsUsing = None

import os
import re
import glob
import numpy
from astropy.io import fits as astrofits
import subprocess

_RAW = 'eLWA_test_raw.tar.gz'
_REF = 'eLWA_test_ref.tar.gz'


class lwa_tests(unittest.TestCase):
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
            
        # Reference data
        if not os.path.exists('ref'):
            os.mkdir('ref')
            
        if not os.path.exists('ref/%s' % _REF):
            subprocess.check_call(['curl',
                                   'https://fornax.phys.unm.edu/lwa/data/%s' % _REF,
                                   '-o', 'ref/%s' % _REF])
            subprocess.check_call(['tar', '-C', 'ref', '-x', '-z', '-f', 'ref/%s' % _REF])
            
        # Other variables
        self._FILES = ['0*', 'LT004_*.tgz']
        self._BASENAME = 'lwaonly'
        
    def test_0_create(self):
        """Build the correlator configuration file."""
        
        files = []
        for regex in self._FILES:
            files.extend(glob.glob(regex))
            
        cmd = [sys.executable, '../createConfigFile.py', '-o', '%s.config' % self._BASENAME]
        cmd.extend(files)
        status = subprocess.check_call(cmd)
        self.assertEqual(status, 0)
        
    def test_1_correlate(self):
        """Run the correlator on eLWA data."""
        
        cmd = [sys.executable, '../superCorrelator.py', '-t', '1', '-l', '256', '-w', '1', 
               '-g', '%sL' % self._BASENAME, '%s.config' % self._BASENAME]
        with open('%s-correlate-L.log' % self._BASENAME, 'w') as logfile:
            status = subprocess.check_call(cmd, stdout=logfile)
        self.assertEqual(status, 0)
        cmd = [sys.executable, '../superCorrelator.py', '-t', '1', '-l', '256', '-w', '2', 
               '-g', '%sH' % self._BASENAME, '%s.config' % self._BASENAME]
        with open('%s-correlate-H.log' % self._BASENAME, 'w') as logfile:
            status = subprocess.check_call(cmd, stdout=logfile)
        
    def test_2_build(self):
        """Build a FITS-IDI file for the eLWA data."""
        
        files = glob.glob('%s[LH]-*.npz' % self._BASENAME)
        cmd = [sys.executable, '../buildMultiBandIDI.py', '-t', self._BASENAME]
        cmd.extend(files)
        with open('%s-build.log' % self._BASENAME, 'w') as logfile:
            status = subprocess.check_call(cmd, stdout=logfile)
        self.assertEqual(status, 0)
        
    def test_3_flag_rfi(self):
        """Flag interference in the FITS-IDI file."""
        
        cmd = [sys.executable, '../flagIDI.py', 'buildIDI_%s.FITS_1' % self._BASENAME]
        with open('%s-flag.log' % self._BASENAME, 'w') as logfile:
            status = subprocess.check_call(cmd, stdout=logfile)
        self.assertEqual(status, 0)
        
    def test_4_validate_headers(self):
        """Validate the headers of the flagged FITS-IDI file against the reference."""
        
        _revRE = re.compile('\$Rev.*?\$')
        
        hdulist1 = astrofits.open('buildIDI_%s_flagged.FITS_1' % self._BASENAME,
                               mode='readonly')
        hdulist2 = astrofits.open('./ref/buildIDI_%s_flagged_flagged.FITS_1' % self._BASENAME,
                               mode='readonly')
        
        # Loop through the HDUs
        for hdu1,hdu2 in zip(hdulist1, hdulist2):
            ## Skip over the FLAG header
            if hdu1.name in ('FLAG',):
                continue
                
            ## Check the header values, modulo the old $Rev$ tag
            for key in hdu1.header:
                if key in ('DATE-MAP', 'UT1UTC', 'POLARX', 'POLARY'):
                    continue
                h1 = re.sub(_revRE, '', str(hdu1.header[key]))
                h2 = re.sub(_revRE, '', str(hdu2.header[key]))
                self.assertEqual(h1, h2, "Mis-match on %s: '%s' != '%s'" % (key, h1, h2))
                
        hdulist1.close()
        hdulist2.close()
        
    def test_5_validate_data(self):
        """Validate the data in the flagged FITS-IDI file against the reference."""
        
        hdulist1 = astrofits.open('buildIDI_%s_flagged.FITS_1' % self._BASENAME,
                               mode='readonly')
        hdulist2 = astrofits.open('./ref/buildIDI_%s_flagged_flagged.FITS_1' % self._BASENAME,
                               mode='readonly')
        
        # Loop through the HDUs
        for hdu1,hdu2 in zip(hdulist1, hdulist2):
            ## Skip over the PRIMARY and FLAG headers
            if hdu1.name in ('PRIMARY', 'FLAG'):
                continue
                
            for r,row1,row2 in zip(range(len(hdu1.data)), hdu1.data, hdu2.data):
                for f in range(len(row1)):
                    try:
                        same_value = numpy.allclose(row1[f], row2[f])
                    except TypeError:
                        same_value = numpy.array_equal(row1[f], row2[f])
                    self.assertTrue(same_value, "%s, row %i, field %i (%s) does not match - %s != %s" % (hdu1.name, r, f, hdu1.data.columns[f], row1[f], row2[f]))
                    
        hdulist1.close()
        hdulist2.close()


class lwa_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the LWA correlation tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(lwa_tests)) 


if __name__ == '__main__':
    unittest.main()
    
