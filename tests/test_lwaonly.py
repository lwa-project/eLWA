"""
Unit tests for the a small LWA correlation job.
"""

import unittest
unittest.TestLoader.sortTestMethodsUsing = None

import os
import re
import sys
import glob
import numpy
from astropy.io import fits as astrofits
import subprocess

from lsl.common.data_access import download_file

_RAW = 'eLWA_test_small_raw.tar.gz'
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
            download_file(f"https://fornax.phys.unm.edu/lwa/data/{_RAW}", _RAW)
            subprocess.check_call(['tar', 'xzf', _RAW])
            
        # Other variables
        self._FILES = ['0*', 'LT004_*.tgz']
        self._BASENAME = 'lwaonly'
        
    def test_0_create(self):
        """Build the correlator configuration file."""
        
        files = []
        for regex in self._FILES:
            files.extend(glob.glob(regex))
        files.sort()
        
        cmd = [sys.executable, '../createConfigFile.py', '-o', '%s.config' % self._BASENAME]
        cmd.extend(files)
        status = subprocess.check_call(cmd)
        self.assertEqual(status, 0)
        
    def test_1_correlate(self):
        """Run the correlator on eLWA data."""

        cmd = [sys.executable, '../superCorrelator.py', '-t', '1', '-l', '512', '-w', '1',
               '-g', '%sL' % self._BASENAME, '%s.config' % self._BASENAME]
        with open('%s-correlate-L.log' % self._BASENAME, 'w') as logfile:
            status = subprocess.check_call(cmd, stdout=logfile)
        self.assertEqual(status, 0)
        cmd = [sys.executable, '../superCorrelator.py', '-t', '1', '-l', '512', '-w', '2',
               '-g', '%sH' % self._BASENAME, '%s.config' % self._BASENAME]
        with open('%s-correlate-H.log' % self._BASENAME, 'w') as logfile:
            status = subprocess.check_call(cmd, stdout=logfile)
        self.assertEqual(status, 0)

    def test_1a_correlate_dual_tuning(self):
        """Run the correlator on eLWA data with dual-tuning mode."""

        # Run correlator with -w 0 to process both tunings at once
        cmd = [sys.executable, '../superCorrelator.py', '-t', '1', '-l', '512', '-w', '0',
               '-g', '%s_dual' % self._BASENAME, '%s.config' % self._BASENAME]
        with open('%s-correlate-dual.log' % self._BASENAME, 'w') as logfile:
            status = subprocess.check_call(cmd, stdout=logfile)
        self.assertEqual(status, 0)

        # Verify both tuning output files were created
        # Dual-tuning mode uses 'L' and 'H' suffixes for compatibility with buildMultiBandIDI.py
        files_t1 = glob.glob('%s_dualL-*.npz' % self._BASENAME)
        files_t2 = glob.glob('%s_dualH-*.npz' % self._BASENAME)
        self.assertGreater(len(files_t1), 0, "No tuning 1 output files found")
        self.assertGreater(len(files_t2), 0, "No tuning 2 output files found")
        self.assertEqual(len(files_t1), len(files_t2), "Tuning 1 and 2 should have same number of files")

        # Basic sanity check: compare file sizes (should be similar)
        for f1, f2 in zip(sorted(files_t1), sorted(files_t2)):
            size1 = os.path.getsize(f1)
            size2 = os.path.getsize(f2)
            # Sizes should be within 10% of each other
            self.assertLess(abs(size1 - size2) / max(size1, size2), 0.1,
                           f"File sizes differ too much: {f1} ({size1}) vs {f2} ({size2})")

        # Verify file format is correct
        data1 = numpy.load(files_t1[0])
        data2 = numpy.load(files_t2[0])

        # Check required fields are present
        for key in ['config', 'srate', 'freq1', 'vis1XX', 'vis1XY', 'vis1YX', 'vis1YY', 'tStart', 'tInt']:
            self.assertIn(key, data1.files, f"Missing key {key} in tuning 1 file")
            self.assertIn(key, data2.files, f"Missing key {key} in tuning 2 file")

        # Check that frequencies are different between tunings
        freq1 = data1['freq1']
        freq2 = data2['freq1']
        self.assertFalse(numpy.allclose(freq1, freq2), "Tuning 1 and 2 frequencies should be different")

    def test_1b_compare_dual_vs_single(self):
        """Compare dual-tuning output with single-tuning output to verify correctness."""

        # Load dual-tuning outputs
        files_dual_t1 = sorted(glob.glob('%s_dualL-*.npz' % self._BASENAME))
        files_dual_t2 = sorted(glob.glob('%s_dualH-*.npz' % self._BASENAME))

        # Load single-tuning outputs (from test_1_correlate)
        files_single_t1 = sorted(glob.glob('%sL-*.npz' % self._BASENAME))
        files_single_t2 = sorted(glob.glob('%sH-*.npz' % self._BASENAME))

        # Skip if we don't have both sets
        if not (files_dual_t1 and files_dual_t2 and files_single_t1 and files_single_t2):
            self.skipTest("Missing output files for comparison")

        # Compare first file from each set
        dual_t1_data = numpy.load(files_dual_t1[0])
        dual_t2_data = numpy.load(files_dual_t2[0])
        single_t1_data = numpy.load(files_single_t1[0])
        single_t2_data = numpy.load(files_single_t2[0])

        # Compare visibilities: dual-tuning t1 should match single-tuning L (tuning 1)
        for key in ['vis1XX', 'vis1XY', 'vis1YX', 'vis1YY']:
            # Allow small numerical differences due to floating point precision
            self.assertTrue(
                numpy.allclose(dual_t1_data[key], single_t1_data[key], rtol=1e-5, atol=1e-8),
                f"Dual-tuning T1 {key} doesn't match single-tuning L output"
            )
            self.assertTrue(
                numpy.allclose(dual_t2_data[key], single_t2_data[key], rtol=1e-5, atol=1e-8),
                f"Dual-tuning T2 {key} doesn't match single-tuning H output"
            )

        # Compare frequencies
        self.assertTrue(
            numpy.allclose(dual_t1_data['freq1'], single_t1_data['freq1'], rtol=1e-10),
            "Dual-tuning T1 frequencies don't match single-tuning L"
        )
        self.assertTrue(
            numpy.allclose(dual_t2_data['freq1'], single_t2_data['freq1'], rtol=1e-10),
            "Dual-tuning T2 frequencies don't match single-tuning H"
        )

        # Compare timestamps
        self.assertAlmostEqual(
            dual_t1_data['tStart'], single_t1_data['tStart'], places=3,
            msg="Dual-tuning T1 timestamp doesn't match single-tuning L"
        )
        self.assertAlmostEqual(
            dual_t2_data['tStart'], single_t2_data['tStart'], places=3,
            msg="Dual-tuning T2 timestamp doesn't match single-tuning H"
        )

    def test_2_build(self):
        """Build a FITS-IDI file for the eLWA data."""
        
        files = glob.glob('%s[LH]-*.npz' % self._BASENAME)
        cmd = [sys.executable, '../buildMultiBandIDI.py', '-f', '-t', self._BASENAME]
        cmd.extend(files)
        with open('%s-build.log' % self._BASENAME, 'w') as logfile:
            status = subprocess.check_call(cmd, stdout=logfile)
        self.assertEqual(status, 0)
        
    def test_3_flag_rfi(self):
        """Flag interference in the FITS-IDI file."""
        
        cmd = [sys.executable, '../flagIDI.py', '-f', 'buildIDI_%s.FITS_1' % self._BASENAME]
        with open('%s-flag.log' % self._BASENAME, 'w') as logfile:
            status = subprocess.check_call(cmd, stdout=logfile)
        self.assertEqual(status, 0)
        
    def test_4_fringe_search(self):
        """Seach for fringes in the FITS-IDI file."""
        
        cmd = [sys.executable, '../fringeSearchIDI.py', '-r', 'LWA1', 'buildIDI_%s_flagged.FITS_1' % self._BASENAME]
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
                    self.assertTrue(snr >= 11)


class lwa_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the LWA correlation tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(lwa_tests)) 


if __name__ == '__main__':
    unittest.main()
    
