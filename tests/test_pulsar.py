"""
Unit tests for superPulsarCorrelator.py with synthetic pulsar data.
"""

import unittest
unittest.TestLoader.sortTestMethodsUsing = None

import os
import sys
import glob
import numpy
import subprocess
from astropy.time import Time

from lsl.common.data_access import download_file

_RAW = 'eLWA_test_small_raw.tar.gz'


class pulsar_tests(unittest.TestCase):
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

        # Raw data (reuse existing test data)
        if not os.path.exists(_RAW):
            download_file(f"https://fornax.phys.unm.edu/lwa/data/{_RAW}", _RAW)
            subprocess.check_call(['tar', 'xzf', _RAW])

        # Other variables
        self._FILES = ['0*']  # DRX files only (no VDIF for this test)
        self._BASENAME = 'pulsar_test'

    def test_0_create_polyco(self):
        """Create a synthetic polyco file with a trivial pulsar."""

        # Create a simple polyco file for a 1-second period pulsar
        # Format based on TEMPO polyco format
        # Note: pulsar name must match config name with 'PSR' removed
        # MJD 58761.98 matches the test data at MJD ~58761.984133
        polyco_content = """TEST      05-Oct-19   23:30:00.00   58761.98000000    0.000  1.0000    -8.50
   0.000000    1.000000000   6   120   12   1400.00
   1.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
   0.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
   0.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
   0.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
"""

        with open(f'{self._BASENAME}.polyco', 'w') as fh:
            fh.write(polyco_content)

        self.assertTrue(os.path.exists(f'{self._BASENAME}.polyco'))

    def test_1_create_config(self):
        """Build the correlator configuration file with polyco reference."""

        files = []
        for regex in self._FILES:
            files.extend(glob.glob(regex))
        files.sort()

        if len(files) < 2:
            self.skipTest("Need at least 2 DRX files (LWA1 and LWA-SV)")

        # Manually create a minimal config file since createConfigFile.py
        # doesn't work well with DRX-only data without source info
        # Need two antennas for correlation: LWA1 and LWA-SV
        config_content = f"""Context
Observer TestObserver
Project TestProject
Session TestSession
EndContext
Configuration
Channels 512
IntTime 1.0
EndConfiguration
Source
Intent target
Name TESTPSR
RA2000 19:59:28.35
Dec2000 40:44:02.0
Polyco {self._BASENAME}.polyco
Duration 2.0
SourceDone
Input
File {files[0]}
Type DRX
Antenna LWA1
Pols X, Y
Location -187.973523, 380.893372, 1.203428
ClockOffset 0.0, 0.0
InputDone
Input
File {files[1]}
Type DRX
Antenna LWA-SV
Pols X, Y
Location 68127.541828, 31632.511219, -1091.455199
ClockOffset 0.0, 0.0
InputDone
"""

        with open(f'{self._BASENAME}.config', 'w') as fh:
            fh.write(config_content)

        # Verify polyco is in config
        with open(f'{self._BASENAME}.config', 'r') as fh:
            config_text = fh.read()
        self.assertTrue('Polyco' in config_text)
        self.assertTrue(os.path.exists(f'{self._BASENAME}.config'))

    def test_2_correlate_single_tuning(self):
        """Run the pulsar correlator on single tuning."""

        cmd = [sys.executable, '../superPulsarCorrelator.py',
               '-t', '1',           # 1 second integration
               '-s', '0.5',         # 0.5 second sub-integration
               '-l', '512',         # 512 channel FFT
               '-w', '1',           # Process tuning 1 only
               '-b', '2',           # 2 profile bins (enough for ~2s of test data)
               '-g', f'{self._BASENAME}_t1',
               f'{self._BASENAME}.config']

        with open(f'{self._BASENAME}-correlate-t1.log', 'w') as logfile:
            try:
                status = subprocess.check_call(cmd, stdout=logfile, stderr=subprocess.STDOUT, timeout=120)
            except subprocess.CalledProcessError as e:
                status = e.returncode
            except subprocess.TimeoutExpired:
                self.fail("Correlator timed out after 120 seconds")

        if status != 0:
            with open(f'{self._BASENAME}-correlate-t1.log', 'r') as logfile:
                print(logfile.read())

        self.assertEqual(status, 0)

        # Check that output files were created
        files = glob.glob(f'{self._BASENAME}_t1-vis2-bin*.npz')
        self.assertTrue(len(files) > 0, "No output files created")

        # Verify file format
        data = numpy.load(files[0])
        self.assertTrue('vis1XX' in data)
        self.assertTrue('freq1' in data)
        self.assertTrue('polycos' in data)

    def test_3_correlate_dual_tuning(self):
        """Run the pulsar correlator on both tunings."""

        cmd = [sys.executable, '../superPulsarCorrelator.py',
               '-t', '1',
               '-s', '0.5',
               '-l', '512',
               '-w', '0',           # Process both tunings
               '-b', '2',           # 2 profile bins (enough for ~2s of test data)
               '-g', f'{self._BASENAME}_dual',
               f'{self._BASENAME}.config']

        with open(f'{self._BASENAME}-correlate-dual.log', 'w') as logfile:
            try:
                status = subprocess.check_call(cmd, stdout=logfile, stderr=subprocess.STDOUT, timeout=180)
            except subprocess.CalledProcessError as e:
                status = e.returncode
            except subprocess.TimeoutExpired:
                self.fail("Correlator timed out after 180 seconds")

        if status != 0:
            with open(f'{self._BASENAME}-correlate-dual.log', 'r') as logfile:
                print(logfile.read())

        self.assertEqual(status, 0)

        # Check that output files were created for both tunings
        files_L = glob.glob(f'{self._BASENAME}_dualL-vis2-bin*.npz')
        files_H = glob.glob(f'{self._BASENAME}_dualH-vis2-bin*.npz')

        self.assertTrue(len(files_L) > 0, "No L-band output files created")
        self.assertTrue(len(files_H) > 0, "No H-band output files created")

        # Verify both tunings have the same number of files
        self.assertEqual(len(files_L), len(files_H),
                        f"Different number of files for L ({len(files_L)}) and H ({len(files_H)})")

        # Verify file format
        data_L = numpy.load(files_L[0])
        data_H = numpy.load(files_H[0])

        self.assertTrue('vis1XX' in data_L)
        self.assertTrue('vis1XX' in data_H)
        self.assertTrue('freq1' in data_L)
        self.assertTrue('freq1' in data_H)

        # Verify frequencies are different
        freq_L = data_L['freq1']
        freq_H = data_H['freq1']
        self.assertFalse(numpy.allclose(freq_L, freq_H),
                        "L and H tuning frequencies should be different")

    def test_4_build_single(self):
        """Build a FITS-IDI file from single-tuning pulsar data."""

        # Use all bin files for building FITS-IDI (more data for fringe search)
        files = glob.glob(f'{self._BASENAME}_t1-vis2-bin*.npz')
        if len(files) == 0:
            self.skipTest("No output files found")

        cmd = [sys.executable, '../buildIDI.py', '-f', '-t', f'{self._BASENAME}_t1']
        cmd.extend(files)
        with open(f'{self._BASENAME}_t1-build.log', 'w') as logfile:
            try:
                status = subprocess.check_call(cmd, stdout=logfile, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                status = e.returncode

        if status != 0:
            with open(f'{self._BASENAME}_t1-build.log', 'r') as logfile:
                print(logfile.read())

        self.assertEqual(status, 0)

    def test_5_flag_single(self):
        """Flag interference in the single-tuning FITS-IDI file."""

        cmd = [sys.executable, '../flagIDI.py', '-f', f'buildIDI_{self._BASENAME}_t1.FITS_1']
        with open(f'{self._BASENAME}_t1-flag.log', 'w') as logfile:
            try:
                status = subprocess.check_call(cmd, stdout=logfile, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                status = e.returncode

        if status != 0:
            with open(f'{self._BASENAME}_t1-flag.log', 'r') as logfile:
                print(logfile.read())

        self.assertEqual(status, 0)

    def test_6_fringe_search_single(self):
        """Search for fringes in the single-tuning FITS-IDI file."""

        cmd = [sys.executable, '../fringeSearchIDI.py', '-r', 'LWA1',
               f'buildIDI_{self._BASENAME}_t1_flagged.FITS_1']
        with open(f'{self._BASENAME}_t1-fringe.log', 'w') as logfile:
            try:
                status = subprocess.check_call(cmd, stdout=logfile, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                status = e.returncode

        if status != 0:
            with open(f'{self._BASENAME}_t1-fringe.log', 'r') as logfile:
                print(logfile.read())

        self.assertEqual(status, 0)

    def test_7_build_dual(self):
        """Build a FITS-IDI file from dual-tuning pulsar data."""

        # Use all bin files for both tunings (more data for fringe search)
        files = glob.glob(f'{self._BASENAME}_dual[LH]-vis2-bin*.npz')
        if len(files) == 0:
            self.skipTest("No output files found for dual tuning")

        cmd = [sys.executable, '../buildMultiBandIDI.py', '-f', '-t', f'{self._BASENAME}_dual']
        cmd.extend(files)
        with open(f'{self._BASENAME}_dual-build.log', 'w') as logfile:
            try:
                status = subprocess.check_call(cmd, stdout=logfile, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                status = e.returncode

        if status != 0:
            with open(f'{self._BASENAME}_dual-build.log', 'r') as logfile:
                print(logfile.read())

        self.assertEqual(status, 0)

    def test_8_flag_dual(self):
        """Flag interference in the dual-tuning FITS-IDI file."""

        cmd = [sys.executable, '../flagIDI.py', '-f', f'buildIDI_{self._BASENAME}_dual.FITS_1']
        with open(f'{self._BASENAME}_dual-flag.log', 'w') as logfile:
            try:
                status = subprocess.check_call(cmd, stdout=logfile, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                status = e.returncode

        if status != 0:
            with open(f'{self._BASENAME}_dual-flag.log', 'r') as logfile:
                print(logfile.read())

        self.assertEqual(status, 0)

    def test_9_fringe_search_dual(self):
        """Search for fringes in the dual-tuning FITS-IDI file."""

        cmd = [sys.executable, '../fringeSearchIDI.py', '-r', 'LWA1',
               f'buildIDI_{self._BASENAME}_dual_flagged.FITS_1']
        with open(f'{self._BASENAME}_dual-fringe.log', 'w') as logfile:
            try:
                status = subprocess.check_call(cmd, stdout=logfile, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                status = e.returncode

        if status != 0:
            with open(f'{self._BASENAME}_dual-fringe.log', 'r') as logfile:
                print(logfile.read())

        self.assertEqual(status, 0)



class pulsar_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the pulsar correlator tests."""

    def __init__(self):
        unittest.TestSuite.__init__(self)

        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(pulsar_tests))


if __name__ == '__main__':
    unittest.main()
