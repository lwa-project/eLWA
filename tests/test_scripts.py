"""
Unit tests for the various eLWA scripts.
"""

import unittest
import json
import glob
import sys
import os

currentDir = os.path.abspath(os.getcwd())
if os.path.exists(os.path.join(currentDir, 'test_scripts.py')):
    MODULE_BUILD = currentDir
else:
    MODULE_BUILD = None
    
run_scripts_tests = False
try:
    from io import StringIO
    from pylint.lint import Run
    from pylint.reporters.json_reporter import JSONReporter
    if MODULE_BUILD is not None:
        run_scripts_tests = True
        
        # Pre-seed data.py
        os.system("%s ../data.py" % sys.executable)
        
except ImportError:
    pass


__version__  = "0.2"
__author__   = "Jayce Dowell"


_PYLINT_IGNORES = [('no-member',              "Module 'ephem' has no"),
                   ('no-member',              "Instance of 'HDUList'"),
                   ('no-member',              "Instance of 'Exception' has no 'GitError' member"),
                   ('no-member',              "Instance of 'GitError' has no 'GitError' member"),
                   ('no-name-in-module',      "No name 'c' in module 'astropy.constants'"),
                   ('bad-string-format-type', "Argument '.ndarray' does not match format"),
                   ('bad-option-value',       "Bad option value 'possibly-")]


@unittest.skipUnless(run_scripts_tests, "requires the 'pylint' module")
class scripts_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the commissioning scripts."""
    
    def setUp(self):
        """Make sure we have get_vla_ant_pos.py in place."""
        
        # get_vla_ant_pos.py
        if not os.path.exists('../get_vla_ant_pos.py'):
            with open('../get_vla_ant_pos.py', 'w') as fh:
                fh.write("""import numpy as np
class database(object):
    def __init__(self, *args, **kwds):
        self._ready = True
    def get_pad(self,ant,date):
        return 'W40', None
    def get_xyz(self,ant,date):
        return np.array((-6777.0613, -360.7018, -3550.9465), dtype=np.float64)
    def close(self):
        return True""")
                
    def test_scripts(self):
        """Static analysis of the eLWA scripts."""
        
        _SCRIPTS = glob.glob(os.path.join(MODULE_BUILD, '..', '*.py'))
        for depth in range(1, 3):
            path = [MODULE_BUILD, '..']
            path.extend(['*',]*depth)
            path.append('*.py')
            _SCRIPTS.extend(glob.glob(os.path.join(*path)))
        _SCRIPTS = list(filter(lambda x: x.find('2018Feb') == -1, _SCRIPTS))
        _SCRIPTS = list(filter(lambda x: x.find('2018Mar') == -1, _SCRIPTS))
        _SCRIPTS = list(filter(lambda x: x.find('2018Apr') == -1, _SCRIPTS))
        _SCRIPTS = list(filter(lambda x: x.find('mini_presto') == -1, _SCRIPTS))
        _SCRIPTS = list(filter(lambda x: x.find('test_scripts.py') == -1, _SCRIPTS))
        _SCRIPTS.sort()
        try:
            import cupy
        except ImportError:
            while True:
                idx = None
                for i,script in enumerate(_SCRIPTS):
                    if script.find('cupy') != -1:
                        idx = i
                        break
                if idx is None:
                    break
                else:
                    del _SCRIPTS[idx]
                    
        for script in _SCRIPTS:
            name = _name_to_name(script)
            with self.subTest(script=name):
                pylint_output = StringIO()
                reporter = JSONReporter(pylint_output)
                Run([script, '-E', '--extension-pkg-whitelist=numpy,ephem,scipy.special,_utils', "--init-hook='import sys; sys.path=[%s]; sys.path.insert(0, \"%s\")'" % (",".join(['"%s"' % p for p in sys.path]), os.path.dirname(MODULE_BUILD))], reporter=reporter, exit=False)
                results = json.loads(pylint_output.getvalue())
                
                for i,entry in enumerate(results):
                    with self.subTest(error_number=i+1):
                        false_positive = False
                        for isym,imesg in _PYLINT_IGNORES:
                            if entry['symbol'] == isym and entry['message'].startswith(imesg):
                                false_positive = True
                                break
                        if false_positive:
                            continue
                            
                        self.assertTrue(False, f"{entry['path']}:{entry['line']} - {entry['symbol']} - {entry['message']}")


def _name_to_name(filename):
    filename = os.path.splitext(filename)[0]
    parts = filename.split(os.path.sep)
    start = parts.index('..')
    parts = parts[start+1:]
    return '_'.join(parts)


class scripts_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the commissioning script
    tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(scripts_tests))


if __name__ == '__main__':
    unittest.main()
    
