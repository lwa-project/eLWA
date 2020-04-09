# -*- coding: utf-8 -*-

"""
Unit tests for the various eLWA scripts.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import unittest
import glob
import sys
import imp
import re
import os

currentDir = os.path.abspath(os.getcwd())
if os.path.exists(os.path.join(currentDir, 'test_scripts.py')):
    MODULE_BUILD = currentDir
else:
    MODULE_BUILD = None
    
run_scripts_tests = False
try:
    from pylint import epylint as lint
    if MODULE_BUILD is not None:
        run_scripts_tests = True
except ImportError:
    pass


__version__  = "0.1"
__author__   = "Jayce Dowell"


_LINT_RE = re.compile('(?P<module>.*?)\:(?P<line>\d+)\: (error )?[\[\(](?P<type>.*?)[\]\)] (?P<info>.*)')


_SAFE_TO_IGNORE = ["Possible",
                   "Module 'numpy",
                   "Module 'ephem",
                   "Instance of 'HDUList'",
                   "Unable to import 'polycos"]


def _get_context(filename, line, before=0, after=0):
    to_save = range(line-before, line+after+1)
    context = []
    with open(filename, 'r') as fh:
        i = 0
        for line in fh:
            if i in to_save:
                context.append(line)
            i += 1
    return context


@unittest.skipUnless(run_scripts_tests, "requires the 'pylint' module")
class scripts_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the commissioning scripts."""
    
    def setUp(self):
        """Make sure we have get_vla_ant_pos.py in place."""
        
        # get_vla_ant_pos.py
        if not os.path.exists('../get_vla_ant_pos.py'):
            with open('../get_vla_ant_pos.py', 'w') as fh:
                fh.write("""class database(object):
    def __init__(self, *args, **kwds):\
        self._ready = True
    def get_pad(self,ant,date):
        return 'W40', None
    def get_xyz(self,ant,date):
        return (-6777.0613, -360.7018, -3550.9465)
    def close(self):
        return True""")


def _test_generator(script):
    """
    Function to build a test method for each script that is provided.  
    Returns a function that is suitable as a method inside a unittest.TestCase
    class
    """
    
    def test(self):
        out, err = lint.py_run("%s -E" % script, return_std=True)
        out_lines = out.read().split('\n')
        err_lines = err.read().split('\n')
        out.close()
        err.close()
        
        for line in out_lines:
            ignore = False
            for phrase in _SAFE_TO_IGNORE:
                if line.find(phrase) != -1:
                    ignore = True
                    break
            if ignore:
                continue
                
            mtch = _LINT_RE.match(line)
            if mtch is not None:
                line_no, type, info = mtch.group('line'), mtch.group('type'), mtch.group('info')
                ignore = False
                if line.find('before assignment') != -1:
                    context = _get_context(script, int(line_no), before=20, after=20)
                    loc = context[20-1]
                    level = len(loc) - len(loc.lstrip()) - 4
                    found_try = None
                    for i in range(2, 20):
                        if context[20-i][level:level+3] == 'try':
                            found_try = i
                            break
                    found_except = None
                    for i in range(0, 20):
                        if context[20+i][level:level+6] == 'except':
                            found_except = i
                            break
                    if found_try is not None and found_except is not None:
                        if context[20+found_except].find('NameError') != -1:
                            ignore = True
                if ignore:
                    continue
                    
                self.assertEqual(type, None, "%s:%s - %s" % (os.path.basename(script), line_no, info))
    return test


if run_scripts_tests:
    _SCRIPTS = glob.glob(os.path.join(MODULE_BUILD, '..', '*.py'))
    for depth in range(1, 3):
        path = [MODULE_BUILD, '..']
        path.extend(['*',]*depth)
        path.append('*.py')
        _SCRIPTS.extend(glob.glob(os.path.join(*path)))
    _SCRIPTS = list(filter(lambda x: x.find('test_scripts.py') == -1, _SCRIPTS))
    _SCRIPTS.sort()
    
    _NAMES = {}
    for script in _SCRIPTS:
        test = _test_generator(script)
        name = 'test_%s' % os.path.splitext(os.path.basename(script))[0]
        try:
            testname = name+('_%s' % _NAMES[name])
        except KeyError:
            testname = name
            _NAMES[name] = 0
        _NAMES[name] += 1
        doc = """Static analysis of the '%s' script.""" % os.path.basename(script)
        setattr(test, '__doc__', doc)
        setattr(scripts_tests, testname, test)


class scripts_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the commissioning script
    tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(scripts_tests))


if __name__ == '__main__':
    unittest.main()
    
