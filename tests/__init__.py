"""
Modules defining package tests.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
__version__   = "0.3"
__author__    = "Jayce Dowell"

from . import test_elwa
from . import test_jit
from . import test_gpu
from . import test_lwaonly
from . import test_scripts
