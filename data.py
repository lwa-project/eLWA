"""
Stub module for downloading the latest data.py from https://github.com/lwa-project/commissioning
"""

from __future__ import print_function

import os
import time
try:
    from urllib import request as urlrequest
except ImportError:
    import urllib2 as urlrequest


# URL to download
MODULE_URL = 'https://raw.githubusercontent.com/lwa-project/commissioning/master/DRX/HDF5/data.py'


# Maximum file age to accept without checking for a newer version
MAX_AGE_SEC = 86400


# Get the current file age and entity tag (if it exists)
age = 1e6
etag = ''
if os.path.exists('_data.py'):
    age = time.time() - os.path.getmtime('_data.py')
    if os.path.exists('_data.etag'):
        with open('_data.etag', 'r') as fh:
            etag = fh.read()
            
# If the file is more than MAX_AGE_SEC old, check for an update
if age > MAX_AGE_SEC:
    request = urlrequest.Request(MODULE_URL)
    opener = urlrequest.build_opener()
    data = opener.open(request)
    if data.headers['etag'] != etag:
        with open('_data.py', 'wb') as fh:
            fh.write(data.read())
        with open('_data.etag', 'w') as fh:
            fh.write(data.headers['etag'])
            
# Load in everything from the module
from _data import *

