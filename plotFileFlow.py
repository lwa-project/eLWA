#!/usr/bin/env python3

"""
Show which files are used when for a set of superCorrelator.py configuration
files.
"""

import os
import argparse
from datetime import datetime

from matplotlib import pyplot as plt


def main(args):
    # Parse the config files to pull out filenames and associated start/stop times
    lwa1 = {}
    lwasv = {}
    vla = {}
    for filename in args.filename:
        with open(filename, 'r') as fh:
            lines = fh.read()
            lines = lines.split('\n')
            
            for i,line in enumerate(lines):
                ## A LWA1 or LWA-SV Type TabNine::no_sem to suppress this message.?
                if line.find('LWA1') != -1 or line.find('LWA-SV') != -1:
                    start = lines[i-8].rsplit('is ', 1)[1]
                    stop = lines[i-7].rsplit('is ', 1)[1]
                    filename = lines[i-3].rsplit(None, 1)[1]
                    
                    start = datetime.strptime(start, "%Y-%m-%d %H:%M:%S.%f")
                    stop = datetime.strptime(stop, "%Y-%m-%d %H:%M:%S.%f")
                    filename = os.path.basename(filename)
                    
                    ### Sort it out
                    if line.find('LWA1') != -1:
                        lwa1[filename] = (start,stop)
                    elif line.find('LWA-SV') != -1:
                        lwasv[filename] = (start,stop)
                        
                ## A VLA entry?
                if line.find('AC-0') != -1 or line.find('BD-0') != -1:
                    start = lines[i-4].rsplit('is ', 1)[1]
                    stop = lines[i-3].rsplit('is ', 1)[1]
                    filename = lines[i-0].rsplit(None, 1)[1]
                    
                    start = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
                    try:
                        stop = datetime.strptime(stop, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        stop = datetime.strptime(stop, "%Y-%m-%d %H:%M:%S.%f")
                    filename = os.path.basename(filename)
                    filename = filename.split('.')[-5]  ### We only want the scan number
                    
                    vla[filename] = (start,stop)
                    
    # Find the first scan in the set
    ref = None
    for site in (lwa1, lwasv, vla):
        for filename in site:
            if ref is None or site[filename][0] < ref:
                ref = site[filename][0]
    ref = ref.replace(microsecond=0)
    
    # Make the plot
    fig = plt.figure()
    ax = fig.gca()
    #ax2 = ax.twiny()
    for offset,site in enumerate((vla,lwa1,lwasv)):
        for filename in site:
            start, stop = site[filename]
            
            dur = stop - start
            dur = dur.total_seconds() / 60.0    # sec -> min
            start = start - ref
            start = start.total_seconds() / 60.0 # sec -> min
            
            try:
                c, = ax.barh(offset, dur, left=start, label=filename, color=c.get_facecolor())
            except NameError:
                c, = ax.barh(offset, dur, left=start, label=filename)
            ax.text(start+dur/2, offset, filename, fontsize=9,
                                                   verticalalignment='center',
                                                   horizontalalignment='center',
                                                   rotation=90)
            
        ## Cleanup to move on to the next color
        try:
            del c
        except NameError:
            pass
            
    ## Plot ranges and labels
    ax.set_title('eLWA Run: %s UTC' % (ref.strftime("%Y/%m/%d %H:%M:%S"),))
    ax.set_ylim((-0.5, 2.5))
    ax.set_xlabel('Elapsed Time [min]')
    ax.set_yticks((0,1,2))
    ax.set_yticklabels(('VLA', 'LWA1', 'LWA-SV'))
    #ax2.set_xlim((ax.get_xlim()[0]/60.0, ax.get_xlim()[1]/60.0))
    #ax2.set_xlabel('Elapsed Time [hr]')
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='show which files are used when for a set of superCorrelator.py configuration files', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', type=str, nargs='+',
                        help='configuration file to parse')
    args = parser.parse_args()
    main(args)
    
