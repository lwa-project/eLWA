#!/usr/bin/env python3

"""
Monitor jobs launch on the LWAUCF by launchJobs.py.
"""

import os
import re
import sys
import time
import shlex
import argparse
import subprocess
from datetime import datetime

from lsl.misc import parser as aph


def run_command(cmd, node=None, cwd=None, quiet=False):
    if node is None:
        if type(cmd) is list:
            pcmd = cmd
        else:
            pcmd = shlex.split(cmd)
    elif cwd is None:
        pcmd = ['ssh', '-t', '-t', node, f'bash -c "{cmd}"']
    else:
        pcmd = ['ssh', '-t', '-t', node, f'bash -c "cd {cwd} && {cmd}"']
        
    outdev = subprocess.PIPE
    if quiet:
        outdev = subprocess.DEVNULL
    p = subprocess.Popen(pcmd, stdout=outdev, stderr=outdev)
    stdout, stderr = p.communicate()
    status = p.returncode
    stdout = stdout.decode('ascii', errors='ignore')
    stderr = stderr.decode('ascii', errors='ignore')
    
    return status, stdout, stderr


def get_directories(node):
    status, dirnames, errors = run_command('ls -d -1 /tmp/correlator-*', node=node)
    if status != 0:
        dirnames = []
    else:
        dirnames = dirnames.split('\n')[:-1]
        dirnames = [dirname.strip().rstrip() for dirname in dirnames]
    return dirnames


def get_processes(node):
    status, processes, errors = run_command('ps aux | grep -e superCorrelator -e superPulsarCorrelator | grep bash | grep -v grep | grep -v ssh', node=node)
    if status != 0:
        processes = []
    else:
        processes = processes.split('\n')[:-1]
        processes = [process.strip().rstrip() for process in processes]
    return processes


def get_directory_contents(node, dirname):
    status, filenames, errors = run_command(f"ls -d -1 {dirname}/*", node=node)
    if status != 0:
        filenames = []
    else:
        filenames = filenames.split('\n')[:-1]
        filenames = [filename.strip().rstrip() for filename in filenames]
    return filenames


def get_logfile_speed(node, logname):
    status, speedtime, error = run_command(f"grep -i -e average -e estimated {logname} | tail -n2", node=node)
    if status != 0:
        speed  = '---'
        remain = '---'
        done = False
    else:
        speedtime = speedtime.split('\n')[:-1]
        speedtime = [entry.strip().rstrip() for entry in speedtime]
        if len(speedtime) == 0:
            speed  = '---'
            remain = '---'
            done = False
        else:
            speed  = '---'
            remain = '---'
            done = False
            
            for entry in speedtime:
                if entry.find('verage') != -1:
                    try:
                        _, speed = entry.rsplit('is', 1)
                        speed = speed.strip().rstrip()
                        if entry.find('Average') != -1:
                            done = True
                    except ValueError:
                        speed = '---'
                elif entry.find('estimated') != -1:
                    try:
                        _, remain = entry.rsplit('is', 1)
                        remain = remain.strip().rstrip()
                    except ValueError:
                        remain = '---'
    return speed, remain, done


def main(args):
    while True:
        try:
            t0 = time.time()
            
            status = {}
            for node in args.nodes:
                ## Find out which directories exist
                dirnames = get_directories(node)
                if len(dirnames) == 0:
                    continue
                    
                ## Create an entry for this node since there seems to be
                ## something to report
                status[node] = {'dirnames' :[], 
                                'processes':[],
                                'active'   :{},
                                'progress' :{},
                                'altconfig':{},
                                'speed'    :{},
                                'remaining':{},
                                'complete': {}}
                status[node]['dirnames'] = dirnames
                
                ## Get running superCorrelator.py/superPulsarCorrelator.py processes
                status[node]['processes'] = get_processes(node)
                
                ## For each process, get the configuration file being
                ## processes
                for process in status[node]['processes']:
                    dirname, cmdname = process.rsplit('&&', 1)
                    _, dirname = dirname.split('cd', 1)
                    dirname = dirname.strip().rstrip()
                    cmdname, _ = cmdname.split('>', 1)
                    _, configname = cmdname.rsplit(None, 1)
                    status[node]['active'][dirname] = configname
                    
                ## For each directory, get the progress toward completion
                ## (the number of NPZ files) and the latest values from 
                ## the logfile for the average time per integration and the
                ## estimated time remaining
                for dirname in status[node]['dirnames']:
                    ### Filenames inside the directory
                    filenames = get_directory_contents(node, dirname)
                    
                    ### Count the number of .npz files and find the .log
                    ### file
                    nNPZ = 0
                    logname = None
                    configname = None
                    for filename in filenames:
                        filename = filename.strip().rstrip()
                        _, ext = os.path.splitext(filename)
                        if ext == '.npz':
                            if filename.find('-vis2-bin') == -1:
                                ## Standard files
                                nNPZ += 1
                            elif filename.find('-vis2-bin000') != -1:
                                ## Binning mode files - we only want one bin
                                nNPZ += 1
                        elif ext == '.log':
                            logname = filename
                        elif ext[:7] == '.config':
                            configname = os.path.basename(filename)
                            
                    ### Parse the logfile, if we have one
                    if logname is not None:
                        cspeed, cremain, cdone = get_logfile_speed(node, logname)
                    else:
                        cspeed, cremain, cdone = '---', '---', False
                        
                    ### Save
                    status[node]['progress'][dirname] = nNPZ
                    status[node]['altconfig'][dirname] = configname
                    status[node]['speed'][dirname] = cspeed
                    status[node]['remaining'][dirname] = cremain
                    status[node]['complete'][dirname] = cdone
            t1 = time.time()
            
            # Report
            print("=== %s ===" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            for node in sorted(status.keys()):
                entry = status[node]
                
                print(f"{node}:")
                for dirname in entry['dirnames']:
                    nFiles = entry['progress'][dirname]
                    if dirname in entry['active']:
                        configfile = entry['active'][dirname]
                        speed = entry['speed'][dirname]
                        remaining = entry['remaining'][dirname]
                        done = entry['complete'][dirname]
                        
                        active = 'active'
                        if done:
                            active += ' - complete'
                        pid = 0
                        for process in entry['processes']:
                            if process.split('>', 1)[0].rsplit(None, 1)[1] == configfile:
                                pid = int(process.split(None)[1], 10)
                                if process.find('superPulsarCorrelator.py') != -1:
                                    if active.find('pulsar') == -1:
                                        active += ' - pulsar'
                                        
                        info = f"{configfile} @ {pid}; {speed} per integration, {remaining} remaining"
                        
                    else:
                        try:
                            configfile = entry['altconfig'][dirname]
                        except KeyError:
                            configfile = None
                        done = entry['complete'][dirname]
                        
                        active = 'stale'
                        if done:
                            active += ' - complete'
                            
                        info = '%s (?)' % configfile if configfile is not None else None
                        
                    print(f"  {dirname} ({active})")
                    print(f"    {nFiles} integrations processed")
                    if info is not None:
                        print(f"    {info}")
                        
            t2 = time.time()
            print(f"query {t1-t0:.3f}, report {t2-t1:.3f}")
            
            # Sleep
            while (time.time() - t0) < 60:
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='monitor a collection of nodes for eLWA correlator activity', 
        epilog="NOTE:  The -n/--nodes option also supports numerical node ranges using the '~' character to indicate a decimal range.  For example, 'lwaucf1~2' is expanded to 'lwaucf1' and 'lwaucf2'.  The range exansion can also be combined with other comma separated entries to specify more complex node lists.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('-n', '--nodes', type=aph.csv_hostname_list, default='localhost', 
                        help='comma seperated lists of nodes to examine')
    args = parser.parse_args()
    main(args)
    
