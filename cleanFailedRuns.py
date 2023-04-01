#!/usr/bin/env python3

"""
Script to clean up stale/failed correlator runs created by launchJobs.py.
"""

import os
import re
import sys
import shlex
import argparse
import subprocess

from lsl.misc import parser as aph


def run_command(cmd, node=None, cwd=None, quiet=False):
    if node is None:
        if type(cmd) is list:
            pcmd = cmd
        else:
            pcmd = shlex.split(cmd)
    elif cwd is None:
        pcmd = ['ssh', '-t', '-t', node, 'bash -c "%s"' % cmd]
    else:
        pcmd = ['ssh', '-t', '-t', node, 'bash -c "cd %s && %s"' % (cwd, cmd)]
        
    outdev = subprocess.PIPE
    if quiet:
        outdev = open(os.devnull, 'wb')
    p = subprocess.Popen(pcmd, stdout=outdev, stderr=outdev)
    stdout, stderr = p.communicate()
    status = p.returncode
    if quiet:
        outdev.close()
        
    return status, stdout, stderr


def get_directories(node):
    status, dirnames, errors = run_command('ls -d -1 /tmp/correlator-*', node=node)
    if status != 0:
        dirnames = []
    else:
        dirnames = dirnames.decode(encoding='ascii', errors='ignore')
        dirnames = dirnames.split('\n')[:-1]
        dirnames = [dirname.strip().rstrip() for dirname in dirnames]
    return dirnames


def get_processes(node):
    status, processes, errors = run_command('ps aux | grep -e superCorrelator -e superPulsarCorrelator | grep bash | grep -v grep | grep -v ssh', node=node)
    if status != 0:
        processes = []
    else:
        processes = processes.decode(encoding='ascii', errors='ignore')
        processes = processes.split('\n')[:-1]
        processes = [process.strip().rstrip() for process in processes]
    return processes


def get_directory_contents(node, dirname):
    status, filenames, errors = run_command('ls -d -1 %s/*' % dirname, node=node)
    if status != 0:
        filenames = []
    else:
        filenames = filenames.decode(encoding='ascii', errors='ignore')
        filenames = filenames.split('\n')[:-1]
        filenames = [filename.strip().rstrip() for filename in filenames]
    return filenames


def remove_directory(node, dirname):
    status, _, errors = run_command('rm -rf %s' % dirname, node=node)
    return True if status == 0 else False


def main(args):
    for node in args.nodes:
        ## Find out which directories exist
        dirnames = get_directories(node)
        if len(dirnames) == 0:
            continue
            
        ## Create an entry for this node since there seems to be
        ## something to report
        status = {'dirnames' :[], 
                  'processes':[],
                  'active'   :{},
                  'progress' :{}}
        status['dirnames'] = dirnames
        
        ## Get running superCorrelator.py/superPulsarCorrelator.py processes
        status['processes'] = get_processes(node)
        
        ## For each process, get the configuration file being
        ## processes
        for process in status['processes']:
            dirname, cmdname = process.rsplit('&&', 1)
            _, dirname = dirname.split('cd', 1)
            dirname = dirname.strip().rstrip()
            cmdname, _ = cmdname.split('>', 1)
            _, configname = cmdname.rsplit(None, 1)
            status['active'][dirname] = configname
            
        ## For each directory, get the progress toward completion
        ## (the number of NPZ files) and the latest values from 
        ## the logfile for the average time per integration and the
        ## estimated time remaining
        for dirname in status['dirnames']:
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
                    
            ### Save
            status['progress'][dirname] = nNPZ
            
        ## Clean
        for dirname in status['dirnames']:
            nFiles = status['progress'][dirname]
            if dirname not in status['active']:
                if nFiles == 0:
                    print("%s @ %s -> stale and empty" % (node, dirname))
                    if not args.dry_run:
                        if remove_directory(node, dirname):
                            print("  removed")
                else:
                     print("%s @ %s -> stale and *not* empty" % (node, dirname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='removed stale/failed correlator runs', 
        epilog="NOTE:  The -n/--nodes option also supports numerical node ranges using the '~' character to indicate a decimal range.  For example, 'lwaucf1~2' is expanded to 'lwaucf1' and 'lwaucf2'.  The range exansion can also be combined with other comma separated entries to specify more complex node lists.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('-n', '--nodes', type=aph.csv_hostname_list, default='localhost', 
                        help='comma seperated lists of nodes to examine')
    parser.add_argument('-d', '--dry-run', action='store_true', 
                        help='dry run; report but do not clean')
    args = parser.parse_args()
    main(args)
    
