#!/usr/bin/env python

"""
Run a collection of pulsar binning mode correlation jobs on the LWAUCF.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import os
import re
import sys
import time
import shlex
import argparse
import tempfile
import threading
import subprocess
try:
    from Queue import Queue
except ImportError:
    from queue import Queue
from getpass import getuser

from lsl.misc import parser as aph


FAILED_QUEUE = Queue()


def check_for_other_instances(quiet=True):
    filename = os.path.basename(__file__)
    pcmd = 'ps aux | grep python | grep %s | grep -v %i | grep -v grep' % (filename, os.getpid())
    
    DEVNULL = None
    if quiet:
        DEVNULL = open(os.devnull, 'wb')
    p = subprocess.Popen(pcmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
    status = p.wait()
    if quiet:
        DEVNULL.close()
        
    return True if status == 0 else False


def configfile_is_pulsar(configfile, quiet=True):
    gcmd = ['grep', 'Polyco', configfile]
    
    DEVNULL = None
    if quiet:
        DEVNULL = open(os.devnull, 'wb')
    p = subprocess.Popen(gcmd, stdout=DEVNULL, stderr=DEVNULL)
    status = p.wait()
    if quiet:
        DEVNULL.close()
        
    return True if status == 0 else False


def configfile_is_lwa_only(configfile, quiet=True):
    gcmd = ['grep', '-e VDIF', '-e GUPPI', configfile]
    
    DEVNULL = None
    if quiet:
        DEVNULL = open(os.devnull, 'wb')
    p = subprocess.Popen(gcmd, stdout=DEVNULL, stderr=DEVNULL)
    status = p.wait()
    if quiet:
        DEVNULL.close()
        
    return False if status == 0 else True


def run_command(cmd, node=None, cwd=None, quiet=False):
    if node is None:
        if type(cmd) is list:
            pcmd = cmd
        else:
            pcmd = shlex.split(cmd)
    elif cwd is None:
        pcmd = ['ssh', node, 'shopt -s huponexit && bash -c "%s"' % cmd]
    else:
        pcmd = ['ssh', node, 'shopt -s huponexit && bash -c "cd %s && %s"' % (cwd, cmd)]
        
    DEVNULL = None
    if quiet:
        DEVNULL = open(os.devnull, 'wb')
    p = subprocess.Popen(pcmd, stdout=DEVNULL, stderr=DEVNULL)
    status = p.wait()
    if quiet:
        DEVNULL.close()
        
    return status


def job(node, configfile, options='-l 256 -t 1 -j', softwareDir=None, resultsDir=None, returnQueue=FAILED_QUEUE):
    code = 0
    
    # Create a temporary directory to use
    cwd = tempfile.mkdtemp(prefix='correlator-')
    os.rmdir(cwd)
    code += run_command('mkdir %s' % cwd, node=node, quiet=True)
    if code != 0:
        print("WARNING: failed to create directory on %s - %s" % (node, os.path.basename(configfile)))
        returnQueue.put(False)
        return False
        
    # Find the polyco file to use from the configuration file
    p = subprocess.Popen(['grep', 'Polyco', configfile], stdout=subprocess.PIPE)
    polyfile, err = p.communicate()
    try:
        try:
            polyfile = polyfile.decode(encoding='ascii', errors='ignore')
        except AttributeError:
            pass
        polyfile = polyfile.split(None, 1)[1].strip().rstrip()
        polyfile = os.path.join(os.path.dirname(configfile), polyfile)
    except IndexError:
        print("WARNING: failed to find polyco file on %s - %s" % (node, os.path.basename(configfile)))
        returnQueue.put(False)
        return False
        
    # Copy the software over
    if softwareDir is None:
        softwareDir = os.path.dirname(__file__)
    for filename in ['buffer.py', 'guppi.py', 'jones.py', 'multirate.py', 'superPulsarCorrelator.py', 'utils.py', 'jit']:
        filename = os.path.join(softwareDir, filename)
        code += run_command('rsync -e ssh -avH %s %s:%s/' % (filename, node, cwd), quiet=True)
    if code != 0:
        print("WARNING: failed to sync software on %s - %s" % (node, os.path.basename(configfile)))
        returnQueue.put(False)
        return False
        
    # Copy the configuration over
    for filename in [configfile, polyfile]:
        code += run_command('rsync -e ssh -avH %s %s:%s/' % (filename, node, cwd), quiet=True)
    if code != 0:
        print("WARNING: failed to sync configuration on %s - %s" % (node, os.path.basename(configfile)))
        returnQueue.put(False)
        return False
        
    # Run the correlator
    configfile = os.path.basename(configfile)
    outname, count = os.path.splitext(configfile)
    try:
        count = int(re.sub(r'\D*', '', count), 10)
        outname = "%s%03i" % (outname, count)
    except ValueError:
        pass
    if options.find('-w 1') != -1 or options.find('-w1') != -1:
        outname += 'L'
    elif options.find('-w 2') != -1 or options.find('-w2') != -1:
        outname += 'H'
    logfile = outname+".log"
    ## Sort out the Python path envirnoment variable so that we can find PRESTO
    pythonPath = ''
    pythonPathVariable = os.environ.get('PYTHONPATH')
    if pythonPathVariable is not None:
        pythonPath = 'PYTHONPATH=%s' % pythonPathVariable
    code += run_command('%s %s ./superPulsarCorrelator.py %s -g %s %s > %s 2>&1' % (pythonPath, sys.executable, options, outname, configfile, logfile), node=node, cwd=cwd)
    if code != 0:
        print("WARNING: failed to run pulsar correlator on %s - %s" % (node, os.path.basename(configfile)))
        returnQueue.put(False)
        return False
        
    # Gather the results
    if resultsDir is None:
        resultsDir = os.path.dirname(__file__)
    code += run_command('rsync -e ssh -avH %s:%s/*.npz %s' % (node, cwd, resultsDir), quiet=True)
    code += run_command('rsync -e ssh -avH %s:%s/*.log %s' % (node, cwd, resultsDir), quiet=True)
    if code != 0:
        print("WARNING: failed to sync results on %s - %s" % (node, os.path.basename(configfile)))
        returnQueue.put(False)
        return False
        
    # Cleanup
    code += run_command('rm -rf %s' % cwd, node=node, quiet=True)
    if code != 0:
        print("WARNING: failed to remove directory on %s - %s" % (node, os.path.basename(configfile)))
        returnQueue.put(False)
        return False
        
    returnQueue.put(True)
    return True


def any_active(threads, node=None):
    active = False
    for slot in threads.keys():
        if node is not None:
            cnode, _ = slot.split('-', 1)
            if cnode != node:
                continue
        if threads[slot] is not None:
            active = True
            break
    return active


def get_done_slot(threads):
    done = None
    for slot in threads.keys():
        if threads[slot] is None:
            continue
        if not threads[slot].is_alive():
            done = slot
            break
    return done


def get_idle_slot(threads):
    idle = None
    for slot in threads.keys():
        if threads[slot] is None:
            idle = slot
            break
    return idle


def create_lock_file(node):
    if node[:6] == 'lwaucf':
        fh = open('/home/%s/correlator%s' % (getuser(), node[6:]), 'w')
        fh.close()
    return True


def remove_lock_file(node):
    if node[:6] == 'lwaucf':
        try:
            os.unlink('/home/%s/correlator%s' % (getuser(), node[6:]))
        except OSError:
            pass
    return True


def main(args):
    # Setup
    ## Time mark
    tStart = time.time()
    ## Sort
    configfiles = args.filename
    configfiles.sort(key=lambda x:[int(v) if v.isdigit() else v for v in re.findall(r'[^0-9]|[0-9]+', x)])
    ## Threads - processes by nodes so that small jobs are spread across jobs
    threads = {}
    for p in xrange(args.processes_per_node):
        for node in args.nodes:
            threads['%s-%02i' % (node, p)] = None
    ## Build the configfile/correlation options/results directory sets
    jobs = []
    for configfile in configfiles:
        if not configfile_is_pulsar(configfile):
            continue
            
        if args.both_tunings and configfile_is_lwa_only(configfile):
            coptions = args.options
            coptions = coptions.replace('-w 1', '').replace('-w1', '')
            coptions = coptions.replace('-w 2', '').replace('-w2', '')
            jobs.append( (configfile, coptions+' -w 1', args.results_dir) )
            jobs.append( (configfile, coptions+' -w 2', args.results_dir) )
        else:
            jobs.append( (configfile, args.options, args.results_dir) )
    nJobs = len(jobs)
    
    # Start
    for slot in sorted(threads.keys()):
        node, _ = slot.split('-', 1)
        
        try:
            configfile, coptions, resultsdir = jobs.pop(0)
            threads[slot] = threading.Thread(name=configfile, target=job, args=(node, configfile,), kwargs={'options':coptions, 'resultsDir':resultsdir})
            threads[slot].daemon = True
            threads[slot].start()
            create_lock_file(node)
            print("%s - %s started" % (slot, threads[slot].name))
            time.sleep(2)
        except IndexError:
            pass
            
    # Wait and processes more
    while any_active(threads):
        ## Check for completed jobs
        done = get_done_slot(threads)
        if done is not None:
            print("%s - %s finished" % (done, threads[done].name))
            threads[done] = None
            
        ## Schedule new jobs
        slot = get_idle_slot(threads)
        if slot is not None:
            node, _ = slot.split('-', 1)
            
            try:
                configfile, coptions, resultsdir = jobs.pop(0)
                threads[slot] = threading.Thread(name=configfile, target=job, args=(node, configfile,), kwargs={'options':coptions, 'resultsDir':resultsdir})
                threads[slot].daemon = True
                threads[slot].start()
                print("%s - %s started" % (slot, threads[slot].name))
                time.sleep(2)
            except IndexError:
                pass
                
        ## Lock file maintenance
        if not check_for_other_instances():
            for node in args.nodes:
                if not any_active(threads, node=node):
                    remove_lock_file(node)
                    
        ## Rest
        time.sleep(10)
        
    # Teardown
    ## Lock files
    if not check_for_other_instances():
        for node in args.nodes:
            ## Lock file cleanup
            remove_lock_file(node)
    ## Stop time
    tFinish = time.time()
    tElapsed = tFinish - tStart
    
    h = int(tElapsed / 60.0) / 60
    m = int(tElapsed / 60.0) % 60
    s = tElapsed % 60.0
    
    # Final exit code evaluation
    failed = 0
    while not FAILED_QUEUE.empty():
        try:
            status = FAILED_QUEUE.get_nowait()
            FAILED_QUEUE.task_done()
            if not status:
                failed += 1
        except:
            pass
            
    print("Completed %i jobs (with %i failues) in %i hr, %i min, %.0f sec" % (nJobs, failed, h, m, s))
    sys.exit(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="given a collection of superPulsarCorrelator.py configuration files, process the runs and aggregate the results",
        epilog="NOTE:  The -n/--nodes option also supports numerical node ranges using the '~' character to indicate a decimal range.  For example, 'lwaucf1~2' is expanded to 'lwaucf1' and 'lwaucf2'.  The range exansion can also be combined with other comma separated entries to specify more complex node lists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, nargs='+',
                        help='correlator configuration file')
    parser.add_argument('-p', '--processes-per-node', type=aph.positive_int, default=1,
                        help='number of processes to run per node')
    parser.add_argument('-n', '--nodes', type=aph.csv_hostname_list, default='localhost',
                        help='comma seperated lists of nodes to use')
    parser.add_argument('-o', '--options', type=str, default="-l 256 -t 1 -j",
                        help='correlator options to use')
    parser.add_argument('-b', '--both-tunings', action='store_true',
                        help='for LWA-only configuration files, process both tunings')
    parser.add_argument('-r', '--results-dir', type=str, default="./results",
                        help='directory to put the results in')
    args = parser.parse_args()
    if not os.path.exists(args.results_dir):
        print("Warning: %s does not exist, creating" % args.results_dir)
        os.mkdir(args.results_dir)
    elif not os.path.isdir(args.results_dir):
        raise RuntimeError('%s is not a directory' % args.results_dir)
    main(args)
    
