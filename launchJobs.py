#!/usr/bin/env python3

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
from collections import OrderedDict

from lsl.misc import parser as aph


FAILED_QUEUE = Queue()


def check_for_other_instances(quiet=True):
    filename = os.path.basename(__file__)
    pcmd = f"ps aux | grep python | grep {filename} | grep -v {os.getpid()} | grep -v grep"
    
    DEVNULL = None
    if quiet:
        DEVNULL = subprocess.DEVNULL
    p = subprocess.Popen(pcmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
    status = p.wait()
    
    return True if status == 0 else False


def configfile_is_pulsar(configfile, quiet=True):
    gcmd = ['grep', 'Polyco', configfile]
    
    DEVNULL = None
    if quiet:
        DEVNULL = subprocess.DEVNULL
    p = subprocess.Popen(gcmd, stdout=DEVNULL, stderr=DEVNULL)
    status = p.wait()
    
    return True if status == 0 else False


def configfile_is_lwa_only(configfile, quiet=True):
    gcmd = ['grep', '-e VDIF', configfile]
    
    DEVNULL = None
    if quiet:
        DEVNULL = subprocess.DEVNULL
    p = subprocess.Popen(gcmd, stdout=DEVNULL, stderr=DEVNULL)
    status = p.wait()
    
    return False if status == 0 else True


def run_command(cmd, node=None, socket=None, cwd=None, return_output=False, quiet=False):
    if return_output and quiet:
        raise RuntimeError("Cannot be both quiet and return output")
        
    if node is None:
        if type(cmd) is list:
            pcmd = cmd
        else:
            pcmd = shlex.split(cmd)
    elif cwd is None:
        pcmd = ['ssh', node, 'shopt -s huponexit && bash -c "']
        if socket is not None:
            pcmd[-1] += f"numactl --cpunodebind={socket} --membind={socket} -- "
        pcmd[-1] += f'{cmd}"'
    else:
        pcmd = ['ssh', node, f'shopt -s huponexit && bash -c "cd {cwd} && ']
        if socket is not None:
            pcmd[-1] += f"numactl --cpunodebind={socket} --membind={socket} -- "
        pcmd[-1] += f'{cmd}"'
        
    OUT, ERR = None, None
    if quiet:
        OUT = subprocess.DEVNULL
        ERR = subprocess.DEVNULL
    if return_output:
        OUT = subprocess.PIPE
        ERR = subprocess.PIPE
    p = subprocess.Popen(pcmd, stdout=OUT, stderr=ERR)
    output, err = p.communicate()
    status = p.returncode
    
    if return_output:
        output = output.decode('ascii', errors='ignore')
        err = err.decode('ascii', errors='ignore')
        status = (status, output)
        
    return status


def job(node, socket, configfile, options='-l 256 -t 1 -j', softwareDir=None, resultsDir=None, returnQueue=FAILED_QUEUE, isPulsar=False):
    code = 0
    
    # Create a temporary directory to use
    cwd = tempfile.mkdtemp(prefix='correlator-')
    os.rmdir(cwd)
    code += run_command(f"mkdir {cwd}", node=node, quiet=True)
    if code != 0:
        print(f"WARNING: failed to create directory on {node} - {os.path.basename(configfile)}")
        returnQueue.put(False)
        return False
        
    # Set the correlator mode and check for a polyco if this is a binning job
    corr_mode = 'superCorrelator.py'
    polyfile = None
    if isPulsar:
        corr_mode = 'superPulsarCorrelator.py'
        # Find the polyco file to use from the configuration file
        p = subprocess.Popen(['grep', 'Polyco', configfile], stdout=subprocess.PIPE)
        polyfile, err = p.communicate()
        try:
            polyfile = polyfile.decode(encoding='ascii', errors='ignore')
            polyfile = polyfile.split(None, 1)[1].strip().rstrip()
            polyfile = os.path.join(os.path.dirname(configfile), polyfile)
        except IndexError:
            print(f"WARNING: failed to find polyco file on {node} - {os.path.basename(configfile)}")
            returnQueue.put(False)
            return False
            
    # Copy the software over
    if softwareDir is None:
        softwareDir = os.path.dirname(__file__)
    for filename in ['jones.py', 'multirate.py', 'superCorrelator.py', 'superPulsarCorrelator.py', 'utils.py', 'mini_presto']:
        filename = os.path.join(softwareDir, filename)
        code += run_command(f"rsync -e ssh -avH {filename} {node}:{cwd}/", quiet=True)
    if code != 0:
        print(f"WARNING: failed to sync software on {node} - {os.path.basename(configfile)}")
        returnQueue.put(False)
        return False
        
    # Copy the configuration over
    for filename in [configfile, polyfile]:
        if filename is None:
            continue
        code += run_command(f"rsync -e ssh -avH {filename} {node}:{cwd}/", quiet=True)
    if code != 0:
        print(f"WARNING: failed to sync configuration on {node} - {os.path.basename(configfile)}")
        returnQueue.put(False)
        return False
        
    # Query the NUMA status
    scode, numa_status = run_command(f"{sys.executable} -c 'import utils; print(utils.get_numa_support(), utils.get_numa_node_count())'", node=node, cwd=cwd, return_output=True)
    code += scode
    if code != 0:
        print(f"WARNING: failed to determine NUMA status on {node} - {os.path.basename(configfile)}")
        returnQueue.put(False)
        return False
    numa_support, numa_node_count = numa_status.split(None, 1)
    if numa_support == 'False':
        ## Nope, drop the socket number
        socket = None
    else:
        ## Yep, make sure the socket number is in range
        socket = socket % int(numa_node_count, 10)
        
    if options.find('--gpu') != -1 and options.find('--gpu=') == -1:
        # Query the GPU status
        scode, gpu_status = run_command("%s -c 'import utils; print(utils.get_gpu_support(), utils.get_gpu_count())'" % (sys.executable,), node=node, cwd=cwd, return_output=True)
        if scode != 0:
            print(f"WARNING: failed to determine GPU status on {node} - {os.path.basename(configfile)}")
            ## Unknown, drop the GPU option
            options = options.replace('--gpu', '')
        else:
            gpu_support, gpu_count = gpu_status.split(None, 1)
            if gpu_support == 'False':
                ## Nope, drop the GPU option
                options = options.replace('--gpu', '')
            else:
                ## Yep, now figure out if we want to set the GPU number
                gpu = 0
                if socket is not None:
                    gpu = socket % int(gpu_count, 10)
                options = options.replace('--gpu', f"--gpu={gpu}")
                
    # Run the correlator
    configfile = os.path.basename(configfile)
    outname, count = os.path.splitext(configfile)
    try:
        count = int(re.sub(r'\D*', '', count), 10)
        outname = f"{outname}{count:03d}"
    except ValueError:
        pass
    if options.find('-w 1') != -1 or options.find('-w1') != -1:
        outname += 'L'
    elif options.find('-w 2') != -1 or options.find('-w2') != -1:
        outname += 'H'
    logfile = outname+".log"
    code += run_command(f"{sys.executable} ./{corr_mode} {options} -g {outname} {configfile} > {logfile} 2>&1", node=node, socket=socket, cwd=cwd)
    if code != 0:
        print(f"WARNING: failed to run correlator on {node} - {os.path.basename(configfile)}")
        returnQueue.put(False)
        return False
        
    # Gather the results
    if resultsDir is None:
        resultsDir = os.path.dirname(__file__)
    code += run_command(f"rsync -e ssh -avH {node}:{cwd}/*.npz {resultsDir}", quiet=True)
    code += run_command(f"rsync -e ssh -avH {node}:{cwd}/*.log {resultsDir}", quiet=True)
    if code != 0:
        print(f"WARNING: failed to sync results on {node} - {os.path.basename(configfile)}")
        returnQueue.put(False)
        return False
        
    # Cleanup
    code += run_command(f"rm -rf {cwd}", node=node, quiet=True)
    if code != 0:
        print(f"WARNING: failed to remove directory on {node} - {os.path.basename(configfile)}")
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
    node = node.replace('10.1.1.10', 'lwaucf')
    if node[:6] == 'lwaucf':
        fh = open(f"/home/{getuser()}/correlator{node[6:]}", 'w')
        fh.close()
    return True


def remove_lock_file(node):
    node = node.replace('10.1.1.10', 'lwaucf')
    if node[:6] == 'lwaucf':
        try:
            os.unlink(f"/home/{getuser()}/correlator{node[6:]}")
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
    threads = OrderedDict()
    for p in range(args.processes_per_node):
        for node in args.nodes:
            n = p + 0
            while f"{node}-{n:02d}" in threads:
                n += 1
            threads[f"{node}-{n:02d}"] = None
    ## Build the configfile/correlation options/results directory sets
    jobs = []
    for configfile in configfiles:
        is_pulsar = configfile_is_pulsar(configfile)
        
        if args.both_tunings and configfile_is_lwa_only(configfile):
            coptions = args.options
            coptions = coptions.replace('-w 1', '').replace('-w1', '')
            coptions = coptions.replace('-w 2', '').replace('-w2', '')
            jobs.append( (configfile, coptions+' -w 1', args.results_dir, is_pulsar) )
            jobs.append( (configfile, coptions+' -w 2', args.results_dir, is_pulsar) )
        else:
            jobs.append( (configfile, args.options, args.results_dir, is_pulsar) )
    nJobs = len(jobs)
    
    # Start
    for slot in sorted(threads.keys()):
        node, socket = slot.split('-', 1)
        socket = int(socket, 10)
        
        try:
            configfile, coptions, resultsdir, is_pulsar = jobs.pop(0)
            threads[slot] = threading.Thread(name=configfile, target=job, args=(node, socket, configfile,), kwargs={'options':coptions, 'resultsDir':resultsdir, 'isPulsar':is_pulsar})
            threads[slot].daemon = True
            threads[slot].start()
            create_lock_file(node)
            print(f"{slot} - {threads[slot].name} started")
            time.sleep(2)
        except IndexError:
            pass
            
    # Wait and processes more
    while any_active(threads):
        ## Check for completed jobs
        done = get_done_slot(threads)
        if done is not None:
            print(f"{done} - {threads[done].name} finished")
            threads[done] = None
            
        ## Schedule new jobs
        slot = get_idle_slot(threads)
        if slot is not None:
            node, socket = slot.split('-', 1)
            socket = int(socket, 10)
            
            try:
                configfile, coptions, resultsdir, is_pulsar = jobs.pop(0)
                threads[slot] = threading.Thread(name=configfile, target=job, args=(node, socket, configfile,), kwargs={'options':coptions, 'resultsDir':resultsdir, 'isPulsar':is_pulsar})
                threads[slot].daemon = True
                threads[slot].start()
                print(f"{slot} - {threads[slot].name} started")
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
            
    print(f"Completed {nJobs} jobs (with {failed} failues) in {h} hr, {m} min, {s:.0f} sec")
    sys.exit(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="given a collection of superCorrelator.py/superPulsarCorrelator.py configuration files, process the runs and aggregate the results",
        epilog="NOTE:  The -n/--nodes option also supports numerical node ranges using the '~' character to indicate a decimal range.  For example, 'lwaucf1~2' is expanded to 'lwaucf1' and 'lwaucf2'.  The range exansion can also be combined with other comma separated entries to specify more complex node lists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, nargs='+',
                        help='correlator configuration file')
    parser.add_argument('-p', '--processes-per-node', type=aph.positive_int, default=1,
                        help='number of processes to run per node')
    parser.add_argument('-n', '--nodes', type=aph.csv_hostname_list, default='localhost',
                        help='comma seperated lists of nodes to use')
    parser.add_argument('--gpu', action='store_true',
                        help='enable the experimental GPU X-engine')
    parser.add_argument('-o', '--options', type=str, default="-l 256 -t 1",
                        help='correlator options to use')
    parser.add_argument('-b', '--both-tunings', action='store_true',
                        help='for LWA-only configuration files, process both tunings')
    parser.add_argument('-r', '--results-dir', type=str, default="./results",
                        help='directory to put the results in')
    args = parser.parse_args()
    if args.gpu:
        args.options += " --gpu"
    if not os.path.exists(args.results_dir):
        print(f"Warning: {args.results_dir} does not exist, creating")
        os.mkdir(args.results_dir)
    elif not os.path.isdir(args.results_dir):
        raise RuntimeError(f"{args.results_dir} is not a directory")
    main(args)
