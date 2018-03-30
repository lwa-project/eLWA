#!/usr/bin/env python

"""
Run a collection of correlation jobs on the LWAUCF

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import re
import sys
import time
import shlex
import getopt
import tempfile
import threading
import subprocess


def usage(exitCode=None):
	print """launchJobs.py - Given a collection of superCorrelator.py configuration files, 
process the runs and aggregate the results.

Usage:
launchJobs.py [OPTIONS] config [config [...]]

Options:
-h, --help                Display this help information
-p, --processes-per-node  Number of processes to run per node (Default = 1)
-n, --nodes               Comma seperated lists of nodes to use 
                          (Default = current)
-o, --options             Correlator options to use
                          (Default = -l 128 -t 1 -j)
-r, --results-dir         Directory to put the results in 
                          (Default = ./results')
"""
	
	if exitCode is not None:
		sys.exit(exitCode)
	else:
		return True


def parseConfig(args):
	config = {}
	# Command line flags - default values
	config['processes'] = 1
	config['nodes'] = ['localhost',]
	config['options'] = '-l 128 -t 1 -j'
	config['results'] = './results'
	config['args'] = []
	
	# Read in and process the command line flags
	try:
		opts, args = getopt.getopt(args, "hp:n:o:r:", ["help", "processes-per-node=", "nodes=", "options=", "results-dir="])
	except getopt.GetoptError, err:
		# Print help information and exit:
		print str(err) # will print something like "option -a not recognized"
		usage(exitCode=2)
		
	# Work through opts
	for opt, value in opts:
		if opt in ('-h', '--help'):
			usage(exitCode=0)
		elif opt in ('-p', '--processes-per-node'):
			config['processes'] = int(value, 10)
		elif opt in ('-n', '--nodes'):
			config['nodes'] = [v.strip().rstrip() for v in value.split(',')]
		elif opt in ('-o', '--options'):
			config['options'] = value
		elif opt in ('-r', '--results-dir'):
			config['results'] = value
		else:
			assert False
			
	# Add in arguments
	config['args'] = args
	
	# Validate
	if config['processes'] <= 0:
		raise RuntimeError('Invalid number of processes per node')
	if os.path.exists(config['results']):
		if not os.path.isdir(config['results']):
			raise RuntimeError('%s is not a directory' % config['results'])
	else:
		print "Warning: %s does not exist, creating" % config['results']
		os.mkdir(config['results'])
		
	# Return configuration
	return config


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


def run_command(cmd, node=None, cwd=None, quiet=False):
	if node is None:
		if type(cmd) is list:
			pcmd = cmd
		else:
			pcmd = shlex.split(cmd)
	elif cwd is None:
		pcmd = ['ssh', '-t', '-t', node, 'bash -c "%s" | cat' % cmd]
	else:
		pcmd = ['ssh', '-t', '-t', node, 'bash -c "cd %s && %s" | cat' % (cwd, cmd)]
		
	DEVNULL = None
	if quiet:
		DEVNULL = open(os.devnull, 'wb')
	p = subprocess.Popen(pcmd, stdout=DEVNULL, stderr=DEVNULL)
	status = p.wait()
	if quiet:
		DEVNULL.close()
		
	return status


def job(node, configfile, options='-l 128 -t 1 -j', softwareDir=None, resultsDir=None):
	code = 0
	
	# Create a temporary directory to use
	cwd = tempfile.mkdtemp(prefix='correlator-')
	os.rmdir(cwd)
	code += run_command('mkdir %s' % cwd, node=node, quiet=True)
	if code != 0:
		print "WARNING: failed to create directory on %s - %s" % (node, os.path.basename(configfile))
		return False
		
	# Copy the software over
	if softwareDir is None:
		softwareDir = os.path.dirname(__file__)
	for filename in ['buffer.py', 'guppi.py', 'jones.py', 'superCorrelator.py', 'utils.py', 'jit', configfile]:
		filename = os.path.join(softwareDir, filename)
		code += run_command('rsync -e ssh -avH %s %s:%s/' % (filename, node, cwd), quiet=True)
	if code != 0:
		print "WARNING: failed to sync software on %s - %s" % (node, os.path.basename(configfile))
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
	#code += run_command('touch %s && touch %s.npz' % (logfile, outname), node=node, cwd=cwd)
	code += run_command('./superCorrelator.py %s -g %s %s > %s 2>&1' % (options, outname, configfile, logfile), node=node, cwd=cwd)
	if code != 0:
		print "WARNING: failed to run correlator on %s - %s" % (node, os.path.basename(configfile))
		return False
		
	# Gather the results
	if resultsDir is None:
		resultsDir = os.path.dirname(__file__)
	code += run_command('rsync -e ssh -avH %s:%s/*.npz %s' % (node, cwd, resultsDir), quiet=True)
	code += run_command('rsync -e ssh -avH %s:%s/*.log %s' % (node, cwd, resultsDir), quiet=True)
	if code != 0:
		print "WARNING: failed to sync results on %s - %s" % (node, os.path.basename(configfile))
		return False
		
	# Cleanup
	code += run_command('rm -rf %s' % cwd, node=node, quiet=True)
	if code != 0:
		print "WARNING: failed to remove directory on %s - %s" % (node, os.path.basename(configfile))
		return False
		
	return True


def any_active(threads, node=None):
	active = False
	for slot in threads.keys():
		if node is not None and slot.split('-')[0] != node:
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
		fh = open('/home/jdowell/correlator%s' % node[6:], 'w')
		fh.close()
	return True


def remove_lock_file(node):
	if node[:6] == 'lwaucf':
		try:
			os.unlink('/home/jdowell/correlator%s' % node[6:])
		except OSError:
			pass
	return True


def main(args):
	# Parse the command line
	config = parseConfig(args)
	
	# Setup
	## Sort
	configfiles = config['args']
	configfiles.sort(key=lambda x:[int(v) if v.isdigit() else v for v in re.findall(r'[^0-9]|[0-9]+', x)])
	## Threads - processes by nodes so that small jobs are spread across jobs
	threads = {}
	for p in xrange(config['processes']):
		for node in config['nodes']:
			threads['%s-%02i' % (node, p)] = None
			
	# Start
	for slot in sorted(threads.keys()):
		node, _ = slot.split('-', 1)
		
		try:
			configfile = configfiles.pop(0)
			threads[slot] = threading.Thread(name=configfile, target=job, args=(node, configfile,), kwargs={'options':config['options'], 'resultsDir':config['results']})
			threads[slot].daemon = True
			threads[slot].start()
			create_lock_file(node)
			print "%s - %s started" % (slot, threads[slot].name)
			time.sleep(2)
		except IndexError:
			continue
			
	# Wait and processes more
	while any_active(threads):
		## Check for completed jobs
		done = get_done_slot(threads)
		if done is not None:
			print "%s - %s finished" % (done, threads[done].name)
			threads[done] = None
			
		## Schedule new jobs
		slot = get_idle_slot(threads)
		if slot is not None:
			node, _ = slot.split('-', 1)
			
			try:
				configfile = configfiles.pop(0)
				threads[slot] = threading.Thread(name=configfile, target=job, args=(node, configfile,), kwargs={'options':config['options'], 'resultsDir':config['results']})
				threads[slot].daemon = True
				threads[slot].start()
				print "%s - %s started" % (slot, threads[slot].name)
				time.sleep(2)
			except IndexError:
				continue
				
		## Lock file maintenance
		if not check_for_other_instances():
			for node in config['nodes']:
				exited = any_active(threads, node=node)
				if exited is None:
					remove_lock_file(node)
					
		## Rest
		time.sleep(10)
		
	# Teardown
	if not check_for_other_instances():
		for node in config['nodes']:
			## Lock file cleanup
			remove_lock_file(node)
			
	print "Done"


if __name__ == "__main__":
	main(sys.argv[1:])
	