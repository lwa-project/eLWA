#!/usr/bin/env python

"""
Monitor jobs launch on the LWAUCF by launchJobs.py.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import re
import sys
import time
import shlex
import subprocess
from datetime import datetime


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
		dirnames = dirnames.split('\n')[:-1]
		dirnames = [dirname.strip().rstrip() for dirname in dirnames]
	return dirnames


def get_processes(node):
	status, processes, errors = run_command('ps aux | grep superCorrelator | grep bash | grep -v grep | grep -v ssh', node=node)
	if status != 0:
		processes = []
	else:
		processes = processes.split('\n')[:-1]
		processes = [process.strip().rstrip() for process in processes]
	return processes


def get_directory_contents(node, dirname):
	status, filenames, errors = run_command('ls -d -1 %s/*' % dirname, node=node)
	if status != 0:
		filenames = []
	else:
		filenames = filenames.split('\n')[:-1]
		filenames = [filename.strip().rstrip() for filename in filenames]
	return filenames


def get_logfile_speed(node, logname):
	status, speedtime, error = run_command('grep -i -e average -e estimated %s | tail -n2' % logname, node=node)
	if status != 0:
		speed  = '---'
		remain = '---'
	else:
		speedtime = speedtime.split('\n')[:-1]
		speedtime = [entry.strip().rstrip() for entry in speedtime]
		if len(speedtime) == 0:
			speed  = '---'
			remain = '---'
		else:
			speed  = '---'
			remain = '---'
			
			for entry in speedtime:
				if entry.find('verage') != -1:
					try:
						_, speed = entry.rsplit('is', 1)
						speed = speed.strip().rstrip()
					except ValueError:
						speed = '---'
				elif entry.find('estimated') != -1:
					try:
						_, remain = entry.rsplit('is', 1)
						remain = remain.strip().rstrip()
					except ValueError:
						remain = '---'
	return speed, remain


def main(args):
	while True:
		try:
			t0 = time.time()
			
			status = {}
			for i in xrange(6):
				node = 'lwaucf%i' % (i+1,)
				
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
				                'remaining':{}}
				status[node]['dirnames'] = dirnames
				
				## Get running superCorrelator.py processes
				status[node]['processes'] = get_processes(node)
				
				## For each process, get the configuration file being
				## processes
				for process in status[node]['processes']:
					dirname, cmdname = process.split('&&', 1)
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
							nNPZ += 1
						elif ext == '.log':
							logname = filename
						elif ext[:6] == '.config':
							configname = filename
							
					### Parse the logfile, if we have one
					if logname is not None:
						cspeed, cremain = get_logfile_speed(node, logname)
					else:
						cspeed, cremain = '---', '---'
						
					### Save
					status[node]['progress'][dirname] = nNPZ
					status[node]['altconfig'][dirname] = configname
					status[node]['speed'][dirname] = cspeed
					status[node]['remaining'][dirname] = cremain
			t1 = time.time()
			
			# Report
			print "=== %s ===" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			for node in sorted(status.keys()):
				entry = status[node]
				
				print "%s:" % node
				for dirname in entry['dirnames']:
					nFiles = entry['progress'][dirname]
					if dirname in entry['active']:
						active = 'active'
						configfile = entry['active'][dirname]
						pid = 0
						for process in entry['processes']:
							if process.split('>', 1)[0].rsplit(None, 1)[1] == configfile:
								pid = int(process.split(None)[1], 10)
						speed = entry['speed'][dirname]
						remaining = entry['remaining'][dirname]
						info = '%s @ %i; %s per integration, %s remaning' % (configfile, pid, speed, remaining)
					else:
						active = 'stale'
						try:
							info = '%s (?)' % entry['altconfig'][dirname]
						except KeyError:
							info = None
					print '  %s (%s)' % (dirname, active)
					print '    %i integrations processed' % nFiles
					if info is not None:
						print '    %s' % info
						
			t2 = time.time()
			print "query %.3f, report %.3f" % (t1-t0, t2-t1)
			
			# Sleep
			time.sleep(min([0, 60-(t2-t0)]))
			
		except KeyboardInterrupt:
			break


if __name__ == "__main__":
	main(sys.argv[1:])
	