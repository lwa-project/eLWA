# -*- coding: utf-8 -*-
"""
Simple ring buffer class for VDIF data.  This module is heavily based on the 
lsl.reader.buffer module.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""


import copy
from collections import deque
try:
	from collections import OrderedDict
except ImportError:
	from lsl.misc.OrderedDict import OrderedDict


class FrameBuffer(object):
	"""
	Frame buffer for re-ordering TBN and DRX frames in time order.  
	This class is filled with frames and a returns a frame list when 
	the 'nSegments' starts filling.  In that case, the oldest segment 
	is returned.

	The buffer also keeps track of what has already been read out so 
	that tardy frames are just dropped.  For buffers that are read out,
	missing frames are replaced with frames filled with zeros.
	
	.. note::
		Due to the nature of the buffer, it is possible that there will
		still be 'nSegments'-1 segements in the buffer that are either
		full or partially full.  This can be retrieved using the buffer's 
		'flush()' function.
	"""
	
	def __init__(self, mode='VDIF', threads=[], nSegments=6, ReorderFrames=False):
		"""
		Initialize the buffer with a list of:
		  * VDIF
		      * list of threads
		By doing this, we should be able to keep up with when the buffer 
		is full and to help figure out which stands/beam/tunings/pols. are 
		missing.
		"""
		
		# Input validation
		if mode.upper() not in ('VDIF'):
			raise RuntimeError("Invalid observing mode '%s'" % mode)
			
		# The buffer itself
		self.nSegments = nSegments
		self.buffer = OrderedDict()
		self.done = deque([0,], maxlen=self.nSegments)
		
		# Buffer statistics
		self.full = 0		# Number of times a full buffer was emptied
		self.partial = 0	# Number of times a partial buffer was emptied
		self.dropped = 0	# Number of late or duplicate frames dropped
		self.missing = 0	# Number of missing frames
		
		# Information describing the observations
		self.mode = mode.upper()
		self.threads = threads
		
		# If we should reorder the returned frames by stand/pol or not
		self.reorder = ReorderFrames
		
		# Figure out how many frames fill the buffer and the list of all
		# possible frames in the data set
		self.nFrames, self.possibleFrames = self.calcFrames()
		
	def calcFrames(self):
		"""
		Calculate the maximum number of frames that we expect from 
		the setup of the observations and a list of tuples that describes
		all of the possible stand/pol combination.
		
		This will be overridden by sub-classes of FrameBuffer.
		"""
		
		pass
		
	def figureOfMerit(self, frame):
		"""
		Figure of merit for storing/sorting frames in the ring buffer.
		
		This will be overridden by sub-classes of FrameBuffer.
		"""
		
		pass
		
	def createFill(self, key, frameParameters):
		"""
		Create a 'fill' frame of zeros using an existing good
		packet as a template.
		
		This will be overridden by sub-classes of FrameBuffer.
		"""
		
		pass
		
	def append(self, frames):
		"""
		Append a new frame to the buffer with the appropriate time tag.  
		True is returned if the frame was added to the buffer and False if 
		the frame was dropped because it belongs to a buffer that has 
		already been returned.
		"""
		
		# Convert input to a deque (if needed)
		typeName = type(frames).__name__
		if typeName == 'deque':
			pass
		elif typeName == 'list':
			frames = deque(frames)
		else:
			frames = deque([frames,])
			
		# Loop over frames
		while True:
			try:
				frame = frames.popleft()
			except IndexError:
				break
				
			# Make sure that it is not in the `done' list.  If it is,
			# disgaurd the frame and make a note of it.
			fom = self.figureOfMerit(frame)
			if fom < self.done[-1]:
				self.dropped += 1
				continue
				
			# If that time tag hasn't been done yet, add it to the 
			# buffer in the correct place.
			try:
				self.buffer[fom].append(frame)
			except KeyError:
				self.buffer[fom] = deque()
				self.buffer[fom].append(frame)
				
		return True
		
	def get(self, keyToReturn=None):
		"""
		Return a list of frames that consitute a 'full' buffer.  Afterwards, 
		delete that buffer and mark it as closed so that any missing frames that
		are recieved late are dropped.  If none of the buffers are ready to be 
		dumped, None is returned.
		"""
		
		# Get the current status of the buffer
		keys = self.buffer.keys()
		
		if keyToReturn is None:
			# If the ring is full, dump the oldest
			if len(keys) < self.nSegments:
				return None
				
			keyToReturn = keys[0]
		returnCount = len(self.buffer[keyToReturn])
		
		if returnCount == self.nFrames:
			## Everything is good (Is it really???)
			self.full = self.full + 1
			
			output = self.buffer[keyToReturn]
		elif returnCount < self.nFrames:
			## There are too few frames
			self.partial = self.partial + 1
			self.missing = self.missing + (self.nFrames - returnCount)
			
			output = self.buffer[keyToReturn]
			for frame in self._missingList(keyToReturn):
				output.append( self.createFill(keyToReturn, frame) )
		else:
			## There are too many frames
			self.full = self.full + 1
			
			output = []
			frameIDs = []
			for frame in self.buffer[keyToReturn]:
				newID = frame.parseID()
				if newID not in frameIDs:
					output.append(frame)
					frameIDs.append(newID)
				else:
					self.dropped += 1
					
		del(self.buffer[keyToReturn])
		self.done.append(keyToReturn)
		
		# Sort and return
		if self.reorder:
			output = list(output)
			output.sort(cmp=_cmpStands)
		return output
		
	def flush(self):
		"""
		Return a list of lists containing all remaining frames in the 
		buffer from buffers that are considered 'full'.  Afterwards, 
		delete all buffers.  This is useful for emptying the buffer after
		reading in all of the data.
		
		.. note::
			It is possible for this function to return list of packets that
			are mostly invalid.
		"""
		
		remainingKeys = self.buffer.keys()
		
		output = []
		for key in remainingKeys:
			output2 = self.get(keyToReturn=key)
			output.append( output2 )
			
		return output
		
	def _missingList(self, key):
		"""
		Create a list of tuples of missing frame information.
		"""
		
		# Find out what frames we have
		frameList = []
		for frame in self.buffer[key]:
			frameList.append(frame.parseID()[1])
			
		# Compare the existing list with the possible list stored in the 
		# FrameBuffer object to build a list of missing frames.
		missingList = []
		for frame in self.possibleFrames:
			if frame not in frameList:
				missingList.append(frame)
				
		return missingList
		
	def status(self):
		"""
		Print out the status of the buffer.  This contains information about:
		  1.  The current buffer fill level
		  2. The numer of full and partial buffer dumps preformed
		  3. The number of missing frames that fill packets needed to be created
		     for
		  4. The number of frames that arrived too late to be incorporated into 
		     one of the ring buffers
		"""
		
		nf = 0
		for key in self.buffer.keys():
			nf = nf + len(self.buffer[key])
			
		outString = ''
		outString = '\n'.join([outString, "Current buffer level:  %i frames" % nf])
		outString = '\n'.join([outString, "Buffer dumps:  %i full / %i partial" % (self.full, self.partial)])
		outString = '\n'.join([outString, "--"])
		outString = '\n'.join([outString, "Missing frames:  %i" % self.missing])
		outString = '\n'.join([outString, "Dropped frames:  %i" % self.dropped])
		
		print outString


class VDIFFrameBuffer(FrameBuffer):
	def __init__(self, threads=[0,1], nSegments=10, ReorderFrames=False):
		super(VDIFFrameBuffer, self).__init__(mode='VDIF', threads=threads, nSegments=nSegments, ReorderFrames=ReorderFrames)
		
	def calcFrames(self):
		"""
		Calculate the maximum number of frames that we expect from 
		the setup of the observations and a list of tuples that describes
		all of the possible stand/pol combination.
		"""
		
		nFrames = 0
		frameList = []
		
		nFrames = len(self.threads)
		for thread in self.threads:
			frameList.append(thread)
			
		return (nFrames, frameList)
		
	def figureOfMerit(self, frame):
		"""
		Figure of merit for sorting frames.  For DRX it is:
		    <frame timetag in ticks>
		"""
		
		return frame.header.secondsFromEpoch*10000 + frame.header.frameInSecond
		
	def createFill(self, key, frameParameters):
		"""
		Create a 'fill' frame of zeros using an existing good
		packet as a template.
		"""

		# Get a template based on the first frame for the current buffer
		fillFrame = copy.deepcopy(self.buffer[key][0])
		
		# Get out the frame parameters and fix-up the header
		thread = frameParameters
		fillFrame.header.threadID = thread
		
		# Zero the data for the fill packet
		fillFrame.data.data *= 0
		
		# Invalidate the frame
		fillFrame.valid = False
		
		return fillFrame
