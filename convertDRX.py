#!/usr/bin/env python

# Python2 compatibility
from __future__ import print_function, division

import os
import sys
import copy
import struct
from datetime import datetime
from collections import deque

from lsl.common.dp import fS
fS = int(fS)
from lsl.reader.ldp import DRXFile
from lsl.reader import drx, errors, buffer
from lsl.writer import vdif


class RawDRXFrame(object):
    """
    Class to help hold and work with a raw (packed) DRX frame.
    """
    
    def __init__(self, contents):
        self.contents = bytearray(contents)
        if len(self.contents) != drx.FRAME_SIZE:
            raise errors.EOFError
        if self.contents[0] != 0xDE or self.contents[1] != 0xC0 or self.contents[2] != 0xDE or self.contents[3] != 0x5c:
            raise errors.SyncError
            
    def __getitem__(self, key):
        return self.contents[key]
        
    def __setitem__(self, key, value):
        self.contents[key] = value
        
    @property
    def id(self):
        _id = self.contents[4]
        _id = (_id & 7), ((_id >> 3) & 7), ((_id >> 7) & 1)
        return _id
        
    @property
    def timetag(self):
        time_tag = 0
        time_tag |= self.contents[16] << 56
        time_tag |= self.contents[17] << 48
        time_tag |= self.contents[18] << 40
        time_tag |= self.contents[19] << 32
        time_tag |= self.contents[20] << 24
        time_tag |= self.contents[21] << 16
        time_tag |= self.contents[22] <<  8
        time_tag |= self.contents[23]
        return time_tag
        
    @property
    def tNom(self):
        t_nom = (self.contents[14] << 8) | self.contents[15]
        return t_nom


class RawDRXFrameBuffer(buffer.FrameBufferBase):
    """
    A sub-type of FrameBufferBase specifically for dealing with raw (packed) DRX 
    frames.  See :class:`lsl.reader.buffer.FrameBufferBase` for a description of 
    how the buffering is implemented.
    
    Keywords:
    beams
    list of beam to expect packets for
    
    tunes
    list of tunings to expect packets for
    
    pols
    list of polarizations to expect packets for
    
    nsegments
    number of ring segments to use for the buffer (default is 20)
    
    reorder
    whether or not to reorder frames returned by get() or flush() by 
    stand/polarization (default is False)
    
    The number of segements in the ring can be converted to a buffer time in 
    seconds:
    
    +----------+--------------------------------------------------+
    |          |                 DRX Filter Code                  |
    | Segments +------+------+------+------+------+-------+-------+
    |          |  1   |  2   |  3   |  4   |  5   |  6    |  7    |
    +----------+------+------+------+------+------+-------+-------+
    |    10    | 0.16 | 0.08 | 0.04 | 0.02 | 0.01 | 0.004 | 0.002 |
    +----------+------+------+------+------+------+-------+-------+
    |    20    | 0.33 | 0.16 | 0.08 | 0.04 | 0.02 | 0.008 | 0.004 |
    +----------+------+------+------+------+------+-------+-------+
    |    30    | 0.49 | 0.25 | 0.12 | 0.06 | 0.03 | 0.013 | 0.006 |
    +----------+------+------+------+------+------+-------+-------+
    |    40    | 0.66 | 0.33 | 0.16 | 0.08 | 0.03 | 0.017 | 0.008 |
    +----------+------+------+------+------+------+-------+-------+
    |    50    | 0.82 | 0.41 | 0.20 | 0.10 | 0.04 | 0.021 | 0.010 |
    +----------+------+------+------+------+------+-------+-------+
    |   100    | 1.64 | 0.82 | 0.41 | 0.20 | 0.08 | 0.042 | 0.021 |
    +----------+------+------+------+------+------+-------+-------+
    
    """
    
    def __init__(self, beams=[], tunes=[1,2], pols=[0, 1], nsegments=20, reorder=False):
        super(RawDRXFrameBuffer, self).__init__(mode='DRX', beams=beams, tunes=tunes, pols=pols, nsegments=nsegments, reorder=reorder)
        
    def get_max_frames(self):
        """
        Calculate the maximum number of frames that we expect from 
        the setup of the observations and a list of tuples that describes
        all of the possible stand/pol combination.
        """
        
        nFrames = 0
        frameList = []
        
        nFrames = len(self.beams)*len(self.tunes)*len(self.pols)
        for beam in self.beams:
            for tune in self.tunes:
                for pol in self.pols:
                    frameList.append((beam,tune,pol))
                    
        return (nFrames, frameList)
        
    def get_figure_of_merit(self, frame):
        """
        Figure of merit for sorting frames.  For DRX it is:
            <frame timetag in ticks>
        """
        
        return frame.timetag
    
    def create_fill(self, key, frameParameters):
        """
        Create a 'fill' frame of zeros using an existing good
        packet as a template.
        """

        # Get a template based on the first frame for the current buffer
        fillFrame = copy.deepcopy(self.buffer[key][0])
        
        # Get out the frame parameters and fix-up the header
        beam, tune, pol = frameParameters
        fillFrame[4] = (beam & 7) | ((tune & 7) << 3) | ((pol & 1) << 7)
        
        # Zero the data for the fill packet
        fillFrame[32:] = '\x00'*4096
        
        return fillFrame


class VDIFFrame(object):
    def __init__(self, id, timetag, frame_size, frames_per_second, data=None):
        self.id = id
        self.timetag = timetag
        self.frame_size = frame_size
        self.frames_per_second = frames_per_second
        
        self.seconds = self.timetag // fS - 946684800
        self.frame = int((self.timetag % fS) / fS * frames_per_second)
        self.frame %= self.frames_per_second
        
        self.data = bytearray()
        if data is not None:
            self.data.extend(data)
            
    def extend(self, data):
        self.data.extend(data)
        
    @property
    def is_ready(self):
        return len(self.data) >= self.frame_size
        
    def write(self, fh):
        raw = bytearray([b'\x00',]*(32 + self.frame_size))
        
        # Valid data, standard (not legacy) 32-bit header, and seconds since 
        # the 01/01/2000 epoch.
        raw[3] = (0 << 7) | (0 << 6) | ((self.seconds >> 24) & 0x3F)
        raw[2] = (self.seconds >> 16) & 0xFF
        raw[1] = (self.seconds >> 8) & 0xFF
        raw[0] = self.seconds & 0xFF
        
        # Reference epoch (0 == 01/01/2000) and frame count
        raw[7] = 0 & 0x3F
        raw[6] = (self.frame >> 16) & 0xFF
        raw[5] = (self.frame >> 8) & 0xFF
        raw[4] = self.frame & 0xFF

        # VDIF version number, number of channels (just 1), and data frame 
        # length in units to 8-bytes (8 raw array elements)
        raw[11] = (1 << 6) | (0 & 0x1F)
        raw[10] = (((self.frame_size+32) // 8) >> 16) & 0xFF
        raw[9] = (((self.frame_size+32) // 8) >> 8) & 0xFF
        raw[8] = ((self.frame_size+32) // 8) & 0xFF

        # Data type, bits per sample, thread ID, and station ID
        raw[15] = (1 << 7) | (((4-1) & 0x1F) << 2) | ((0 >> 8) & 0x03)
        raw[14] = (self.id % 10) & 0xFF
        raw[13] = ((self.id//10) >> 8) & 0xFF
        raw[12] = (self.id//10) & 0xFF
        
        # The data itself
        raw[32:] = self.data[:self.frame_size]
        
        fh.write(raw)
        
        return self.data[self.frame_size:]


def main(args):
    filename = args[0]
    # Open the file
    idf = DRXFile(filename)
    
    # Load in basic information about the data
    nFramesFile = idf.get_info('nframe')
    srate = idf.get_info('sample_rate')
    ttSkip = int(round(196e6/srate))*4096
    beam = idf.get_info('beam')
    beampols = idf.get_info('nbeampol')
    tunepol = beampols
    
    ## Date
    beginDate = idf.get_info('start_time')
    beginTime = beginDate.datetime
    mjd = beginDate.mjd
    mjd_day = int(mjd)
    mjd_sec = (mjd-mjd_day)*86400
    
    ## Tuning frequencies
    central_freq1 = idf.get_info('freq1')
    central_freq2 = idf.get_info('freq2')
    beam = idf.get_info('beam')
    
    # File summary
    print("Input Filename: %s" % filename)
    print("Date of First Frame: %s (MJD=%f)" % (str(beginDate),mjd))
    print("Tune/Pols: %i" % tunepol)
    print("Tunings: %.1f Hz, %.1f Hz" % (central_freq1, central_freq2))
    print("Sample Rate: %i Hz" % srate)
    print("Frames: %i (%.3f s)" % (nFramesFile, 4096.0*nFramesFile / srate / tunepol))
    
    # Output names
    vdifname1 = "%s_%.0fMHz.vdif" % (os.path.basename(filename), central_freq1/1e6)
    vdifname2 = "%s_%.0fMHz.vdif" % (os.path.basename(filename), central_freq2/1e6)
    print(vdifname1, vdifname2)
    
    # Output formatting
    vdif_bits = 8
    vdif_complex = True
    vdif_frame_size = 5000
    while int(srate) % vdif_frame_size != 0:
        vdif_frame_size += 1
    vdif_frames_per_second = int(srate) // vdif_frame_size
    
    # Output summary
    print("Output Filenames: %s, %s" % (vdifname1, vdifname2))
    print("Bits: %i" % vdif_bits)
    print("Complex data: %s" % vdif_complex)
    print("Samples per frame: %i" % vdif_frame_size)
    print("Frames per second: %i" % vdif_frames_per_second)
    
    # Ready the internal interface for file access
    fh = idf.fh
    
    # Ready the output files - one for each tune/pol
    fhOut = []
    for vdifname in (vdifname1, vdifname2):
        fhOut.append(open(vdifname, 'wb'))
        
    # Setup the buffer
    buffer = RawDRXFrameBuffer(beams=[beam,], reorder=True)
    
    # Go!
    c1X, c1Y = 0, 0
    c2X, c2Y = 0, 0
    started = False
    eofFound = False
    while True:
        if eofFound:
            break
            
        ## Load in some frames
        if not buffer.overfilled:
            rFrames = deque()
            for i in range(tunepol):
                try:
                    rFrames.append( RawDRXFrame(fh.read(drx.FRAME_SIZE)) )
                    #print rFrames[-1].id, rFrames[-1].timetag, c, i
                except errors.EOFError:
                    eofFound = True
                    buffer.append(rFrames)
                    break
                except errors.SyncError:
                    continue
                
            buffer.append(rFrames)
            
        timetag = buffer.peek()
        if timetag is None:
            # Continue adding frames if nothing comes out.
            continue
        else:
            # Otherwise, make sure we are on track
            try:
                timetag = timetag - tNomX # T_NOM has been subtracted from ttLast
                if timetag != ttLast + ttSkip:
                    missing = (timetag - ttLast - ttSkip) / float(ttSkip)
                    if int(missing) == missing and missing < 50:
                        ## This is kind of black magic down here
                        for m in range(int(missing)):
                            m = ttLast + ttSkip*(m+1) + tNomX   # T_NOM has been subtracted from ttLast
                            baseframe = copy.deepcopy(rFrames[0])
                            baseframe[14:24] = struct.pack('>HQ', struct.unpack('>HQ', baseframe[14:24])[0], m)
                            baseframe[32:] = '\x00'*4096
                            buffer.append(baseframe)
            except NameError:
                pass
        rFrames = buffer.get()
        
        ## Continue adding frames if nothing comes out.
        if rFrames is None:
            continue
            
        ## If something comes out, process it
        for tuning in (1, 2):
            ### Load
            pairX = rFrames[2*(tuning-1)+0]
            pairY = rFrames[2*(tuning-1)+1]
            
            ### Time tag manipulation to remove the T_NOM offset
            tNomX, timetagX = pairX.tNom, pairX.timetag
            #tNomY, timetagX = pairY.tNom, pairY.timetag
            tNom = tNomX - tNomX
            timetag = timetagX - tNomX
            timetag = timetag // (fS // int(srate)) * (fS // int(srate))
            
            if timetag % fS > (fS - ttSkip) or timetag % fS == 0:
                started = True
                
                sample_offset = (fS - timetag % fS) % fS
                sample_offset = sample_offset // (fS // int(srate))
                print(timetag, fS, sample_offset, timetag+sample_offset*(fS // int(srate)))
                
                if tuning == 1:
                    f1X = VDIFFrame(10, timetag + sample_offset*(fS // int(srate)),
                                    vdif_frame_size, vdif_frames_per_second,
                                    data=pairX[32+sample_offset:])
                    f1Y = VDIFFrame(11, timetag + sample_offset*(fS // int(srate)),
                                    vdif_frame_size, vdif_frames_per_second,
                                    data=pairY[32+sample_offset:])
                    
                else:
                    f2X = VDIFFrame(20, timetag + sample_offset*(fS // int(srate)),
                                    vdif_frame_size, vdif_frames_per_second,
                                    data=pairX[32+sample_offset:])
                    f2Y = VDIFFrame(21, timetag + sample_offset*(fS // int(srate)),
                                    vdif_frame_size, vdif_frames_per_second,
                                    data=pairY[32+sample_offset:])
                                    
                #print(timetag, timetag/fS)
                
            elif started:
                if tuning == 1:
                    f1X.extend(pairX[32:])
                    f1Y.extend(pairY[32:])
                    
                    if f1X.is_ready:
                        remainder = f1X.write(fhOut[tuning-1])
                        c1X += 1
                        c1X %= vdif_frames_per_second
                        f1X = VDIFFrame(10, timetag + (4096-len(remainder))*(fS // int(srate)),
                                        vdif_frame_size, vdif_frames_per_second,
                                        data=remainder)
                        #print('X', f1X.seconds, f1X.frame, c1X)
                        
                    if f1Y.is_ready:
                        remainder = f1Y.write(fhOut[tuning-1])
                        c1Y += 1
                        c1Y %= vdif_frames_per_second
                        f1Y = VDIFFrame(11, timetag + (4096-len(remainder))*(fS // int(srate)),
                                        vdif_frame_size, vdif_frames_per_second,
                                        data=remainder)
                        #print('Y', f1Y.seconds, f1Y.frame, c1Y)
                    
                else:
                    f2X.extend(pairX[32:])
                    f2Y.extend(pairY[32:])
                    
                    if f2X.is_ready:
                        remainder = f2X.write(fhOut[tuning-1])
                        c2X += 1
                        c2X %= vdif_frames_per_second
                        f2X = VDIFFrame(20, timetag + (4096-len(remainder))*(fS // int(srate)),
                                        vdif_frame_size, vdif_frames_per_second,
                                        data=remainder)
                        
                    if f2Y.is_ready:
                        remainder = f2Y.write(fhOut[tuning-1])
                        c2Y += 1
                        c2Y %= vdif_frames_per_second
                        f2Y = VDIFFrame(21, timetag + (4096-len(remainder))*(fS // int(srate)),
                                        vdif_frame_size, vdif_frames_per_second,
                                        data=remainder)
                    


if __name__ == "__main__":
    main(sys.argv[1:])
    
