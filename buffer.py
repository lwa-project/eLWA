# -*- coding: utf-8 -*-
"""
Simple ring buffer class for GUPPI data.  This module is heavily based on the 
lsl.reader.buffer module.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

from lsl.reader.buffer import VDIFFrameBuffer

__version__ = '0.3'
__revision__ = '$Rev$'
__all__ = ['GUPPIFrameBuffer',]


class GUPPIFrameBuffer(VDIFFrameBuffer):
    def get_figure_of_merit(self, frame):
        """
        Figure of merit for sorting frames.  For DRX it is:
            <frame timetag in ticks>
        """
        
        return frame.header.offset
