"""
Copyright (c) 1998-2021 Scott M. Ransom <sransom@nrao.edu>
This software was originally released under GPLv2.
"""

from __future__ import absolute_import

import numpy as Num
from mini_presto.psr_constants import *

def rad_to_dms(rad):
    """
    rad_to_dms(rad):
       Convert radians to degrees, minutes, and seconds of arc.
    """
    if (rad < 0.0): sign = -1
    else: sign = 1
    arc = RADTODEG * Num.fmod(Num.fabs(rad), PI)
    d = int(arc)
    arc = (arc - d) * 60.0
    m = int(arc)
    s = (arc - m) * 60.0
    if sign==-1 and d==0:
        return (sign * d, sign * m, sign * s)
    else:
        return (sign * d, m, s)

def rad_to_hms(rad):
    """
    rad_to_hms(rad):
       Convert radians to hours, minutes, and seconds of arc.
    """
    rad = Num.fmod(rad, TWOPI)
    if (rad < 0.0): rad = rad + TWOPI
    arc = RADTOHRS * rad
    h = int(arc)
    arc = (arc - h) * 60.0
    m = int(arc)
    s = (arc - m) * 60.0
    return (h, m, s)

def coord_to_string(h_or_d, m, s):
    """
    coord_to_string(h_or_d, m, s):
       Return a formatted string of RA or DEC values as
       'hh:mm:ss.ssss' if RA, or 'dd:mm:ss.ssss' if DEC.
    """
    retstr = ""
    if h_or_d < 0:
        retstr = "-"
    elif abs(h_or_d)==0:
        if (m < 0.0) or (s < 0.0):
            retstr = "-"
    h_or_d, m, s = abs(h_or_d), abs(m), abs(s)
    if (s >= 9.9995):
        return retstr+"%.2d:%.2d:%.4f" % (h_or_d, m, s)
    else:
        return retstr+"%.2d:%.2d:0%.4f" % (h_or_d, m, s)

def ra_to_rad(ra_string):
    """
    ra_to_rad(ar_string):
       Given a string containing RA information as
       'hh:mm:ss.ssss', return the equivalent decimal
       radians.
    """
    h, m, s = ra_string.split(":")
    return hms_to_rad(int(h), int(m), float(s))

def dec_to_rad(dec_string):
    """
    dec_to_rad(dec_string):
       Given a string containing DEC information as
       'dd:mm:ss.ssss', return the equivalent decimal
       radians.
    """
    d, m, s = dec_string.split(":")
    if "-" in d and int(d)==0:
        m, s = '-'+m, '-'+s
    return dms_to_rad(int(d), int(m), float(s))

def p_to_f(p, pd, pdd=None):
   """
   p_to_f(p, pd, pdd=None):
      Convert period, period derivative and period second
      derivative to the equivalent frequency counterparts.
      Will also convert from f to p.
   """
   f = 1.0 / p
   fd = -pd / (p * p)
   if (pdd is None):
       return [f, fd]
   else:
       if (pdd==0.0):
           fdd = 0.0
       else:
           fdd = 2.0 * pd * pd / (p**3.0) - pdd / (p * p)
       return [f, fd, fdd]

def pferrs(porf, porferr, pdorfd=None, pdorfderr=None):
    """
    pferrs(porf, porferr, pdorfd=None, pdorfderr=None):
       Calculate the period or frequency errors and
       the pdot or fdot errors from the opposite one.
    """
    if (pdorfd is None):
        return [1.0 / porf, porferr / porf**2.0]
    else:
        forperr = porferr / porf**2.0
        fdorpderr = Num.sqrt((4.0 * pdorfd**2.0 * porferr**2.0) / porf**6.0 +
                               pdorfderr**2.0 / porf**4.0)
        [forp, fdorpd] = p_to_f(porf, pdorfd)
        return [forp, forperr, fdorpd, fdorpderr]
