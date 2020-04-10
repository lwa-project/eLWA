#!/usr/bin/env python

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import os
import sys
import ephem
import numpy
from scipy.optimize import leastsq

from lsl.common.stations import ecef2geo, lwa1, lwasv
from lsl.common.constants import c as vLight


_SOURCES = {'3C295' : ('14:11:20.45',  '52:12:09.36' ),
            '3C286' : ('13:31:08.28',  '30:30:32.76' ),
            'VIRGOA': ('12:30:49.423', '12:23:28.044'),
            '3C196' : ('8:13:36.03',   '48:13:02.60' ),
            '3C48'  : ('1:37:41.30',   '33:09:35.11' ), 
            '3C84'  : ('3:19:48.16',   '41:30:42.11' ), 
            '3C147' : ('5:42:36.14',   '49:51:07.20' ),
            '3C216' : ('9:09:33.50',   '42:53:46.51' ),}


def main(args):
    filename = args[0]
    src, time, freq, az, el, snr, delay, rate = [], [], [], [], [], [], [], []
    fh = open(filename, 'r')
    for line in fh:
        fields = line.split()
        if fields[0] in ('3C84', 'VirgoA', '3C309.1'):
            continue
        src.append( fields[0] )
        time.append( float(fields[1]) )
        freq.append( float(fields[2]) )
        az.append( float(fields[3])*numpy.pi/180 )
        el.append( float(fields[4])*numpy.pi/180 )
        snr.append( float(fields[7]) )
        delay.append( float(fields[8])*1e-6 )
        rate.append( float(fields[10])*1e-3 )
    fh.close()

    az = numpy.array(az)
    el = numpy.array(el)
    snr = numpy.array(snr)
    delay = numpy.array(delay)
    valid = numpy.where( (snr >= 15) )[0]#& (el > 25*numpy.pi/180) )[0]
    az = az[valid]
    el = el[valid]
    snr = snr[valid]
    delay = delay[valid]
    print(az.shape, el.shape, delay.shape)
    
    import pylab
    pylab.title('Before')
    pylab.subplot(1, 2, 1)
    pylab.plot(az*180/numpy.pi, delay*1e6, linestyle='-', marker='x')
    pylab.xlabel('Azimuth [deg]')
    pylab.ylabel('Delay [$\\mu$s]')
    pylab.subplot(1, 2, 2)
    pylab.plot(el*180/numpy.pi, delay*1e6, linestyle='-', marker='x')
    pylab.xlabel('Elevation [deg]')
    pylab.ylabel('Delay [$\\mu$s]')
    pylab.show()
    
    def fnc(p, x):
        lwasv = numpy.array([p[0], p[1], p[2]])
        offset = p[3]
        
        azs = x[:x.size/2]
        els = x[x.size/2:]
    
        y = numpy.zeros(azs.size, dtype=azs.dtype)
        for i in xrange(azs.size):
            azPC, elPC = azs[i], els[i]
            dc = numpy.array([numpy.cos(elPC)*numpy.sin(azPC), 
                        numpy.cos(elPC)*numpy.cos(azPC), 
                        numpy.sin(elPC)])
        
            y[i] = numpy.dot(dc, lwasv)/vLight + offset
            
        return y

    def err(p, x, y):
        yFit = fnc(p, x)
        return y - yFit

    x = numpy.zeros((az.size*2), dtype=az.dtype)
    x[:x.size/2] = az
    x[x.size/2:] = el
    p0 = [0, 0, 0, delay.min()]
    p, stat = leastsq(err, p0, args=(x, delay))
    print(stat, p)
    print('!!', (delay-fnc(p,x))*1e9)
    print('II', numpy.sqrt((delay-delay.mean()*1e9)**2).mean(), numpy.sqrt((((delay-fnc(p,x))*1e9)**2).mean()))
    
    delayFixed = err(p, x, delay)
    pylab.title('After')
    pylab.subplot(1, 2, 1)
    pylab.plot(az*180/numpy.pi, delayFixed*1e9, linestyle='-', marker='x')
    pylab.xlabel('Azimuth [deg]')
    pylab.ylabel('Delay [ns]')
    pylab.subplot(1, 2, 2)
    pylab.plot(el*180/numpy.pi, delayFixed*1e9, linestyle='-', marker='x')
    pylab.xlabel('Elevation [deg]')
    pylab.ylabel('Delay [ns]')
    pylab.show()
    
    ## Derived from the 2018 Feb 28 observations of 3C295 and Virgo A
    ## with LWA1 and EA03/EA01
    LWA1_ECEF = numpy.array((-1602235.14380825, -5042302.73757814, 3553980.03506238))
    LWA1_LAT =   34.068956328 * numpy.pi/180
    LWA1_LON = -107.628103026 * numpy.pi/180
    LWA1_ROT = numpy.array([[ numpy.sin(LWA1_LAT)*numpy.cos(LWA1_LON), numpy.sin(LWA1_LAT)*numpy.sin(LWA1_LON), -numpy.cos(LWA1_LAT)], 
                            [-numpy.sin(LWA1_LON),                     numpy.cos(LWA1_LON),                      0                  ],
                            [ numpy.cos(LWA1_LAT)*numpy.cos(LWA1_LON), numpy.cos(LWA1_LAT)*numpy.sin(LWA1_LON),  numpy.sin(LWA1_LAT)]])
    
    ## Derived from the 2018 Feb 23 observations of 3C295 and 3C286
    ## with LWA1 and LWA-SV.  This also includes the shift detailed
    ## above for LWA1
    LWASV_ECEF = numpy.array((-1531538.16270273, -5045434.96744258, 3579251.37144792))
    LWASV_LAT =   34.348422098 * numpy.pi/180
    LWASV_LON = -106.885629291 * numpy.pi/180
    LWASV_ROT = numpy.array([[ numpy.sin(LWASV_LAT)*numpy.cos(LWASV_LON), numpy.sin(LWASV_LAT)*numpy.sin(LWASV_LON), -numpy.cos(LWASV_LAT)], 
                             [-numpy.sin(LWASV_LON),                      numpy.cos(LWASV_LON),                       0                   ],
                             [ numpy.cos(LWASV_LAT)*numpy.cos(LWASV_LON), numpy.cos(LWASV_LAT)*numpy.sin(LWASV_LON),  numpy.sin(LWASV_LAT)]])
    
    xyz = LWASV_ECEF
    rho = xyz - LWA1_ECEF
    sez = numpy.dot(LWA1_ROT, rho)
    enz = sez[[1,0,2]]
    enz[1] *= -1
    
    enz += p[0:3]
    print('Offsets:', 'ENZ', enz, 'Clock', p[3])
    enz[1] *= -1
    sez = enz[[1,0,2]]
    rho = numpy.dot(numpy.linalg.inv(LWA1_ROT), sez)
    xyz = rho + LWA1_ECEF
    print(xyz, xyz-LWASV_ECEF)
    lat, lon, elev = ecef2geo(*xyz)
    print(LWASV_LAT*180/numpy.pi, lat*180/numpy.pi)
    print(LWASV_LON*180/numpy.pi, lon*180/numpy.pi)
    print(lwasv.elev, elev)


if __name__ == "__main__":
    main(sys.argv[1:])
    