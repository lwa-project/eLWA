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

import prtab


_SOURCES = {'3C295' : ('14:11:20.45',  '52:12:09.36' ),
            '3C286' : ('13:31:08.28',  '30:30:32.76' ),
            'VIRGOA': ('12:30:49.423', '12:23:28.044'),
            '3C196' : ('8:13:36.030',  '48:13:02.500'),
            '3C48'  : ('1:37:41.30',   '33:09:35.11' ), 
            '3C84'  : ('3:19:48.16',   '41:30:42.11' ), 
            '3C147' : ('5:42:36.130',  '49:51:07.200'),
            '3C216' : ('9:09:33.50',   '42:53:46.51' ),
            'TAUA'  : ('5:34:31.940',  '22:00:52.200'),
            '3C123' : ('4:37:04.370',  '29:40:13.800'),
            '3C161' : ('6:27:10.110',  '-5:53:05.100')}


MAX_DELAY = 250e-9
MIN_WEIGHT = 100


def main(args):
    filenames = args
    delays = {}
    for filename in filenames:
        srcName = filename.split('.', 1)[0]
        delays[srcName] = {}
        
        header, data = prtab.parse(filename)
        i = prtab.mindex(header, 'TIME')
        j = prtab.mindex(header, 'ANTENNA')
        k = prtab.mindex(header, 'DELAY 1')
        l = prtab.mindex(header, 'DELAY 2')
        m, n = prtab.mindex(header, 'WEIGHT')
        for rowID in sorted(data.keys()):
            row = data[rowID]
            
            entry = [row[i], row[j], None, None]
            if abs(row[k]) < MAX_DELAY and row[m] >= MIN_WEIGHT:
                entry[2] = row[k]
            if abs(row[l]) < MAX_DELAY and row[n] >= MIN_WEIGHT:
                entry[3] = row[l]
                
            delays[srcName][rowID] = entry
            
    obs = lwa1.getObserver()
    ant, az, el, delay = [], [], [], []
    for srcName in delays:
        ra, dec = _SOURCES[srcName]
        bdy = ephem.FixedBody()
        bdy._ra = ra
        bdy._dec = dec
        bdy._epoch = ephem.J2000
        info = delays[srcName]
        
        for row in info:
            time, antenna, delay1, delay2 = info[row]
            ds = [d for d in [delay1,delay2] if d is not None]
            if antenna in (52,):
                continue
            if len(ds) == 0:
                continue
            obs.date = '2018/4/20 %s' % time
            bdy.compute(obs)
            ant.append( antenna )
            az.append( bdy.az*1.0  )
            el.append( bdy.alt*1.0 )
            delay.append( sum(ds)/float(len(ds)) )
    az = numpy.array(az)
    el = numpy.array(el)
    delay = numpy.array(delay)
    print(az.shape, el.shape, delay.shape)
    
    oaz = numpy.argsort(az)
    oel = numpy.argsort(el)
    
    import pylab
    pylab.subplot(1, 2, 1)
    pylab.plot(az[oaz]*180/numpy.pi, delay[oaz]*1e9, linestyle='-', marker='x')
    pylab.xlabel('Azimuth [deg]')
    pylab.ylabel('Delay [ns]')
    pylab.subplot(1, 2, 2)
    pylab.plot(el[oel]*180/numpy.pi, delay[oel]*1e9, linestyle='-', marker='x')
    pylab.xlabel('Elevation [deg]')
    pylab.ylabel('Delay [ns]')
    pylab.draw()
    
    uant = numpy.unique(ant)
    uant = list(uant)
    print(uant)
    
    def fnc(p, x, ant=ant, uant=uant):
        lwasv = numpy.array([p[0], p[1], p[2]])
        offsets = p[3:]
        
        azs = x[:x.size/2]
        els = x[x.size/2:]
        
        order = [uant.index(ant[i]) for i in xrange(azs.size)]
    
        y = numpy.zeros(azs.size, dtype=azs.dtype)
        for i in xrange(azs.size):
            azPC, elPC = azs[i], els[i]
            dc = numpy.array([numpy.cos(elPC)*numpy.sin(azPC), 
                            numpy.cos(elPC)*numpy.cos(azPC), 
                            numpy.sin(elPC)])    
            y[i] = numpy.dot(dc, lwasv)/vLight + offsets[order[i]]
            
        return y
        
    def err(p, x, y, ant=ant, uant=uant):
        yFit = fnc(p, x, ant=ant, uant=uant)
        return y - yFit
        
    x = numpy.zeros((az.size*2), dtype=az.dtype)
    x[:x.size/2] = az
    x[x.size/2:] = el
    p0 = [0, 0, 0]
    p0.extend( [delay.min() for u in numpy.unique(ant)] )
    p, stat = leastsq(err, p0, args=(x, delay))
    print(stat, p)
    print('!!', (delay-fnc(p,x))*1e9)
    print('II', numpy.sqrt((delay-delay.mean()*1e9)**2).mean(), numpy.sqrt((((delay-fnc(p,x))*1e9)**2).mean()))
    print('->', numpy.sqrt((((delay-fnc(p,x))*1e9)**2).mean()) / numpy.sqrt((delay-delay.mean()*1e9)**2).mean())
    
    delayFixed = err(p, x, delay)
    pylab.subplot(1, 2, 1)
    pylab.plot(az[oaz]*180/numpy.pi, delayFixed[oaz]*1e9, linestyle='-', marker='x')
    pylab.xlabel('Azimuth [deg]')
    pylab.ylabel('Delay [ns]')
    pylab.subplot(1, 2, 2)
    pylab.plot(el[oel]*180/numpy.pi, delayFixed[oel]*1e9, linestyle='-', marker='x')
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
    print(ecef2geo(*LWA1_ECEF), LWA1_LAT, LWA1_LON, lwa1.lat*1.0, lwa1.lon*1.0)

    ## Derived from the 2018 Feb 23 observations of 3C295 and 3C286
    ## with LWA1 and LWA-SV.  This also includes the shift detailed
    ## above for LWA1
    LWASV_ECEF = numpy.array((-1531556.98709475, -5045435.8720832, 3579254.27947458))
    LWASV_LAT =   34.34841153053564 * numpy.pi/180
    LWASV_LON = -106.88582216960029 * numpy.pi/180
    LWASV_ROT = numpy.array([[ numpy.sin(LWASV_LAT)*numpy.cos(LWASV_LON), numpy.sin(LWASV_LAT)*numpy.sin(LWASV_LON), -numpy.cos(LWASV_LAT)], 
                             [-numpy.sin(LWASV_LON),                      numpy.cos(LWASV_LON),                       0                   ],
                             [ numpy.cos(LWASV_LAT)*numpy.cos(LWASV_LON), numpy.cos(LWASV_LAT)*numpy.sin(LWASV_LON),  numpy.sin(LWASV_LAT)]])
    
    print("***")
    
    enz = numpy.array([0.0, 0.0, 0.0])
    enz -= p[0:3]
    enz[1] *= -1
    sez = enz[[1,0,2]]
    rho = numpy.dot(numpy.linalg.inv(LWA1_ROT), sez)
    xyz = rho + LWA1_ECEF
    print(xyz, xyz-LWA1_ECEF)
    lat, lon, elev = ecef2geo(*xyz)
    print(LWA1_LAT*180/numpy.pi, lat*180/numpy.pi)
    print(LWA1_LON*180/numpy.pi, lon*180/numpy.pi)
    print(lwa1.elev, elev)
    
    print("===")
    
    xyz = LWASV_ECEF
    rho = xyz - LWA1_ECEF
    sez = numpy.dot(LWA1_ROT, rho)
    enz = sez[[1,0,2]]
    enz[1] *= -1
    enz -= p[0:3]
    #print(enz, p[3])
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
    