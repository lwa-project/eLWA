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

from astropy.constants import c as vLight
vLight = vLight.to('m/s').value

from lsl.common.stations import ecef_to_geo, lwa1, lwasv


_SOURCES = {'3C295' : ('14:11:20.45',  '52:12:09.36' ),
            '3C286' : ('13:31:08.28',  '30:30:32.76' ),
            'VIRGOA': ('12:30:49.423', '12:23:28.044')}



def main(args):
    filenames = args
    delays = {}
    for filename in filenames:
        srcName = filename.split('_', 1)[0]
        
        fh = open(filename, 'r')
        info = {}
        for line in fh:
            if len(line) < 3:
                continue
            fields = line.split()
            if len(fields) == 13 and line.find(':') != -1:
                row = int(fields[0], 10)
                time = fields[1]
                ant = int(fields[4], 10)
                info[row] = [time, ant, None, None]
            if len(fields) == 13 and (line.find('E+') != -1 or line.find('E-') != -1):
                try:
                    row = int(fields[0], 10)
                    delay1 = float(fields[2])
                    weight = float(fields[4])
                    delay2 = float(fields[11])
                    
                    h, m, s = info[row][0].split(':', 2)
                    h = int(h, 10) + int(m, 10)/60.0 + float(s)/3600.0
                    if h >= 8 + 36/60.:
                        offset = 102e-9
                    else:
                        offset = 0
                        
                    info[row][2] = delay1 + offset
                    info[row][3] = delay2 + offset
                    if abs(delay1) > 200e-9 or abs(delay2) > 200e-9 or weight < 20:
                        ## Skip things that are all zero delay
                        try:
                            del info[row]
                        except KeyError:
                            pass
                        continue
                except ValueError:
                    ## Skip unparsable entries
                    try:
                        del info[row]
                    except KeyError:
                        pass
                except KeyError:
                    pass
        fh.close()
        
        delays[srcName] = info
        
    obs = lwa1.getObserver()
    az, el, delay = [], [], []
    for srcName in delays:
        ra, dec = _SOURCES[srcName]
        bdy = ephem.FixedBody()
        bdy._ra = ra
        bdy._dec = dec
        bdy._epoch = ephem.J2000
        info = delays[srcName]
        
        for row in info:
            time, ant, delay1, delay2 = info[row]
            ds = [d for d in [delay1, delay2] if d is not None]
            if ant not in (3, 1):
                continue
            if len(ds) == 0:
                continue
            obs.date = '2018/2/28 %s' % time
            bdy.compute(obs)
            az.append( bdy.az*1.0  )
            el.append( bdy.alt*1.0 )
            delay.append( sum(ds)/float(len(ds)) )
    az = numpy.array(az)
    el = numpy.array(el)
    delay = numpy.array(delay)
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
        
            y[i] = numpy.dot(dc, lwasv)/vLight + 0*offset
            
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
    
    ## Derived from the 2017 Oct 31 LWA1 SSMIF
    LWA1_ECEF = numpy.array((-1602258.2104158669, -5042300.0220439518, 3553974.6599673284))
    LWA1_LAT =   34.068894 * numpy.pi/180
    LWA1_LON = -107.628350 * numpy.pi/180
    LWA1_ROT = numpy.array([[ numpy.sin(LWA1_LAT)*numpy.cos(LWA1_LON), numpy.sin(LWA1_LAT)*numpy.sin(LWA1_LON), -numpy.cos(LWA1_LAT)], 
                            [-numpy.sin(LWA1_LON),                     numpy.cos(LWA1_LON),                      0                  ],
                            [ numpy.cos(LWA1_LAT)*numpy.cos(LWA1_LON), numpy.cos(LWA1_LAT)*numpy.sin(LWA1_LON),  numpy.sin(LWA1_LAT)]])
    print(ecef_to_geo(*LWA1_ECEF), LWA1_LAT, LWA1_LON, lwa1.lat*1.0, lwa1.lon*1.0)

    ## Derived from the 2017 Oct 27 LWA-SV SSMIF
    LWASV_ECEF = numpy.array((-1531554.7717322097, -5045440.9839560054, 3579249.988606174))
    LWASV_LAT =   34.348358 * numpy.pi/180
    LWASV_LON = -106.885783 * numpy.pi/180
    LWASV_ROT = numpy.array([[ numpy.sin(LWASV_LAT)*numpy.cos(LWASV_LON), numpy.sin(LWASV_LAT)*numpy.sin(LWASV_LON), -numpy.cos(LWASV_LAT)], 
                             [-numpy.sin(LWASV_LON),                      numpy.cos(LWASV_LON),                       0                   ],
                             [ numpy.cos(LWASV_LAT)*numpy.cos(LWASV_LON), numpy.cos(LWASV_LAT)*numpy.sin(LWASV_LON),  numpy.sin(LWASV_LAT)]])
    
    ##xyz = LWASV_ECEF
    ##rho = xyz - LWA1_ECEF
    ##sez = numpy.dot(LWA1_ROT, rho)
    ##enz = sez[[1,0,2]]
    ##enz[1] *= -1
    
    ##enz += p[0:3]
    ##print(enz, p[3])
    ##enz[1] *= -1
    ##sez = enz[[1,0,2]]
    ##rho = numpy.dot(numpy.linalg.inv(LWA1_ROT), sez)
    ##xyz = rho + LWA1_ECEF
    ##print(xyz, xyz-LWASV_ECEF)
    ##lat, lon, elev = ecef_to_geo(*xyz)
    ##print(LWASV_LAT*180/numpy.pi, lat*180/numpy.pi)
    ##print(LWASV_LON*180/numpy.pi, lon*180/numpy.pi)
    ##print(lwasv.elev, elev)
    

    enz = numpy.array([0.0, 0.0, 0.0])
    enz -= p[0:3]
    enz[1] *= -1
    sez = enz[[1,0,2]]
    rho = numpy.dot(numpy.linalg.inv(LWA1_ROT), sez)
    xyz = rho + LWA1_ECEF
    print(xyz, xyz-LWA1_ECEF)
    lat, lon, elev = ecef_to_geo(*xyz)
    print(LWA1_LAT*180/numpy.pi, lat*180/numpy.pi)
    print(LWA1_LON*180/numpy.pi, lon*180/numpy.pi)
    print(lwa1.elev, elev)


if __name__ == "__main__":
    main(sys.argv[1:])
    