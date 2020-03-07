# This file is part of the LSST Solar System Processing lsst_dm_ssp.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""
transforms

LSST Solar System Processing

Transformations between common coordinate and time systems
Implementation: Python 3.6, S. Eggl 20191115
"""
# Accelerators
import numpy as np
import numba

# NASA NAIF Spice wrapper 
import spiceypy as sp

# Constants such as the speed of light and GM
import constants as cnst

# time scale transforms
from astropy.time import Time

import vector as vec

__all__ = ['mjd2jd', 'jd2mjd', 'frameCheck', 
           'keplerian2cartesian', 'cartesian2keplerian',
           'cartesian2cometary','cometary2keplerian',
           'cometary2cartesian', 'sphereLineIntercept',
           'radec2heliocentric', 'radec2icrfu',
           'icrf2ephemeris', 'topocentric2ephemeris',
           'state2ephemeris', 'icrf2ecliptic',
           'ecliptic2icrf','coordinateTransform']

############################################
# MODULE VARIABLES FROM CONSTANTS
###########################################

OBLRAD=np.deg2rad(cnst.EARTH_OBLIQUITY)
COSOBS=np.cos(OBLRAD)
SINOBL=np.sin(OBLRAD)

ICRF2ECL = np.array([[1., 0., 0.],
                     [0., COSOBS, SINOBL],
                     [0., -SINOBL, COSOBS]], dtype='float64')

ECL2ICRF = np.array([[1., 0., 0.],
                     [0., COSOBS, -sinobl],
                     [0., SINOBL, COSOBS]], dtype='float64')

PIX2 = np.pi*2.

multiply=np.multiply
divide=np.divide

############################################
# MODULE SPECIFIC EXCEPTION
###########################################

class Error(Exception):
    """Transforms module specific exception."""
    pass

############################################
# TIME TRANSFORMS
###########################################

def mjd2jd(mjd):
    """Transform modified Julian date to Julian date.
    
    Parameters:
    -----------
    mjd ... epoch [modified Julian date]
    
    
    Returns:
    --------
    jd ... epoch [Julian date]
    """
    jd = mjd+2400000.5
    return jd

def jd2mjd(jd):
    """Transform Julian date to modifiedJulian date.
    
    Parameters:
    -----------
    jd ... epoch [Julian date]
    
    Returns:
    --------
    mjd ... epoch [modified Julian date]
    """
    
    mjd=jd-2400000.5
    return mjd

############################################
# ORBITAL ELEMENT TRANSFORMS
###########################################

def frameCheck(cart, frame):
    """Transform coordinates into the correct (ecliptic) frame
    to convert to and from orbital elements.
    
    Parameters:
    -----------
    cart ... Cartesian state vector (x,y,z,vx,vy,vz)
    frame... string: 'ICRF' or 'ecliptic'
    
    Returns:
    --------
    ecart ... Cartesian state vector in ecliptic coordinates
    """   
    
    ecart = []
    if(frame=='ICRF' or frame=='icrf'):
        ecart.extend(icrf2ecliptic(cart[0:3]))
        ecart.extend(icrf2ecliptic(cart[3:6]))          
    elif(frame=='ecliptic'):
        ecart = cart
    else:
        raise Exception('Error in Framecheck: Coordinate frame not supported') 
                       
    return np.array(ecart)  


def keplerian2cartesian(epoch, kep, frame='ecliptic', mu=cnst.GM):
    """Uses spiceypy to convert Keplerian orbital elements
    to heliocentric Cartesian states.

    Parameters:
    -----------
    epoch ... epoch of orbital elements [JD]
    kep   ... orbital element array [a,e,inc(deg),peri/w(deg),node(deg),M(deg)]
    frame ... Coordinate frame of Cartesian states: 'ecliptic', 'icrf'
    mu    ... Gravitational parameter

    Returns:
    --------
    cart ... heliocentric ecliptic Cartesian state (x,y,z,vx,vy,vz).

    External dependencies:
    ----------------------
    spiceypy (imported as sp)
    numpy (imported as np)


    """
    q = kep[0]*(1-kep[1])

#   Input for spiceypy.conics:
#   q    = pericenter distance
#   e    = eccentricity
#   i    = inclination (deg)
#   node = longitude of the ascending node (deg)
#   w    = argument of pericenter (deg)
#   M    = mean anomaly at epoch (deg)
#   T0   = epoch
#   mu   = gravitational parameter

    cart = sp.conics(np.array([q, kep[1], np.deg2rad(kep[2]),
                               np.deg2rad(kep[4]), np.deg2rad(kep[3]),
                               np.deg2rad(kep[5]), 0, mu]), 0)
                   
    res = frameCheck(cart, frame)
    return res


def cartesian2keplerian(epoch, state, frame='ecliptic', mu=cnst.GM):
    """Uses spiceypy to convert heliocentric
    cartesian states to Keplerian orbital elements.

    Parameters:
    -----------
    epoch ... epoch of orbital elements [JD]
    state ... Cartesian state (x,y,z,vx,vy,vz)
    frame ... coordinate frame of Cartesian states: 'ecliptic', 'icrf'
    mu ... Gravitational parameter

    Returns:
    --------
    kep ... orbital elements array
            a    = pericenter distance
            e    = eccentricity
            i    = inclination (deg)
            w    = argument of pericenter (deg)
            node = longitude of the ascending node (deg)
            M    = mean anomaly at epoch (deg)
            T0   = epoch
            mu   = gravitational parameter

    External dependencies:
    ----------------------
    spiceypy (imported as sp)
    numpy (imported as np)

    """

#   Output for spiceypy.oscelt:
#   q    = pericenter distance
#   e    = eccentricity
#   i    = inclination (deg)
#   node = longitude of the ascending node (deg)
#   w    = argument of pericenter (deg)
#   M    = mean anomaly at epoch (deg)
#   T0   = epoch
#   mu   = gravitational parameter

    estate = frameCheck(state, frame)                
                   
    oscelt = sp.oscelt(np.array(estate), epoch, mu)

    kep = []
    # semimajor axis a from q
    kep.append(oscelt[0]/(1-oscelt[1]))
    # eccentricity
    kep.append(oscelt[1])
    # inclination
    kep.append(np.rad2deg(oscelt[2]))
    # w: argument of pericenter
    kep.append(np.rad2deg(oscelt[4]))
    # node
    kep.append(np.rad2deg(oscelt[3]))
    # mean anomaly
    kep.append(np.rad2deg(oscelt[5]))

    return kep, epoch


def cartesian2cometary(epoch, state, frame='ecliptic', mu=cnst.GM):
    """Spiceypy conversion from heliocentric
    cartesian states to cometary orbital elements.

    Parameters:
    -----------
    epoch ... epoch of orbital elements [time units]
    state ... heliocentric ecliptic cartesian state (x,y,z,vx,vy,vz)
    frame ... Coordinate frame of Cartesian states: 'ecliptic', 'icrf'
    mu    ... Gravitational parameter
    
    Returns:
    --------
    com ... orbital elements array
            q    = pericenter distance
            e    = eccentricity
            i    = inclination (deg)
            node = longitude of the ascending node (deg)
            w    = argument of pericenter (deg)
            Tp   = time of (next) pericenter passage [time units]
            
    epoch ... epoch of orbital elements
    period... orbital period [time units]

    External dependencies:
    ----------------------
    spiceypy (imported as sp)
    numpy (imported as np)

    """

#   Output for spiceypy.oscltx:
#   q    = pericenter distance
#   e    = eccentricity
#   i    = inclination (deg)
#   node = longitude of the ascending node (deg)
#   w    = argument of pericenter (deg)
#   M    = mean anomaly at epoch (deg)
#   T0   = epoch
#   mu   = gravitational parameter
#   NU   = True anomaly at epoch.
#   a    = Semi-major axis. A is set to zero if it is not computable.
#   TAU  = Orbital period. Applicable only for elliptical orbits. Set to zero otherwise.

    estate = frameCheck(state, frame)
                   
    oscltx = sp.oscltx(estate, 0, mu)

    com = []
    com_add=com.append
    # pericenter distance q
    com_add(oscltx[0])
    # eccentricity
    com_add(oscltx[1])
    # inclination
    com_add(np.rad2deg(oscltx[2]))
    # node
    com_add(np.rad2deg(oscltx[3]))
    # w: argument of pericenter
    com_add(np.rad2deg(oscltx[4]))
    # period
    period = oscltx[10]
    # mean anomaly
    man = oscltx[5]
    print(np.rad2deg(man))
    # epoch of pericenter passage
    com_add(epoch-man/PIX2*period)

    return com, epoch, period

def cometary2keplerian(epoch, elements, mu=cnst.GM):
    """Convert cometary orbital elements to Keplerian orbital elements

    Parameters:
    -----------
    epoch      ... epoch of cometary elements [JD]
    elements   ... cometary elements [q[au],e,inc[deg],node[deg],peri[deg],tp[JD]]

    Optional:
    ---------
    mu ... Gravitational parameter (e.g. k**2*(M+m))

    Returns:
    --------
    kep ... Keplerian orbital elements
            [a, e, i[deg], w[deg], node[deg], M[deg]]

    """
    ele=np.array(elements)
    
    a = ele[0] / (1. - ele[1])
    
    M = np.sqrt(mu/a**3)*(epoch - ele[5])
    while(M < 0):
        M = M+PIX2
    while(M > PIX2):
        M = M-PIX2

    # a, e, i, w, node, M
    kep = [a, ele[1], ele[2], ele[4], ele[3], np.rad2deg(M)]

    return kep, epoch

def cometary2cartesian(epoch, com, frame='ecliptic', mu=cnst.GM):
    """Uses spiceypy to convert cometary orbital elements to
    HELIOCENTRIC (!) Cartesian states

    Parameters:
    -----------
    epoch    ... epoch of orbital elements [JD]
    com      ... cometary element array [q,e,inc(deg),node(deg),peri(deg),tp(JD)]]
    frame    ... reference frame of output Cartesian state ('ecliptic', 'icrf')
    mu       ... Gravitational parameter

    Returns:
    --------
    cart     ... Cartesian state (x,y,z,vx,vy,vz)

    External dependencies:
    ----------------------
    spiceypy (imported as sp)
    numpy (imported as np)
    cometary2keplerian

    """
    kep = cometary2keplerian(epoch, com, mu)[0]

#   Input for spiceypy.conics:
#   q    = pericenter distance
#   e    = eccentricity
#   i    = inclination (rad)
#   node = longitude of the ascending node (rad)
#   w    = argument of pericenter (rad)
#   M    = mean anomaly at epoch (rad)
#   T0   = epoch
#   mu   = gravitational parameter

    cart = sp.conics(np.array([com[0], com[1], np.deg2rad(com[2]),
                     np.deg2rad(com[3]), np.deg2rad(com[4]),
                     np.deg2rad(kep[5]), 0, mu]), 0)
            
    res = frameCheck(cart, frame)
    return res          

############################################
# RA DEC TRANSFORMS
###########################################
                     
def sphereLineIntercept(l, o, r):
    """Calculate intercept point between line y = l.x + o
       and a sphere around the center of origin of the
       coordinate system: c=0

       Parameters:
       ------------
       l ... vector of line of sight
       o ... observer position
       r ... vector of distances (radii on the heliocecntric sphere)

       Returns:
       --------
       x_intercept  ... position of intersection
                        between the line and the sphere
                        along the line of sight
    """

    x_intercept = np.full(np.shape(l), np.nan)

    ln = unitVector(l)

    for i in range(len(l[:, 0])):

        lo = np.vdot(ln[i, :], o[i, :])

        if (r.size == 1):
            r2 = r*r
        else:
            r2 = r[i]*r[i]

        discrim = lo**2 - (np.vdot(o[i, :], o[i, :]) - r2)
        if(discrim >= 0):
            # line and sphere do actually intersect
            d = -lo + np.sqrt(discrim)
            x_intercept[i, :] = o[i, :]+d*ln[i, :]

    return x_intercept
                     
                     
def radec2heliocentric(tref, epoch, ra, dec, r=1, drdt=0, deg=True, frame='ecliptic'):
    """Transform topocentric Right Ascension (RA) and Declination (DEC)
    to heliocentric coordinates.
    
    Parameters:
    -----------
    tref   ... reference time (used to calculate radial distance)
    epoch  ... epoch of observation
    ra     ... Right Ascension, default [deg]
    dec    ... Declination, default [deg]
    r      ... heliocentric distance [au]
    drdt   ... heliocentric radial velocity [au/day]
    deg    ... True: angles in degrees, False: angles in radians
    frame  ... reference frame for coordinates ('icrf' or 'ecliptic')
    
    Returns:
    --------
    pos ... heliocientric positions

    """
    # speed of light in au/day
    c_aupd = cnst.CAUPD

    # Transform RADEC observations into positions on the unit sphere (US)
    xyz = radec2icrfu(ra, dec, deg)

    # Those are the line of sight (LOS) vectors
    los = np.array([xyz[0], xyz[1], xyz[2]]).T

    # Calculate how much the heliocentric distance changes
    # during the obsevations based on assumed dr/dt
    dt = tref-epoch
    dr = drdt*dt
    r_plus_dr = r+dr

    # Heliocentric postions of the observed asteroids
    pos = sphereLineIntercept(los, observer, r_plus_dr)
    
    if(frame == 'ecliptic'):
        posh = ecliptic2icrf(pos)
    elif(frame == 'icrf'):
        posh = pos
    else:
        raise Exception('Error in radec2heliocentric_xyz: frame unknown.')
                        
    return posh


# def heliocentric2radec(epoch, state_ast, state_obs, time_format='mjd', 
#                              time_scale='utc', frame='icrf', deg=True, lttc=True):
#     """Transform heliocentric coordinates to 
#     topocentric Right Ascension (RA) and Declination (DEC)
#     Asteroid and observatory states are assumed to have TDB timescale.
    
#     Parameters:
#     -----------
#     epoch             ... epoch of states 
#     state_ast         ... heliocentric state vectors of asteroid  [au, au/day]
#     state_observer    ... heliocentric state vectors of observer  [au, au/day]
#     time_format       ... time format of epoch of state ('mjd','jd', astropy.time format)
#     time_scale        ... time scale for epoch ('utc', 'tdb')
#     frame             ... coordinate frame for states ('icrf','ecliptic')
#     deg               ... True: return Right Ascension and Declination in [deg]
#                           False: return Right Ascension and Declination in [rad]
#     Returns:
#     --------
#     ra ... Right Ascension [deg]
#     dec ... Declination [deg]
#     """
    
#     #observer to asteroid vectors
#     xyz_oa=xyz_ast-xyz_obs
    
  
#     if(frame == 'ecliptic'):
#         state_a=np.array([ecliptic2icrf(state_ast[0:3]),ecliptic2icrf(state_ast[3:6])])
#         state_b=np.array([ecliptic2icrf(state_ast[0:3]),ecliptic2icrf(state_ast[3:6])])
#     elif(frame == 'icrf'):
#         pass
#     else:
#         raise Exception('Error in heliocentric2radec: unknown frame')
                        
    
# #     icrf2radec(epoch, state, time_format=time_format, time_scale=time_scale, 
# #                deg=True, lttc=False)
    
#     return xyz_oa


#@numba.njit
def radec2icrfu(ra, dec, deg=True):
    """Convert Right Ascension and Declination to ICRF xyz unit vector.
    Geometric states on unit sphere, no light travel time/aberration correction.

    Parameters:
    -----------
    ra ... Right Ascension [deg]
    dec ... Declination [deg]
    deg ... True: angles in degrees, False: angles in radians
    
    Returns:
    --------
    x,y,z ... 3D vector of unit length (ICRF)
    """
    
    if(deg):
        a = np.deg2rad(ra)
        d = np.deg2rad(dec)
    else:
        a = np.array(ra)
        d = np.array(dec)
       
    cosd = np.cos(d)
    x = cosd*np.cos(a)
    y = cosd*np.sin(a)
    z = np.sin(d)

    return np.array([x, y, z])


def icrf2ephemeris(epoch, state, timescale_epoch='utc', 
                   timescale_state='tdb', time_format='mjd', deg=True, lttc=True):
    """Transform topocentric ICRF states to
    Right Ascension (RA) and Declination (DEC) observations.
    
    Parameters:
    -----------
    epoch             ... epoch of RADEC observation (astropy.time object)
    state             ... topocentric state vector of object (ICRF) [au] 
                      ... (given at epoch in TDB)
    epoch_timescale   ... time scale for epoch ('utc', 'tdb'), default utc
    state_timescale   ... time scale for state epoch ('utc', 'tdb'), default tdb
    deg               ... True (default): angles in degrees, False: angles in radians
    lttc              ... True (default): correct for light travel time \
                          (needs entire state including velocities)
    
    Returns:
    --------
    RA               ... Right Ascension [default deg]
    DEC              ... Declination [default deg]
    dRA/dt*cos(DEC)  ... sky plane motion in RA [default deg/day]
    dDEC/dt          ... sky plane motion in DEC [default deg/day]
    r                ... distance to object [au]
    dr/dt            ... line of sight velocity [au/day]

    """
    
    # Correct for time shift between UTC and TDB
    time=Time(epoch,scale=timescale_epoch, format=time_format)
    
    if(timescale_state != timescale_epoch):
        dt = ((Time(time,scale=timescale_epoch,format=time_format)).value - 
             (Time(time,scale=timescale_state,format=time_format)).value)
    else:
        dt = 0.
      
    if(state.ndim == 1):  
        # Check if we have the full state 
        if(len(state)!=6):
                   raise Exception('Error in icrf2radec: Full state including velocities \
                        needed for light travel time and timescale correcion.')           
           
        # Add Light travel time correction to TDB-UTC
        if(lttc):
            # determine state corrected for light travel time
            r0 = np.linalg.norm(state[0:3]) 
            dt = dt + r0/cnst.c_aupd
        
        state_corrected = np.hstack([state[0:3] - dt*state[3:6] 
                                    # - dt**2*cnst.GM/r0**3*state[0:3], 
                                     , state[3:6]])
       
        # Calculate Right Ascension and Declination
        r1 = np.linalg.norm(state_corrected[0:3]) 
        
        rn = state_corrected[0:3]/r1 
        rdot = np.dot(rn,state_corrected[3:6])
        
        RA = np.mod(np.arctan2(rn[1],rn[0])+pix2,pix2)
        DEC = np.arcsin(rn[2])
        

        # Analytic Derivatives
        # dalpha/dt = (x dy/dt - y dx/dt)/(x^2+y^2)
        # ddelta/dt = (dz/dt r - z dr/dt)/(r^2 sqrt(1-z^2/r^2))
        dRAdt = (state_corrected[0]*state_corrected[4]-
                 state_corrected[1]*state_corrected[3]
                 )/(state_corrected[0]**2+state_corrected[1]**2)
        dDECdt = (state_corrected[5]*r1-state_corrected[2]*rdot)/(r1*
                  np.sqrt(r1**2-state_corrected[3]**2))
        
#         print('dRAdt,dDECdt, analytics') 
#         print([np.rad2deg(dRAdt)*np.cos(DEC),np.rad2deg(dDECdt)])
        
         # Finite Differences for derivatives
#         RA_later = np.mod(np.arctan2(state[1],state[0])+pix2,pix2)
#         DEC_later = np.arcsin(state[2]/r0)
#         dRAdt=(RA_later-RA)/dt
#         dDECdt=(DEC_later-DEC)/dt
       
#         print('dRAdt,dDECdt, finite diff')     
#         print([np.rad2deg(dRAdt)*np.cos(DEC),np.rad2deg(dDECdt)])
        
    else:
        if(state.shape[1]!=6):
                raise Exception('Error in icrf2radec: Full state including velocities \
                        needed for light travel time and timescale correcion.')  
        
         # Add Light travel time correction to TDB-UTC
        if(lttc):
            # determine state corrected for light travel time
            r0 = np.linalg.norm(state[:,0:3],axis=1)
            # print(dt)
            # print(np.divide(r0,cnst.c_aupd))
            dt = np.add(dt,np.divide(r0,cnst.c_aupd))
            # print(dt)
            
        state_xyz = np.array([state[:,0] - multiply(dt,state[:,3]),
                              state[:,1] - multiply(dt,state[:,4]),
                              state[:,2] - multiply(dt,state[:,5])]).T 
        
        r1 = np.linalg.norm(state_xyz[:,0:3],axis=1)                     
        rn = np.array([divide(state_xyz[:,0],r1),
                     divide(state_xyz[:,1],r1),
                     divide(state_xyz[:,2],r1)]).T
        
        
        #rdot = np.tensordot(rn[:,0:3],state[:,3:6],axes=1)
        rdot = vec.dot2D(rn,state,
                          np.array([0,1,2],dtype='Int32'),np.array([3,4,5],dtype='Int32'))
        
        RA = np.mod(np.arctan2(rn[:,1],rn[:,0])+pix2,pix2)
        DEC = np.arcsin(rn[:,2])
        
        dRAdt = divide((state_xyz[:,0]*state[:,4]-state_xyz[:,1]*state[:,3]),
                        state_xyz[:,0]**2+state_xyz[:,1]**2)
        #dDECdt=divide(divide(state[:,5]),rdot)
        dDECdt = divide(multiply(state[:,5],r1)-multiply(state_xyz[:,2],rdot),
                        multiply(r1,np.sqrt(r1**2 - state[:,5]**2)))
        
    if(deg):
#         print('RA, DEC')
#         print(np.rad2deg(RA))
#         print(np.rad2deg(DEC))
#         print('dRAdt,dDecdt')
#         print(np.rad2deg(multiply(dRAdt,np.cos(DEC))))
#         print(np.rad2deg(dDECdt))
#         print('r, rdot')
#         print(r1)
#         print(rdot)
        
        radecr = np.array([np.rad2deg(RA), np.rad2deg(DEC),
                           np.rad2deg(multiply(dRAdt,np.cos(DEC))),np.rad2deg(dDECdt),
                           r1, rdot]).T
    else:
        radecr = np.array([RA, DEC,  multiply(dRAdt,np.cos(DEC)), dDECdt, r1, rdot]).T
    
    return radecr

def topocentric2ephemeris(epoch, state, frame='icrf', **kwargs):
    """Transform topocentric ICRF states to
    Right Ascension (RA) and Declination (DEC) observations.
    
    Parameters:
    -----------
    epoch             ... epoch of RADEC observation (astropy.time object)
    state             ... topocentric state vector of object [au] 
    frame             ... reference frame ('icrf' or 'ecliptic')
                      ... (given at epoch in TDB)
    Kwargs:
    -------
    epoch_timescale   ... time scale for epoch ('utc', 'tdb'), default 'utc'
    state_timescale   ... time scale for state epoch ('utc', 'tdb'), default 'tdb'
    deg               ... True (default): angles in degrees, False: angles in radians
    lttc              ... True (default): correct for light travel time (needs full state including velocities)
    
    Returns:
    --------
    RA               ... Right Ascension [default deg]
    DEC              ... Declination [default deg]
    dRA/dt*cos(DEC)  ... sky plane motion in RA [default deg/day]
    dDEC/dt          ... sky plane motion in DEC [default deg/day]
    r                ... distance to object [au]
    dr/dt            ... line of sight velocity [au/day]
    """

    if (frame == 'ecliptic'):    
        state_icrf = ecliptic2icrf(state)
        ephemeris = icrf2ephemeris(epoch, state_icrf, **kwargs)
        
    elif(frame == 'icrf'):
        ephemeris = icrf2ephemeris(epoch, state, **kwargs)
         
    else:
        raise Exception('Error in coordinate_transform: \
                         3D positions or 6D state vector required') 
        
    return ephemeris   


def state2ephemeris(epoch, state_asteroid, state_observer, **kwargs):
    """Transform topocentric ICRF states to
    Right Ascension (RA) and Declination (DEC) observations.
    
    Parameters:
    -----------
    epoch             ... epoch of RADEC observation (astropy.time object)
    state_asteroid    ... state vector of asteroid [au, au/day] 
    state_observer    ... state vector of observer [au, au/day]    
    
    Kwargs:
    -------
    frame             ... reference frame for both states ('icrf' or 'ecliptic'), default 'icrf'
    epoch_timescale   ... time scale for epoch ('utc', 'tdb'), default 'utc'
    state_timescale   ... time scale for state epoch ('utc', 'tdb'), default 'tdb'
    deg               ... True: angles in degrees, False: angles in radians
    lttc              ... True: correct for light travel time (needs full state including velocities)
    
    Returns:
    --------
    RA               ... Right Ascension [default deg]
    DEC              ... Declination [default deg]
    dRA/dt*cos(DEC)  ... sky plane motion in RA [default deg/day]
    dDEC/dt          ... sky plane motion in DEC [default deg/day]
    r                ... distance to object [au]
    dr/dt            ... line of sight velocity [au/day]
    """

    # observer to asteroid vectors
    topocentric_state = state_asteroid - state_observer
    # calculate ephemeris
    ephemeris = topocentric2ephemeris(epoch, topocentric_state, **kwargs)
        
    return ephemeris 
        
# def radec2eclip(ra, dec, deg=True):
#     """Convert Right Ascension and Declination to ecliptic xyz unit vector

#     Parameters:
#     -----------
#     ra  ... Right Ascension
#     dec ... Declination
#     deg ... True: angles in degrees, False: angles in radians

#     Returns:
#     --------
#     x_ecl ... 3D vectors of unit length (ecliptic)
#     """
#     x_icrf = radec2icrfu(ra, dec, deg)
#     x_ecl = icrf2ecliptic(x_icrf)
#     return x_ecl

# def ecliptic2radec(x_ecl, deg=True):
#     """Convert ecliptic xyz unit vector to Right Ascension and Declination (ICRF)

#     Parameters:
#     -----------
#     x_ecl ... 3D vectors of unit length (ecliptic coordinate system)
#     deg   ... True: angles in degrees, False: angles in radians

#     Returns:
#     --------
#     ra ... Right Ascension
#     dec ... Declination

#     """
#     x_icrf = ecliptic2icrf(x_ecl)
#     radec = icrf2radec(x_icrf, deg=deg)
#     return radec

def icrf2ecliptic(x_icrf):
    """Convert vector in ICRF coordinate system 
    to vector in ecliptic coordinate system.
    
    Parameters:
    -----------
    x_icrf      ... 1D or 2D numpy array, position/state vectors in ICRF coordinate system
 
    Returns:
    --------
    x_ecl       ... 1D or 2D numpy array, position/state vectors in ecliptic coordinate system
    
    External:
    ---------
    numpy
    coordinate_transform
    """ 
                      
    # tranfromation matrix ecliptic to ICRF coordinate system
    M = icrf2ecl
    x_ecl = coordinateTransform(M, x_icrf)                     
    return x_ecl

def ecliptic2icrf(x_ecl):
    """Convert vector in ecliptic coordinate system 
    to vector in ICRF coordinate system.
    
    Parameters:
    -----------
    x_ecl     ... 1D or 2D numpy array, position/state vectors in ecliptic coordinate system
 
    Returns:
    --------
    x_icrf    ... 1D or 2D numpy array, position/state vectors in ICRF coordinate system
    
    External:
    ---------
    numpy
    coordinate_transform
    """
    # tranfromation matrix ecliptic to ICRF coordinate system
    M = ecl2icrf                   
    x_icrf = coordinateTransform(M, x_ecl)
    return x_icrf

def coordinateTransform(M, x_in):
    """Convert 3D and 6D vectors or array of vectors between coordinates frames by multiplying with 
    a transformation function M. 
    
    Parameters:
    -----------
    M        ... transformation matrix, 3x3 numpy array
    x_in     ... 1D or 2D numpy array, 3 or 6 entries per row (positions / states)

 
    Returns:
    --------
    x_out    ... 1D or 2D numpy array, 3 or 6 entries per row (transformed positions / states) 
    """
    matmul=np.matmul
    MT=M.T
    
    if(x_in.ndim == 1):      
            if(x_in.size == 3):
                x_out = matmul(M, x_in)  
            elif(x_in.size == 6 ):  
                x_out = np.ravel(np.array([matmul(M, x_in[0:3]),matmul(M,x_in[3:6])]))
            else:
                raise Exception('Error in coordinate_transform: \
                                 3D positions or 6D state vector required')               
    else:
            print('x_in.shape',x_in.shape)
            if(x_in.shape[1] == 3):  
                x_out = matmul(x_in[:,0:3],MT)   
            elif (x_in.shape[1] == 6):
                x_out = np.hstack([matmul(x_in[:,0:3],MT),matmul(x_in[:,3:6],MT)])
            else:
                raise Exception('Error in coordinate_transform: \
                                 3D positions or 6D state vector required') 
    return x_out
    