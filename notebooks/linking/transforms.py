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
           'cometary2cartesian', 
           'radec2heliocentric', 'radec2icrfu',
           'icrf2ecliptic',
           'ecliptic2icrf','coordinateTransform']

############################################
# MODULE VARIABLES FROM CONSTANTS
###########################################

OBLRAD = np.deg2rad(cnst.EARTH_OBLIQUITY)
COSOBS = np.cos(OBLRAD)
SINOBL = np.sin(OBLRAD)

ICRF2ECL = np.array([[1., 0., 0.],
                     [0., COSOBS, SINOBL],
                     [0., -SINOBL, COSOBS]], dtype='float64')

ECL2ICRF = np.array([[1., 0., 0.],
                     [0., COSOBS, -SINOBL],
                     [0., SINOBL, COSOBS]], dtype='float64')

PIX2 = np.pi*2.

multiply = np.multiply
divide = np.divide

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
    pos = vec.sphereLineIntercept(los, observer, r_plus_dr)
    
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
    M = ICRF2ECL
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
    M = ECL2ICRF                   
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
    