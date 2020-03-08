# This file is part of the LSST Solar System Processing lsstssp.
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
ephemeris

LSST Solar System Processing routines for
calculating ephemerides of moving objects


Implementation: S. Eggl 20120308
"""

# Accelerators
import numpy as np
import numba

# Accelerated vector operations
import vector

# Constants such as the speed of light and GM
import constants as cn

# Coordinate transfroms
import transforms as tr

# State Propagation routines
import propagate as pr

# External API's
from astroquery.jplhorizons import Horizons

#Interpolation
import scipy.interpolate as spi

# time scale transforms
from astropy.time import Time


__all__ = ['getObserverStates', 'observerStatesFromHorizons',  'icrf2ephemeris', 
           'topocentric2ephemeris', 'state2ephemeris', 'radecResiduals']

############################################
# MODULE SPECIFIC EXCEPTION
###########################################
class Error(Exception):
    """Vector module specific exception."""
    
    pass

###########################################
# OBSERVER STATES 
###########################################

def get_observer_states(observation_epochs,observer_location='I11',ephemeris_dt='1h'):
    """Produce sun-observer state vectors at observation epochs.
    
    Parameters:
    -----------
    observation_epochs         ... Numpy array of observation epochs [JD] 
    observer_location          ... Horizons identifyer of observer location, e.g. 'I11'
    ephemeris_dt               ... Time step for ephemeris query. 
                                   Typically 1h since the actual times will be interpolated later.
    
    Returns:
    --------
    observer_positions         ... Heliocentric observer positions at observation epochs in [au].
    
    
    External Function Requirements:
    -------------------------------

    # Interpolation
    import scipy.interpolate as spi
    
    # time transform
    mjd2jd                         ... change modified Julian date to Julian date, timescale TDB)
    
    # NASA JPL HORIZONS API call wrapper
    observer_states_from_horizons  ... Wrapper function for JPL Horizons state query via astropy
    
    """

    tmin = np.min(observation_epochs)
    tmax = np.max(observation_epochs)
    
    #Start and stop times of the survey
    tstart = 'JD'+str(tmin-1.)
    tstop = 'JD'+str(tmax+1.)

    epochs = np.unique(observation_epochs)


    [observer_jd,observer_xyz,observer_vxyz] = observer_states_from_horizons(observation_epochs,
                                                                             observer_location,
                                                                             tstart,tstop, ephemeris_dt)
        
    # Interpolate heliocentric observer positions to the actual observation epochs
    ir = spi.CubicHermiteSpline(observer_jd, observer_xyz,observer_vxyz, axis=1, extrapolate=None)
    observer_positions = ir(observation_epochs).T
    # Interpolate heliocentric observer velocities to the actual observation epochs
    dirdt=ir.derivative(nu=1)
    observer_velocities = dirdt(observation_epochs).T
    
    return observer_positions, observer_velocities

def observer_states_from_horizons(epochs_of_observation, observer_location, 
                                  tstart, tstop, ephemeris_dt='1h'):
    """Query JPL Horizons via astroquery to get sun-observer state vectors.
    
    Parameters:
    -----------
    observer_location  ... Horizons identifyer of observer location, e.g. 'I11'
    tstart             ... start time for ephemeris in Horizons format, e.g. 'JD2456789.5'
    tstop              ... end time for ephemeris in Horizons format, e.g. 'JD2456799.5'
    ephemeris_dt       ... Time step for ephemeris query. 
                           Typically 1h since the actual times will be interpolated later.
    
    Returns:
    --------
    observer_xyz       ... Heliocentric observer positions at gridded epochs in [au].
    observer_vxyz      ... Heliocentric observer velocities at gridded epochs in [au].
    observer_jd        ... Gridded ephemeris epochs (JD / TDB)
    
    External Function Requirements:
    -------------------------------
    # External API's
    from astroquery.jplhorizons import Horizons
    """
    try:
        # Get observer locations (caution: choose the right plane of reference and direction of the vectors!)
        # check query by copy/pasting the output of print(observer_sun.uri) into a webbrowser if there are problems.
        observer_sun = Horizons(id='Sun', location=observer_location, id_type='majorbody',
                      epochs = {'start':tstart, 'stop':tstop,
                      'step':ephemeris_dt})

        xyz = observer_sun.vectors()['x','y','z']
        vxyz = (observer_sun.vectors())['vx','vy','vz']
        jd = (observer_sun.vectors())['datetime_jd']
        
        #We need the sun-observer vector not the observer-sun vector
        observer_xyz = (-1)*np.array([xyz['x'],xyz['y'],xyz['z']])
        observer_vxyz =(-1)*np.array([vxyz['vx'],vxyz['vy'],vxyz['vz']])
        observer_jd = np.array(jd)
        
    except:
        print("Error in observer_state_from_horizons: potential online ephemeris query failure.")
        raise
        
    return observer_jd, observer_xyz, observer_vxyz

###########################################
# EPHEMERIS CALCULATION
###########################################

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
    
    #numpy functions for math and vector operations
    multiply = np.multiply
    divide = np.divide
    add = np.add
    norm = np.linalg.norm
    dot = np.dot
    mod = np.mod
    arcsin = np.arcsin
    sqrt = np.sqrt
    cos = np.cos
    
    # other numpy functions and constants
    array = np.array
    hstack = np.hstack
    rad2deg = np.rad2deg
    
    #PI * 2
    PIX2=np.pi*2
    

    # Correct for time shift between UTC and TDB
    time=Time(epoch, scale=timescale_epoch, format=time_format)
    
    if(timescale_state != timescale_epoch):
        dt = ((Time(time, scale=timescale_epoch,format=time_format)).value - 
             (Time(time, scale=timescale_state,format=time_format)).value)
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
            r0 = norm(state[0:3]) 
            dt = dt + r0/cn.CAUPD
        
        state_corrected = hstack([state[0:3] - dt*state[3:6] 
                                    # - dt**2*cnst.GM/r0**3*state[0:3], 
                                     , state[3:6]])
       
        # Calculate Right Ascension and Declination
        r1 = norm(state_corrected[0:3]) 
        
        rn = state_corrected[0:3]/r1 
        rdot = dot(rn, state_corrected[3:6])
        
        RA = mod(np.arctan2(rn[1], rn[0])+PIX2, PIX2)
        DEC = arcsin(rn[2])
        

        # Analytic Derivatives
        # dalpha/dt = (x dy/dt - y dx/dt)/(x^2+y^2)
        # ddelta/dt = (dz/dt r - z dr/dt)/(r^2 sqrt(1-z^2/r^2))
        dRAdt = (state_corrected[0]*state_corrected[4]-
                 state_corrected[1]*state_corrected[3]
                 )/(state_corrected[0]**2+state_corrected[1]**2)
        dDECdt = (state_corrected[5]*r1-state_corrected[2]*rdot)/(r1*
                  sqrt(r1**2-state_corrected[3]**2))
        
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
            r0 = norm(state[:,0:3],axis=1)
            # print(dt)
            # print(np.divide(r0,cnst.c_aupd))
            dt = add(dt,divide(r0,cn.CAUPD))
            # print(dt)
            
        state_xyz = array([state[:,0] - multiply(dt,state[:,3]),
                           state[:,1] - multiply(dt,state[:,4]),
                           state[:,2] - multiply(dt,state[:,5])]).T 
        
        r1 = norm(state_xyz[:,0:3],axis=1)                     
        rn = array([divide(state_xyz[:,0],r1),
                    divide(state_xyz[:,1],r1),
                    divide(state_xyz[:,2],r1)]).T
        
        
        #rdot = np.tensordot(rn[:,0:3],state[:,3:6],axes=1)
        rdot = vec.dot2D(rn,state,
                          array([0,1,2],dtype='Int32'),array([3,4,5],dtype='Int32'))
        
        RA = mod(np.arctan2(rn[:,1],rn[:,0])+PIX2, PIX2)
        DEC = arcsin(rn[:,2])
        
        dRAdt = divide((state_xyz[:,0]*state[:,4]-state_xyz[:,1]*state[:,3]),
                        state_xyz[:,0]**2+state_xyz[:,1]**2)
        #dDECdt=divide(divide(state[:,5]),rdot)
        dDECdt = divide(multiply(state[:,5],r1)-multiply(state_xyz[:,2],rdot),
                        multiply(r1,sqrt(r1**2 - state[:,5]**2)))
        
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
        
        radecr = array([rad2deg(RA), rad2deg(DEC),
                           rad2deg(multiply(dRAdt,cos(DEC))),rad2deg(dDECdt),
                           r1, rdot]).T
    else:
        radecr = array([RA, DEC,  multiply(dRAdt,cos(DEC)), dDECdt, r1, rdot]).T
    
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
        state_icrf = tr.ecliptic2icrf(state)
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

def radecResiduals(df, epoch, state_asteroid, output_units='deg', **kwargs):
    """Calculate O-C values in Right Ascension (RA) and Declination (DEC) for a given asteroid state. 
    The state is propagated to all observation epochs and the corresponding RA and DEC values are compared
    to the corresponding observations. Heliocentric ecliptic states for the observer are required for
    every obserbation epoch. 
    
    Parameters:
    -----------
    df                ... Pandas DataFrame containing nightly RA and DEC [deg], time [JD, MJD] UTC,
                          heliocentric ecliptic observer positions and velocities [au]
                
    epoch             ... epoch of asteroid state [JD, MJD], timescale: TDB
    state_asteroid    ... heliocentric ecliptic state of asteroid at epoch (x,y,z,vx,vy,vz), [au, au/day]
    
    Keyword arguments:
    ------------------
    output_units      ... units for O-C results: 'deg', 'arcsec', 'rad'
    epoch_timescale   ... time scale for observation epoch ('utc', 'tdb'), default utc
    state_timescale   ... time scale for asteroid and observer state epoch ('utc', 'tdb'), default tdb
    deg               ... True (default): angles in degrees, False: angles in radians
    lttc              ... True (default): correct for light travel time \
                          (needs entire state including velocities) 
    propagator        ... propagation algorithm for asteroid state propagation: 
                          '2body' (default), 'linear', 'nbody'                      
    Returns:
    --------
    rms               ... root mean square (RMS) of RA and DEC O-Cs [arcseconds]
    dra               ... Right Ascension O-C values for all observations [arcseconds]
    ddec              ... Declination O-C values for all observations [arcseconds]
    """
    dt = epoch-df['time'].values

    ephemeris = []
    ephemeris_app = ephemeris.append
    nobs = len(dt)
    for i in range(nobs):
        state_observer = df[['x_obs', 'y_obs', 'z_obs','vx_obs', 'vy_obs', 'vz_obs']].values[i]
    
    # propagate orbit to all observation time
        pstate = pr.propagateState(state_asteroid[0:3],state_asteroid[3:6], 
                                     epoch, df['time'].values[i], propagator='2body')
        state_asteroid_prop = np.array(pstate[0:2]).flatten()
        ephemeris_app(state2ephemeris(epoch, state_asteroid_prop, state_observer, frame='ecliptic', lttc=True, timescale_state='tdb',
                                        timescale_epoch='utc')) 
    
    # O-C
    ephemeris_array = np.array(ephemeris)
    
#     print('observed', np.array([df['RA'].values, df['DEC'].values]).T)
#     print('calculated', np.array(ephemeris)[:,0:2])
    
    dra = df['RA'].values - ephemeris_array[:,0]
    ddec =  df['DEC'].values - ephemeris_array[:,1]
    
    dradec = np.array([dra, ddec]).flatten()
    
    rms = np.sqrt(np.dot(dradec.T, dradec)/nobs)

    if(output_units == 'deg'):
        return rms, dra, ddec
    elif(output_units == 'rad'):
        return np.deg2rad(rms), np.deg2rad(dra), np.deg2rad(ddec)
    elif(output_units == 'arcsec'):
        return rms*3600, dra*3600, ddec*3600
    else:
        raise Exception('Error in radecResiduals: unknown output unit.')