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
lsstssp

LSST Solar System Processing routines for
linking of observations of moving objects.

Formerly known as MOPS. Algorithm: Based on HelioLinC (Holman et al. 2018)
we transform topocentric observations to heliocentric states assuming a
distance and radial velocity. The resulting 3D positions are collected into
tracklets. Tracklets contain at least two observations and can,
thus, be used to create velocity vectors. A tracklet + velocity vector is
called an "arrow". Arrows are propagated to a common epoch using spiceypy's
2body propagator, and then clustered using dbscan.

Implementation: S. Eggl 20191215
"""

# Accelerators
import numpy as np
import numba

# Database
# import pandas as pd

# Orbital Dynamics
import spiceypy as sp

# Clustering
import scipy.spatial as scsp
# import sklearn.cluster as cluster

__all__ = ['norm', 'unit_vector', 'rotate_vector', 'sphere_line_intercept',
           'radec2icrfu', 'RaDec2IcrfU_deg', 'RaDec2IcrfU_rad', 'radec2eclip',
           'SelectTrackletsFromObsData', 'CullSameTimePairs',
           'create_heliocentric_arrows', 'propagate_arrows_linear',
           'propagate_arrows_great_circle', 'propagate_arrows_2body'
           ]

############################################
# VECTOR OPERATIONS
###########################################
@numba.njit
def norm(v):
    """Calculate 2-norm for vectors 1D and 2D arrays.

    Parameters:
    -----------
    v ... vector or 2d array of vectors

    Returns:
    --------
    u ... norm (length) of vector(s)

    """

    if(v.ndim == 1):
        n = np.vdot(v, v)
    elif(v.ndim == 2):
        lv = len(v[:, 0])
        n = np.zeros(lv)
        for i in range(lv):
            n[i] = np.vdot(v[i, :], v[i, :])
    else:
        raise TypeError

    return np.sqrt(n)


@numba.njit
def unit_vector(v):
    """Normalize vectors (1D and 2D arrays).

    Parameters:
    -----------
    v ... vector or 2d array of vectors

    Returns:
    --------
    u ... unit lenght vector or 2d array of vectors of unit lenght

    """

    if(v.ndim == 1):
        u = v/norm(v)
    elif(v.ndim == 2):
        lv = len(v[:, 0])
        dim = len(v[0, :])
        u = np.zeros((lv, dim))
        for i in range(lv):
            n = norm(v[i, :])
            for j in range(dim):
                u[i, j] = v[i, j]/n
    else:
        raise TypeError

    return u


def rotate_vector(angle, axis, vector):
    """Rotate vector about arbitrary axis by angle[rad].

    Parameters:
    -----------
    angle  ... rotation angle [rad]
    axis   ... rotation axis: array (n,3)
    vector ... vector to be rotated: array(n,3)
    """
    sina = np.sin(angle)
    cosa = np.cos(angle)
    u = unit_vector(axis)
    print('u:', np.shape(u), 'v:', np.shape(vector))
    uxv = np.cross(u, vector)
    uuv = u*(np.vdot(u, vector))
    vrot = uuv.T+cosa*np.cross(uxv, u).T+sina*uxv.T
    return vrot.T


# @numba.njit
def sphere_line_intercept(l, o, r):
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

    ln = unit_vector(l)

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


############################################
# COORDINATE TRANSFORMS
###########################################

def radec2icrfu(ra, dec, **kwargs):
    """Convert Right Ascension and Declination to ICRF xyz unit vector

    Parameters:
    -----------
    ra ... Right Ascension
    dec ... Declination

    Keyword Arguments:
    ------------------
    deg ... Are angles in degrees: True or radians: False

    Returns:
    --------
    x,y,z ... 3D vector of unit length (ICRF)
    """
    options = {'deg': False}

    options.update(kwargs)

    if (options['deg']):
        a = np.deg2rad(ra)
        d = np.deg2rad(dec)
    else:
        a = ra
        d = dec

    cosd = np.cos(d)
    x = cosd*np.cos(a)
    y = cosd*np.sin(a)
    z = np.sin(d)

    return np.array([x, y, z])


@numba.njit
def RaDec2IcrfU_deg(ra, dec):
    """Convert Right Ascension and Declination to ICRF xyz unit vector

    Parameters:
    -----------
    ra ... Right Ascension [deg]
    dec ... Declination [deg]

    Returns:
    --------
    x,y,z ... 3D vector of unit length (ICRF)
    """

    a = np.deg2rad(ra)
    d = np.deg2rad(dec)

    cosd = np.cos(d)
    x = cosd*np.cos(a)
    y = cosd*np.sin(a)
    z = np.sin(d)

    return np.array([x, y, z])


@numba.njit
def RaDec2IcrfU_rad(ra, dec):
    """Convert Right Ascension and Declination to ICRF xyz unit vector

    Parameters:
    -----------
    ra ... Right Ascension [rad]
    dec ... Declination [rad]

    Returns:
    --------
    x,y,z ... 3D vector of unit length (ICRF)
    """

    cosd = np.cos(dec)
    x = cosd*np.cos(ra)
    y = cosd*np.sin(ra)
    z = np.sin(dec)

    return np.array([x, y, z])

def IcrfHel2RaDecTopo_deg(xyz_hel_ast,xyz_hel_observer):
    """Transform heliocentric ICRF coordinates to 
    topocentric Right Ascension (RA) and Declination (DEC)
    
    Parameters:
    --------
    xyz_hel_ast      ... 3D heliocentric position vectors of asteroid (ICRF) [au]
    xyz_hel_observer ... 3D heliocentric position vectors of observer (ICRF) [au]
    
    Returns:
    --------
    ra ... Right Ascension [deg]
    dec ... Declination [deg]

    """
    pix2=np.pi*2
    
    #observer to asteroid vectors
    xyz_obs=xyz_ast-xyz_observer
    
    r=np.linalg.norm(xyz_obs,axis=1)
    rn=np.array([np.divide(xyz_obs[:,0],r),np.divide(xyz_obs[:,1],r),np.divide(xyz_obs[:,2],r)]).T
    
    RA=np.mod(np.arctan2(rn[:,1],rn[:,0])+pix2,pix2)
    DEC=np.arcsin(rn[:,2])
    
    return np.rad2deg([RA, DEC]).T

def Icrf2RaDec_deg(xyz_topo_ast,):
    """Transform topocentric ICRF coordinates to
    Right Ascension (RA) and Declination (DEC)
    
    Parameters:
    --------
    xyz_topo_ast      ... 3D topocentric position vectors of asteroid (ICRF) [au]
 
    
    Returns:
    --------
    ra ... Right Ascension [deg]
    dec ... Declination [deg]

    """
    pix2=np.pi*2
    
    #observer to asteroid vectors
    
    r=np.linalg.norm(xyz_topo_ast,axis=1)
    rn=np.array([np.divide(xyz_topo_ast[:,0],r),
                 np.divide(xyz_topo_ast[:,1],r),
                 np.divide(xyz_topo_ast[:,2],r)]).T
    
    RA=np.mod(np.arctan2(rn[:,1],rn[:,0])+pix2,pix2)
    DEC=np.arcsin(rn[:,2])
    
    return np.rad2deg([RA, DEC]).T

def radec2eclip(ra, dec, **kwargs):
    """Convert Right Ascension and Declination to ecliptic xyz unit vector

    Parameters:
    -----------
    ra ... Right Ascension
    dec ... Declination

    Keyword Arguments:
    ------------------
    deg ... Are angles in degrees: True or radians: False

    Returns:
    --------
    x_ecl ... 3D vectors of unit length (ecliptic)
    """

    x_icrf = radec2icrfu(ra, dec, **kwargs)

    icrf2ecl = np.array([[1., 0., 0.],
                        [0., 0.91748206, 0.39777716],
                        [0., -0.39777716, 0.91748206]])

    x_ecl = icrf2ecl.dot(x_icrf)
    return x_ecl


############################################
# OBSERVATIONS, TRACKLETS AND ARROWS
###########################################

def SelectTrackletsFromObsData(pairs, df, dt_min, dt_max, time_column_name):
    """Select data in trackelts from observations data frame.

    Parameters:
    -----------
    pairs             ... array of observation indices that form a tracklet
                          (only 2 entries are supported for now)
    df                ... pandas dataframe containing observation data
    dt_min            ... minimum timespan between two observations 
                          to be considered a pair, e.g. exposure duration (days)
    dt_max            ... maximum timespan between two observations 
                          to be considered a pair (days)
    time_column_name  ... string, the name of the pandas dataframe column
                          containing the epoch of each observation

    Returns:
    --------
    df2               ... pandas dataframe containing only observations
                          that occur in tracklets.
    goodpairs         ... array of observation indices that form possible
                          tracklets (only 2 observation entries per
                          tracklet are supported for now)
    """
    goodpairs = CullSameTimePairs(pairs, df, dt_min, dt_max, time_column_name)
    index_list = np.unique(goodpairs.flatten())
    #df2 = (df.iloc[index_list]).reset_index()

    return df, goodpairs


def CullSameTimePairs(pairs, df, dt_min, dt_max, time_column_name):
    """Cull all observation pairs that occur at the same time.

    Parameters:
    -----------
    pairs             ... array of observation indices that form a tracklet
                          (only 2 entries are supported for now)
    df                ... pandas dataframe containing observation data
    dt_min            ... minimum timespan between two observations 
                          to be considered a pair, e.g. exposure duration (days)
    dt_max            ... maximum timespan between two observations 
                          to be considered a pair (days)
    time_column_name  ... string, the name of the pandas dataframe column
                          containing the epoch of each observation

    Returns:
    --------
    goodpairs         ... array of observation indices that form possible
                          tracklets (only 2 observation entries
                          per tracklet are supported for now)
    """

    tn = time_column_name
    # Tracklets from observation paris cannot be constructed from contemporal observations (delta_t==0)
    # nor observations that are too far apart (>= dt_max)
    delta_t = np.abs(df[tn][pairs[:,1]].values-df[tn][pairs[:,0]].values)
    goodpairs = pairs[(delta_t>dt_min) & (delta_t<dt_max)]
    return np.array(goodpairs)


def create_heliocentric_arrows(df, r, drdt, tref, cr, ct_min, ct_max, v_max=1., lttc=False, filtering=True, verbose=True, eps=0):
    """Create tracklets/arrows from dataframe containing nightly RADEC observations
       and observer positions.

    Parameters:
    -----------
    df       ... Pandas DataFrame containing nightly RA and DEC [deg], time [JD, MJD],
                 (x,y,z)_observer positions [au, ICRF]
    r        ... assumed radius of heliocentric sphere used for arrow creation[au]
    drdt     ... assumed radial velocity
    tref     ... reference time for arrow generation
    cr       ... maximum spacial clustering radius for arrow creation (au)
    ct_min   ... minimum temporal clusting radius for arrow creation (days)
    ct_max   ... maximum temporal clusting radius for arrow creation (days)


    Keyword arguments:
    ------------------
    v_max (optional)       ... velocity cutoff [au/day]
    lttc (optional)        ... light travel time correction
    filtering (optional)   ... filter created tracklets (exclude tracklets built 
                                from data with the same timestamp) 
    verbose (optional)     ... print verbose progress statements  
    eps (optional)         ... Branches of the Kdtree are not explored if their 
                               nearest points are further than r/(1+eps), 
                               and branches are added in bulk if their furthest points 
                               are nearer than r * (1+eps). eps has to be non-negative.

    Returns:
    --------
    x         ... tracklet/arrow position (3D) [au]
    y         ... tracklet/arrow velocity (3D) [au]
    t         ... tracklet/arrow reference epoch [JD/MJD]
    goodpairs ... index pairs of observations that go into each tracklet/arrow
    """
    
    goodpairs=[]
    paris=[]
    
    # speed of light in au/day
    c_aupd = 173.145

    # Transform RADEC observations into positions on the unit sphere (US)
    xyz = radec2icrfu(df['RA'], df['DEC'], deg=True)

    # Those are the line of sight (LOS) vectors
    los = np.array([xyz[0], xyz[1], xyz[2]]).T

    # Use the position of the observer and the LOS to project the position of
    # the asteroid onto a heliocentric great circle with radius r
    observer = df[['x_obs', 'y_obs', 'z_obs']].values

    # Calculate how much the heliocentric distance changes
    # during the obsevations based on assumed dr/dt
    dt = tref-df['time'].values
    dr = drdt*dt
    r_plus_dr = r+dr

    # Heliocentric postions of the observed asteroids
    posu = sphere_line_intercept(los, observer, r_plus_dr)

    if(verbose):
        print('Heliocentric positions generated.')
        print('Building spacial KDTree...')
        
    # To generate tracklets we build our KDTree based on the positions
    # in heliocentric space
    kdtree_s = scsp.cKDTree(posu, leafsize=16, compact_nodes=True,
                          copy_data=False, balanced_tree=True, boxsize=None)
    # rule out un-physical combinations of observations with kdtree

    # Query KDTree for good pairs of observations that lie within
    # the clustering radius cr
    if(verbose):
        print('KDTree generated. Creating tracklets...')
        
    pairs = kdtree_s.query_pairs(cr,p=2., eps=eps, output_type='ndarray')

    if(verbose):
        print('Tracklet candidates found:',len(pairs))

    if (filtering):
        if(verbose):
            print('Filtering arrows by time between observations...')
        
        # Discard impossible pairs (same timestamp)
        [df2, goodpairs] = SelectTrackletsFromObsData(pairs, df, ct_min, ct_max, 'time')
        
        if(verbose):
            print('Tracklets filtered. New number of tracklets:',len(goodpairs))
    
    else:
        goodpairs=pairs
    
    
    # tracklet position for filtered pairs
    x = posu[goodpairs[:,0]]
    # tracklet time
    t = df['time'][goodpairs[:,0]].values
    # tracklet velocity through forward differencing
    va = []
    vapp = va.append
    dt = df['time'][goodpairs[:,1]].values-df['time'][goodpairs[:,0]].values
    dx = posu[goodpairs[:,1]]-posu[goodpairs[:,0]]
    for d in range(0,3):
        vapp(np.divide(dx[:,d],dt))
    v = np.array(va).T
    
    if (filtering):
        if(verbose):
            print('Filtering arrows by max velocity..')
        vnorm=norm(v)
        v_idx=np.where(vnorm<=v_max)[0]
    
        goodpairs=np.take(goodpairs,v_idx,axis=0)
        x=np.take(x,v_idx,axis=0)
        v=np.take(v,v_idx,axis=0)
        t=np.take(t,v_idx)
    
#         print('lenx_filtered',len(x))
#         print('lenv_filtered',len(v))
#         print('lent_filtered',len(t))
#         print('x',x)
#         print('v',v)
#         print('t',t)
#         print('goodpairs',goodpairs)
    
    if(verbose):
        print('Tracklets created:',len(goodpairs))
    
    # correct arrows for light travel time
    if(lttc):
        if(verbose):
            print('(Linear correction for light travel time aberration...')
        xo = observer[goodpairs[:, 0]]
        dist = norm(x-xo)
        xl = x.T-dist/c_aupd*v.T
        return xl.T, v, t, goodpairs

    else:
        return x, v, t, goodpairs


def propagate_arrows_linear(x, v, t, tp):
    """Linear propagattion of arrows to the same time.

    Parameters:
    -----------
    x  ... array of 3D positions
    v  ... array of 3D velocities
    t  ... array of epochs for states (x,v)
    tp ... epoch to propagate state to

    Returns:
    --------
    xp ... array of propagated 3D positions
    dt ... array of time deltas wrt the propatation epoch: tp-t
    """
    dt = tp-t
    xp = x + (v*np.array([dt, dt, dt]).T)
    return xp, dt


def propagate_arrows_great_circle(x, v, t, tp):
    """Propagate arrows to the same time along a Great Circle.

    Parameters:
    -----------
    x  ... array of 3D positions
    v  ... array of 3D velocities
    t  ... array of epochs for states (x,v)
    tp ... epoch to propagate state to

    Returns:
    --------
    xp ... array of propagated 3D positions
    vp ... array of propagated 3D velocities
    dt ... array of time deltas wrt the propatation epoch: tp-t
    """
    dt = tp-t

    # define rotation axis as r x v (orbital angular momemtum)
    h = np.cross(x, v)
    # define rotation angle via w = v/r where w is dphi/dt
    # so that phi = w*dt = v/r*dt
    phi = norm(v)/norm(x)*dt

    xp = rotate_vector(phi, h, x)
    vp = rotate_vector(phi, h, v)
    return xp, vp, dt


def propagate_arrows_2body(x, v, t, tp):
    """ Propagate arrows to the same time using spicepy's 2body propagation.

    Parameters:
    -----------
    x  ... array of 3D positions
    v  ... array of 3D velocities
    t  ... array of epochs for states (x,v)
    tp ... epoch to propagate state to

    Returns:
    --------
    xp ... array of propagated 3D positions
    vp ... array of propagated 3D velocities
    dt ... array of time deltas wrt the propatation epoch: tp-t
    """
    dt = np.array(tp-t)

    # Gaussian Gravitational Constant [au^1.5/Msun^0.5/D]
    gaussk = 0.01720209894846
    # default gravitational parameter [Gaussian units]
    gm = gaussk*gaussk

    if(x.ndim == 1):
        state = sp.prop2b(gm, np.hstack((x, v)), dt)
        xp = state[0:3]
        vp = state[3:6]

    elif(x.ndim == 2):
        lenx = len(x[:, 0])
        dimx = len(x[0, :])
        dimv = len(v[0, :])
        try:
            assert(dimx == dimv)
        except(TypeError):
            raise TypeError

        xp = []
        xp_add = xp.append
        vp = []
        vp_add = vp.append
        for i in range(lenx):
            state = sp.prop2b(gm, np.hstack((x[i, :], v[i, :])), dt[i])
            xp_add(state[0:3])
            vp_add(state[3:6])

    return np.array(xp), np.array(vp), dt
