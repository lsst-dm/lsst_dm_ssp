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

def SelectTrackletsFromObsData(pairs, df, time_column_name):
    """Select data in trackelts from observations data frame.

    Parameters:
    -----------
    pairs             ... array of observation indices that form a tracklet
                          (only 2 entries are supported for now)
    df                ... pandas dataframe containing observation data
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
    goodpairs = CullSameTimePairs(pairs, df, time_column_name)
    index_list = np.unique(goodpairs.flatten())
    df2 = (df.iloc[index_list]).reset_index()

    return df2, goodpairs


def CullSameTimePairs(pairs, df, time_column_name):
    """Cull all observation pairs that occur at the same time.

    Parameters:
    -----------
    pairs             ... array of observation indices that form a tracklet
                          (only 2 entries are supported for now)
    df                ... pandas dataframe containing observation data
    time_column_name  ... string, the name of the pandas dataframe column
                          containing the epoch of each observation

    Returns:
    --------
    goodpairs         ... array of observation indices that form possible
                          tracklets (only 2 observation entries
                          per tracklet are supported for now)
    """

    tn = time_column_name
    goodpairs = []
    goodpairs_app = goodpairs.append
    for p in pairs:
        if (df[tn][p[0]] < df[tn][p[1]]):
            goodpairs_app(p)
    return np.array(goodpairs)


def create_heliocentric_arrows(df, r, drdt, tref, cr, lttc=False):
    """Create tracklets/arrows from dataframe containing RADEC observations
       and observer positions.

    Parameters:
    -----------
    df   ... pandas DataFrame containing RA and DEC [deg], time [JD,MJD],
             (x,y,z)_observer positions [au, ICRF]
    r    ... assumed radius of heliocentric sphere used for arrow creation[au]
    drdt ... assumed radial velocity
    tref ... reference time for arrow generation
    cr   ... maximum clustering radius for arrow creation (au)


    Keyword arguments:
    ------------------
    lttc (optional) ... light travel time correction

    Returns:
    --------
    x         ... tracklet/arrow position (3D) [au]
    y         ... tracklet/arrow velocity (3D) [au]
    t         ... tracklet/arrow reference epoch [JD/MJD]
    goodpairs ... index pairs of observations that go into each tracklet/arrow
    """
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

    # To generate tracklets we build our KDTree based on the positions
    # in heliocentric space
    kdtree = scsp.cKDTree(posu, leafsize=16, compact_nodes=True,
                          copy_data=False, balanced_tree=True, boxsize=None)
    # rule out un-physical combinations of observations with kdtree

    # Query KDTree for good pairs of observations that lie within
    # the clustering radius cr
    pairs = kdtree.query_pairs(cr)

    # Discard impossible pairs (same timestamp)
    [df2, goodpairs] = SelectTrackletsFromObsData(pairs, df, 'time')

    x = []
    x_add = x.append
    v = []
    v_add = v.append
    t = []
    t_add = t.append
    for p in goodpairs:
        x_add(posu[p[0]])
        t_add(df['time'][p[0]])
        v_add((posu[p[1]]-posu[p[0]])/(df['time'][p[1]]-df['time'][p[0]]))

# correct arrows for light travel time
    if(lttc):
        xo = observer[goodpairs[:, 0]]
        dist = norm(np.array(x)-xo)
        xl = np.array(x).T-dist/c_aupd*np.array(v).T
        return xl.T, np.array(v), np.array(t), goodpairs

    else:
        return np.array(x), np.array(v), np.array(t), goodpairs


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
