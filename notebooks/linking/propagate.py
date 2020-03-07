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
propagate

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


__all__ = ['propagateState', 'propagate2body', 'propagateLinear']

class State:
    def __init__(self, data):
        self.data = np.array(data)
        if(state.ndim == 1): 
            self.x = self.data[0:3]
            self.v = self.data[3:6]
        elif(state.ndim == 2):
            self.x = self.data[:,0:3]
            self.v = self.data[:,3:6]
        else:
            raise Exception('Error in class State: only 
          
        self.frame = 'icrf'
        self.timescale = 'tdb'
        self.epoch = np.array([0])
        self.timeunit = 'mjd'
        self.lengthunit='au'
        self.center='sun'
    
    
    def propagate(self, tp, propagator='2body'):
        """Propagate state to epoch tp."""
        
        [xp, vp, dt] = PropagateState(self.x, self.v, self.epoch, tp, propagator=propagator)
        
        return [xp, vp, dt]
    
    def toFrame(self,frame):
        
        
    def toTimeunit(self, timeunit):
        
    def toCenter(self, center):
        
        
        
    


def propagateState(x, v, t, tp, propagator='linear'):
    """Propagation of states with choce of propagator.

    Parameters:
    -----------
    x            ... array of 3D positions
    v            ... array of 3D velocities
    t            ... array of epochs for states (x,v)
    tp           ... epoch to propagate state to
    propagator   ... select propagator from 'linear, 2body, nbody'
    
    Returns:
    --------
    xp           ... array of propagated 3D positions
    vp           ... array of propagated 3D velocities
    dt           ... array of time deltas wrt the propatation epoch: tp-t
    """

    if(propagator == 'linear'):
        [xp, vp, dt] = PropagateLinear(x, v, t, tp)
        
    elif(propagator == '2body'):
        [xp, vp, dt] = Propagate2body(x, v, t, tp)
        
    elif(propagator == 'nbody'):
        raise Exception('N-body propagation not yet implemented.')
        
    else:
        raise Exception('Error in Propagate Arrows: select valid propagator.')
    
    return xp, vp, dt


def propagate2body(x, v, t, tp):
    """ Propagate states to the same time using spicepy's 2body propagation.

    Parameters:
    -----------
    x  ... array of 3D heliocentric/barycentric positions
    v  ... array of 3D heliocentric/barycentric velocities
    t  ... array of epochs for states (x,v)
    tp ... epoch to propagate state to

    Returns:
    --------
    xp ... array of propagated 3D positions
    vp ... array of propagated 3D velocities
    dt ... array of time deltas wrt the propatation epoch: tp-t
    """
    dt = np.array(tp-t)
    
#     print('x.ndim',x.ndim)
#     print('x.shape',x.shape)
#     print('x',x)
#     print('v',v)
    
    if (len(x)<1):
        # empty
        xp =[]
        vp =[]
        
    elif(x.ndim==1):
        state = sp.prop2b(cnst.GM, np.hstack((x, v)), dt)
        xp = state[0:3]
        vp = state[3:6]

    elif(x.ndim==2):
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
            state = sp.prop2b(cnst.GM, np.hstack((x[i, :], v[i, :])), dt[i])
            xp_add(state[0:3])
            vp_add(state[3:6])

    else:
        raise TypeError
        
    return np.array(xp), np.array(vp), dt


def propagateLinear(x, v, t, tp):
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
    v  ... array of velocities
    dt ... array of time deltas wrt the propatation epoch: tp-t
    """
    dt = tp-t
    
    if (len(x)<1):
        # empty
        xp =[]
        vp =[]
        
    elif(x.ndim==1):
        xp = x + v*dt
        
    elif(x.ndim==2):    
        xp = x + (v*np.array([dt, dt, dt]).T)
        
    return xp, v, dt
