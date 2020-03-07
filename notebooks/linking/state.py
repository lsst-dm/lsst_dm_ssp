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
State

LSST Solar System Processing

Defines the state of a moving object

Implementation: Python 3.6, S. Eggl 20200306
"""
# Accelerators
import numpy as np

# time scale transforms
from astropy.time import Time

# Constants such as the speed of light and GM
import constants as cnst

# State Propagation routines
import propagate as prop

# Coordinate transforms
import transforms as tr


__all__ = ['State']


############################################
# MODULE SPECIFIC EXCEPTION
###########################################
class Error(Exception):
    """Vector module specific exception."""
    
    pass

############################################
# STATE CLASS
###########################################
class State:
    def __init__(self, data, epoch):
        self.data = np.array(data)
#         if(state.ndim == 1): 
#             self.x = self.data[0:3]
#             self.v = self.data[3:6]
#         elif(state.ndim == 2):
#             self.x = self.data[:,0:3]
#             self.v = self.data[:,3:6]
#         else:
#             raise Exception('Error in class State: only 1D and 2D arrays suppored.')
          
        self.frame = 'icrf'
        self.timescale = 'tdb'
        self.epoch = np.array(epoch)
        self.timeformat = 'mjd'
        self.xunit='au'
        self.vunit='au/day'
        self.center='sun'
        
        self.data2xv()
        
    
    def data2xv(self):
        """Split state data into positions (x) and velocities (v)"""
        
        if(self.data.ndim == 1): 
                self.x = self.data[0:3]
                self.v = self.data[3:6]
        elif(self.data.ndim == 2):
                self.x = self.data[:,0:3]
                self.v = self.data[:,3:6]
        else:
            raise Exception('Error in class State: only 1D and 2D arrays suppored.')
            
        return 

    def propagate(self, tp, propagator='2body'):
        """Propagate state to epoch tp.
        
        Parameters:
        -----------
        self        ... State class object
        tp          ... time to propagate state to; 
                        tp must share the same units and timescales as .epoch
        
        Returns:
        --------
        self         ... propagated state at new epoch = old epoch + tp
            .x 
            .v 
            .data 
            .epoch 
        
        """
        try:                 
            [xp, vp, dt] = prop.PropagateState(self.x, self.v, self.epoch, tp, propagator=propagator)
        
            self.x = xp
            self.v = vp
            self.data = np.hstack(xp, vp)
            self.epoch = self.epoch + dt                    
        except:
            raise Exception("Error in class State: propagation failed.") 
        
        return 
    
    def toFrame(self, frame):
        """Transform from current coodrinate frame to 'frame'

        Parameters:
        -----------
        self       ... State class object
        frame      ... coordinate frame: 'ecliptic' or 'icrf'
        
        Returns:
        --------
        self       ... State class object in new coordinate frame
            .x 
            .v 
            .data 
            .frame  
        """
                            
        if (frame == self.frame):                    
            data_new = self.data
            
        elif (frame == 'icrf' and self.frame == 'ecliptic'):
            data_new = tr.ecliptic2icrf(self.data)
            self.frame = 'icrf'
            
        elif (frame == 'ecliptic' and self.frame == 'icrf'):             
            data_new = tr.icrf2ecliptic(self.data)
            self.frame = 'ecliptic'
            
        else:
            raise Exception("Error in class State: unknown coordinate frame.")                  
        
        self.data = data_new                        
        self.data2xv()  
        
        return
        
    def toEpoch(self, timeformat='mjd', timescale='tdb'):
        """Change current time unit and/or timescale of state epoch.

        Parameters:
        -----------
        self      ... State class object
        timeunit  ... astropy.Time object supported units (e.g. mjd, jd)
        timescale ... astropy.Time 
        
        Returns:
        --------
        self          ... State class object in new coordinate frame
            .epoch  
            .timescale
            .timeunit
        """
        try:
            epoch_old = Time(self.epoch, scale=self.timescale, format=self.timeformat)
            epoch_new = Time(epoch_old, scale=timescale, format=timeformat)
            self.epoch = epoch_new.value
            self.timeformat = timeformat
            self.timescale = timescale
            
        except:
            raise Exception("Error in class State: epoch conversion failed.") 
            
        return 