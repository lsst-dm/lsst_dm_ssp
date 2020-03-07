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

# Accelerated vector operations
import vector

# Constants such as the speed of light and GM
import constants as const

# Coordinate transfroms
import transforms

# Orbital Dynamics 
import propagate

# Database
# import pandas as pd

# Orbital Dynamics
import spiceypy as sp

# Clustering
import scipy.spatial as scsp
# import sklearn.cluster as cluster

__all__ = [ 'lsstNight','sphereLineIntercept',
           'selectTrackletsFromObsData', 'cullSameTimePairs',
           'makeHeliocentricArrows', 
           'Heliolinc2','CollapseClusters']

############################################
# AUXLIARY FUNCTIONS
###########################################

def lsstNight(expMJD, minexpMJD):
    """Calculate the night for a given observation epoch and a survey start date.
    
    Parameters:
    -----------
    expMJD ... epoch of observation / exposure [modified Julian date, MJD]
    minexpMJD ... start date of survey [modified Julian date, MJD]
    
    Returns:
    --------
    night ... the night of a given observation epoch with respect to the survey start date.
    
    """
    local_midnight = 0.16
    const = minexpMJD + local_midnight - 0.5
    night = np.floor(expMJD - const)
    return night

############################################
# OBSERVATIONS, TRACKLETS AND ARROWS
###########################################

def cullSameTimePairs(pairs, df, dt_min, dt_max, time_column_name):
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
    
    
def selectTrackletsFromObsData(pairs, df, dt_min, dt_max, time_column_name):
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
    goodpairs = cullSameTimePairs(pairs, df, dt_min, dt_max, time_column_name)
    index_list = np.unique(goodpairs.flatten())
    #df2 = (df.iloc[index_list]).reset_index()

    return df, goodpairs

def makeHeliocentricArrows(df, r, drdt, tref, cr, ct_min, ct_max, v_max=1., lttc=False, filtering=True, verbose=True, eps=0):
    """Create tracklets/arrows from dataframe containing nightly RADEC observations
    and observer positions.

    Parameters:
    -----------
    df       ... Pandas DataFrame containing nightly RA and DEC [deg], time [JD, MJD],
                 (x,y,z)_observer positions [au, ICRF]
    r        ... assumed radius of heliocentric sphere used for arrow creation[au]
    drdt     ... assumed radial velocity
    tref     ... reference time for arrow generation. Used to calculate how much the 
                 heliocentric distance changes between observations based on assumed dr/dt
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
    posu = sphereLineIntercept(los, observer, r_plus_dr)

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
            print('Filtering arrows by max velocity...')
        vnorm=norm(v)
        v_idx=np.where(vnorm<=v_max)[0]
    
        goodpairs=np.take(goodpairs,v_idx,axis=0)
        x=np.take(x,v_idx,axis=0)
        v=np.take(v,v_idx,axis=0)
        t=np.take(t,v_idx,axis=0)
    
#         print('lenx_filtered',len(x))
#         print('lenv_filtered',len(v))
#         print('lent_filtered',len(t))
#         print('x',x)
#         print('v',v)
 #        print('t',t)
#         print('goodpairs',goodpairs)
    
    if(verbose):
        print('Tracklets created:',len(goodpairs))
    
    # correct arrows for light travel time
    if(lttc):
        if(verbose):
            print('(Linear correction for light travel time aberration...')
        xo = observer[goodpairs[:, 0]]
        dist = norm(x-xo)
        xl = x.T-dist/const.CAUPD*v.T
        return xl.T, v, t, goodpairs

    else:
        return x, v, t, goodpairs

def helioLinc2(dfobs, r, drdt, cr, ct_min, ct_max, clustering_algorithm='dbscan', lttc=False, verbose=True):
    """HelioLinC2 (Heliocentric Linking in Cartesian Coordinates) algorithm.

    Parameters:
    -----------
    dfobs ... Pandas DataFrame containing object ID (objId), observation ID (obsId),
               time, night, RA [deg], DEC[deg], observer x,y,z [au]


    r      ... assumed heliocentric distance [au]
    drdt   ... dr/dt assumed heliocentric radial velocity [au/day]
    cr     ... clustering radius [au]
    ct_min ... minimum timespan between observations to allow for trackelt making [days]
    ct_max ... maximum timespan between observations to allow for tracklet making [days]
    clustering_algorithm ... clustering_algorithm (currently either 'dbscan' or 'kdtree')
    lttc   ... Light travel time correction
    verbose... Print progress statements

    Returns:
    --------
    obs_in_cluster_df ... Pandas DataFrame containing linked observation clusters (no prereduction)
    """

#     xpall=[]
#     vpall=[]
    xar=[]
    var=[]
    tar=[]
    obsids_night=[]

    # the following two arrays are for testing purposes only
    objid_night=[]
    tobs_night=[]

    for n in df_grouped_by_night.groups.keys():
        if (verbose):
            print('Processing night ',n)
        # SELECT NIGHT FROM OBSERVATIONS DATA BASE
        idx=df_grouped_by_night.groups[n].values
        df=dfobs.loc[idx,:].reset_index(drop=True)
        #tref=(df['time'].max()+df['time'].min())*0.5
        tref=(dfobs['time'].max()+dfobs['time'].min())*0.5

        # GENERATE ARROWS / TRACKLETS FOR THIS NIGHT
        [xarrow_night, 
         varrow_night, 
         tarrow_night, 
         goodpairs_night]=MakeHeliocentricArrows(df,r,drdt,tref,cr,ct_min,
                                                     ct_max,v_max=1,lttc=False,
                                                     filtering=True,verbose=True,eps=cr)
        # ADD TO PREVIOUS ARROWS
        if (len(xarrow_night)<1):
            if (verbose):
                print('no data in night ',n)
        else:
            xar.append(xarrow_night)
            var.append(varrow_night)
            tar.append(tarrow_night)
            obsids_night.append(df['obsId'].values[goodpairs_night])
            objid_night.append(df['objId'].values[goodpairs_night])
            tobs_night.append(df['time'].values[goodpairs_night])


    if (len(xar)<1):
        if (verbose):
            print('No arrows for the current r, dr/dt pair. ',n)
    else:    
        xarrow=np.vstack(xar)
        varrow=np.vstack(var)
        tarrow=np.hstack(tar)
        obsids=np.vstack(obsids_night)

    # the following two arrays are for testing purposes only
        objids=np.vstack(objid_night)
        tobs=np.vstack(tobs_night)

    # PROPAGATE ARROWS TO COMMON EPOCH
        if (verbose):
            print('Propagating arrows...')
        tprop=(dfobs['time'].max()+dfobs['time'].min())*0.5
    #tprop=dfobs['time'].max()+180
        #[xp,vp,dt] = PropagateArrows2body(xarrow,varrow,tarrow,tprop)
        [xp, vp, dt] = propagateState(xarrow, varrow, tarrow, tprop, propagator='2body')
    #[xp,vp,dt] = ls.propagate_arrows_2body(xarrow,varrow,tarrow,dfobs['time'].max()+360)

        rnorm=(r/ls.norm(vp))
        vpn=vp*np.array([rnorm,rnorm,rnorm]).T
        xpvp=np.hstack([xp,vpn])

#       # CLUSTER WITH DBSCAN
        if (verbose):
            print('Clustering arrows...')
            
#       # CLUSTER PROPAGATED STATES (HERE IN REAL SPACE, BUT COULD BE PHASE SPACE)               
        if(clustering_algorithm=='dbscan'):
            db=cluster.DBSCAN(eps=eps,min_samples=min_samples,n_jobs=4).fit(xp)

#       # CONVERT CLUSTER INDICES TO OBSERVATION INDICES IN EACH CLUSTER
            try:
                obs_in_cluster, labels = observationsInCluster(dfobs,obsids,db,garbage=False)
                obs_in_cluster_df=pd.DataFrame(zip(labels,obs_in_cluster),columns=['clusterId','obsId'])
            except: 
                print('Error in constructing cluster dataframe.')

        elif (clustering_algorithm=='kdtree'):
        # CLUSTER WITH KDTree
            if (verbose):
                print('Clustering arrows...')
            tree = scsp.KDTree(xp)
            db = tree.query(xp, k=8, p=2, distance_upper_bound=eps)

            if (verbose):
                print('Deduplicating observations in clusters...')
            obs = []
            obs_app = obs.append
            arrow_idx = np.array(db,dtype="int64")[1]
            nan_idx = arrow_idx.shape[0]
            for i in arrow_idx:
                entries = i[i<nan_idx]
                if(len(entries)) > 1:
                     obs_app([np.unique(np.ravel(obsids[entries]))])

            obs_in_cluster_df = pd.DataFrame(obs,columns=['obsId'])
            obs_in_cluster_df['clusterId']=obs_in_cluster_df.index.values
            obs_in_cluster_df=obs_in_cluster_df[['clusterId','obsId']]

        else:
            raise ('Error in heliolinc2: no valid clustering algorithm selected') 

        # COMMON CODE
        obs_in_cluster_df['r'] = r
        obs_in_cluster_df['drdt'] = drdt
        obs_in_cluster_df['cluster_epoch'] = tprop
        #xpall.append(xp)
        #vpall.append(vp)
        return obs_in_cluster_df
    

def deduplicateClusters:    
     """Deduplicate clusters produced by helioLinC2 
     based on similar observations (r,rdot are discarded)

    Parameters:
    -----------
    cdf ... Pandas DataFrame containing object ID (objId), observation ID (obsId)
              
    Returns:
    --------
    cdf2     ... deduplicated Pandas DataFrame 
    """
 
    dup_idx = np.where(cdf.astype(str).duplicated(subset='obsId',keep='first'))[0]
    cdf2 = cdf.iloc[dup_idx]
     
    return cdf2
        
        
    
def collapseClusterSubsets(cdf):
    """Merge clusters that are subsets of each other 
    as produced by HelioLinC2.

    Parameters:
    -----------
    cdf ... Pandas DataFrame containing object ID (objId), observation ID (obsId)
              
    Returns:
    --------
    cdf2                ... collapsed Pandas DataFrame 
    subset_clusters     ... indices of input dataframe (cdf) that are subsets
    subset_cluster_ids  ... linked list of cluster ids [i,j] where j is a subset of i
    """
    
   
    #for index, row in clusters_df.iterrows():
    vals=cdf.obsId.values
    subset_clusters=[]
    subset_clusters_app=subset_clusters.append
    subset_cluster_ids=[]
    subset_cluster_ids_app=subset_cluster_ids.append

    cdf_idx=range(0,len(cdf))

    vals_set=[]
    vals_set_app=vals_set.append
    vals_min=[]
    vals_min_app=vals_min.append
    vals_max=[]
    vals_max_app=vals_max.append

    for i in cdf_idx:
        vals_set_app(set(vals[i]))          
        vals_min_app(np.min(vals[i]))
        vals_max_app(np.max(vals[i]))         

    vmin=np.array(vals_min)
    vmax=np.array(vals_max)

    for i in cdf_idx:
        for j in cdf_idx:
            if(i != j):
                    #use the fact that we have ordered sets here
                    if(vmax[i]<vmin[j]):
                        break
                    elif(vmin[i]>vmax[j]):
                        break
                    else:
                        is_subset=vals_set[i].issubset(vals_set[j])
                        #print(i,j,is_subset)
                        if(is_subset):
                            subset_clusters_app(i)
                            subset_cluster_ids_app([i,j])
                            break
        if(np.mod(i, 1000) == 0):                
            print('Progress [%], ', i/cdf_idx[-1]*100)

    idx_to_drop = subset_clusters
    
    cdf2=cdf.drop(index=idx_to_drop)
    return cdf2, subset_clusters, subset_cluster_ids 
