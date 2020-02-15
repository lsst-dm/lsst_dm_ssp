# LSST Solar System Processing test data.

Simulated LSST obervations and orbital elements of roughly 500 main belt asteroids. The LSST survey used was baseline2018b. 
https://github.com/lsst-pst/survey_strategy/tree/master/baseline2018b-doc

Survey simulator:
https://github.com/eggls6/openobs

Files in this folder:
lsst_sso_534.com ... contains orbital elements of target asteroids in openorb format. Header info:
                      objId ... object ID
                      pdes  ... primary designation of orbital element data (COM : Cometary)
                      q     ... perhelion distance [au]
                      e     ... orbital eccentricity
                      i     ... inclination [deg]
                      om    ... longitude of the ascending node [deg]
                      w     ... argument of pericenter [deg]
                      tp_cal... time of pericenter passage in calender format
                      H     ... absolute magnitude (mag)
                      epoch_mjd epoch of orbital elements [Modified Julian Date]
                      
lsst_sso_534.csv ... contains the same orbital elements of target asteroids in comma separated value format
lsst_sso_534_obs.tar.gz ... simulated LSST observations of asteroids given in the .com file.




