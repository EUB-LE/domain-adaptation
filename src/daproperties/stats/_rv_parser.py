from __future__ import annotations

from daproperties.stats.rv_discrete import rv_discrete
from daproperties.stats.rv_continuous import rv_continuous
from daproperties.stats.rv_mixed import rv_mixed
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


def rv_from_discrete(x:np.ndarray, **kwargs) -> rv_discrete:

    df = pd.DataFrame(x)
    pmf = df.value_counts(ascending=True, normalize=True, sort=False)
    xk = list(pmf.index.values)
    pk = pmf.values

    return rv_discrete(xk = xk, pk = pk, **kwargs)


def rv_from_continuous(x:np.ndarray, bandwidth_range=np.logspace(-1,1,20), **kwargs) -> rv_continuous:
   
    params = {"bandwidth": bandwidth_range}
    kde = None
    grid = GridSearchCV(KernelDensity(), params, verbose=1, n_jobs=-1)    
    grid.fit(x)
    kde = grid.best_estimator_

    shape = (x.shape[1],)
    coverage = _value_range_from_data(x)

    return rv_continuous(kde, shape, coverage, **kwargs)

def rv_from_mixed(x:np.ndarray, y:np.ndarray, rv_x:rv_continuous=None, rv_y:rv_discrete=None,  bandwidth_range=np.logspace(-1,1,20), **kwargs) -> rv_mixed:
    params = {"bandwidth": bandwidth_range}
    cond_kdes = {} 
    labels = pd.DataFrame(y).value_counts(ascending=True, normalize=True, sort=False).index.values
    print(f"Fitting KDEs for every label in {labels}. This may take a while. Consider saving the cond_kdes and using the direct constructor of rv_mixed to save time.")
    for label in labels: 
        sample = x[y == label]
        grid = GridSearchCV(KernelDensity(), params, verbose=1, n_jobs=-1)    
        grid.fit(sample)
        cond_kdes[label] = grid.best_estimator_
    
    rv_x = rv_x if rv_x else rv_from_continuous(x, bandwidth_range, name="continuous")
    rv_y = rv_y if rv_y else rv_from_discrete(y, name="discrete")

    return rv_mixed(rv_x, rv_y, cond_kdes, **kwargs)

def rvs_from_mixed(x:np.ndarray, y:np.ndarray, bandwidth_range=np.logspace(-1,1,20)) -> tuple[rv_continuous, rv_discrete, rv_mixed]:
    rv_xy = rv_from_mixed(x, y, bandwidth_range=bandwidth_range, name="xy")
    rv_x = rv_xy.rv1
    rv_x.name = "x"
    rv_y = rv_xy.rv2
    rv_y.name = "y"
    return (rv_x, rv_y, rv_xy)

    


    

    


def _value_range_from_data(x:np.ndarray):
    mins = x.min(axis=0)
    maxs = x.max(axis=0)

    minsmaxs = np.vstack((mins,maxs))
    return list(map(tuple, minsmaxs.T))

def _reject_outliers(data:np.ndarray, m = 2.) -> np.ndarray:
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]


    
        
   