from __future__ import annotations
import numpy as np
from domainadaptation.helper._monte_carlo import generate_mc_points
from domainadaptation.stats import rv_continuous, rv_discrete, rv_mixed


def common_coverage_of_discrete_rvs(rv1: rv_discrete, rv2: rv_discrete) -> np.ndarray:
    xk1 = np.array(rv1.xk)
    xk2 = np.array(rv2.xk)

    return np.intersect1d(xk1, xk2)

def _get_common_limits_of_continuous(rv1: rv_continuous, rv2: rv_continuous):
    c1 = np.array(rv1.coverage)
    c2 = np.array(rv2.coverage)

    # Get for the highest minimum and the lowest maximum for every dimension to define the common coverage
    mins = np.maximum(c1[:,0], c2[:,0])
    maxs = np.minimum(c1[:,1], c2[:,1])

    minsmaxs = np.vstack((mins,maxs)).T
    limits = list(map(tuple, minsmaxs))
    
    return limits

def common_coverage_of_continuous_rvs(rv1: rv_continuous, rv2: rv_continuous, eval_pts:int=1000) -> np.ndarray: 
    limits = _get_common_limits_of_continuous(rv1, rv2)
    return generate_mc_points(limits, eval_pts)

def common_coverage_of_mixed_rvs(rv1: rv_mixed, rv2: rv_mixed, eval_pts:int = 1000) -> tuple[np.ndarray, np.ndarray]: 
    common_coverage_continuous = common_coverage_of_continuous_rvs(rv1.rv1, rv2.rv1)
    common_coverage_discrete = common_coverage_of_discrete_rvs(rv1.rv2, rv2.rv2)
   
    X = None
    y = []

    for label in common_coverage_discrete:
        y.append(np.repeat(label, common_coverage_continuous.shape[0]))
        if X is None:
            X = common_coverage_continuous
        else:
            X = np.concatenate((X,common_coverage_continuous))
    
    y = np.array(y).reshape(-1,)

    return (X, y)




