import numpy as np
import numpy.typing as npt
from domainadaptation.divergence import kld, rv_discrete

def jsd(pd_x: npt.ArrayLike, pd_y:npt.ArrayLike) -> float:
    pd_x = np.array(pd_x)
    pd_y = np.array(pd_y)
    
    m = (1./2.) * (pd_x + pd_y)
    return (1./2.) * kld(pd_x, m) + (1./2.) * kld(pd_y, m)

def jsd_from_discrete(rv1: rv_discrete, rv2: rv_discrete) -> float:
    xk = np.intersect1d(rv1.xk, rv2.xk)

    pd_x = rv1.score_samples(xk)
    pd_y = rv2.score_samples(xk)

    return jsd(pd_x, pd_y)
