from __future__ import annotations
import numpy as np
import numpy.typing as npt
from domainadaptation.measures import kld

def jsd(pd_x: npt.ArrayLike, pd_y:npt.ArrayLike) -> float:
    pd_x = np.array(pd_x)
    pd_y = np.array(pd_y)
    
    m = (1./2.) * (pd_x + pd_y)
    kldxm = kld(pd_x, m)
    kldym = kld(pd_y, m)
    return (1./2.) * kldxm + (1./2.) * kldym


