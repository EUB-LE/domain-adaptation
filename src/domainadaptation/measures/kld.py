from __future__ import annotations
import numpy as np
import numpy.typing as npt
from scipy.stats import entropy

def kld(pd_x: npt.ArrayLike, pd_y: npt.ArrayLike) -> float:
    pd_x = np.array(pd_x)
    pd_y = np.array(pd_y)
    
    entropy(pk= pd_x, qk= pd_y)

