from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.stats import entropy

if TYPE_CHECKING:
    import numpy.typing as npt

def kld(pd_x: npt.ArrayLike, pd_y: npt.ArrayLike) -> float:
    pd_x = np.array(pd_x)
    pd_y = np.array(pd_y)
    
    return entropy(pk= pd_x, qk= pd_y)

