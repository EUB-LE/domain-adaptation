from __future__ import annotations
import numpy as np
from domainadaptation.measures import jsd, kld
from numpy.typing import ArrayLike
from domainadaptation.helper._coverage 

# Insert new divergence measures here
# TODO: Automatically detect divergence measures.
divergence_measures = {
        "jsd": jsd,
        "kld": kld
}


def _divergence(pd_x:np.ndarray, pd_y:np.ndarray, div_type:str="jsd") -> float: 
    try:
        divergence_measure = divergence_measures[div_type]
    except KeyError:
        raise KeyError(f"div_type must be one of {[key for key in divergence_measures.keys()]} but is {div_type}.")

    return divergence_measure(pd_x, pd_y)


    
    







