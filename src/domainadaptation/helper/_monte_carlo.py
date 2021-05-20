from __future__ import annotations
import numpy as np
from numpy import number, random


def generate_mc_points(limits:list[tuple[float, float]] = [(0,1)], eval_points_per_dim:int = 1000) -> np.ndarray:
    dims = len(limits)
    eval_points = eval_points_per_dim
    
    mc_points = np.empty(shape=(dims, eval_points))
    for index, limit in enumerate(limits): 
        mc_points[index] = random.uniform(low=limit[0], high=limit[1], size=eval_points)
    
    # adapt shape to correct layout for datapoints
    mc_points = mc_points.T

    return mc_points



