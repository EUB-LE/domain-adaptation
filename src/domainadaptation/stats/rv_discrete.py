from __future__ import annotations
import numpy as np
import numpy.typing as npt
from domainadaptation.stats.rv_base import rv_base



class rv_discrete(rv_base):
    def __init__(self, xk:npt.ArrayLike, pk:npt.ArrayLike, name:str = None, badvalue:float = np.nan) -> None:
        
        self.badvalue = badvalue
        self.xk = xk
        self.pk = pk

        if len(xk) is not len(pk):
            raise ValueError(f"xk: {xk} and pk: {pk} must have the same length.")
        
        if np.array(pk).ndim != 1:
            raise ValueError(f"pk: {pk} must be a 1d array-like that defines probabilities.")
    
        pmf_dict = {}
        for x,p in zip(self.xk, self.pk):
            pmf_dict[x] = p
        
        self.pmf_dict = pmf_dict
        self.name = name

  
    def _prob_of(self, key: npt.ArrayLike) -> float: 
        if np.isscalar(key):
            key = [key]
        
        key = tuple(key) 
        try:
            return self.pmf_dict[key]
        except KeyError:
            return self.badvalue

    def pmf(self, X:npt.ArrayLike) -> np.ndarray:
        X = np.array(X)

        try:
            return np.apply_along_axis(self._prob_of, 1, X)
        except np.AxisError:
            # Deal with 0-d arrays
            X = X.reshape(-1,1)
            return np.apply_along_axis(self._prob_of, 1, X)
    
    def score_samples(self, X:npt.ArrayLike) -> np.ndarray:
        return np.log(self.pmf(X))

    
    def _common_coverage(self, rv:rv_discrete) -> np.ndarray:
        xk1 = np.array(self.xk)
        xk2 = np.array(rv.xk)

        return np.intersect1d(xk1, xk2)
    
    def divergence(self, rv:rv_discrete, div_type:str="jsd") -> float:
        common_coverage = self._common_coverage(rv)
        pd_x = self.score_samples(common_coverage)
        pd_y = rv.score_samples(common_coverage)

        divergence = self.divergence_from_distribution(pd_x, pd_y, div_type)
        return divergence



