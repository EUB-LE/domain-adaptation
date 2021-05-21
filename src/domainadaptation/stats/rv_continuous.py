from __future__ import annotations
import numpy as np
from sklearn.neighbors import KernelDensity
import numpy.typing as npt 
from typing import Union
from domainadaptation.helper import generate_mc_points
from domainadaptation.stats.rv_base import rv_base



class rv_continuous(rv_base):
    def __init__(self, kde:KernelDensity, shape:tuple, coverage:Union[list[tuple]] = None, name:str = None) -> None:
        self.kde = kde
        self.name = name
        self.shape = shape

        # inferred values 
        self.coverage = self._process_coverage_parameter(coverage)

        # sanity checks
        # TODO: add sanity checks for coverage bounds
    
    def _process_coverage_parameter(self, coverage:list[tuple[float, float]]) -> list[tuple]:
        dims = self.shape[0]
        if coverage is None:
            return [(-np.inf, np.inf) for dim in range(dims)]
        elif type(coverage) is tuple:
            return [coverage for dim in range(dims)]
        elif type(coverage) is list and len(coverage) == 1:
            coverage = coverage[0]
            return [coverage for dim in range(dims)]
        elif type(coverage) is list and len(coverage) == dims:
            return coverage
        else:
            raise ValueError(f"Parameter coverage {coverage} is not as expected. Check the documentation for valid inputs.")
    
    def _sanity_check(self):
        # check shape
        if not type(self.shape) is tuple and len(self.shape) == 1:
            raise ValueError("Shape of the distribution is {self.shape} but should be a tuple indicating dimensions like (3,). This is probably due to the parameter kde.")
        # check coverage
        # TODO: add sanity check for coverage

    
    def _get_common_limits_of_continuous(self, rv: rv_continuous):
        c1 = np.array(self.coverage)
        c2 = np.array(rv.coverage)

        # Get for the highest minimum and the lowest maximum for every dimension to define the common coverage
        mins = np.maximum(c1[:,0], c2[:,0])
        maxs = np.minimum(c1[:,1], c2[:,1])

        minsmaxs = np.vstack((mins,maxs)).T
        limits = list(map(tuple, minsmaxs))

        return limits
    
    
    def _common_coverage(self, rv: rv_continuous, eval_pts:int=1000) -> np.ndarray: 
        limits = self._get_common_limits_of_continuous(rv)
        return generate_mc_points(limits, eval_pts)
    
    def divergence(self, rv: rv_continuous, div_type:str = "jsd") -> float:
        common_coverage = self._common_coverage(rv)
        pd_x = self.score_samples(common_coverage)
        pd_y = rv.score_samples(common_coverage)

        return self.divergence_from_distribution(pd_x, pd_y, div_type)
            

    def pdf(self, X:npt.ArrayLike) -> np.ndarray:
        """Returns the probability density function at x.

        Args:
            x (np.ndarray): scalar or vector

        Returns:
            np.ndarray: 1-D array of probabilities
        """
        score = self.kde.score_samples(X)
        return np.exp(score)

    def score_samples(self, X:npt.ArrayLike) -> np.ndarray:
        return self.kde.score_samples(X)

 