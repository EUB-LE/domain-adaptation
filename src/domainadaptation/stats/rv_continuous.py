import numpy as np
from sklearn.neighbors import KernelDensity
import numpy.typing as npt 


class rv_continuous():
    def __init__(self, kde:KernelDensity, coverage:list[tuple] = None, name:str = None) -> None:
        self.kde = kde
        self.name = name

        # inferred values 
        self.shape:tuple = self.kde.sample().reshape(-1).shape
        self.coverage = coverage if coverage else [(-np.inf, np.inf) for dim in range(self.shape[0])]

        # sanity checks
        # TODO: add sanity checks for coverage bounds
       

    def pdf(self, X:npt.ArrayLike) -> np.ndarray:
        """Returns the probability density function at x.

        Args:
            x (np.ndarray): scalar or vector

        Returns:
            np.ndarray: 1-D array of probabilities
        """
        score = self.kde.score_samples(X)
        return np.exp(score)

    # implement abstract method(s)
    def score_samples(self, X:npt.ArrayLike) -> np.ndarray:
        return self.kde.score_samples(X)