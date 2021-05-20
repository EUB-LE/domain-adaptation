from __future__ import annotations
import numpy as np
from numpy.lib import Arrayterator
from domainadaptation.stats import rv_continuous, rv_discrete
from sklearn.neighbors import KernelDensity
import numpy.typing as npt
from domainadaptation.helper._coverage import common_coverage_of_mixed_rvs



class rv_mixed():
    def __init__(self, rv1: rv_continuous, rv2: rv_discrete, cond_kdes: dict = None, name:str = None) -> None:
        self.rv1 = rv1
        self.rv2 = rv2
        self.name = name

        keys = np.array([key for key in cond_kdes.keys()])

        """
        TODO: adapt check here to deal with tuple <-> array conversion for numpy.array_equal

        if not np.array_equal(keys, rv2.xk):
            raise KeyError(
                "Keys of cond_kdes must be identical to possible values for rv2, i.e. rv2.xk.")
        """
        self.cond_kdes = cond_kdes
        

    def _get_cond_kde(self, key:npt.ArrayLike) -> KernelDensity:
        # cast to single-item list if not already iterable
        try:
            iter(key)
        except TypeError:
            key = [key]

        # cast iterable key to tuple
        if type(key) is not tuple:
            key = tuple(key)

        return self.cond_kdes[key]

    def pdf_given_y(self, x:npt.ArrayLike, y:npt.ArrayLike) -> np.ndarray: 
        x = np.array(x)
        y = np.array(y)
        if y.ndim == 2 or y.ndim == 0:
            y = y.reshape(-1)
        
        p_x_given_y = np.empty(shape=x.shape[0])

        labels = np.unique(y, axis=0)
   
        for label in labels:
            try: 
                kde = self._get_cond_kde(label)
                p_x_given_y[y == label] = np.exp(kde.score_samples(x[y == label]))
            except KeyError:
                # if no kde for the label is present, the entries can remain 0. 
                p_x_given_y[y == label] = 0
        
        return p_x_given_y

    def pdf(self, x:npt.ArrayLike, y:npt.ArrayLike) -> np.ndarray: 
        p_x_given_y = self.pdf_given_y(x,y)
        p_y = self.rv2.pmf(y)

        p_x_and_y = p_x_given_y * p_y 
        return p_x_and_y
    
    def pmf_given_x(self, x:npt.ArrayLike, y:npt.ArrayLike) -> np.ndarray:
        p_x_and_y = self.pdf(x,y)
        p_x = self.rv1.pdf(x)

        p_y_given_x = p_x_and_y / p_x
        return p_x_and_y
        
    def score_samples(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.log(self.pdf(X,y))

