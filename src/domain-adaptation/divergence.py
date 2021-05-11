from typing import Tuple
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.base import clone
from sklearn.base import BaseEstimator 
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from abc import ABC, abstractmethod

class rv(ABC):

    @abstractmethod
    def score_samples(self, X:np.ndarray) -> np.ndarray:
        """Returns the log probabilities for sample X.

        Args:
            X (np.ndarray): sample 

        Returns:
            np.ndarray: log probabilities 
        """
        pass


class rv_discrete(rv):
    def __init__(self, values:Tuple[Tuple, np.ndarray], name:str=None) -> None:
        self.xk = values[0]
        self.pk = values[1]
        self.name = name

        self._lookup = dict()
        for (x, p) in zip (self.xk, self.pk):
            self._lookup[x] = p

    # TODO: this does not work properly yet, bc lookup is not the right solution. 

    def pmf(self, X:np.ndarray) -> np.ndarray:
        return np.array([self._lookup[(x,)] for x in X])
    
    def score_samples(self, X:np.ndarray) -> np.ndarray:
        return np.log(self.pmf(X))

class rv_continuous(rv):
    def __init__(self, kde:KernelDensity, name:str=None ) -> None:
        self.kde = kde
        self.name = name

    def pdf(self, x:np.ndarray) -> np.ndarray:
        return np.exp(self.kde.score_samples(x))
    
    def score_samples(self, X:np.ndarray) -> np.ndarray:
        return self.pdf(X)

def rv_from_discrete(x:np.ndarray, name:str=None) -> rv_discrete:

    df = pd.DataFrame(x)
    pmf = df.value_counts(ascending=True, normalize=True)

    name = name if name else "custom"
    xk = pmf.index
    pk = pmf.values
    return rv_discrete(name=name, values=(xk,pk))



def rv_from_continuous(x:np.ndarray, estimator:KernelDensity=None, name:str=None, params:dict={"bandwidth": np.logspace(-1,2,20)}) -> rv_continious:
    kde = None
    if estimator is None:
        grid = GridSearchCV(KernelDensity(), params, verbose=4, n_jobs=-1)    
        grid.fit(x)
        kde = grid.best_estimator_
    
    else:
        estimator.fit(x)
        kde = estimator
    
    return rv_continuous(kde, name)


def jsd(pd_x: np.ndarray, pd_y:np.ndarray) -> float:
  m = (1./2.) * (pd_x + pd_y)
  return (1./2.) * entropy(pd_x, m) + (1./2.) * entropy(pd_y, m)

def prior_shift_discrete(y_s:np.ndarray, y_t:np.ndarray):
    pmf_s = rv_from_discrete(y_s)
    pmf_t = rv_from_discrete(y_t)



    pd_s, pd_t = pmf_s.align(pmf_t, join="inner")
    return jsd(pd_s, pd_t)

def make_pd_from_pdf(x:np.ndarray, pdf:KernelDensity) -> np.ndarray:
    return pdf.score_samples(x)













