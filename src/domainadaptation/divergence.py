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


    def pmf(self, X:np.ndarray) -> np.ndarray:
        """Returns the probability mass function at X. 

        Args:
            X (np.ndarray): Scalar or vector 

        Returns:
            np.ndarray: 1-D array of probabilities
        """
        mask = self.xk == X
        prob_matrix = mask * self.pk.reshape(-1,1) 
        pmf = prob_matrix.sum(axis=0)
        return pmf
    
    # implemenent abstract method(s)
    def score_samples(self, X:np.ndarray) -> np.ndarray:
        return np.log(self.pmf(X))

class rv_continuous(rv):
    def __init__(self, kde:KernelDensity, a:np.ndarray = -np.inf, b:np.ndarray = np.inf, name:str=None ) -> None:
        self.kde = kde
        self.name = name
        self.dim = kde.sample().reshape(-1,).shape
        
        if np.ndim(a) == 0:
            self.a = np.full((self.dim), a)
        elif a.shape  == self.dim:
            self.a = a
        else:
            raise ValueError(f"Lower bound of support (a) must be scalar or {self.dim}" )
        
        if np.ndim(b) == 0:
            self.b = np.full((self.dim), b)
        elif a.shape == self.dim:
            self.b = b
        else:
            raise ValueError(f"Upper bound of support (b) must be scalar or {self.dim}" )
    
    def set_a(self, a:np.ndarray):
        """Set lower bound for support of RV. 

        Args:
            a (np.ndarray): lower bound, must be scalar or vector

        Raises:
            ValueError: if a is not of the correct shape
        """
        if np.ndim(a) == 0:
            self.a = np.full((self.dim), a)
        elif a.shape  == self.dim:
            self.a = a
        else:
            raise ValueError(f"Lower bound of support (a) must be scalar or {self.dim}" )
    
    
    def set_b(self, b:np.ndarray):
        """Set upper bound for support of RV. 

        Args:
            b (np.ndarray): upper bound, must be scalar or vector

        Raises:
            ValueError: if b is not of the correct shape
        """
        if np.ndim(b) == 0:
            self.b = np.full((self.dim), b)
        elif b.shape == self.dim:
            self.b = b
        else:
            raise ValueError(f"Upper bound of support (b) must be scalar or {self.dim}" )




    def pdf(self, x:np.ndarray) -> np.ndarray:
        """Returns the probability density function at x. 

        Args:
            x (np.ndarray): scalar or vector

        Returns:
            np.ndarray: 1-D array of probabilities
        """
        return np.exp(self.kde.score_samples(x))
    
    #implement abstract method(s)
    def score_samples(self, X:np.ndarray) -> np.ndarray:
        return self.kde.score_samples(X)

def rv_from_discrete(x:np.ndarray, name:str=None) -> rv_discrete:
    """Create a random variable (rv_discrete) instance from x. 
    X must contain realizations of a discrete random variable.

    Args:
        x (np.ndarray): Vector of samples.
        name (str, optional): Name of the random variable. Defaults to None.

    Returns:
        rv_discrete: Random variable 
    """

    df = pd.DataFrame(x)
    pmf = df.value_counts(ascending=True, normalize=True, sort=False)

    name = name if name else "custom"
    xk = np.array([i for i in pmf.index.values])
    pk = pmf.values

    return rv_discrete(name=name, values=(xk,pk))



def rv_from_continuous(x:np.ndarray, estimator:KernelDensity=None, name:str=None, params:dict={"bandwidth": np.logspace(-1,1,20)}) -> rv_continuous:
    """[summary]

    Args:
        x (np.ndarray): [description]
        estimator (KernelDensity, optional): [description]. Defaults to None.
        name (str, optional): [description]. Defaults to None.
        params (dict, optional): [description]. Defaults to {"bandwidth": np.logspace(-1,1,20)}.

    Returns:
        rv_continuous: [description]
    """
    kde = None
    if estimator is None:
        grid = GridSearchCV(KernelDensity(), params, verbose=1, n_jobs=-1)    
        grid.fit(x)
        kde = grid.best_estimator_
    
    else:
        estimator.fit(x)
        kde = estimator
        
    return rv_continuous(kde, name)


def jsd(pd_x: np.ndarray, pd_y:np.ndarray) -> float:
    m = (1./2.) * (pd_x + pd_y)
    return (1./2.) * entropy(pd_x, m) + (1./2.) * entropy(pd_y, m)

def jsd_from_discrete(rv_x: rv_discrete, rv_y: rv_discrete) -> float:
    xk = np.intersect1d(rv_x.xk, rv_y.xk)

    pd_x = rv_x.score_samples(xk)
    pd_y = rv_y.score_samples(xk)

    return jsd(pd_x, pd_y)

def jsd_from_continuous(rv_x: rv_continuous, rv_y: rv_continuous, eval_pts:int=10000) -> float: 

    # Get lower and upper bounds from RVs
    x_min = rv_x.a 
    y_min = rv_y.a
    x_max = rv_x.b
    y_max = rv_x.b 

    if True in np.isinf([x_min,y_min,x_max,y_max]):
        raise ValueError("Lower and upper bounds for rv_x, rv_y must not contain infinity. Set explicit limits via rv_x.set_a() and rv_x.set_b().")


    # Create uniform distributed sample points in the area that is covered by both RVs
    # Monte Carlo approach
    mc_min = np.array([x_min, y_min]).max(axis=0)
    mc_max = np.array([x_max, y_max]).min(axis=0)
    mc_points = np.random.uniform(low=mc_min, high=mc_max, size=(eval_pts, *x_min.shape))

    # Approximate JSD usind MC sample points
    pd_x = rv_x.score_samples(mc_points)
    pd_y = rv_y.score_samples(mc_points)

    return jsd(pd_x, pd_y)


















