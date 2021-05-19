from operator import pos
from typing import Tuple, List, Union
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
    def score_samples(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Returns the log probabilities for sample X.

        Args:
            X (np.ndarray): sample

        Returns:
            np.ndarray: log probabilities
        """
        pass


class rv_discrete(rv):
    def __init__(self, values: Tuple[Tuple, np.ndarray], name: str = None) -> None:
        self.xk = values[0]
        self.pk = values[1]
        
        pmf_dict = {}
        for i in range(0, len(self.xk)):
            pmf_dict[self.xk[i]] = self.pk[i]
        
        self.pmf_dict = pmf_dict
        self.name = name

    def pmf(self, X: np.ndarray) -> np.ndarray:
        """Returns the probability mass function at X.

        Args:
            X (np.ndarray): Scalar or vector

        Returns:
            np.ndarray: 1-D array of probabilities
        """

        #TODO implement multidimensional adaptation. This requires a rework.
        # mask = self.xk == X
        # prob_matrix = mask * self.pk.reshape(-1, 1)
        # pmf = prob_matrix.sum(axis=0)
        # return pmf

        # TODO implement 0 for values that are not present

        return np.array([self.pmf_dict[i] for i in X])

    # implemenent abstract method(s)
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return np.log(self.pmf(X))


class rv_continuous(rv):
    def __init__(self, kde: KernelDensity, a: np.ndarray = -np.inf, b: np.ndarray = np.inf, name: str = None) -> None:
        self.kde = kde
        self.name = name
        self.dim = kde.sample().reshape(-1,).shape

        if np.ndim(a) == 0:
            self.a = np.full((self.dim), a)
        elif a.shape == self.dim:
            self.a = a
        else:
            raise ValueError(
                f"Lower bound of support (a) must be scalar or {self.dim}")

        if np.ndim(b) == 0:
            self.b = np.full((self.dim), b)
        elif a.shape == self.dim:
            self.b = b
        else:
            raise ValueError(
                f"Upper bound of support (b) must be scalar or {self.dim}")

    def set_a(self, a: np.ndarray):
        """Set lower bound for support of RV.

        Args:
            a (np.ndarray): lower bound, must be scalar or vector

        Raises:
            ValueError: if a is not of the correct shape
        """
        if np.ndim(a) == 0:
            self.a = np.full((self.dim), a)
        elif a.shape == self.dim:
            self.a = a
        else:
            raise ValueError(
                f"Lower bound of support (a) must be scalar or {self.dim}")

    def set_b(self, b: np.ndarray):
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
            raise ValueError(
                f"Upper bound of support (b) must be scalar or {self.dim}")

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Returns the probability density function at x.

        Args:
            x (np.ndarray): scalar or vector

        Returns:
            np.ndarray: 1-D array of probabilities
        """
        return np.exp(self.kde.score_samples(x))

    # implement abstract method(s)
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self.kde.score_samples(X)


class rv_mixed_joint(rv):
    def __init__(self, rv1: rv_continuous, rv2: rv_discrete, cond_kdes: dict = None, name:str = None) -> None:
        self.rv1 = rv1
        self.rv2 = rv2
        self.name = name

        keys = np.array([key for key in cond_kdes.keys()])
        if not np.array_equal(keys, rv2.xk):
            raise KeyError(
                "Keys of cond_kdes must be identical to possible values for rv2, i.e. rv2.xk.")
        self.cond_kdes = cond_kdes

    def _get_cond_kde(self, key: Union[tuple, np.ndarray, list]) -> KernelDensity:
        # try to use np if key is not already iterable
        try:
            iter(key)
        except TypeError:
            key = [key]

        # cast iterable key to tuple
        if type(key) is not tuple:
            key = tuple(key)

        return self.cond_kdes[key]

    def pdf(self, x:np.ndarray, y:np.ndarray): 
        result = np.empty(shape=x.shape[0])
        for label in np.unique(y, axis=0):
            kde = self._get_cond_kde(label)
            result[y == label] = kde.score_samples(x[y == label])
        return np.exp(result)         
        
    def score_samples(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.log(self.pdf(X,y))


def rv_from_discrete(x:np.ndarray, name:str=None) -> rv_discrete:
    """Create a random variable (rv_discrete) instance from x. 
    X must contain realizations of a discrete random variable.

    Args:
        x (np.ndarray): Vector of samples.
        name (str, optional): Name of the random variable. Defaults to None.

    Returns:
        rv_discrete: Random variable 
    """
    name = name 

    df = pd.DataFrame(x)
    pmf = df.value_counts(ascending=True, normalize=True, sort=False)
    xk = pmf.index.values
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

def rv_from_joint(x:np.ndarray, y:np.ndarray, rv_x:rv, rv_y:rv, name:str=None, params:dict={"bandwidth": np.logspace(-1,1,20)}) -> rv:
    if rv_x.pmf is not None and rv_x.pmf is not None:
        xy = np.hstack((x,y))
        return rv_from_discrete(xy, name)

    elif type(rv_x) is rv_continuous and type(rv_y) is rv_continuous: 
        xy = np.hstack((x,y))
        return rv_from_continuous(xy, name=name)
    
    elif type(rv_x) is rv_discrete and type(rv_y) is rv_continuous:
        raise ValueError("If two RVs have different types, the continuous RV must be at position x, rv_x.")
    
    else:

        cond_kdes = {} 
        labels = pd.DataFrame(y).value_counts(ascending=True, normalize=True, sort=False).index.values
        print(f"Fitting KDEs for every label in {labels}. This may take a while. Consider saving the cond_kdes and using the direct constructor of rv_mixed_joint to save time.")
        for label in labels: 
            sample = x[y == label]
            grid = GridSearchCV(KernelDensity(), params, verbose=1, n_jobs=-1)    
            grid.fit(sample)
            cond_kdes[label] = grid.best_estimator_
    
        return rv_mixed_joint(rv_x, rv_y, cond_kdes, name)


    




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
    y_max = rv_y.b 

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






















