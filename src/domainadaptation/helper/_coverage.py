from domainadaptation.stats import rv_discrete, rv_continuous, rv_mixed
import numpy as np

def common_coverage_of_discrete_rvs(rv1: rv_discrete, rv2: rv_discrete) -> np.ndarray:
    xk1 = np.array(rv1.xk)
    xk2 = np.array(rv2.xk)

    return np.intersect1d(xk1, xk2)

def common_coverage_of_continuous_rvs(rv1: rv_continuous, rv2: rv_continuous) -> np.ndarray: 
    pass
