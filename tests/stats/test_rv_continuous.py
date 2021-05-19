import unittest
import numpy as np
from domainadaptation.stats import rv_continuous
from numpy import random
from sklearn.neighbors import KernelDensity

class TestRVContinuous(unittest.TestCase):
    X = None
    y = None
    kde = None

    # probability of 2-dimensional standardnormal distribution for (0,0)
    # https://de.wikipedia.org/wiki/Mehrdimensionale_Normalverteilung
    p_00 = 0.1591

    @classmethod
    def setUpClass(cls):
        """Construct Kernel Density Estimation with predetermined bandwidth 0.3 
        from standard normal distribution.
        """
        cls.X = random.standard_normal(size=(1000,2))
        cls.y = np.apply_along_axis(lambda x: 1 if x[0]+x[1] > 0 else 0, 1, cls.X)
        cls.kde = KernelDensity(bandwidth=0.3).fit(cls.X)
    
    def test_bounds_from_scalar(self):
        rv = rv_continuous(self.kde)
        a_target = [0,0]
        a_is = rv.a
        b_target = [1,1]
        b_is = rv.b
        self.assertTrue(np.array_equal(a_is, a_target))
        self.assertTrue(np.array_equal(b_is, b_target))
            
    
    def test_pdf_of_2d_array(self): 
        rv = rv_continuous(self.kde)
        X = [[0,0], [0,0]]
        P_is = rv.pdf(X)
        P_target = np.array([self.p_00, self.p_00])
        is_close = np.allclose(P_is, P_target, atol=0.02)
        self.assertTrue(is_close, f"P_is: {P_is} is not close enough to P_target: {P_target}.")
    
    def test_pdf_of_1d_array_raises_Error(self):
        rv = rv_continuous(self.kde)
        X = [0,0]
        with self.assertRaises(ValueError):
            P_is = rv.pdf(X)
    
    def test_score_samples(self):
        rv = rv_continuous(self.kde)
        X = [[0,0], [0,0]]
        P_is = rv.score_samples(X)
        P_target = np.log(np.array([self.p_00, self.p_00]))
        is_close = np.allclose(P_is, P_target, atol=0.2)
        self.assertTrue(is_close, f"P_is: {P_is} is not close enough to P_target: {P_target}.")

 


        