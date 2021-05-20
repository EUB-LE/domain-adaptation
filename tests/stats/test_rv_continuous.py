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
    
    def test_coverage_default(self):
        rv = rv_continuous(self.kde, (2,))
        coverage_is = rv.coverage
        coverage_target = [(-np.inf, np.inf), (-np.inf, np.inf)]
        self.assertTrue(np.array_equal(coverage_is, coverage_target), f"coverage_is: {coverage_is} is not coverage_target: {coverage_target}.")
    
    def test_coverage_from_tuple(self):
        rv = rv_continuous(self.kde, (2,), coverage=(-1,1))
        coverage_is = rv.coverage
        coverage_target = [(-1,1),(-1,1)]
        self.assertTrue(np.array_equal(coverage_is, coverage_target), f"coverage_is: {coverage_is} is not coverage_target: {coverage_target}.")
    
    def test_coverage_from_single_item_list(self):
        rv = rv_continuous(self.kde, (2,), coverage=[(-1,1)])
        coverage_is = rv.coverage
        coverage_target = [(-1,1),(-1,1)]
        self.assertTrue(np.array_equal(coverage_is, coverage_target), f"coverage_is: {coverage_is} is not coverage_target: {coverage_target}.")
    
    def test_coverage_from_multiple_item_list(self):
        rv = rv_continuous(self.kde, (2,), coverage=[(-1,1),(-2,2)])
        coverage_is = rv.coverage
        coverage_target = [(-1,1),(-2,2)]
        self.assertTrue(np.array_equal(coverage_is, coverage_target), f"coverage_is: {coverage_is} is not coverage_target: {coverage_target}.")
    
    def test_coverage_from_invalid_parameter(self):
        with self.assertRaises(ValueError):
            rv = rv_continuous(self.kde, (2,), coverage=[(-1,+1), (-2,+2), (-3,+3)])

    
            
    
    def test_pdf_of_2d_array(self): 
        rv = rv_continuous(self.kde, (2,))
        X = [[0,0], [0,0]]
        P_is = rv.pdf(X)
        P_target = np.array([self.p_00, self.p_00])
        is_close = np.allclose(P_is, P_target, atol=0.03)
        self.assertTrue(is_close, f"P_is: {P_is} is not close enough to P_target: {P_target}.")
    
    def test_pdf_of_1d_array_raises_Error(self):
        rv = rv_continuous(self.kde, (2,))
        X = [0,0]
        with self.assertRaises(ValueError):
            P_is = rv.pdf(X)
    
    def test_score_samples(self):
        rv = rv_continuous(self.kde, (2,))
        X = [[0,0], [0,0]]
        P_is = rv.score_samples(X)
        P_target = np.log(np.array([self.p_00, self.p_00]))
        is_close = np.allclose(P_is, P_target, atol=0.2)
        self.assertTrue(is_close, f"P_is: {P_is} is not close enough to P_target: {P_target}.")

if __name__ == '__main__':
    unittest.main()


        