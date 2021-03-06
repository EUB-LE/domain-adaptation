import unittest
import numpy as np
from numpy import random
from sklearn.neighbors import KernelDensity
from daproperties.stats import rv_continuous, rv_discrete, rv_mixed


class TestRVMixed(unittest.TestCase):
    
    y = None
    X = None
    kde = None

    # conditional kdes
    kde0 = None
    kde1 = None

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

        # Fit conditional kdes
        cls.kde0 = KernelDensity(bandwidth=0.3).fit(cls.X[cls.y == 0])
        cls.kde1 = KernelDensity(bandwidth=0.3).fit(cls.X[cls.y == 1])
    

    def test_get_cond_kdes_from_scalar(self):
        rv_x = rv_continuous(self.kde, (2,))
        rv_y = rv_discrete(xk=[(0,), (1,)], pk=[.5, .5])

        cond_kdes = {(0,): KernelDensity(bandwidth=1.0), (1,): KernelDensity(bandwidth=99.0)}
        rv = rv_mixed(rv_x, rv_y, cond_kdes)

        kde_is = rv._get_cond_kde(0)
        kde_target = cond_kdes[(0,)]

        self.assertEqual(kde_is, kde_target)
    
    def test_get_cond_kdes_from_array(self): 
        rv_x = rv_continuous(self.kde, (2,))
        rv_y = rv_discrete(xk=[(0,0), (1,0)], pk=[.5, .5])

        cond_kdes = {(0,0): KernelDensity(bandwidth=1.0), (1,0): KernelDensity(bandwidth=99.0)}
        rv = rv_mixed(rv_x, rv_y, cond_kdes)

        kde_is = rv._get_cond_kde([0,0])
        kde_target = cond_kdes[(0,0)]

        self.assertEqual(kde_is, kde_target)
    
    def test_get_cond_kdes_from_tuple(self): 
        rv_x = rv_continuous(self.kde,(2,))
        rv_y = rv_discrete(xk=[(0,0), (1,0)], pk=[.5, .5])

        cond_kdes = {(0,0): KernelDensity(bandwidth=1.0), (1,0): KernelDensity(bandwidth=99.0)}
        rv = rv_mixed(rv_x, rv_y, cond_kdes)

        kde_is = rv._get_cond_kde((0,0))
        kde_target = cond_kdes[(0,0)]

        self.assertEqual(kde_is, kde_target)
    
    def test_get_cond_kdes_wrong_key(self):
        rv_x = rv_continuous(self.kde, (2,))
        rv_y = rv_discrete(xk=[(0,), (1,)], pk=[.5, .5])

        cond_kdes = {(0,): KernelDensity(bandwidth=1.0), (1,): KernelDensity(bandwidth=99.0)}
        rv = rv_mixed(rv_x, rv_y, cond_kdes)

        with self.assertRaises(KeyError):
            kde_is = rv._get_cond_kde(3)
    
    def test_pdf_for_1d_discrete(self):
        rv_x = rv_continuous(self.kde,(2,))
        rv_y = rv_discrete(xk=[(0,), (1,)], pk=[.5, .5])

        cond_kdes = {(0,): self.kde0, (1,): self.kde1}
        rv = rv_mixed(rv_x, rv_y, cond_kdes)

        P_is = rv.pdf(x=[[0,0],[0,0]], y=[1,0]) 
        P_target = [self.p_00 * 0.5, self.p_00 * 0.5]

        is_close = np.isclose(P_is, P_target, rtol=0.2)

        self.assertTrue(is_close.all(), f"P_is: {P_is} is not close enough to P_target: {P_target}. ")
    
    def test_pdf_for_invalid_discrete(self):
        rv_x = rv_continuous(self.kde, (2,))
        rv_y = rv_discrete(xk=[(0,), (1,)], pk=[.5, .5], badvalue=0)

        cond_kdes = {(0,): self.kde0, (1,): self.kde1}
        rv = rv_mixed(rv_x, rv_y, cond_kdes)

        P_is = rv.pdf(x=[[0,0],[0,0],[1,0]], y=[1,0,99]) 
        P_target = [self.p_00 * 0.5, self.p_00 * 0.5, 0]

        is_close = np.isclose(P_is, P_target, rtol=0.3, equal_nan=True)
        
        self.assertTrue(is_close.all(), f"P_is: {P_is} is not close enough to P_target: {P_target}. ")

    @unittest.skip("No support for multidimensional discrete variables at the moment.")
    def test_pdf_for_2d_discrete(self):
        rv_x = rv_continuous(self.kde, (2,))
        
        # 2d discrete (0,0) if for (x1,x2) : x1 < 0 and x2 < 0, (1,0) if x1 >= 0 and x2 < 0 and so on.
        y0 = np.apply_along_axis(lambda x: 1 if x[0] >= 0 else 0, 1, self.X)
        y1 = np.apply_along_axis(lambda x: 1 if x[1] >= 0 else 0, 1, self.X)
        y = np.vstack((y0,y1)).T

        # TODO: Implement sensible behavior for multidimensional discrete variables.
        # At the moment this fails, beause numpy does not support fast row selection via a multidimensional index, 
        # e.g.: if y = [[0,1], [1,1]], then y == [0,1] results in [[True, True], [False, True]] 
        # and not in [True, False] as required for this use case in order to use the result as a selection for X and train KDEs. 
        pass

    
    def test_common_coverage(self): 
        rv1d = rv_discrete([(0,),(1,),(2,)], pk=[0.3, 0.3, 0.4])
        rv2d = rv_discrete([(0,),(1,),(4,)], pk=[0.2, 0.4, 0.4])

        c1 = [(-2,2),(1,5)]
        c2 = [(-1,4),(-3,3)]
        rv1c = rv_continuous(KernelDensity().fit([[0,1]]), (2,), c1)
        rv2c = rv_continuous(KernelDensity().fit([[0,1]]), (2,), c2)


        rv1 = rv_mixed(rv1c, rv1d, {(0,): KernelDensity().fit([[1,1]])})
        rv2 = rv_mixed(rv2c, rv2d, {(0,): KernelDensity().fit([[1,0]])})

        y_target = np.repeat([0,1],1000)

        x, y = rv1._common_coverage(rv2, 1000)
        self.assertEqual(x.shape[0], 2000)
        self.assertEqual(x.shape[1], 2)
        self.assertEqual(y.shape[0], x.shape[0])
        self.assertTrue(np.array_equal(y, y_target), f"y {y} is not equal y_target {y_target}.")
        # TODO: actual check if the common coverage for X is correct. 
    
    def test_divergence(self): 
        rv1d = rv_discrete([(0,),(1,),(2,)], pk=[0.3, 0.3, 0.4])
        rv2d = rv_discrete([(0,),(1,),(4,)], pk=[0.2, 0.4, 0.4])

        c1 = [(-2,2),(1,5)]
        c2 = [(-1,4),(-3,3)]
        # c_target = [(-1,2),(1,3)]
        rv1c = rv_continuous(KernelDensity().fit([[-1,2]]), (2,), c1)
        rv2c = rv_continuous(KernelDensity().fit([[1,3]]), (2,), c2)

        # the data makes no sense, this is just a pro forma test
        rv1 = rv_mixed(rv1c, rv1d, {(0,): KernelDensity().fit([[-1,2]]), (1,): KernelDensity().fit([[-2,3], [-1,4]])})
        rv2 = rv_mixed(rv2c, rv2d, {(0,): KernelDensity().fit([[-1,2]]), (1,): KernelDensity().fit([[-3,4], [0,0]])})

        divergence = rv1.divergence(rv2) 
        self.assertEqual(type(divergence), np.float64)
        self.assertGreater(divergence, 0.0)

    

if __name__ == '__main__':
    unittest.main()
    





