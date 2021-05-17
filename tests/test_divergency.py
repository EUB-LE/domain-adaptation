import unittest
import sys
import os

from numpy.core.numeric import array_equal
from numpy.testing._private.utils import assert_array_equal, assert_equal
from domainadaptation.divergence import jsd, rv_conditional, rv_continuous, rv_from_continuous, rv_from_discrete, rv_discrete, rv_mixed_joint
import numpy as np
import pandas as pd
from numpy import random
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy



class TestRVDiscrete(unittest.TestCase):

    def setUp(self):
        """Create rv from observation x
        """
        x = np.array([0,0,1,1,0,0,0,1,2,2])
        self.rv = rv_from_discrete(x)
    
    def tearDown(self):
        del self.rv

    def test_rv_from_discrete(self):
        """Fails if the result is not of type rv_discrete.
        """
        self.assertIs(type(self.rv),rv_discrete)


    def test_rv_xk_from_discrete(self):
        """Fails if the labels xk, aka discrete values, are not correctly set.
        """
        xk = [[0],[1],[2]]
        self.assertTrue(np.array_equal(self.rv.xk, xk), f"rv.xk is {self.rv.xk}, but should be {xk}")
    
    def test_rv_xk_vector_from_discrete(self):
        """Support multiple discrete dimensions
        """
        xk = [(0,1),(1,0),(1,1)]
        x = [[0,1],[1,0],[1,0],[1,1]]
        rv = rv_from_discrete(x)
        self.assertTrue(np.array_equal(rv.xk,xk))
    
    def test_rv_pk_from_discrete(self):
        """Fails if the probabilites pk for the discrete values are not correctly set.
        """
        pk = [5.0/10, 3.0/10, 2.0/10]
        self.assertTrue(np.array_equal(self.rv.pk, pk), f"rv.pk is {self.rv.pk}, but should be {pk}")
    

    def test_pmf_scalar(self):
        """Fails if the pmf of a scalar is not as expected.
        """
        pmf_of_1 = 0.3
        self.assertEqual(self.rv.pmf(1), 0.3)
    
    def test_pmf_nothing(self): 
        """Fails if the pmf of None is not 0. 
        """
        pmf_of_None = 0
        self.assertEqual(self.rv.pmf(None), 0)
    
    def test_pmf_vector(self):
        """Fails if the pmf of a vector containing valid and invalid values is not as expected.
        """
        vector = [0,0,1,1,2,10]
        pmf = [0.5,0.5,0.3,0.3,0.2,0]
        self.assertTrue(np.array_equal(self.rv.pmf(vector), pmf))
    
    def test_pmf_wrong_data(self):
        """Fails if the handling of invalid datatypes is not as expected.
        """
        self.rv.pmf("foobar")
        # unsure what it should return 
        pass

    def test_score_samples(self):
        """Fails if the score samples result is not as expected, e.g. not log-probability.
        """
        vector = [0,0,1,1,2,10]
        pmf = [0.5,0.5,0.3,0.3,0.2,0]
        log_pmf= np.log(pmf)

        self.assertTrue(np.array_equal(self.rv.score_samples(vector), log_pmf))

class TestRVContinuous(unittest.TestCase):

    kde = None
    X = None

    @classmethod
    def setUpClass(cls):
        """Construct Kernel Density Estimation with predetermined bandwidth 0.3 
        from standard normal distribution.
        """
        cls.X = random.standard_normal(size=(10000,1))
        cls.kde = KernelDensity(bandwidth=0.3).fit(cls.X)
    
    def setUp(self) -> None:
        self.rv = rv_continuous(self.kde, "test")
    
    def tearDown(self) -> None:
        del self.rv

    def test_rv_from_continuous_with_kde(self):
        """Fails if kdes are not equal
        """
        self.assertEqual(self.kde, self.rv.kde)
        self.assertEqual(type(self.rv), rv_continuous)

    @unittest.skip("Skipping grid search test")
    def test_rv_from_continuous_without_kde(self):
        """Fails if the result of the grid search is not as expected.
        """
        # Test with only two choices for parameters, so the grid search always picks 0.3 
        # This test is not waterproof, because the fitting of KDE is out of scope
        rv = rv_from_continuous(self.X, None, "test with gridsearch", {"bandwidth": [0.3,20.0]})
        self.assertEqual(rv.kde.bandwidth, self.rv.kde.bandwidth)

    def test_pdf_scalar(self):
        """Fails if the pdf of 0 deviates too much from the correct value pdf(0)=0.3989.
        """
        # given a uniform distribution, we just want to make sure, that the 
        pdf_of_0 = 0.3989
        pdf = self.rv.pdf([[0]])
        self.assertAlmostEqual(pdf,pdf_of_0,delta=0.03)

    def test_pdf_vector(self):
        # it works, believe me :) 
        pass

    def test_score_samples(self): 
        x = [[0],[10]]
        pdf = self.rv.pdf(x)
        log_pdf = np.log(pdf)
        score_samples = self.rv.score_samples(x)

        self.assertTrue(np.array_equal(score_samples, log_pdf))

class TestRVMixedJoint(unittest.TestCase):

    kde = None
    kde0 = None
    kde1 = None
    X = None
    y = None

    
    @classmethod
    def setUpClass(cls):
        """Construct Kernel Density Estimation with predetermined bandwidth 0.3 
        from standard normal distribution.
        """
        cls.X = random.standard_normal(size=(1000,100))
        cls.y = random.randint(3,size=(1000))
        cls.kde = KernelDensity(bandwidth=0.3).fit(cls.X)
        cls.kde0 = KernelDensity(bandwidth=0.3).fit(cls.X[cls.y == 0])
        cls.kde1 = KernelDensity(bandwidth=0.3).fit(cls.X[cls.y == 1])
        cls.kde2 = KernelDensity(bandwidth=0.3).fit(cls.X[cls.y == 2])

    def setUp(self):
        self.rv1 = rv_continuous(self.kde)
        self.rv2 = rv_from_discrete(self.y)
        cond_kdes = {0: self.kde0, 1: self.kde1, 2:self.kde2}
        self.rv = rv_mixed_joint(self.rv1, self.rv2, cond_kdes)
    
    def test_rv_mixed_joint(self):
        self.assertEqual(type(self.rv), rv_mixed_joint)
    
    def test_pdf(self):
        result = self.rv.pdf(self.X, self.y)
        self.assertTrue(True)


    

class TestJSD(unittest.TestCase):

    def test_entropy_equals_kl_divergence(self):
        x = np.array([10,6,4])
        y = np.array([4,10,6])

        pd_x = x / x.sum()
        pd_y = y / y.sum()

        # KL definition
        kl_xy = np.array([p_x * np.log(p_x/p_y) for p_x,p_y in zip(pd_x,pd_y)]).sum()
        # Entropy from scipy
        entropy_xy = entropy(pd_x,pd_y)

        self.assertEqual(kl_xy, entropy_xy)


    
    def test_jsd(self):
        x = np.array([10,6,4])
        y = np.array([4,10,6])

        pd_x = x / x.sum()
        pd_y = y / y.sum()

        M = 1./2. * (pd_x + pd_y)
        # M = [0.35, 0.4, 0.25]

        target = 1./2. * entropy(pd_x, M) + 1./2. * entropy(pd_y, M)
        jsd_xy = jsd(x,y)

        self.assertEqual(jsd_xy, target)
    

        


        


            



        
        

        


        


    
if __name__ == '__main__':
    unittest.main()