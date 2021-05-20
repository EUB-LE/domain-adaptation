from __future__ import annotations
import unittest
from domainadaptation.stats import rv_continuous, rv_discrete, rv_mixed
from sklearn.neighbors import KernelDensity
from domainadaptation.helper._coverage import _get_common_limits_of_continuous, common_coverage_of_continuous_rvs, common_coverage_of_discrete_rvs, common_coverage_of_mixed_rvs
from domainadaptation.helper import rv_from_continuous, rv_from_discrete, rv_from_mixed
import numpy as np
from numpy import random

class TestCoverage(unittest.TestCase):

    def test_common_coverage_discrete(self):
        rv1 = rv_discrete([(0,),(1,),(2,)], pk=[0.3, 0.3, 0.4])
        rv2 = rv_discrete([(0,),(1,),(4,)], pk=[0.2, 0.4, 0.4])

        coverage_is = common_coverage_of_discrete_rvs(rv1, rv2)
        coverage_target = np.array([(0,),(1,)]).reshape(-1)

        self.assertTrue(np.array_equal(coverage_is, coverage_target), f"covarage_is {coverage_is} not equal to coverage_target {coverage_target}.")


    def test_common_limits_of_continuous(self):
        c1 = [(-2,2),(1,5)]
        c2 = [(-1,4),(-3,3)]
        
        rv1 = rv_continuous(KernelDensity().fit([[0,1]]), c1)
        rv2 = rv_continuous(KernelDensity().fit([[0,1]]), c2)

        c_is = _get_common_limits_of_continuous(rv1, rv2)
        c_target = [(-1,2),(1,3)]

        self.assertListEqual(c_is, c_target)

    def test_common_coverage_of_continuous(self):
        c1 = [(-2,2),(1,5)]
        c2 = [(-1,4),(-3,3)]
        
        rv1 = rv_continuous(KernelDensity().fit([[0,1]]), c1)
        rv2 = rv_continuous(KernelDensity().fit([[0,1]]), c2)

        mc_points = common_coverage_of_continuous_rvs(rv1, rv2, 1000)
        self.assertEqual(mc_points.shape[0], 1000)
        self.assertEqual(mc_points.shape[1], rv1.shape[0])


    def test_common_coverage_of_mixed(self): 
        y1 = random.randint(0,2, 1000)
        y2 = random.randint(0,3, 1000)

        rv1d = rv_from_discrete(y1)
        rv2d = rv_from_discrete(y2)

        X1 = random.uniform(-5, 5, size=(1000,2))
        X2 = random.uniform(0,10, size=(1000,2))



        rv1c = rv_from_continuous(X1)
        rv2c = rv_from_continuous(X2)

        rv1 = rv_from_mixed(X1, y1, rv1c, rv1d)
        rv2 = rv_from_mixed(X2, y2, rv2c, rv2d)

        x, y = common_coverage_of_mixed_rvs(rv1, rv2, 1000)
        self.assertEqual(x.shape[0], 2000)
        self.assertEqual(x.shape[1], 2)
        self.assertEqual(y.shape[0], x.shape[0])

    