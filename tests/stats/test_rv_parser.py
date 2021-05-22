from __future__ import annotations
import unittest
import numpy as np
from numpy import random
from daproperties.stats._rv_parser import rv_from_continuous, rv_from_discrete, rv_from_mixed
from daproperties.stats import rv_discrete, rv_continuous, rv_mixed

class TestRVParsers(unittest.TestCase):
    
    def setUp(self):
        pass

    
    def tearDown(self):
        #del self.rv
        pass

    def test_rv_from_discrete(self): 
        y = [0,0,1,1,0,0,1,1,0,0]
        rv = rv_from_discrete(y, name="discrete")

        xk_is = rv.xk
        pk_is = rv.pk
        name_is = rv.name

        xk_target = [(0,),(1,)]
        pk_target = [0.6, 0.4]
        name_target = "discrete"

        self.assertEqual(type(rv), rv_discrete)
        self.assertTrue(np.array_equal(pk_is, pk_target), f"pk_is {pk_is} is not pk_target {pk_target}.")
        self.assertListEqual(rv.xk, xk_target)
        self.assertEqual(rv.pmf(0), 0.6)
        self.assertEqual(name_is, name_target)

    def test_rv_from_continuous(self): 
        X = random.uniform(size=(100,2))
        rv = rv_from_continuous(X, name="continuous")

        coverage_is = rv.coverage
        coverage_target = [(0,1),(0,1)]

        self.assertEqual(type(rv), rv_continuous)
        # TODO: add tests for coverage
        self.assertGreater(rv.pdf([[0,1]]), 0)
    
    def test_rv_from_mixed_only_data(self):
        X = random.standard_normal(size=(1000,2))
        y = np.apply_along_axis(lambda x: 1 if x[0]+x[1] > 0 else 0, 1, X)

        rv = rv_from_mixed(X, y)
        self.assertEqual(type(rv), rv_mixed)
        self.assertEqual(type(rv.rv1), rv_continuous)
        self.assertEqual(type(rv.rv2), rv_discrete)
        self.assertGreater(rv.pdf([[0.5,0.5]], [1]), 0)

    def test_rv_from_data_and_rvs(self):
        X = random.standard_normal(size=(1000,2))
        y = np.apply_along_axis(lambda x: 1 if x[0]+x[1] > 0 else 0, 1, X)

        rv_x = rv_from_continuous(X)
        rv_y = rv_from_discrete(y)

        rv = rv_from_mixed(X,y, rv_x, rv_y)
        self.assertEqual(type(rv), rv_mixed)
        

if __name__ == '__main__':
    unittest.main()

    