import unittest
import numpy as np
from scipy.stats import entropy
#from domainadaptation.measures import jsd

class TestJSD(unittest.TestCase):
    
    def test_ok(self):
        self.assertTrue(True)
"""
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
"""
    
    