import unittest
import numpy as np
from scipy.stats import entropy

class TestKLD(unittest.TestCase):

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
