from __future__ import annotations
import unittest
from domainadaptation.helper._rv_divergence import _divergence

class TestRVDivergence(unittest.TestCase):

    def test_divergence_private(self):
        pd_x = [1,1,1,1,0,0]
        pd_y = [0,0,0,0,1,1]

        divergence_is = _divergence(pd_x, pd_y)
        divergence_target = 0
        pass
