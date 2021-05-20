import unittest
import numpy as np
from domainadaptation.stats import rv_discrete



class TestRVDiscrete(unittest.TestCase):

    def setUp(self):
        pass

    
    def tearDown(self):
        #del self.rv
        pass

    def test_xk_and_pk_dimensions(self): 
        xk = [(0,1),(1,0), (1,1)]
        pk = [0.6, 0.4]
        with self.assertRaises(ValueError):
            rv = rv_discrete(xk, pk)

    
    def test_prob_of_tuple(self):
        xk = [(0,1),(1,0)]
        pk = [0.6, 0.4]
        rv = rv_discrete(xk, pk)
        p = rv._prob_of((0,1))
        self.assertEqual(p, 0.6)
    
    def test_prob_of_scalar(self):
        xk = [(0,),(1,)]
        pk = [0.6, 0.4]
        rv = rv_discrete(xk, pk)
        p = rv._prob_of(0)
        self.assertEqual(p, 0.6)
    
    def test_prob_of_list(self):
        xk = [(0,),(1,)]
        pk = [0.6, 0.4]
        rv = rv_discrete(xk, pk)
        p = rv._prob_of([0])
        self.assertIs(p, 0.6)
    
    def test_prob_of_invalid(self):
        xk = [(0,),(1,)]
        pk = [0.6, 0.4]
        rv = rv_discrete(xk, pk)
        p = rv._prob_of(3)
        self.assertIs(p, rv.badvalue)
    
    def test_pmf_tuple(self): 
        xk = [(0,1),(1,0)]
        pk = [0.6, 0.4]
        rv = rv_discrete(xk, pk, badvalue=0)

        x= [(0,1),(0,1),(1,0),(0,0)]
        P_is = rv.pmf(x)
        P_target = np.array([0.6, 0.6, 0.4, 0])
        
        self.assertTrue(np.array_equal(P_is, P_target), f"P_is: {P_is} is not P_target: {P_target}.")
    
    def test_pmf_array(self):
        xk = [(0,1),(1,0)]
        pk = [0.6, 0.4]
        rv = rv_discrete(xk, pk, badvalue=0)

        x= [[0,1],[0,1],[1,0],[0,0]]
        P_is = rv.pmf(x)
        P_target = np.array([0.6, 0.6, 0.4, 0])
        
        self.assertTrue(np.array_equal(P_is, P_target), f"P_is: {P_is} is not P_target: {P_target}.")
    
    def test_pmf_scalar(self): 
        xk = [(0,),(1,)]
        pk = [0.6, 0.4]
        rv = rv_discrete(xk, pk, badvalue=0)

        x= [0,0,1,1,2]
        P_is = rv.pmf(x)
        P_target = np.array([0.6, 0.6, 0.4, 0.4, 0])
        
        self.assertTrue(np.array_equal(P_is, P_target), f"P_is: {P_is} is not P_target: {P_target}.")
    
    def test_score_samples(self):
        xk = [(0,),(1,)]
        pk = [0.6, 0.4]
        rv = rv_discrete(xk, pk, badvalue=0)

        x= [0,0,1,1,2]
        score_is = rv.score_samples(x)
        score_target = np.log(np.array([0.6, 0.6, 0.4, 0.4, 0]))
        
        self.assertTrue(np.array_equal(score_is, score_target), f"P_is: {score_is} is not P_target: {score_target}.")
    
if __name__ == '__main__':
    unittest.main()


    
        

    