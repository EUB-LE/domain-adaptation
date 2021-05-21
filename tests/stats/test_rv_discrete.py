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
    
    def test_common_coverage(self):
        rv1 = rv_discrete([(0,),(1,),(2,)], pk=[0.3, 0.3, 0.4])
        rv2 = rv_discrete([(0,),(1,),(4,)], pk=[0.2, 0.4, 0.4])

        coverage_is = rv1._common_coverage(rv2)
        coverage_reverse = rv2._common_coverage(rv1)
        coverage_target = np.array([(0,),(1,)]).reshape(-1)

        self.assertTrue(np.array_equal(coverage_is, coverage_target), f"covarage_is {coverage_is} not equal to coverage_target {coverage_target}.")
        self.assertTrue(np.array_equal(coverage_is, coverage_reverse), f"covarage_is {coverage_is} not equal to coverage_target {coverage_reverse}.")

    def test_divergence(self):
        rv1 = rv_discrete([(0,),(1,),(2,)], pk=[0.3, 0.3, 0.4])
        rv2 = rv_discrete([(0,),(1,),(4,)], pk=[0.2, 0.4, 0.4])
        
        pd_1 = rv1.score_samples([(0,), (1,)])
        pd_2 = rv2.score_samples([(0,), (1,)])
        divergence_target = rv1.divergence_from_distribution(pd_1, pd_2)
        divergence_is = rv1.divergence(rv2)

        self.assertEqual(divergence_is, divergence_target)
    
if __name__ == '__main__':
    unittest.main()


    
        

    