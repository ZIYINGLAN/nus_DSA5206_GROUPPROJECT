import unittest
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian
from duong_methods import tolerance_accuracy, parameter_label, format_correlation, saliency_pair

class MethodsTest(unittest.TestCase):
    def test_overprediction(self):
        self.assertEqual(tolerance_accuracy([0,0], [.5,100]), .5)
    def test_underprediction(self):
        self.assertEqual(tolerance_accuracy([0,0], [-.5,-100]), .5)
    def test_strict_threshold(self):
        self.assertEqual(tolerance_accuracy([0,0,0,0], [.7,-.7,.699,-.699]), .5)
    def test_baseline(self):
        self.assertEqual(parameter_label(np.zeros((5,1))), '0+1')
        self.assertEqual(parameter_label(np.ones((5,4))), '4+1')
        self.assertEqual(format_correlation(float('nan')), 'N/A')
    def test_a_matches_released_skimage(self):
        g=np.random.default_rng(0).normal(size=(40,40))
        a,b=saliency_pair(g)
        np.testing.assert_allclose(a,gaussian(g*g,sigma=(5,5),truncate=3.5))
    def test_b_kernel_support_and_sign(self):
        g=np.zeros((41,41));g[20,20]=1
        a,b=saliency_pair(g)
        self.assertEqual(np.count_nonzero(b),25)
        self.assertAlmostEqual(b.sum(),1)
        a_neg,b_neg=saliency_pair(-g)
        np.testing.assert_allclose(a,a_neg)
        np.testing.assert_allclose(b,-b_neg)
    def test_b_reflect_preserves_constant_at_edges(self):
        a,b=saliency_pair(np.ones((8,8)))
        np.testing.assert_allclose(b,1)

if __name__=='__main__':unittest.main()
