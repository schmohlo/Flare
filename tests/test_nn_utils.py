
import unittest
import numpy as np
import torch
import numpy.testing as nptest
import torch.testing as totest

from math import pi
from flare.utils.timer import Timer

import flare.nn._utils as nn_utils


inf = float("Inf")
NaN = float("NaN")




###############################################################################
###############################################################################
###############################################################################



class TestProjectZNumpy(unittest.TestCase):

    coords = np.asarray([[-1,  2, 3],
                         [-1,  2, 2],
                         [ 0, -1, -10],
                         [ 0,  1, 0],
                         [ 1,  1, 8],
                         [ 0, -1, 4]])

    features = np.asarray([[ 1],
                           [-4],
                           [ 6],
                           [ 2],
                           [ 0],
                           [ 5]])

    
    
    def test_max(self):
        v, h = nn_utils.project_z_numpy(self.coords, self.features, mode='max')

        h_actual = np.asarray([[NaN, NaN, NaN,   3],
                               [  4, NaN,   0, NaN],
                               [NaN, NaN,   8, NaN]])

        v_actual = np.asarray([[0, 0, 0, 1],
                               [5, 0, 2, 0],
                               [0, 0, 0, 0]])
        v_actual = np.expand_dims(v_actual, 2)

        nptest.assert_allclose(v, v_actual)
        nptest.assert_allclose(h, h_actual)



    def test_min(self):
        v, h = nn_utils.project_z_numpy(self.coords, self.features, mode='min')

        h_actual = np.asarray([[NaN, NaN, NaN,   2],
                               [-10, NaN,   0, NaN],
                               [NaN, NaN,   8, NaN]])

        v_actual = np.asarray([[0, 0, 0, -4],
                               [6, 0, 2,  0],
                               [0, 0, 0,  0]])
        v_actual = np.expand_dims(v_actual, 2)

        totest.assert_allclose(v, v_actual)
        totest.assert_allclose(h, h_actual)



    def test_default(self):
        v, h = nn_utils.project_z_numpy(self.coords, self.features, default=42)

        h_actual = np.asarray([[NaN, NaN, NaN,   3],
                               [  4, NaN,   0, NaN],
                               [NaN, NaN,   8, NaN]])

        v_actual = np.asarray([[42, 42, 42, 1],
                               [ 5, 42,  2, 42],
                               [42, 42,  0, 42]])
        v_actual = np.expand_dims(v_actual, 2)

        totest.assert_allclose(v, v_actual)
        totest.assert_allclose(h, h_actual)



    def test_features_2D(self):
        features = np.hstack((self.features, self.features+10))
        v, h = nn_utils.project_z_numpy(self.coords, features)

        h_actual = np.asarray([[NaN, NaN, NaN,   3],
                               [  4, NaN,   0, NaN],
                               [NaN, NaN,   8, NaN]])

        v_actual = np.asarray([[[0, 0], [0, 0], [0, 0], [1,11]],
                               [[5,15], [0, 0], [2,12], [0, 0]],
                               [[0, 0], [0, 0], [0,10], [0, 0]]])

        totest.assert_allclose(v, v_actual)
        totest.assert_allclose(h, h_actual)



###############################################################################
###############################################################################
###############################################################################



class TestProjectZ_Torch(unittest.TestCase):

    coords = torch.tensor([[-1,  2, 3],
                           [ 0, -1, -10],
                           [ 0,  1, 0],
                           [ 1,  1, 8]])

    features = torch.tensor([[ 1.],
                             [ 6.],
                             [ 2.],
                             [ 0.]])


    
    
    def test_default(self):
        v, h = nn_utils.project_z_torch(self.coords, self.features, default=42)

        h_actual = torch.tensor([[NaN, NaN, NaN,   3],
                                 [-10, NaN,   0, NaN],
                                 [NaN, NaN,   8, NaN]])

        v_actual = torch.tensor([[42., 42., 42.,  1.],
                                 [ 6., 42.,  2., 42.],
                                 [42., 42.,  0., 42.]]).unsqueeze_(2)

        totest.assert_allclose(v, v_actual)
        totest.assert_allclose(h, h_actual)



    def test_gpu(self):
        if not torch.cuda.is_available():
            print("Warning: no gpu available for test_gpu in",
                  self.__class__.__name__)
            return

        v, h = nn_utils.project_z_torch(self.coords, self.features.cuda())

        h_actual = torch.tensor([[NaN, NaN, NaN,   3],
                                 [-10, NaN,   0, NaN],
                                 [NaN, NaN,   8, NaN]])

        v_actual = torch.tensor([[0., 0, 0, 1],
                                 [ 6, 0, 2, 0],
                                 [ 0, 0, 0, 0]]).unsqueeze_(2).cuda()

        self.assertEqual(v.device, torch.device('cuda:0'))
        self.assertEqual(h.device, torch.device('cpu'))
        totest.assert_allclose(v, v_actual)
        totest.assert_allclose(h, h_actual)



    def test_features_2D_int_features(self):
        features = torch.cat((self.features, self.features+10), 1)
        v, h = nn_utils.project_z_torch(self.coords, features)

        h_actual = torch.tensor([[NaN, NaN, NaN,   3],
                                 [-10, NaN,   0, NaN],
                                 [NaN, NaN,   8, NaN]])

        v_actual = torch.tensor([[[0,0.], [0, 0], [0, 0], [1,11]],
                                 [[6,16], [0, 0], [2,12], [0, 0]],
                                 [[0, 0], [0, 0], [0,10], [0, 0]]])

        totest.assert_allclose(v, v_actual)
        totest.assert_allclose(h, h_actual)



if __name__ == '__main__':
    unittest.main()
