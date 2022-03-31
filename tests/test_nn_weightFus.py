
import unittest
import numpy as np
import torch
import numpy.testing as nptest
import torch.testing as totest

from math import pi
from flare.utils.timer import Timer

import flare.nn.weight_fus as weight_fus


inf = float("Inf")
NaN = float("NaN")




###############################################################################
###############################################################################
###############################################################################



class TestWeightLookup(unittest.TestCase):

    reg_values = torch.tensor([[ 1,  2, 3],
                               [ 1,  2, 2],
                               [ 0,  1, 1],
                               [ 0,  1, 0],
                               [ 1,  1, 1],
                               [ 0,  1, 2]], dtype=torch.float32)

    weights = torch.tensor([[1,  2,  3,  4],
                            [5,  6,  7,  8],
                            [9, 10, 11, 12]])
    
    
    def test_device(self):
        wl = weight_fus.WeightLookup(self.weights, 1)
        wl.to('cuda')
        self.assertEqual(wl.weights.device.type, 'cuda')
        wl.to('cpu')
        self.assertEqual(wl.weights.device.type, 'cpu')
        
    
    def test_default_2D(self):
        voxel_size = 0.5
        indices    = [1, 2]
        lookup_values = self.reg_values[:, indices]
        
        weights_actuall = torch.tensor([6, 6, 1, 1, 1, 2])
        
        wl      = weight_fus.WeightLookup(self.weights, voxel_size)
        weights = wl(lookup_values)

        totest.assert_allclose(weights, weights_actuall)


    def test_default_2D_reihenfolge(self):
        voxel_size = 1.0
        indices    = [1, 0]
        lookup_values = self.reg_values[:, indices]
        
        weights_actuall = torch.tensor([10, 10, 5, 5, 6, 5])
        
        wl      = weight_fus.WeightLookup(self.weights, voxel_size)
        weights = wl(lookup_values)
        totest.assert_allclose(weights, weights_actuall)
        
        
    def test_default_1D(self):
        voxel_size = 1
        indices    = [1]
        lookup_values = self.reg_values[:, indices]
        
        weights_actuall = torch.tensor([11, 11, 10, 10, 10, 10])
        
        wl      = weight_fus.WeightLookup(self.weights[2,:], voxel_size)
        weights = wl(lookup_values)

        totest.assert_allclose(weights, weights_actuall)



if __name__ == '__main__':
    unittest.main()
