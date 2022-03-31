import os
import unittest
import numpy as np
import numpy.testing as nptest

import flare.data_prep.voxelizer as voxelizer


IDT = 'float32'  # float32 is what I expect to exclusively use later on.


class TestVoxelizeFields(unittest.TestCase):


    coords = np.asarray([[   0.0,  0.0,  0.0],
                         [   4.4, 18.6, -4.4],
                         [   0.2, 0.3,   0.1],
                         [  -6.6, 16.9, 22.2],
                         [   4.1, 19.4, -4.5],
                         [   4.4, 18.7, -4.3]], dtype=IDT)

    features = np.asarray([[  12.4,  1.4],
                           [   4.5,  8.6],
                           [-123.2, 12.3],
                           [  66.6,  6.9],
                           [  42.0,  4.2],
                           [   2.4,  9.7]], dtype=IDT)

    labels = np.asarray([[1,   0, 2],
                         [4,   8, 5],
                         [1,   1, 0],
                         [6, 188, 0],
                         [4,   3, 5],
                         [3,   3, 5]], dtype=np.uint16)



    def setUp(self):
        # Global state variables:
        voxelizer.IDT = IDT


    def test_vsize50cm(self):

        # Note, that unique returns are sorted, here by coords!
        map        = np.asarray([1, 2, 1, 0, 3, 2])
        coords_v   = np.asarray([[-14, 33, 44], [0, 0, 0], [8, 37, -9], [8, 38, -9]])
        features_v = np.asarray([[ 66.6, 6.90], [-55.4, 6.85], [ 3.45, 9.15], [42.0, 4.2]])
        labels_v   = np.asarray([[6, 188, 0], [1, 0, 0],  [3, 3, 5], [4, 3, 5]], dtype=np.uint16)
        nop        = np.asarray([1, 2, 2, 1])
        acc        = np.asarray([5/6, 4/6, 5/6])

        c, f, l, m, n, a = voxelizer.voxelize(
            self.coords, self.features, self.labels, 0.5)

        self.assertEqual(f.dtype, IDT)
        self.assertEqual(l.dtype, np.uint16)

        nptest.assert_allclose(m, map)
        nptest.assert_allclose(c, coords_v)
        nptest.assert_allclose(f, features_v)
        nptest.assert_allclose(l, labels_v)
        nptest.assert_allclose(n, nop)
        nptest.assert_allclose(a, acc)


    def test_vsize3m(self):

        # Note, that unique returns are sorted, here by coords!
        map        = np.asarray([1, 2, 1, 0, 2, 2])
        coords_v   = np.asarray([[-3, 5, 7], [0, 0, 0], [1, 6, -2]])
        features_v = np.asarray([[ 66.6, 6.90], [-55.4, 6.85], [ 16.3, 7.50]])
        labels_v   = np.asarray([[6, 188, 0], [1, 0, 0],  [4, 3, 5]], dtype=np.uint16)
        nop        = np.asarray([1, 2, 3])
        acc        = np.asarray([5/6, 4/6, 5/6])

        c, f, l, m, n, a = voxelizer.voxelize(
            self.coords, self.features, self.labels, 3)

        self.assertEqual(f.dtype, IDT)
        self.assertEqual(l.dtype, np.uint16)

        nptest.assert_allclose(m, map)
        nptest.assert_allclose(c, coords_v)
        nptest.assert_allclose(f, features_v)
        nptest.assert_allclose(l, labels_v)
        nptest.assert_allclose(n, nop)
        nptest.assert_allclose(a, acc)



if __name__ == '__main__':
    unittest.main()
