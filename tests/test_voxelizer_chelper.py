import unittest
import numpy as np
import numpy.testing as nptest

from flare.utils.timer import Timer

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, language_level=3)
import flare.data_prep.voxelizer_chelper as chelper


IDT = 'float32'  # float32 is what I expect to exclusively use later on.


class TestVoxelizeFields(unittest.TestCase):

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
    
    ignore_label = np.asarray([-1, -1, -1], dtype=np.int32)

    map = np.asarray([0, 1, 0, 2, 1, 1], dtype=np.uint64)

    features_v = np.asarray([[-55.4, 6.85], [ 16.3, 7.50], [ 66.6, 6.90]])
    labels_v   = np.asarray([[1, 0, 0],  [4, 3, 5], [6, 188, 0]], dtype=np.uint16)
    nop        = np.asarray([2, 3, 1])
    m          = 3 # number of voxels



    def multiply_data(self, n, more_labels=True):
        f_big   = np.tile(self.features, ((n, 1)))
        f_v_big = np.tile(self.features_v, (n, 1))

        l_big   = np.tile(self.labels, (n, 1))
        l_b_big = np.tile(self.labels_v, (n, 1))

        map_big = np.tile(self.map, (n,))
        nop_big = np.tile(self.nop, (n,))

        a = np.floor(np.arange(0, n, 1/self.labels.shape[0])).astype(np.uint64)
        b = np.floor(np.arange(0, n, 1/self.labels_v.shape[0])).astype(np.uint64)

        map_big += a*self.m

        if more_labels:
            for i in range(self.labels.shape[1]):
                l_big[:, i] += a
                l_b_big[:, i] += b

        return f_big, f_v_big, l_big, l_b_big, map_big, nop_big


    #%#########################################################################
    ## End preparations                                                       #
    #%#########################################################################


    def test_small_case(self):
        f, l, p = chelper.voxelize_fields(self.features, self.labels,
                                          self.map, self.m, self.ignore_label)

        self.assertEqual(f.dtype, IDT)
        self.assertEqual(l.dtype, np.uint16)

        nptest.assert_allclose(f, self.features_v)
        nptest.assert_allclose(l, self.labels_v)
        nptest.assert_allclose(p, self.nop)


    def test_ignore_label(self):
        ignore_label = np.asarray([4, 3, 0], dtype=np.int32)
        labels_v   = np.asarray([[1, 0, 2],  [3, 8, 5], [6, 188, 0]], dtype=np.uint16)

        f, l, p = chelper.voxelize_fields(self.features, self.labels,
                                          self.map, self.m, ignore_label)

        self.assertEqual(f.dtype, IDT)
        self.assertEqual(l.dtype, np.uint16)

        nptest.assert_allclose(f, self.features_v)
        nptest.assert_allclose(l, labels_v)
        nptest.assert_allclose(p, self.nop)



    def test_medium_case_1K(self):
        # use simple cas and "multiply" data from simpe_case
        n = 1000
        f_big, f_v_big, l_big, l_b_big, map_big, nop_big = self.multiply_data(n)

        f, l, p = chelper.voxelize_fields(f_big, l_big, map_big,
                                          self.m*n, self.ignore_label)

        nptest.assert_allclose(f, f_v_big)
        nptest.assert_allclose(l, l_b_big)
        nptest.assert_allclose(p, nop_big)



    def test_big_case_1M(self):
        # Different to bigger_case, classes keep the same, because they would
        # become to large.
        n = int(1e6)
        f_big, f_v_big, l_big, l_b_big, map_big, nop_big = self.multiply_data(n, False)

        f, l, p = chelper.voxelize_fields(f_big, l_big, map_big,
                                          self.m*n, self.ignore_label)

        nptest.assert_allclose(f, f_v_big)
        nptest.assert_allclose(l, l_b_big)
        nptest.assert_allclose(p, nop_big)



    def test_compare_new_old(self):
        # use simple cas and "multiply" data from simpe_case
        n = 1000
        f_big, f_v_big, l_big, l_b_big, map_big, nop_big = self.multiply_data(n,False)

        timer1 = Timer()
        for i in range(n):
            f, l, p = chelper.voxelize_fields(f_big, l_big[:, 0, None], map_big,
                                              self.m*n, self.ignore_label)
        timer1.stop()

        nptest.assert_allclose(f, f_v_big)
        nptest.assert_allclose(l, l_b_big[:,0,None])
        nptest.assert_allclose(p, nop_big)

        timer2 = Timer()
        for i in range(n):
            f, l, p = chelper.voxelize_fields_old(f_big, l_big[:, 0], map_big,
                                                  self.m*n)
        timer2.stop()

        nptest.assert_allclose(f, f_v_big)
        nptest.assert_allclose(l, l_b_big[:, 0])
        nptest.assert_allclose(p, nop_big)

        rtol = 0.7
        self.assertLess(timer1.time(), timer2.time() * (1 + rtol))



# =============================================================================
#      def test_medium_case_100x(self):
#          """ Runtime analysis for testing during optimization """
#          timer = Timer()
#          for i in range(100):
#              self.test_medium_case_1K()
#          print(timer.stop())
#          # ~ 3.8s
# =============================================================================



# =============================================================================
#      def test_big_case_2x(self):
#          """ Runtime analysis for testing during optimization """
#          timer = Timer()
#          for i in range(2):
#              self.test_big_case_1M()
#          print(timer.stop())
#          # ~ 5.6s
# =============================================================================



if __name__ == '__main__':
    unittest.main()
