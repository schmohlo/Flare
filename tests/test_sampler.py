import os
import math
import numpy as np
import unittest
import numpy.testing as nptest

import flare.utils.las as las
from flare.utils.timer import Timer
from flare.data_prep import sample_fus



def p2tf(filename):
    return os.path.join(os.path.dirname(__file__), filename)



class TestTilling:

    fname = p2tf('./data/test_sampler.las')

    data, fields, precision, offset = las.read(fname, IDT='float32')
    x = data[:, fields.index('x')]
    y = data[:, fields.index('y')]


    def setUp(self):
        x = self.x
        y = self.y
        data = self.data

        # Parameters depending on concrete test case:
        overlap = self.overlap
        t_size = self.t_size

        self.samples, self.indices = sample_fus.tiling(x, y, data, overlap, t_size)


    def test_numb_fields(self):
        for s in self.samples:
            self.assertEqual(s.shape[1], self.data.shape[1])


    def test_numb_of_samples(self):
        extent_x = np.max(self.x) - np.min(self.x)
        extent_y = np.max(self.y) - np.min(self.y)

        ts = self.t_size
        ol = self.overlap

        n_x = math.ceil((extent_x - ts[0]) / ((1 - ol[0]) * ts[0]) + 1)
        n_y = math.ceil((extent_y - ts[1]) / ((1 - ol[1]) * ts[1]) + 1)

        self.assertEqual(len(self.samples), n_x*n_y)


    def test_indices(self):

        n_points_samples = [s.shape[0] for s in self.samples]
        n_points_indices = [len(i) for i in self.indices]

        nptest.assert_allclose(n_points_indices, n_points_samples)

        for i, s in zip(self.indices, self.samples):
            nptest.assert_allclose(self.data[i, :], s)


    def test_statistics(self):

        mins = np.asarray([np.min(s, 0) for s in self.samples])
        maxs = np.asarray([np.max(s, 0) for s in self.samples])

        mins_g = np.min(mins, 0)
        maxs_g = np.max(maxs, 0)

        nptest.assert_allclose(mins_g, np.min(self.data, 0))
        nptest.assert_allclose(maxs_g, np.max(self.data, 0))


    def test_sample_extents(self):
        # Exclude tail samples which may be only of fractional size.
        # Those are the n_x/y ones with highest x/y.
        # Do so by testing only the number of samples with the correct extent.
        mins_x = np.asarray([np.min(s[:, 0]) for s in self.samples])
        maxs_x = np.asarray([np.max(s[:, 0]) for s in self.samples])
        mins_y = np.asarray([np.min(s[:, 1]) for s in self.samples])
        maxs_y = np.asarray([np.max(s[:, 1]) for s in self.samples])

        extents_x = maxs_x - mins_x
        extents_y = maxs_y - mins_y

        extent_x = np.max(self.x) - np.min(self.x)
        extent_y = np.max(self.y) - np.min(self.y)

        ts = self.t_size
        ol = self.overlap

        n_x = math.ceil((extent_x - ts[0]) / ((1 - ol[0]) * ts[0]) + 1)
        n_y = math.ceil((extent_y - ts[1]) / ((1 - ol[1]) * ts[1]) + 1)

        sum_correct = 0
        atol = 0.26  # atol results from density of the point cloud.
        for i, s in enumerate(self.samples):
            if extents_x[i] - ts[0] < atol and extents_y[i] - ts[1] < atol:
                sum_correct += 1

        self.assertGreaterEqual(sum_correct, len(self.samples) - n_x - n_y - 1)


    def test_visually(self):
        subdir = p2tf('test_sampler_visually_' + self.__class__.__name__)
        try:
            os.mkdir(subdir)
        except FileExistsError:
            pass

        for i,s in enumerate(self.samples):
            fpath = os.path.join(subdir, '_{}_a.las'.format(i))
            las.write(fpath, s, self.fields, self.precision, self.offset)




class TestTilingNoOverlapp(TestTilling, unittest.TestCase):

    @property
    def overlap(self):
        return [0, 0]

    @property
    def t_size(self):
        return [10, 13]


    def test_includes_all_points(self):
        n_points = [s.shape[0] for s in self.samples]
        self.assertEqual(sum(n_points), self.data.shape[0])




class TestTilingWithOverlapp(TestTilling, unittest.TestCase):

    @property
    def overlap(self):
        return [0.30, 0.68]

    @property
    def t_size(self):
        return [10, 13]


    def test_includes_min_number_of_points(self):
        n_points = [s.shape[0] for s in self.samples]
        self.assertGreaterEqual(sum(n_points), self.data.shape[0])




# =============================================================================
#  class CompareTilingRuntimeSmall(unittest.TestCase):
#
#      n = 100
#      fname = p2tf('test_sampler.las')
#      overlap = [0, 0]
#      t_size =  [5, 11]
#
#      def setUp(self):
#          self.data, self.fields, self.precision, self.offset = las.read(self.fname, IDT='float32')
#          self.x = self.data[:, self.fields.index('x')]
#          self.y = self.data[:, self.fields.index('y')]
#
#
#      def test_run_kdtree(self):
#
#          timer = Timer()
#          for i in range(self.n):
#              samples, indices = sampler.tiling(self.x, self.y, self.data, self.overlap, self.t_size)
#          print(timer.stop())
# =============================================================================




if __name__ == '__main__':
    unittest.main()
