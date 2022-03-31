import os
import unittest
import numpy as np
import numpy.testing as nptest

import flare.utils.pts as pts


def p2tf(filename):
    return os.path.join(os.path.dirname(__file__), filename)



class TestPtsRead(unittest.TestCase):

    fname = p2tf('./data/test_pts_read.pts')
    IDT = 'float32'  # float32 is what I expect to exclusively use later on.


    def setUp(self):
        pass

    def test_read_all(self):
        data, fields = pts.read(self.fname, IDT=self.IDT)

        self.assertEqual(data.dtype, self.IDT)
        self.assertEqual(fields, ['x', 'y', 'z', 'num_returns', 'return_num',
                                  'intensity', 'classification', 'red',
                                  'green', 'blue'])
        nptest.assert_allclose(data[0, :], [17.74, 0.96, 13.80, 2.00, 2.00,
                               12.00, 2.00, 100.00, 64.00, 58.00])
        nptest.assert_allclose(data[1, :], [17.48, 2.08, 13.92, 2.00, 2.00,
                               13.00, 2.00, 148.00, 95.00, 97.00])
        nptest.assert_allclose(data[-1, :], [326.48, 228.14, 25.58, 1.00, 1.00,
                               152.00, 8.00, 81.00, 41.00, 41.00])


    def test_read_only_coords_implicitly(self):
        data, fields = pts.read(self.fname, fields=[])

        self.assertEqual(fields, ['x', 'y', 'z'])
        nptest.assert_allclose(data[ 0, :], [17.74, 0.96, 13.80,])
        nptest.assert_allclose(data[ 1, :], [17.48, 2.08, 13.92,])
        nptest.assert_allclose(data[-1, :], [326.48, 228.14, 25.58])


    def test_read_one_feature_one_coord_explicitly_in_wrong_order(self):
        data, fields = pts.read(self.fname, fields=['intensity', 'y'])

        self.assertEqual(fields, ['x', 'y', 'z', 'intensity'])
        nptest.assert_allclose(data[ 0, :], [17.74, 0.96, 13.80, 12.00])
        nptest.assert_allclose(data[ 1, :], [17.48, 2.08, 13.92, 13.00])
        nptest.assert_allclose(data[-1, :], [326.48, 228.14, 25.58, 152.00])


    def test_read_nonexistent_feature(self):
        data, fields = pts.read(self.fname, fields=['foo', 'classification'])

        self.assertEqual(fields, ['x', 'y', 'z', 'classification'])
        nptest.assert_allclose(data[ 0, :], [17.74, 0.96, 13.80, 2.00])
        nptest.assert_allclose(data[ 1, :], [17.48, 2.08, 13.92, 2.00])
        nptest.assert_allclose(data[-1, :], [326.48, 228.14, 25.58, 8.00])


    def test_read_big_coords_reduced(self):
        d = np.asarray([[2146483.645,   -2145483.123,   2147442.647,  41],
                        [2146493.789,   -2145412.123,   2147459.645,  42]])

        data, _, offset = pts.read(p2tf('./data/test_pts_big_coords.pts'), 
                                   reduced=True, IDT=self.IDT)

        coords = data[:, 0:3] + offset

        self.assertEqual(data.dtype, self.IDT)
        self.assertEqual(offset.dtype, 'float64')
        nptest.assert_allclose(coords, d[:, 0:3])




class TestPtsWrite(unittest.TestCase):

    fname = p2tf('./data/test_pts_write.pts')
    IDT = 'float32'  # float32 is what I expect to exclusively use later on.

    def setUp(self):
        pass
        

    def tearDown(self):
        os.remove(self.fname)


    def test_write_all(self):
        fields = ['x', 'y', 'z', 'num_returns', 'return_num', 'intensity',
                  'classification', 'red', 'green', 'blue']
        data = np.asarray(
            [[17.74, 0.96, 13.80, 2.00, 2.00, 12.00, 2.00, 100.00, 64.00, 58.00],
             [17.48, 2.08, 13.92, 2.00, 2.00, 13.00, 2.00, 148.00, 95.00, 97.00],
             [326.48, 228.14, 25.58, 1.00, 1.00, 152.00, 8.00, 81.00, 41.00, 41.00]])

        pts.write(self.fname, data, fields)
        data_, fields_ = pts.read(self.fname)

        self.assertEqual(fields_, fields)
        nptest.assert_allclose(data_[ 0, :], data[0,:])
        nptest.assert_allclose(data_[ 1, :], data[1,:])
        nptest.assert_allclose(data_[-1, :], data[-1,:])


    def test_write_big_coords_reduced(self):
        fields = ['x', 'y', 'z', 'intensity']
        data = np.asarray([[2146483.645,   -2145483.123,   2147442.647,  41],
                           [2146493.789,   -2145412.123,   2147459.645,  42]])
        offset = np.mean(data[:, 0:3], axis=0)
        data_reduced = data.copy()
        data_reduced[:, 0:3] -= offset

        pts.write(self.fname, data_reduced, fields, offset=offset)
        data_, _, offset_ = pts.read(self.fname, reduced=True, IDT=self.IDT)

        # breakpoint()
        data_g = data_.astype('float64')
        data_g[:, 0:3] += offset_

        self.assertEqual(data_.dtype, self.IDT)
        self.assertEqual(offset_.dtype, 'float64')
        nptest.assert_allclose(data, data_g)


if __name__ == '__main__':
    unittest.main()
