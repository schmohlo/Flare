import os
import unittest
import numpy as np
import numpy.testing as nptest

import flare.utils.las as las


IDT = 'float32'  # float32 is what I expect to exclusively use later on.

precision_ = [0.001]*3
offset_    = [-44035.7, 258181.0, 409.3]

fields_ = ['x', 'y', 'z', 'intensity', 'classification', 'scan_angle_rank',
           'pt_src_id', 'red', 'green', 'blue', 'roughness', 'linearity',
           'return_num', 'num_returns', 'scan_dir_flag']

data_ = np.asarray(
        [[-4.4035512e+04,  2.5818134e+05,  4.0995001e+02, 1.0700000e+02,
           3.0000000e+01,  1.4000000e+01,  1.6200000e+02, 3.4304000e+04,
           3.4304000e+04,  3.3280000e+04,  1.1695862e-02, 4.8684892e-01,
           1.0000000e+00,  1.0000000e+00,  1.0000000e+00],
         [-4.4035109e+04,  2.5818131e+05,  4.0995001e+02, 9.8000000e+01,
           3.0000000e+01,  1.4000000e+01,  1.6200000e+02, 3.4560000e+04,
           3.4560000e+04,  3.3536000e+04,  9.1018677e-03, 5.0799459e-01,
           1.0000000e+00,  1.0000000e+00,  1.0000000e+00],
         [-4.4029961e+04,  2.5823066e+05,  4.0969000e+02, 2.4200000e+02,
           1.5000000e+01,  1.4000000e+01,  1.6300000e+02, 2.6112000e+04,
           2.6368000e+04,  1.7664000e+04,  6.9732666e-03, 1.0806788e-01,
           1.0000000e+00,  1.0000000e+00,  1.0000000e+00]])

min_ = [-44035.66015625, 258181.07812500, 409.29998779,
        1.0, 15.0, 10.0, 162.0, 9728, 11008, 9728,
        0.0, 0.00041070016, 1.0, 0.0, 1.0]
max_ = [-43972.26953125, 258241.43750000, 456.89001465,
        1200.0, 225.0, 19.0, 163.0, 50176, 50176, 49408,
        2.69488525, 0.99965554, 7.0, 7.0, 1.0]



def p2tf(filename):
    return os.path.join(os.path.dirname(__file__), filename)




class TestLasRead(unittest.TestCase):

    fname = p2tf('./data/test_las_read.las')

    def setUp(self):
        pass


    def test_read_all(self):
        data, fields, precision, offset = las.read(self.fname, IDT=IDT)

        self.assertEqual(data.dtype, IDT)
        self.assertEqual(data.shape, (64332, 15))
        self.assertEqual(precision, precision_)
        nptest.assert_allclose(offset, offset_)
        self.assertEqual(len(fields), 15)
        self.assertEqual(fields[:3], ['x', 'y', 'z'])

        # Other than xyz, the order of the data fields may be random.
        # Same for the point order more or less, but it seems constant.

        for f in fields_:
            self.assertTrue(f in fields)

        for i,f in enumerate(fields):
            j = fields_.index(f)

            nptest.assert_allclose(data[ 0, i], data_[ 0, j])
            nptest.assert_allclose(data[ 1, i], data_[ 1, j])
            nptest.assert_allclose(data[-1, i], data_[-1, j])

            nptest.assert_allclose(np.nanmin(data[:, i]), min_[j])
            nptest.assert_allclose(np.nanmax(data[:, i]), max_[j])


    def test_read_only_coords_implicitly(self):
        data, fields, _, _ = las.read(self.fname, fields=[])

        self.assertEqual(fields, ['x', 'y', 'z'])
        nptest.assert_allclose(data[ 0, :], data_[ 0, 0:3])
        nptest.assert_allclose(data[ 1, :], data_[ 1, 0:3])
        nptest.assert_allclose(data[-1, :], data_[-1, 0:3])


    def test_read_one_feature_one_coord_explicitly_in_wrong_order(self):
        data, fields, _, _ = las.read(self.fname, fields=['intensity', 'y'])

        self.assertEqual(fields, ['x', 'y', 'z', 'intensity'])
        nptest.assert_allclose(data[ 0, :], data_[ 0, 0:4])
        nptest.assert_allclose(data[ 1, :], data_[ 1, 0:4])
        nptest.assert_allclose(data[-1, :], data_[-1, 0:4])


    def test_read_nonexistent_feature(self):
        data, fields, _, _ = las.read(self.fname, fields=['foo', 'classification'])

        self.assertEqual(fields, ['x', 'y', 'z', 'classification'])
        nptest.assert_allclose(data[ 0, :], data_[ 0, [0, 1, 2, 4]])
        nptest.assert_allclose(data[ 1, :], data_[ 1, [0, 1, 2, 4]])
        nptest.assert_allclose(data[-1, :], data_[-1, [0, 1, 2, 4]])


    def test_read_decompose_classification_byte(self):
        data, fields, _, _ = las.read(self.fname, decomp_class_byte=True)

        c = fields.index('classification')
        s = fields.index('synthetic')
        k = fields.index('key_point')
        w = fields.index('withheld')

        #        1 2 4  8 16  32  64 128
        #
        #  raw   |--  c  --|   s   k   w
        #
        #   15   1 1 1  1  0   0   0   0      #   534    c = 15
        #   30   0 1 1  1  1   0   0   0      # 39365    c = 30
        #   45   1 0 1  1  0   1   0   0      #  3081    c = 13
        #   60   0 0 1  1  1   1   0   0      #  8695    c = 28
        #   75   1 1 0  1  0   0   1   0      #  1720    c = 11
        #   90   0 1 0  1  1   0   1   0      #  5756    c = 26
        #  105   1 0 0  1  0   1   1   0      #    74    c =  9
        #  120   0 0 0  1  1   1   1   0      #  2940    c = 24
        #  210   0 1 0  0  1   0   1   1      #  1211    c = 18
        #  225   1 0 0  0  0   1   1   1      #   956    c =  1
        #                                    => S(n_i*c_i) = 1735029

        self.assertEqual(np.min(data[:, c]), 1)
        self.assertEqual(np.max(data[:, c]), 30)
        self.assertEqual(np.sum(data[:, c]), 1735029)

        self.assertEqual(np.sum(data[:, s]), 15746)
        self.assertEqual(np.sum(data[:, k]), 12657)
        self.assertEqual(np.sum(data[:, w]),  2167)



    def test_read_big_coords_reduced(self):
        d = np.asarray([[2146483.645,   -2145483.123,   2147442.647,  41],
                        [2146493.789,   -2145412.123,   2147459.645,  42]])

        data, _, _, offset = las.read(p2tf('./data/test_las_big_coords.las'), 
                                      reduced=True, IDT=IDT)

        coords = data[:, 0:3] + offset

        self.assertEqual(data.dtype, IDT)
        self.assertEqual(type(offset[0]), float)
        nptest.assert_allclose(coords, d[:, 0:3])






class TestLasReadFromFloat(unittest.TestCase):
    """ file has the same fields as the other one, but all named "..._float32"
     and one additional one without, which should not be read! """

    fname = p2tf('./data/test_las_read_from_float.las')


    def test_read_all(self):
        data, fields, precision, offset = las.read_from_float(self.fname, dtype=IDT)

        self.assertEqual(data.dtype, IDT)
        self.assertEqual(data.shape, (64332, 15))
        self.assertEqual(precision, precision_)
        nptest.assert_allclose(offset, offset_)
        self.assertEqual(len(fields), 15)
        self.assertEqual(fields[:3], ['x', 'y', 'z'])

        # Other than xyz, the order of the data fields may be random.
        # Same for the point order more or less, but it seems constant.

        for f in fields_:
            self.assertTrue(f in fields)

        for i,f in enumerate(fields):
            j = fields_.index(f)

            nptest.assert_allclose(data[ 0, i], data_[ 0, j])
            nptest.assert_allclose(data[ 1, i], data_[ 1, j])
            nptest.assert_allclose(data[-1, i], data_[-1, j])

            nptest.assert_allclose(np.nanmin(data[:, i]), min_[j])
            nptest.assert_allclose(np.nanmax(data[:, i]), max_[j])


    def test_read_one_feature_explicitly(self):
        data, fields, _, _ = las.read_from_float(self.fname, fields=['intensity'])

        self.assertEqual(fields, ['x', 'y', 'z', 'intensity'])
        nptest.assert_allclose(data[ 0, :], data_[ 0, 0:4])
        nptest.assert_allclose(data[ 1, :], data_[ 1, 0:4])
        nptest.assert_allclose(data[-1, :], data_[-1, 0:4])




class TestLasWrite(unittest.TestCase):

    fname = p2tf('./data/test_las_write.las')


    def tearDown(self):
        os.remove(self.fname)


    def test_write_all(self):
        las.write(self.fname, data_, fields_, precision=precision_, offset=offset_)
        data, fields, precision, offset = las.read(self.fname, IDT=IDT)

        self.assertEqual(data.dtype, IDT)
        self.assertEqual(data.shape, (3, 15))
        self.assertEqual(precision, precision_)
        nptest.assert_allclose(offset, offset_)
        self.assertEqual(len(fields), 15)
        self.assertEqual(fields[:3], ['x', 'y', 'z'])

        # Other than xyz, the order of the data fields may be random.
        # Same for the point order more or less, but it seems constant.

        for f in fields_:
            self.assertTrue(f in fields)

        for i,f in enumerate(fields):
            j = fields_.index(f)

            nptest.assert_allclose(data[ 0, i], data_[ 0, j])
            nptest.assert_allclose(data[ 1, i], data_[ 1, j])
            nptest.assert_allclose(data[-1, i], data_[-1, j])


    def test_write_decompose_classification(self):
        data__ = np.asarray([[1, 1, 1, 255]])
        fields__ = ['x', 'y', 'z', 'classification']

        with self.assertWarns(Warning):
            las.write(self.fname, data__, fields__, decomp_class_byte=True)

        data, fields, _, _ = las.read(self.fname)

        # Checking fields also implies no other class byte flags.
        self.assertEqual(fields, fields__)
        nptest.assert_allclose(data[0, 3], 31)


    def test_write_big_coords_reduced(self):
        precision = [0.001]*3
        fields = ['x', 'y', 'z', 'intensity']
        data = np.asarray([[2146483.645,   -2145483.123,   2147442.647,  41],
                           [2146493.789,   -2145412.123,   2147459.645,  42]])
        offset = np.mean(data[:, 0:3], axis=0)
        data_reduced = data.copy()
        data_reduced[:, 0:3] -= offset

        las.write(self.fname, data_reduced, fields, precision=precision,
                  offset=offset, reduced=True)
        data_, _, precision_, offset_ = las.read(self.fname, reduced=True, IDT=IDT)

        # breakpoint()
        data_g = data_.astype('float64')
        data_g[:, 0:3] += offset_

        self.assertEqual(data_.dtype, IDT)
        self.assertEqual(type(offset_[0]), float)
        nptest.assert_allclose(data, data_g)
        nptest.assert_allclose(precision_, precision)




class TestLasWriteToFloat(unittest.TestCase):

    fname = p2tf('./data/test_las_write_to_float.las')
    IDT = 'float32'  # float32 is what I expect to exclusively use later on.


    def tearDown(self):
        os.remove(self.fname)


    def test_write_all(self):
        las.write_to_float(self.fname, data_, fields_, precision=precision_, offset=offset_)
        data, fields, precision, offset = las.read_from_float(self.fname, dtype=IDT)

        self.assertEqual(data.dtype, IDT)
        self.assertEqual(data.shape, (3, 15))
        self.assertEqual(precision, precision_)
        nptest.assert_allclose(offset, offset_)
        self.assertEqual(len(fields), 15)
        self.assertEqual(fields[:3], ['x', 'y', 'z'])

        # Other than xyz, the order of the data fields may be random.
        # Same for the point order more or less, but it seems constant.

        for f in fields_:
            self.assertTrue(f in fields)

        for i,f in enumerate(fields):
            j = fields_.index(f)

            nptest.assert_allclose(data[ 0, i], data_[ 0, j])
            nptest.assert_allclose(data[ 1, i], data_[ 1, j])
            nptest.assert_allclose(data[-1, i], data_[-1, j])



if __name__ == '__main__':
    unittest.main()
