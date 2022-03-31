import unittest
import numpy as np
import numpy.testing as nptest

from flare.utils.pointcloud import PointCloud as PC



class TestPC(unittest.TestCase):

    fields = ['x', 'y', 'z', 'num_returns', 'return_num', 'intensity',
              'classification', 'red', 'green', 'blue']
    data = np.asarray([[17.74, 0.96, 13.80, 2.00, 2.00, 12.00, 2.00, 100.00, 64.00, 58.00],
                       [17.48, 2.08, 13.92, 3.00, 2.00, 13.00, 4.00, 148.00, 95.00, 97.00],
                       [17.84, 2.86, 13.94, 1.00, 1.00, 62.00, 5.00,  84.00, 41.00, 42.00]])

    def setUp(self):
        self.pc = PC(self.data.copy(), self.fields.copy())


    def test_create(self):
        nptest.assert_allclose(self.pc.data, self.data)
        self.assertEqual(self.pc.fields, self.fields)


    def test_len(self):
        self.assertEqual(len(self.pc), 3)


    def test_shape(self):
        self.assertEqual(self.pc.shape, (3, 10))


    def test_field_indices(self):
        flds = ['y', 'x', 'z', 'classification', 'blue']
        inds = [1, 0, 2, 6, 9]
        self.assertEqual(self.pc.field_indices(flds), inds)


    def test_get_one(self):
        flds = ['classification']
        inds = [6]
        nptest.assert_allclose(self.pc.get_fdata(flds), self.data[:, inds])


    def test_get_multi(self):
        flds = ['y', 'x', 'z', 'classification', 'blue']
        inds = [1, 0, 2, 6, 9]
        nptest.assert_allclose(self.pc.get_fdata(flds), self.data[:, inds])


    def test_set(self):
        flds = ['y', 'classification', 'blue']
        dats = np.asarray([[1, 2, 3], [4, 5, 6], [6, 7, 8]])
        oflds = [f for f in self.fields if f not in flds]
        oinds = self.pc.field_indices(oflds)
        self.pc.set_fdata(dats, flds)
        nptest.assert_allclose(self.pc.get_fdata(flds), dats)
        nptest.assert_allclose(self.pc.get_fdata(oflds), self.data[:,oinds])


    def test_get_xyz(self):
        nptest.assert_allclose(self.pc.get_xyz(), self.data[:, 0:3])


    def test_set_xyz(self):
        dats = np.asarray([[1, 2, 3], [4, 5, 6], [6, 7, 8]])
        self.pc.set_xyz(dats)
        nptest.assert_allclose(self.pc.get_xyz(), dats)


    def test_add_fields(self):
        flds = ['lin', 'plan']
        dats = np.asarray([[0.1, 1.2], [3.3, 5.4], [2.5, 0.6]])
        self.pc.add_fields(dats, flds)
        self.assertEqual(self.pc.fields, self.fields + flds)
        nptest.assert_allclose(self.pc.data, np.hstack((self.data, dats)))


    def test_add_fields_existing_field(self):
        flds = ['x', 'plan']
        with self.assertRaises(ValueError):
            self.pc.add_fields(None, flds)


    def test_delete_fields(self):
        flds = ['y', 'classification']

        oflds = [f for f in self.fields if f not in flds]
        oinds = self.pc.field_indices(oflds)
        odats = self.pc.data[:, oinds]

        self.pc.delete_fields(flds)

        self.assertEqual(self.pc.fields, oflds)
        nptest.assert_allclose(self.pc.data, odats)






if __name__ == '__main__':
    unittest.main()
