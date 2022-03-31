import math
import unittest
import numpy as np
import numpy.testing as nptest

import flare.utils.coord_trafo as ct


s2 = math.sqrt(2)
# Atol: since some true values are 0, atol becomes relevant, because 0*rtol=0.
atol = 1e-15  


class TestCoordTrafo(unittest.TestCase):

    v_ = np.asarray([[2, 0, 0], [0, 2, 0],  [0, 0, 2],
                     [0, 2, 2], [2, 0, 2],  [2, 2, 0]])


    def test_preserves_original(self):
        v_copy = self.v_.copy()

        m = ct.T([3, -4, 10])
        _ = ct.transform(self.v_, m)

        nptest.assert_allclose(self.v_, v_copy)


    def test_shift(self):
        m = ct.T([3, -4, 10])

        v, m_ = ct.transform(self.v_, m)

        v__ = [[5, -4, 10], [3, -2, 10],  [3, -4, 12],
               [3, -2, 12], [5, -4, 12],  [5, -2, 10]]

        nptest.assert_allclose(m, m_)
        nptest.assert_allclose(v, v__, atol=atol)
        nptest.assert_allclose(ct.transform(v, m, True)[0], self.v_, atol=atol)


    def test_rot_x(self):
        m = ct.R_x(45)

        v, m_ = ct.transform(self.v_, m)

        v__ = [[2, 0,    0], [0,  2/s2, 2/s2], [0, -2/s2, 2/s2],
               [0, 0, 2*s2], [2, -2/s2, 2/s2], [2,  2/s2, 2/s2]]

        nptest.assert_allclose(m, m_)
        nptest.assert_allclose(v, v__, atol=atol)
        nptest.assert_allclose(ct.transform(v, m, True)[0], self.v_, atol=atol)


    def test_rot_y(self):
        m = ct.R_y(45)

        v, m_ = ct.transform(self.v_, m)

        v__ = [[2/s2, 0, -2/s2], [   0, 2, 0], [2/s2, 0,  2/s2],
               [2/s2, 2,  2/s2], [2*s2, 0, 0], [2/s2, 2, -2/s2]]

        nptest.assert_allclose(m, m_)
        nptest.assert_allclose(v, v__, atol=atol)
        nptest.assert_allclose(ct.transform(v, m, True)[0], self.v_, atol=atol)


    def test_rot_z(self):
        m = ct.R_z(45)

        v, m_ = ct.transform(self.v_, m)

        v__ = [[ 2/s2, 2/s2, 0], [-2/s2, 2/s2, 0], [0,    0, 2],
               [-2/s2, 2/s2, 2], [ 2/s2, 2/s2, 2], [0, 2*s2, 0]]

        nptest.assert_allclose(m, m_)

        nptest.assert_allclose(v, v__, atol=atol)
        nptest.assert_allclose(ct.transform(v, m, True)[0], self.v_, atol=atol)


    def test_scale(self):
        m = ct.S(3)

        v, m_ = ct.transform(self.v_, m)

        v__ = self.v_ * 3

        nptest.assert_allclose(m, m_)
        nptest.assert_allclose(v, v__, atol=atol)
        nptest.assert_allclose(ct.transform(v, m, True)[0], self.v_, atol=atol)



if __name__ == '__main__':
    unittest.main()
