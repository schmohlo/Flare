import unittest
import numpy as np
import numpy.testing as nptest

from math import pi

from flare.utils.timer import Timer

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, language_level=3)

from flare.utils.nms_iou import nms as c_nms
from flare.utils.nms_iou import nms_circles as c_circle_nms
from flare.utils.nms_iou import nms_cylinders as c_cylinder_nms
from flare.utils.nms_iou import nms_rectangles as c_rectangle_nms
from flare.utils.nms_iou import nms_cuboids as c_cuboid_nms
from flare.utils.nms_iou import iou_circles as c_circle_iou
from flare.utils.nms_iou import iou as c_iou
from flare.utils.nms_iou import iou_cylinders as c_cylinder_iou
from flare.utils.nms_iou import iou_rectangles as c_rectangle_iou
from flare.utils.nms_iou import iou_cuboids as c_cuboid_iou




###############################################################################
# =============================================================================
# General IoU & NMS
# =============================================================================
# Use specific tests below for a more thorough testing.
###############################################################################

class Test_NMS(unittest.TestCase):

    def test_circle(self):
        ## Inputs:
        circles = np.asarray([[ 5,  5, 5],
                              [ 5,  4, 5],
                              [ 9,  1, 1]], dtype=np.float32)

        scores = np.asarray([ 4,  9, 2], dtype=np.float32)
        iou_max = 0.7

        ## Expected results:
        nms_indices_e = np.asarray([1, 2])

        ## Results:
        nms_indices = c_nms(circles, scores, iou_max)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)



    def test_cylinder(self):
        ## Inputs:
        cylinders = np.asarray([[ 5,  5, 1, 5, 1],
                                [ 5,  4, 1.1, 5, 1],
                                [ 9,  1, 3, 1, 5]], dtype=np.float32)

        scores = np.asarray([ 4,  9, 2], dtype=np.float32)
        iou_max = 0.6

        ## Expected results:
        nms_indices_e = np.asarray([1, 2])

        ## Results:
        nms_indices = c_nms(cylinders, scores, iou_max)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)



    def test_rectangle(self):
        ## Inputs:
        rectangles = np.asarray([[ 5,  5, 4, 4],
                                 [ 5,  4, 4, 4],
                                 [ 5,  6, 4, 4],
                                 [ 9,  1, 6, 6]], dtype=np.float32)

        scores = np.asarray([ 4,  9, 2, 3], dtype=np.float32)
        iou_max = 0.5

        ## Expected results:
        nms_indices_e = np.asarray([1, 3, 2])

        ## Results:
        nms_indices = c_nms(rectangles, scores, iou_max)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)



    def test_cuboid(self):
        ## Inputs:
        cuboids = np.asarray([[ 5,  5, 1, 4, 4, 1],
                              [ 5,  4, 1, 4, 4, 1],
                              [ 5,  4, 1, 4, 4, 2],
                              [ 9,  1, 2, 6, 6, 4]], dtype=np.float32)

        scores = np.asarray([4,  9, 2, 3], dtype=np.float32)
        iou_max = 0.5

        ## Expected results:
        nms_indices_e = np.asarray([1, 3, 2])

        ## Results:
        nms_indices = c_nms(cuboids, scores, iou_max)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)



    def test_input_shape(self):
        ## Inputs:
        to_large = np.ones([3,7], dtype=np.float32)
        to_small = np.ones([3,2], dtype=np.float32)

        scores = np.ones(3, dtype=np.float32)
        iou_max = 0.5

        ## Results:
        self.assertRaises(ValueError, c_nms, to_large, scores, iou_max)
        self.assertRaises(ValueError, c_nms, to_small, scores, iou_max)




class Test_IoU(unittest.TestCase):

    def test_circle(self):
        """
        Values calculated with Matlab and the tool:
        "Analytical intersection area between two circles"
        by Guillaume JACQUENOT
        """

        ## Inputs:
        circles1 = np.asarray([[-9,  1, 1],
                               [ 9, -1, 2],
                               [ 9,  1, 3],
                               [ 4, -2, 9],
                               [ 3,  3, 5]], dtype='float32')

        circles2 = np.asarray([[ 9,  1, 2],
                               [ 9,  5,10]], dtype='float32')

        ## Expected results:
        i_e = np.asarray([[ 0.0         ,   0.0],
                          [ 4.9134787944,  12.566370614359],
                          [12.5663706144,  28.274333882301],
                          [12.5663706143, 125.419116075097],
                          [ 1.21481498543527, 70.5463719878998]])

        u_e = np.asarray([[(1**2 + 2**2) *pi, (1**2 + 10**2) *pi],
                          [(2**2 + 2**2) *pi, (2**2 + 10**2) *pi],
                          [(3**2 + 2**2) *pi, (3**2 + 10**2) *pi],
                          [(9**2 + 2**2) *pi, (9**2 + 10**2) *pi],
                          [(5**2 + 2**2) *pi, (5**2 + 10**2) *pi]]) \
                          - i_e

        iou_e = i_e / u_e

        ## Results:
        iou = c_iou(circles1, circles2)

        ## Tests:
        nptest.assert_allclose(iou, iou_e, atol=1e-7)



    def test_cylinder(self):
        """
        Used the values from circle tests and added the height component
        """

        ## Inputs:
        cylinders1 = np.asarray([[-9,  1, 0, 1, 1],
                                 [ 9, -1, 2, 2, 2],
                                 [ 9,  1, 0, 3, 5],
                                 [ 4, -2, 0, 9, 2],
                                 [ 3,  3, 0, 5, 1]], dtype=np.float32)

        cylinders2 = np.asarray([[ 9,  1, 0, 2, 4],
                                 [ 9,  5, 4,10, 4]], dtype='float32')

        ## Expected results:
        i_e = np.asarray([[     0.0         ,               0.0],
                          [   4.9134787944*2,               0.0],
                          [  12.5663706144*4,   28.274333882301],
                          [  12.5663706143*2,               0.0],
                          [ 1.21481498543527,               0.0]])

        u_e = np.asarray([[(1**2*pi*1 + 2**2*pi*4), (1**2*pi*1 + 10**2*pi*4)],
                          [(2**2*pi*2 + 2**2*pi*4), (2**2*pi*2 + 10**2*pi*4)],
                          [(3**2*pi*5 + 2**2*pi*4), (3**2*pi*5 + 10**2*pi*4)],
                          [(9**2*pi*2 + 2**2*pi*4), (9**2*pi*2 + 10**2*pi*4)],
                          [(5**2*pi*1 + 2**2*pi*4), (5**2*pi*1 + 10**2*pi*4)]]) \
                          - i_e

        iou_e = i_e / u_e

        ## Results:
        iou = c_iou(cylinders1, cylinders2)

        ## Tests:
        nptest.assert_allclose(iou, iou_e, atol=1e-7)



    def test_rectangle(self):
        """
        Values calculated by hand
        """

        ## Inputs:
        rectangles1 = np.asarray([[-9,  1, 2, 2],
                                  [ 9, -1, 2, 2],
                                  [ 9,  1, 4, 4],
                                  [ 4, -2, 8, 4],
                                  [ 3,  3, 6,12],
                                  [13, -1, 2, 2.8]], dtype='float32')

        rectangles2 = np.asarray([[ 9,  1, 4, 4],
                                  [ 9,  5,10,10]], dtype='float32')

        ## Expected results:
        i_e = np.asarray([[ 0.0,   0.0],
                          [ 2.0,   0.0],
                          [16.0,  12.0],
                          [ 1.0,   0.0],
                          [ 0.0,  18.0],
                          [ 0.0,   0.8]])

        u_e = np.asarray([[(2*2)+(4*4),   (2*2)+(10*10)],
                          [(2*2)+(4*4),   (2*2)+(10*10)],
                          [(4*4)+(4*4),   (4*4)+(10*10)],
                          [(8*4)+(4*4),   (8*4)+(10*10)],
                          [(12*6)+(4*4),  (12*6)+(10*10)],
                          [(2*2.8)+(4*4), (2*2.8)+(10*10)]]) \
                          - i_e

        iou_e = i_e / u_e

        ## Results:
        iou = c_iou(rectangles1, rectangles2)

        ## Tests:
        nptest.assert_allclose(iou, iou_e, atol=1e-7)



    def test_cuboid(self):
        """
        Values calculated by hand
        """

        ## Inputs:
        cuboids1 = np.asarray([[-9,  1, 3, 2, 2, 2],
                               [ 9, -1, 1, 2, 2, 3],
                               [13, -1, 1, 2,2.8,4]], dtype='float32')

        cuboids2 = np.asarray([[ 9,  1, 1, 4, 4, 1],
                               [ 9,  5, 2,10,10, 2]], dtype='float32')

        ## Expected results:
        i_e = np.asarray([[ 0.0,   0.0],
                          [ 2.0,   0.0],
                          [ 0.0,   1.6]])

        u_e = np.asarray([[(2*2*2)+(4*4*1),   (2*2*2)+(10*10*2)],
                          [(2*2*3)+(4*4*1),   (2*2*3)+(10*10*2)],
                          [(2*2.8*4)+(4*4*1), (2*2.8*4)+(10*10*2)]]) \
                          - i_e

        iou_e = i_e / u_e

        ## Results:
        iou = c_iou(cuboids1, cuboids2)

        ## Tests:
        nptest.assert_allclose(iou, iou_e, atol=1e-7)


    def test_input_shape(self):
        ## Inputs:
        to_large = np.ones([3,7], dtype=np.float32)
        to_small = np.ones([4,2], dtype=np.float32)

        ## Results:
        self.assertRaises(ValueError, c_iou, to_large, to_large)
        self.assertRaises(ValueError, c_iou, to_small, to_small)
        self.assertRaises(AssertionError, c_iou, to_small, to_large)
        
        
        


###############################################################################
# =============================================================================
# Circles
# =============================================================================
###############################################################################

class Test_Circle_NMS(unittest.TestCase):

    # circle = x, y, radius (x,y = center coordinates)

    def test_various(self):
        ## Inputs:
        circles = np.asarray([[ 5,  5, 5],
                              [ 5,  4, 5],
                              [ 9,  1, 1]], dtype=np.float32)

        scores = np.asarray([ 4,  9, 2], dtype=np.float32)
        iou_max = 0.7

        ## Expected results:
        nms_indices_e = np.asarray([1, 2])

        ## Results:
        nms_indices = c_circle_nms(circles, scores, iou_max)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)



    def test_score_min(self):
        ## Inputs:
        circles = np.asarray([[ 5,  5, 5],
                              [ 5,  4, 5],
                              [ 9,  1, 1],
                              [ 9,  2, 1]], dtype=np.float32)

        scores = np.asarray([ 4,  9, 2, 5], dtype=np.float32)
        iou_max = 0.7
        score_min = 3

        ## Expected results:
        nms_indices_e = np.asarray([1, 3])

        ## Results:
        nms_indices = c_circle_nms(circles, scores, iou_max, score_min=score_min)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)



    def test_nothing_higher_than_score_min(self):
        ## Inputs:
        circles = np.asarray([[ 5,  5, 5],
                              [ 5,  4, 5],
                              [ 9,  1, 1]], dtype=np.float32)

        scores = np.asarray([ 4,  9, 2], dtype=np.float32)
        iou_max = 0.7
        score_min = 10

        ## Expected results:
        nms_indices_e = np.asarray([], dtype=np.int64)

        ## Results:
        nms_indices = c_circle_nms(circles, scores, iou_max, score_min=score_min)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)



    def test_n(self):
        ## Inputs:
        circles = np.asarray([[ 5,  5, 5],
                              [ 5,  4, 5],
                              [ 9,  1, 1]], dtype=np.float32)

        scores = np.asarray([ 4,  9, 2], dtype=np.float32)
        iou_max = 0.7
        n = 1

        ## Expected results:
        nms_indices_e = np.asarray([1])

        ## Results:
        nms_indices = c_circle_nms(circles, scores, iou_max, n=n)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)

# # Various Tests
#     def test_various_1000x(self):
#         timer = Timer()
#         for i in range(1000):
#             self.test_various()
#         print(timer.stop())
#
#
#     def test_random_big_10x(self):
#         def test_random_big():
#             circles = np.random.rand(1000 ,3).astype(np.float32) * 10
#             scores  = np.random.rand(1000).astype(np.float32) * 10
#             iou_max = 0.7
#             nms_indices = c_circle_nms(circles, scores, 0.7)
#
#         timer = Timer()
#         for i in range(10):
#             test_random_big()
#         print(timer.stop())

class Test_Circle_IOU(unittest.TestCase):

    # circle = x, y, radius (x,y = center coordinates)

    def shortDescription(self):
        """ Prevent Unittest to print first line of docstring """
        return None



    def test_various(self):
        """
        Values calculated with Matlab and the tool:
        "Analytical intersection area between two circles"
        by Guillaume JACQUENOT
        """

        ## Inputs:
        circles1 = np.asarray([[-9,  1, 1],
                               [ 9, -1, 2],
                               [ 9,  1, 3],
                               [ 4, -2, 9],
                               [ 3,  3, 5]], dtype='float32')

        circles2 = np.asarray([[ 9,  1, 2],
                               [ 9,  5,10]], dtype='float32')

        ## Expected results:
        i_e = np.asarray([[ 0.0         ,   0.0],
                          [ 4.9134787944,  12.566370614359],
                          [12.5663706144,  28.274333882301],
                          [12.5663706143, 125.419116075097],
                          [ 1.21481498543527, 70.5463719878998]])

        u_e = np.asarray([[(1**2 + 2**2) *pi, (1**2 + 10**2) *pi],
                          [(2**2 + 2**2) *pi, (2**2 + 10**2) *pi],
                          [(3**2 + 2**2) *pi, (3**2 + 10**2) *pi],
                          [(9**2 + 2**2) *pi, (9**2 + 10**2) *pi],
                          [(5**2 + 2**2) *pi, (5**2 + 10**2) *pi]]) \
                          - i_e

        iou_e = i_e / u_e

        ## Results:
        iou = c_circle_iou(circles1, circles2)

        ## Tests:
        nptest.assert_allclose(iou, iou_e, atol=1e-7)



    def test_distance0(self):
        """
        Expect intersection to be the smaller circle.
        Edge case because 0 devision inside cos-1.
        """

        ## Inputs:
        circles1 = np.asarray([[ 9,  1, 1],
                               [ 9,  1, 2],
                               [ 9,  1, 3]], dtype='float32')


        circles2 = np.asarray([[ 9,  1, 2]], dtype='float32')

        ## Expected results:
        iou_e = np.asarray([[1 * pi / (1*pi + 4*pi - 1*pi)],
                            [4 * pi / (4*pi + 4*pi - 4*pi)],
                            [4 * pi / (9*pi + 4*pi - 4*pi)]])

        ## Results:
        iou = c_circle_iou(circles1, circles2)

        ## Tests:
        nptest.assert_allclose(iou, iou_e)



    def test_no_overlapp(self):
        """
        No overlap at all
        """
        ## Inputs:
        circles1 = np.asarray([[-9,  1, 1],
                               [ 9, -1, 2],
                               [ 9,  1, 3],
                               [ 9,  5, 3]], dtype='float32') # direct touch.

        circles2 = np.asarray([[ 9,  10, 2]], dtype='float32')

        ## Expected results:
        iou_e = np.asarray([[0.], [0.], [0.], [0.]])

        ## Results:
        iou = c_circle_iou(circles1, circles2)

        ## Tests:
        nptest.assert_allclose(iou, iou_e)

# # Various Test 1000x
#     def test_various_1000x(self):
#         timer = Timer()
#         for i in range(1000):
#             self.test_various()
#         print(timer.stop())


###############################################################################
# =============================================================================
# Cylinders
# =============================================================================
###############################################################################

class Test_Cylinder_NMS(unittest.TestCase):

    # cylinder = x, y, z, radius, height (x,y,z = center coordinates)

    def test_various(self):
        ## Inputs:
        cylinders = np.asarray([[ 5,  5, 1, 5, 1],
                                [ 5,  4, 1.1, 5, 1],
                                [ 9,  1, 3, 1, 5]], dtype=np.float32)

        scores = np.asarray([ 4,  9, 2], dtype=np.float32)
        iou_max = 0.6

        ## Expected results:
        nms_indices_e = np.asarray([1, 2])

        ## Results:
        nms_indices = c_cylinder_nms(cylinders, scores, iou_max)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)



    def test_score_min(self):
        ## Inputs:
        cylinders = np.asarray([[ 5,  5, 1, 5, 1],
                                [ 5,  4, 1.1, 5, 1],
                                [ 9,  1, 3, 1, 5]], dtype=np.float32)

        scores = np.asarray([ 4,  9, 2], dtype=np.float32)
        iou_max = 0.6
        score_min = 3

        ## Expected results:
        nms_indices_e = np.asarray([1])

        ## Results:
        nms_indices = c_cylinder_nms(cylinders, scores, iou_max, score_min=score_min)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)



    def test_nothing_higher_than_score_min(self):
        ## Inputs:
        cylinders = np.asarray([[ 5,  5, 1, 5, 1],
                                [ 5,  4, 1.1, 5, 1],
                                [ 9,  1, 3, 1, 5]], dtype=np.float32)

        scores = np.asarray([ 4,  9, 2], dtype=np.float32)
        iou_max = 0.6
        score_min = 10

        ## Expected results:
        nms_indices_e = np.asarray([], dtype=np.int64)

        ## Results:
        nms_indices = c_cylinder_nms(cylinders, scores, iou_max, score_min=score_min)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)



    def test_n(self):
        ## Inputs:
        cylinders = np.asarray([[ 5,  5, 1, 5, 1],
                                [ 5,  4, 1.1, 5, 1],
                                [ 9,  1, 3, 1, 5]], dtype=np.float32)

        scores = np.asarray([ 4,  9, 2], dtype=np.float32)
        iou_max = 0.6
        n = 1

        ## Expected results:
        nms_indices_e = np.asarray([1])

        ## Results:
        nms_indices = c_cylinder_nms(cylinders, scores, iou_max, n=n)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)


class Test_Cylinder_IOU(unittest.TestCase):

    # cylinder = x, y, z, radius, height (x,y,z = center coordinates)

    def shortDescription(self):
        """ Prevent Unittest to print first line of docstring """
        return None



    def test_various(self):
        """
        Used the values from circle tests and added the height component
        """

        ## Inputs:
        cylinders1 = np.asarray([[-9,  1, 0, 1, 1],
                                 [ 9, -1, 2, 2, 2],
                                 [ 9,  1, 0, 3, 5],
                                 [ 4, -2, 0, 9, 2],
                                 [ 3,  3, 0, 5, 1]], dtype=np.float32)

        cylinders2 = np.asarray([[ 9,  1, 0, 2, 4],
                                 [ 9,  5, 4,10, 4]], dtype='float32')

        ## Expected results:
        i_e = np.asarray([[     0.0         ,              0.0],
                          [   4.9134787944*2,              0.0],
                          [  12.5663706144*4,  28.274333882301],
                          [  12.5663706143*2,              0.0],
                          [ 1.21481498543527,              0.0]])

        u_e = np.asarray([[(1**2*pi*1 + 2**2*pi*4), (1**2*pi*1 + 10**2*pi*4)],
                          [(2**2*pi*2 + 2**2*pi*4), (2**2*pi*2 + 10**2*pi*4)],
                          [(3**2*pi*5 + 2**2*pi*4), (3**2*pi*5 + 10**2*pi*4)],
                          [(9**2*pi*2 + 2**2*pi*4), (9**2*pi*2 + 10**2*pi*4)],
                          [(5**2*pi*1 + 2**2*pi*4), (5**2*pi*1 + 10**2*pi*4)]]) \
                          - i_e

        iou_e = i_e / u_e

        ## Results:
        iou = c_cylinder_iou(cylinders1, cylinders2)

        ## Tests:
        nptest.assert_allclose(iou, iou_e, atol=1e-7)



    def test_distance0(self):
        """
        Expect intersection to be the smaller cylinder.
        Edge case because 0 devision inside cos-1.
        """

        ## Inputs:
        cylinders1 = np.asarray([[ 9,  1, 0, 1, 1],
                                 [ 9,  1, 0, 2, 2],
                                 [ 9,  1, 0, 3, 4]], dtype='float32')


        cylinders2 = np.asarray([[ 9,  1, 0, 2, 2]], dtype='float32')

        ## Expected results:
        iou_e = np.asarray([[1*pi*1 / (1*pi*1 + 4*pi*2 - 1*pi*1)],
                            [4*pi*2 / (4*pi*2 + 4*pi*2 - 4*pi*2)],
                            [4*pi*2 / (9*pi*4 + 4*pi*2 - 4*pi*2)]])

        ## Results:
        iou = c_cylinder_iou(cylinders1, cylinders2)

        ## Tests:
        nptest.assert_allclose(iou, iou_e)



    def test_no_overlapp(self):
        """
        No overlap at all
        """
        ## Inputs:
        cylinders1 = np.asarray([[-9,  1, 0, 1, 4],
                                 [ 9, -1, 3, 2, 2],
                                 [ 9,  1,-1, 3, 8],
                                 [ 9,  5, 2, 3, 3]], dtype='float32') # direct touch.

        cylinders2 = np.asarray([[ 9,  10, 0, 2, 6]], dtype='float32')

        ## Expected results:
        iou_e = np.asarray([[0.], [0.], [0.], [0.]])

        ## Results:
        iou = c_cylinder_iou(cylinders1, cylinders2)

        ## Tests:
        nptest.assert_allclose(iou, iou_e)


###############################################################################
# =============================================================================
# Rectangles
# =============================================================================
###############################################################################

class Test_Rectangle_NMS(unittest.TestCase):

    # rectangle = x, y, length, width (x,y = center coordinates)

    def test_various(self):
        ## Inputs:
        rectangles = np.asarray([[ 5,  5, 4, 4],
                                 [ 5,  4, 4, 4],
                                 [ 5,  6, 4, 4],
                                 [ 9,  1, 6, 6]], dtype=np.float32)

        scores = np.asarray([ 4,  9, 2, 3], dtype=np.float32)
        iou_max = 0.5

        ## Expected results:
        nms_indices_e = np.asarray([1, 3, 2])

        ## Results:
        nms_indices = c_rectangle_nms(rectangles, scores, iou_max)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)



    def test_score_min(self):
        ## Inputs:
        rectangles = np.asarray([[ 5,  5, 4, 4],
                                 [ 5,  4, 4, 4],
                                 [ 9,  1, 2, 2],
                                 [ 9,  2, 2, 2]], dtype=np.float32)

        scores = np.asarray([ 4,  9, 2, 5], dtype=np.float32)
        iou_max = 0.5
        score_min = 3

        ## Expected results:
        nms_indices_e = np.asarray([1, 3])

        ## Results:
        nms_indices = c_rectangle_nms(rectangles, scores, iou_max, score_min=score_min)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)



    def test_nothing_higher_than_score_min(self):
        ## Inputs:
        rectangles = np.asarray([[ 5,  5, 4, 4],
                                 [ 5,  4, 4, 4],
                                 [ 9,  1, 1, 5]], dtype=np.float32)

        scores = np.asarray([ 4,  9, 2], dtype=np.float32)
        iou_max = 0.5
        score_min = 10

        ## Expected results:
        nms_indices_e = np.asarray([], dtype=np.int64)

        ## Results:
        nms_indices = c_rectangle_nms(rectangles, scores, iou_max, score_min=score_min)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)



    def test_n(self):
        ## Inputs:
        rectangles = np.asarray([[ 5,  5, 5, 3],
                                 [ 5,  4, 3, 3],
                                 [ 9,  1, 1, 1]], dtype=np.float32)

        scores = np.asarray([ 4,  9, 2], dtype=np.float32)
        iou_max = 0.5
        n = 1

        ## Expected results:
        nms_indices_e = np.asarray([1])

        ## Results:
        nms_indices = c_rectangle_nms(rectangles, scores, iou_max, n=n)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)


class Test_Rectangle_IOU(unittest.TestCase):

    # rectangle = x, y, length, width (x,y = center coordinates)

    def shortDescription(self):
        """ Prevent Unittest to print first line of docstring """
        return None


    def test_various(self):
        """
        Values calculated by hand
        """

        ## Inputs:
        rectangles1 = np.asarray([[-9,  1, 2, 2],
                                  [ 9, -1, 2, 2],
                                  [ 9,  1, 4, 4],
                                  [ 4, -2, 8, 4],
                                  [ 3,  3, 6,12],
                                  [13, -1, 2, 2.8]], dtype='float32')

        rectangles2 = np.asarray([[ 9,  1, 4, 4],
                                  [ 9,  5,10,10]], dtype='float32')

        ## Expected results:
        i_e = np.asarray([[ 0.0,   0.0],
                          [ 2.0,   0.0],
                          [16.0,  12.0],
                          [ 1.0,   0.0],
                          [ 0.0,  18.0],
                          [ 0.0,   0.8]])

        u_e = np.asarray([[(2*2)+(4*4),   (2*2)+(10*10)],
                          [(2*2)+(4*4),   (2*2)+(10*10)],
                          [(4*4)+(4*4),   (4*4)+(10*10)],
                          [(8*4)+(4*4),   (8*4)+(10*10)],
                          [(12*6)+(4*4),  (12*6)+(10*10)],
                          [(2*2.8)+(4*4), (2*2.8)+(10*10)]]) \
                          - i_e

        iou_e = i_e / u_e

        ## Results:
        iou = c_rectangle_iou(rectangles1, rectangles2)

        ## Tests:
        nptest.assert_allclose(iou, iou_e, atol=1e-7)



    def test_distance0(self):
        """
        Expect intersection to be the smaller rectangle.
        """

        ## Inputs:
        rectangles1 = np.asarray([[ 9,  1, 3, 3],
                                  [ 9,  1, 4, 4],
                                  [ 9,  1, 5, 5]], dtype='float32')


        rectangles2 = np.asarray([[ 9,  1, 4, 4]], dtype='float32')

        ## Expected results:
        iou_e = np.asarray([[(3*3) / (4*4)],
                            [(4*4) / (4*4)],
                            [(4*4) / (5*5)]])

        ## Results:
        iou = c_rectangle_iou(rectangles1, rectangles2)

        ## Tests:
        nptest.assert_allclose(iou, iou_e)



    def test_no_overlapp(self):
        """
        No overlap at all
        """
        ## Inputs:
        rectangles1 = np.asarray([[-9,  1, 2, 2],
                                  [ 9, -1, 4, 4],
                                  [ 9,  1, 3, 3],
                                  [ 9,  5, 20,4]], dtype='float32') # direct touch.

        rectangles2 = np.asarray([[ 9, 10, 6, 6]], dtype='float32')

        ## Expected results:
        iou_e = np.asarray([[0.], [0.], [0.], [0.]])

        ## Results:
        iou = c_rectangle_iou(rectangles1, rectangles2)

        ## Tests:
        nptest.assert_allclose(iou, iou_e)


###############################################################################
# =============================================================================
# Cuboids
# =============================================================================
###############################################################################

class Test_Cuboid_NMS(unittest.TestCase):

    # cuboid = x, y, z, length, width, height (x,y,z = bottom center coordinates)

    def test_various(self):
        ## Inputs:
        cuboids = np.asarray([[ 5,  5, 1, 4, 4, 1],
                              [ 5,  4, 1, 4, 4, 1],
                              [ 5,  4, 1, 4, 4, 2],
                              [ 9,  1, 2, 6, 6, 4]], dtype=np.float32)

        scores = np.asarray([4,  9, 2, 3], dtype=np.float32)
        iou_max = 0.5

        ## Expected results:
        nms_indices_e = np.asarray([1, 3, 2])

        ## Results:
        nms_indices = c_cuboid_nms(cuboids, scores, iou_max)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)



    def test_score_min(self):
        ## Inputs:
        cuboids = np.asarray([[ 5,  5, 1, 4, 4, 1],
                              [ 5,  4, 1, 4, 4, 1],
                              [ 5,  4, 1, 4, 4, 2],
                              [ 9,  1, 2, 6, 6, 4]], dtype=np.float32)

        scores = np.asarray([ 4,  9, 2, 5], dtype=np.float32)
        iou_max = 0.5
        score_min = 3

        ## Expected results:
        nms_indices_e = np.asarray([1, 3])

        ## Results:
        nms_indices = c_cuboid_nms(cuboids, scores, iou_max, score_min=score_min)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)



    def test_nothing_higher_than_score_min(self):
        ## Inputs:
        cuboids = np.asarray([[ 5,  5, 1, 4, 4, 1],
                              [ 5,  4, 1, 4, 4, 1],
                              [ 5,  4, 1, 4, 4, 2],
                              [ 9,  1, 2, 6, 6, 4]], dtype=np.float32)

        scores = np.asarray([ 4,  9, 2, 3], dtype=np.float32)
        iou_max = 0.5
        score_min = 10

        ## Expected results:
        nms_indices_e = np.asarray([], dtype=np.int64)

        ## Results:
        nms_indices = c_cuboid_nms(cuboids, scores, iou_max, score_min=score_min)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)



    def test_n(self):
        ## Inputs:
        cuboids = np.asarray([[ 5,  5, 1, 4, 4, 1],
                              [ 5,  4, 1, 4, 4, 1],
                              [ 5,  4, 1, 4, 4, 2],
                              [ 9,  1, 2, 6, 6, 4]], dtype=np.float32)

        scores = np.asarray([ 4,  9, 2, 3], dtype=np.float32)
        iou_max = 0.5
        n = 2

        ## Expected results:
        nms_indices_e = np.asarray([1, 3])

        ## Results:
        nms_indices = c_cuboid_nms(cuboids, scores, iou_max, n=n)

        ## Tests:
        nptest.assert_allclose(nms_indices, nms_indices_e)


class Test_cuboid_IOU(unittest.TestCase):

    # cuboid = x, y, z, length, width, height (x,y,z = center coordinates)

    def shortDescription(self):
        """ Prevent Unittest to print first line of docstring """
        return None


    def test_various(self):
        """
        Values calculated by hand
        """

        ## Inputs:
        cuboids1 = np.asarray([[-9,  1,  3, 2,  2,  2],
                               [ 9, -1,  1, 2,  2,  3],
                               [13, -1,  1, 2,2.8,  4],
                               [ 9,  1,0.5, 2,  2,0.8]], dtype='float32')

        cuboids2 = np.asarray([[ 9,  1, 1, 4, 4, 1],
                               [ 9,  5, 2,10,10, 2]], dtype='float32')

        ## Expected results:
        i_e = np.asarray([[ 0.0,   0.0],
                          [ 2.0,   0.0],
                          [ 0.0,   1.6],
                          [ 1.2,   0.0]])

        u_e = np.asarray([[(2*2*2)+(4*4*1),   (2*2*2)+(10*10*2)],
                          [(2*2*3)+(4*4*1),   (2*2*3)+(10*10*2)],
                          [(2*2.8*4)+(4*4*1), (2*2.8*4)+(10*10*2)],
                          [(2*2*0.8)+(4*4*1), (2*2*0.8)+(10*10*2)]]) \
                          - i_e

        iou_e = i_e / u_e

        ## Results:
        iou = c_cuboid_iou(cuboids1, cuboids2)

        ## Tests:
        nptest.assert_allclose(iou, iou_e, atol=1e-7)



    def test_distance0(self):
        """
        Expect intersection to be the smaller cuboid.
        """

        ## Inputs:
        cuboids1 = np.asarray([[ 9,  1, 1, 3, 3, 1],
                               [ 9,  1, 1, 4, 4, 1],
                               [ 9,  1, 1, 5, 5, 5]], dtype='float32')


        cuboids2 = np.asarray([[ 9,  1, 1, 4, 4, 1]], dtype='float32')

        ## Expected results:
        iou_e = np.asarray([[(3*3*1) / (4*4*1)],
                            [(4*4*1) / (4*4*1)],
                            [(4*4*1) / (5*5*5)]])

        ## Results:
        iou = c_cuboid_iou(cuboids1, cuboids2)

        ## Tests:
        nptest.assert_allclose(iou, iou_e)



    def test_no_overlapp(self):
        """
        No overlap at all
        """
        ## Inputs:
        cuboids1 = np.asarray([[-9,  1, 3, 2, 2, 0.1],
                               [ 9, -1, 1, 4, 4, 2],
                               [ 9,  1, 1, 3, 3, 2],
                               [ 9,  5, 1, 20,4, 2],                    # touch in y
                               [ 9, 10, 5, 6, 6, 2]], dtype='float32')  # touch in z

        cuboids2 = np.asarray([[ 9, 10, 1, 6, 6, 4]], dtype='float32')

        ## Expected results:
        iou_e = np.asarray([[0.], [0.], [0.], [0.], [0.]])

        ## Results:
        iou = c_cuboid_iou(cuboids1, cuboids2)

        ## Tests:
        nptest.assert_allclose(iou, iou_e)

if __name__ == '__main__':
    unittest.main()
