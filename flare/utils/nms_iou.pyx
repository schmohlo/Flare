""" This module contains cython implemententation of Non-Max-Supression for circles
    cylinders, rectangles and cuboids.

Content:
    - nms():                NMS for infered object type
    - nms_circles():        NMS for circles
    - nms_cylinders():      NMS for cylinders
    - nms_rectangle():      NMS for rectangles
    - nms_cuboid():         NMS for cuboids
    - iou():                IoU for infered object type
    - iou_circles():        IoU for circles
    - iou_cylinders():      IoU for cylinders
    - iou_rectangle():      IoU for rectangles
    - iou_cuboid():         IoU for cuboids
    - _nms():               Non-maximum-supression for two lists
    - _iou():               Intersection-Over-Union for two lists
    - _..._iou_1pair():     Optimised cython backbone.

Usage:
    import pyximport
    pyximport.install(setup_args={"include_dirs":np.get_include()}, language_level=3)
    import nms_iou

WARNING:
    - Bound checks and wrawp around (=neg indixing) are deactivated for speed.


by Stefan Schmohl, Johannes Ernst 2021
"""

import math
import numpy as np
cimport cython
cimport numpy as cnp

from libc.math cimport acos
from libc.math cimport sqrt
from libc.math cimport pi

# Typedef function pointer
ctypedef float (*f_type)(float[:], float[:])




@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

def nms(float[:, :] objects, cnp.ndarray[cnp.float32_t, ndim=1] scores,
        float iou_max,
        float score_min=float('-inf'), long n=-1):
    """ Non-Maximum Supression (NMS) for given objects.

    The supported object types are inferred from input shape:
        Circles     (x, y, radius)                      --> var = 3
        Rectangles  (x, y, length, width)               --> var = 4
        Cylinders   (x, y, z, radius, height)           --> var = 5
        Cuboids     (x, y, z, length, width, height)    --> var = 6

    Args:
        objects     ndarray[N, var]
        scores:     ndarray[N]
        iou_max:    Specifies max allowed overlap between two objects.
        score_min:  Removes objects with scores < score_min
        n:          Max number of objects to return (keeps n highest
                    scoring objects)

    Returns:
        indices of maximum objects.
    """

    # Infering object type from input shape:
    if objects.shape[1] == 3:
        iou_fu = _circle_iou_1pair
    elif objects.shape[1] == 4:
        iou_fu = _rectangle_iou_1pair
    elif objects.shape[1] == 5:
        iou_fu = _cylinder_iou_1pair
    elif objects.shape[1] == 6:
        iou_fu = _cuboid_iou_1pair
    else:
        raise ValueError("Unsuportet object type inferred from array size")

    # Calculating NMS
    nms_result = _nms(objects, scores, iou_max, score_min, n, iou_fu)

    return nms_result



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

def iou(float[:, :] objects1, float[:, :] objects2):
    """ Return intersection-over-union (Jaccard index) of cylinders.

    The supported object types are inferred from input shape:
        Circles     (x, y, radius)                      --> var = 3
        Rectangles  (x, y, length, width)               --> var = 4
        Cylinders   (x, y, z, radius, height)           --> var = 5
        Cuboids     (x, y, z, length, width, height)    --> var = 6

    Args:
        objects1    ndarray[N, var]
        objects2    ndarray[N, var]

    Returns:
        iou ndarray[N, M]:  NxM matrix containing the pairwise IoU values
                            for every element in cylinders1 and cylinders2.
    """

    assert objects1.shape[1] == objects2.shape[1]
    
    # Infering object type from input shape:
    if objects1.shape[1] == 3:
        iou_fu = _circle_iou_1pair
    elif objects1.shape[1] == 4:
        iou_fu = _rectangle_iou_1pair
    elif objects1.shape[1] == 5:
        iou_fu = _cylinder_iou_1pair
    elif objects1.shape[1] == 6:
        iou_fu = _cuboid_iou_1pair
    else:
        raise ValueError("Unsuportet object type inferred from array size")

    # Calculating IoU
    iou_result = _iou(objects1, objects2, iou_fu)

    return iou_result




    
###############################################################################
###############################################################################
###############################################################################
###############################################################################



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

def nms_circles(cnp.ndarray[cnp.float32_t, ndim=2] circles,
                cnp.ndarray[cnp.float32_t, ndim=1] scores,
                float iou_max,
                float score_min=float('-inf'), long n=-1):
    """ Non-Maximum Supression for circles.

    Args:
        circles:          ndarray[N, 3] (x,y,radius) with x,y = center
        scores:           ndarray[N]
        iou_max:          Specifies max allowed overlap between two circles.
        score_min:        Removes circles with scores < score_min
        n:                Max number of circles to return (keeps n highest
                          scoring circles)

    Returns:
        indices of maximum circles.
    """

    # Checking geometry (x,y,radius)
    assert circles.shape[1] == 3

    # Calculating NMS
    nms_result = _nms(circles, scores, iou_max, score_min, n, _circle_iou_1pair)

    return nms_result



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

def nms_cylinders(cnp.ndarray[cnp.float32_t, ndim=2] cylinders,
                  cnp.ndarray[cnp.float32_t, ndim=1] scores,
                  float iou_max,
                  float score_min=float('-inf'), long n=-1):
    """ Non-Maximum Supression for cylinders.

    Args:
        cylinders:        ndarray[N, 5] (x,y,z,radius,height) with x,y,z = bottom center
        scores:           ndarray[N]
        iou_max:          Specifies max allowed overlap between two cylinders.
        score_min:        Removes cylinders with scores < score_min
        n:                Max number of cylinders to return (keeps n highest
                          scoring cylinders)

    Returns:
        indices of maximum cylinders.
    """

    # Checking geometry (x,y,z,radius,height)
    assert cylinders.shape[1] == 5

    # Calculating NMS
    nms_result = _nms(cylinders, scores, iou_max, score_min, n, _cylinder_iou_1pair)

    return nms_result



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

def nms_rectangles(cnp.ndarray[cnp.float32_t, ndim=2] rectangles,
                   cnp.ndarray[cnp.float32_t, ndim=1] scores,
                   float iou_max,
                   float score_min=float('-inf'), long n=-1):
    """ Non-Maximum Supression for rectangles.

    Args:
        rectangles:       ndarray[N, 4] (x,y,length,width) with x,y = center
        scores:           ndarray[N]
        iou_max:          Specifies max allowed overlap between two rectangles.
        score_min:        Removes rectangles with scores < score_min
        n:                Max number of rectangles to return (keeps n highest
                          scoring rectangles)

    Returns:
        indices of maximum rectangles.
    """

    # Checking geometry (x,y,length,width)
    assert rectangles.shape[1] == 4

    # Calculating NMS
    nms_result = _nms(rectangles, scores, iou_max, score_min, n, _rectangle_iou_1pair)

    return nms_result



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

def nms_cuboids(cnp.ndarray[cnp.float32_t, ndim=2] cuboids,
                cnp.ndarray[cnp.float32_t, ndim=1] scores,
                float iou_max,
                float score_min=float('-inf'), long n=-1):
    """ Non-Maximum Supression for cuboids.

    Args:
        cuboids:          ndarray[N, 6] (x,y,z,length,width,height) with x,y,z = bottom center
        scores:           ndarray[N]
        iou_max:          Specifies max allowed overlap between two cuboids.
        score_min:        Removes cuboids with scores < score_min
        n:                Max number of cuboids to return (keeps n highest
                          scoring cuboids)

    Returns:
        indices of maximum cuboids.
    """

    # Checking geometry (x,y,z,length,width,height)
    assert cuboids.shape[1] == 6

    # Calculating NMS
    nms_result = _nms(cuboids, scores, iou_max, score_min, n, _cuboid_iou_1pair)

    return nms_result



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

def iou_circles(float[:, :] circles1, float[:, :] circles2):
    """
    Return intersection-over-union (Jaccard index) of circles.

    Formular explained at:
    https://mathworld.wolfram.com/Circle-CircleIntersection.html

    Both sets of circles are expected to be in (x, y, r) format,
    where x,y = center.

    Args:
        circles1 ndarray[N, 3]
        circles2 ndarray[M, 3]

    Returns:
        iou ndarray[N, M]:     NxM matrix containing the pairwise IoU values
                                for every element in circles1 and circles2.
    """

    # Checking geometry
    assert circles1.shape[1] == circles2.shape[1] == 3

    # Calculating IoU
    iou_result = _iou(circles1, circles2, &_circle_iou_1pair)

    return iou_result



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

def iou_cylinders(float[:, :] cylinders1, float[:, :] cylinders2):
    """
    Return intersection-over-union (Jaccard index) of cylinders.

    Both sets of cylinders are expected to be in (x, y, z, radius, height) format,
    where x,y,z = bottom center.

    Args:
        cylinders1 ndarray[N, 5]
        cylinders2 ndarray[M, 5]

    Returns:
        iou ndarray[N, M]:  NxM matrix containing the pairwise IoU values
                            for every element in cylinders1 and cylinders2.
    """

    # Checking geometry
    assert cylinders1.shape[1] == cylinders2.shape[1] == 5

    # Calculating IoU
    iou_result = _iou(cylinders1, cylinders2, _cylinder_iou_1pair)

    return iou_result



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

def iou_rectangles(float[:, :] rectangles1, float[:, :] rectangles2):
    """
    Return intersection-over-union (Jaccard index) of rectangles.

    Both sets of rectangles are expected to be in (x, y, length, width) format,
    where x,y = center.

    Args:
        rectangles1 ndarray[N, 4]
        rectangles2 ndarray[M, 4]

    Returns:
        iou ndarray[N, M]:     NxM matrix containing the pairwise IoU values
                                for every element in rectangles1 and rectangles2.
    """

    # Checking geometry
    assert rectangles1.shape[1] == rectangles2.shape[1] == 4

    # Calculating IoU
    iou_result = _iou(rectangles1, rectangles2, _rectangle_iou_1pair)

    return iou_result



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

def iou_cuboids(float[:, :] cuboids1, float[:, :] cuboids2):
    """
    Return intersection-over-union (Jaccard index) of cuboids.

    Both sets of cuboids are expected to be in (x, y, z, length, width, height)
    format, where x,y,z = bottom center.

    Args:
        cuboids1 ndarray[N, 6]
        cuboids2 ndarray[M, 6]

    Returns:
        iou ndarray[N, M]:      NxM matrix containing the pairwise IoU values
                                for every element in cuboids1 and cuboids2.
    """

    # Checking geometry
    assert cuboids1.shape[1] == cuboids2.shape[1] == 6

    # Calculating IoU
    iou_result = _iou(cuboids1, cuboids2, _cuboid_iou_1pair)

    return iou_result



###############################################################################
###############################################################################
###############################################################################
###############################################################################



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

cdef _nms(float[:, :] objects, cnp.ndarray[cnp.float32_t, ndim=1] scores,
          float iou_max, float score_min, long n, f_type iou_1pair):
    """  Core function for calculating Non-Maximum Supression (NMS).

    The supported object types are:
    Circles     (x, y, radius)                      --> var = 3
    Rectangles  (x, y, length, width)               --> var = 4
    Cylinders   (x, y, z, radius, height)           --> var = 5
    Cuboids     (x, y, z, length, width, height)    --> var = 6

    Args:
        objects     ndarray[N, var]
        scores:     ndarray[N]
        iou_max:    Specifies max allowed overlap between two objects.
        score_min:  Removes objects with scores < score_min
        n:          Max number of objects to return (keeps n highest
                    scoring objects)
        iou_1pair:  Function pointer to the respecitve object function
                    (e.g. pointer to _circle_iou_1pair)

    Returns:
        indices of maximum cuboids to nms_object function
    """
    ## Filter out scores to low:
    indices_min = (scores >= score_min).nonzero()[0]
    if indices_min.size == 0:
        return np.asarray([], dtype=np.int64)
    scores = scores[indices_min]

    ## Sort leftover scores descendingly:
    indices_sort = np.argsort(-scores)

    ## Remove objects with scores to low and sort the rest descending by score:
    indices = indices_min[indices_sort]

    ## Non-Maximum Supression:
    # Starting from highest scoring object, add new object to keep-list if it
    # has no overlap to any object already in keep-list.
    cdef list keep_indices = []

    cdef Py_ssize_t[:] indices_view = indices
    cdef float[:, :]   objects_view = objects

    cdef Py_ssize_t i, ind, j
    cdef Py_ssize_t len_indices = indices.shape[0]
    cdef float iou
    cdef bint do_add

    # Start by adding first / highest scoring object
    ind = indices_view[0]
    keep_indices.append(ind)

    for i in range(1, len_indices):
        ind = indices_view[i]
        do_add = True
        for j in keep_indices:
            # Add only, if no iou is greater then threshold:
            iou  = iou_1pair(objects_view[j,:], objects_view[ind, :])
            if iou > iou_max:
                do_add = False
                break
        if do_add:
            keep_indices.append(ind)

    ## Return n highest scoring objects:
    if n > -1 and len(keep_indices) > n:
        keep_indices = keep_indices[0:n]

    return np.stack(keep_indices)



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

cdef _iou(float[:, :] obj1, float[:, :] obj2, f_type iou_1pair):
    """  Core function for calculating intersection-over-union.

    !! Note that both objects have to be of the same type !!
    The supported object types are:
    Circles     (x, y, radius)                      --> var = 3
    Rectangles  (x, y, length, width)               --> var = 4
    Cylinders   (x, y, z, radius, height)           --> var = 5
    Cuboids     (x, y, z, length, width, height)    --> var = 6

    Args:
        obj1:       ndarray[N, var]
        obj2:       ndarray[M, var]
        iou_1pair:  Function pointer to the respecitve object function
                    (e.g. pointer to _circle_iou_1pair)

    Returns:
        iou ndarray[N, M]:  NxM matrix containing the pairwise IoU values
                            for every element in obj1 and obj2.
    """
    cdef Py_ssize_t N = obj1.shape[0]
    cdef Py_ssize_t M = obj2.shape[0]
    cdef Py_ssize_t n, m

    iou = np.zeros((N, M), dtype=np.float32)
    cdef float[:, :] iou_view = iou

    for n in range(N):
        for m in range(M):
           iou_view[n, m] = iou_1pair(obj1[n, :], obj2[m, :])

    return iou



###############################################################################
###############################################################################
###############################################################################
###############################################################################



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Deactivate zero-division exception checking

cdef float _circle_iou_1pair(float[:] circle1, float[:] circle2):
    """ Returns intersection-over-union (Jaccard index) of two circles.

    Formular explained at:
    https://mathworld.wolfram.com/Circle-CircleIntersection.html

    Additional criterion of minimum distance, to prevent very small circles
    lying in center of bigger one. May also speed things up.

    Args:
        circles1 ndarray[3]  [x, y, radius]
        circles2 ndarray[3]  [x, y, radius]

    Returns:
        iou float
    """

    cdef float iou
    cdef float r, r2, R, R2, a, A, x, y, X, Y, d, d2, i
    cdef float temp1, temp2, temp3, temp4, temp5

    ## Intersection-Over-Union:
    # Assigning parameters
    r = circle1[2]
    R = circle2[2]

    x = circle1[0]
    y = circle1[1]
    X = circle2[0]
    Y = circle2[1]

    d2 = (x - X)**2 + (y - Y)**2   # distance squared
    d  = sqrt(d2)              # distance

    # Cases without overlap:
    if d >= r+R:
        return 0.0

    # Note: precomputinng r2, a, etc. in beforehand has no benefit!
    r2 = r ** 2
    R2 = R ** 2

    a = r2 * pi
    A = R2 * pi

    # Cases, where the smaller circle lies within the bigger one:
    if d <= abs(r - R):
        i = min(a, A)
        iou = i / (a + A - i)
        return iou

    # Normal case intersection:
    temp1 = d2 + r2 - R2
    temp2 = 2 * d * r
    temp1 = 0 if temp2 == 0 else temp1 / temp2

    temp3 = d2 + R2 - r2
    temp4 = 2 * d * R
    temp3 = 0 if temp4 == 0 else temp3 / temp4

    temp5 = (-d+r+R) * (d+r-R) * (d-r+R) * (d+r+R)

    i = r2 * acos(temp1) + \
        R2 * acos(temp3) - \
        .5 * sqrt(temp5)

    # Calculating IoU
    iou = i / (a + A - i)
    return iou



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Deactivate zero-division exception checking

cdef float _cylinder_iou_1pair(float[:] cyl1, float[:] cyl2):
    """ Returns intersection-over-union (Jaccard index) of two cylinders.

    Formular for circle intersection explained at:
    https://mathworld.wolfram.com/Circle-CircleIntersection.html

    Additional criterion of minimum distance, to prevent very small cylinders
    lying in center of bigger one. May also speed things up.

    Cylinders are specified by the given radius, height and the bottom center
    coordinates (x,y,z).

    This function uses the same approach as the function for circles but
    with the added height dimension.

    Args:
        cyl1 ndarray[5]  [x, y, z, radius, height]
        cyl2 ndarray[5]  [x, y, z, radius, height]

    Returns:
        iou float
    """

    cdef float iou, i
    cdef float x, y, z, r, h, X, Y, Z, R, H, zb, zt, Zb, Zt, v, V, r2, R2, d, d2
    cdef float temp1, temp2, temp3, temp4, temp5, temp6, temp7

    ## Intersection-Over-Union:
    # Assigning parameters
    x = cyl1[0]
    y = cyl1[1]
    z = cyl1[2]
    r = cyl1[3]
    h = cyl1[4]

    X = cyl2[0]
    Y = cyl2[1]
    Z = cyl2[2]
    R = cyl2[3]
    H = cyl2[4]

    d2 = (x-X)**2 + (y-Y)**2   # distance squared (planar: x and y only)
    d  = sqrt(d2)              # distance (planar)

    # Borders in z-direction
    zb = z
    zt = z+h

    Zb = Z
    Zt = Z+H

    # Cases without overlap:
    if d >= r+R or Zb >= zt or Zt <= zb:
        return 0.0

    # Computing volumes
    r2 = r ** 2
    R2 = R ** 2

    v = r2 * pi * h
    V = R2 * pi * H

    # Computing borders in height
    temp6 = max(zb, Zb)
    temp7 = min(zt, Zt)

    # Cases, where the smaller cylinder lies within the bigger one:
    if d <= abs(r - R):
        i = min(r2, R2) * pi * (temp7 - temp6)
        iou = i / (v + V - i)
        return iou

    # Normal case intersection:
    temp1 = d2 + r2 - R2
    temp2 = 2 * d * r
    temp1 = 0 if temp2 == 0 else temp1 / temp2

    temp3 = d2 + R2 - r2
    temp4 = 2 * d * R
    temp3 = 0 if temp4 == 0 else temp3 / temp4

    temp5 = (-d+r+R) * (d+r-R) * (d-r+R) * (d+r+R)

    i = (r2 * acos(temp1) + \
         R2 * acos(temp3) - \
         .5 * sqrt(temp5)) * (temp7 - temp6)

    # Calculating IoU
    iou = i / (v + V - i)
    return iou



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Deactivate zero-division exception checking

cdef float _rectangle_iou_1pair(float[:] rect1, float[:] rect2):
    """ Returns intersection-over-union (Jaccard index) of two rectangles.

    Rectangle borders are specified by the given length, width and the
    center coordinates (x,y).

    Args:
        rect1 ndarray[4]  [x (center), y (center), length, width]
        rect2 ndarray[4]  [x (center), y (center), length, width]

    Returns:
        iou float
    """

    cdef float iou, i
    cdef float x, y, l, w, X, Y, L, W, a, A
    cdef float left, right, bottom, top, Left, Right, Bottom, Top
    cdef float temp1, temp2, temp3, temp4

    ## Intersection-Over-Union:
    # Assigning parameters
    x = rect1[0]
    y = rect1[1]
    l = rect1[2]
    w = rect1[3]

    X = rect2[0]
    Y = rect2[1]
    L = rect2[2]
    W = rect2[3]

    # Borders of the rectangles
    left   = x - l/2
    right  = x + l/2
    bottom = y - w/2
    top    = y + w/2

    Left   = X - L/2
    Right  = X + L/2
    Bottom = Y - W/2
    Top    = Y + W/2

    # Cases without overlap:
    if Left >= right or Right <= left or Bottom >= top or Top <= bottom:
        return 0.0

    # Calculating area
    a = l * w
    A = L * W

    # Calculating intersection:
    temp1 = max(left, Left)
    temp2 = min(right, Right)
    temp3 = max(bottom, Bottom)
    temp4 = min(top, Top)

    i = (temp2 - temp1) * (temp4 - temp3)

    # Calculating IoU
    iou = i / (a + A - i)
    return iou



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Deactivate zero-division exception checking

cdef float _cuboid_iou_1pair(float[:] cub1, float[:] cub2):
    """ Returns intersection-over-union (Jaccard index) of two cuboids.

    Cuboid borders are specified by the given length, width, height and the
    bottom center coordinates (x,y,z).
    This function uses the same approach as the function for rectangles but
    in 3 dimensions.

    Args:
        cub1 ndarray[6]  [x, y, z, length, width, height]
        cub2 ndarray[6]  [x, y, z, length, width, height]

    Returns:
        iou float
    """

    cdef float iou, i
    cdef float x, y, z, l, w, h, X, Y, Z, L, W, H, v, V
    cdef float xb, xt, yb, yt, zb, zt, Xb, Xt, Yb, Yt, Zb, Zt
    cdef float temp1, temp2, temp3, temp4, temp5, temp6

    ## Intersection-Over-Union:
    # Assigning parameters
    x = cub1[0]
    y = cub1[1]
    z = cub1[2]
    l = cub1[3]
    w = cub1[4]
    h = cub1[5]

    X = cub2[0]
    Y = cub2[1]
    Z = cub2[2]
    L = cub2[3]
    W = cub2[4]
    H = cub2[5]

    # Borders of the cuboids (e.g. (bottom in x) = xb, (top in x) = xt)
    xb = x - l/2
    xt = x + l/2
    yb = y - w/2
    yt = y + w/2
    zb = z 
    zt = z + h

    Xb = X - L/2
    Xt = X + L/2
    Yb = Y - W/2
    Yt = Y + W/2
    Zb = Z 
    Zt = Z + H

    # Cases without intersection
    if Xb >= xt or Xt <= xb or Yb >= yt or Yt <= yb or Zb >= zt or Zt <= zb:
        return 0.0

    # Calculating volume
    v = l * w * h
    V = L * W * H

    # Calculating intersection:
    temp1 = max(xb, Xb)
    temp2 = min(xt, Xt)
    temp3 = max(yb, Yb)
    temp4 = min(yt, Yt)
    temp5 = max(zb, Zb)
    temp6 = min(zt, Zt)

    i = (temp2 - temp1) * (temp4 - temp3) * (temp6 - temp5)

    # Calculating IoU
    iou = i / (v + V - i)
    return iou
