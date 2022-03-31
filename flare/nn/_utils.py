""" Utility function for machine learning stuff

by Stefan Schmohl, Johannes Ernst 2021
"""

import math
import numpy as np
import torch


###############################################################################
# Tree-Stuff                                                                  #
#                                                                             #
# - Detections -> Regression targets transform and back                       #
# (- Circle IoU, now in flare.utils)                                          #
###############################################################################



# def bcylinder_transform_inv(deltas, anchors):
    # """ Inverse transform of residual encodings into bounding circles.
    # See bcircles_transform() for details.
    # deltas: [[dx, dy, dz, dr, dh]]
    # anchors: [[x, y, z, r, h]] <- x,y from locations
    # """

    # x = deltas[:, 0] * anchors[:, 3] + anchors[:, 0]
    # y = deltas[:, 1] * anchors[:, 3] + anchors[:, 1]
    # z = deltas[:, 2] * anchors[:, 4] + anchors[:, 2]

    # r = deltas[:, 3] * anchors[:, 3] + anchors[:, 3]
    # h = deltas[:, 4] * anchors[:, 4] + anchors[:, 4]

    # circles = torch.stack((x, y, z, r, h), 1)
    # return circles



# def bcylinder_transform(circles, anchors):
    # """ Transform bounding circles into residual encoding:
    # Faster-RCNN / VoxelNet-style.
    # circles: [[x, y, z, r, h]]
    # anchors: [[x, y, z, r, h]] <- x,y from locations
    # """

    # d_x = (circles[:, 0] - anchors[:, 0]) / anchors[:, 3]  # norm by radius
    # d_y = (circles[:, 1] - anchors[:, 1]) / anchors[:, 3]  # norm by radius
    # d_z = (circles[:, 2] - anchors[:, 2]) / anchors[:, 4]  # norm by height

    # d_r = (circles[:, 3] - anchors[:, 3]) / anchors[:, 3]
    # d_h = (circles[:, 4] - anchors[:, 4]) / anchors[:, 4]

    # delta = torch.stack((d_x, d_y, d_z, d_r, d_h), 1)
    # return delta



def bdetection_transform_inv(deltas, anchors, geometry):
    """ Inverse transform of residual encodings into bounding geometry.

    The supported geometries are:
    Circles     (x, y, radius)
    Rectangles  (x, y, length, width)
    Cylinders   (x, y, z, radius, height)
    Cuboids     (x, y, z, length, width, height)

    deltas:  [[dx, dy, ...]] (e.g. cylinder [[dx, dy, dz, dr, dh]])
    anchors: [[x, y,   ...]] <- x,y from locations
    """
    if geometry == "circles":
        x = deltas[:, 0] * anchors[:, 2] + anchors[:, 0]
        y = deltas[:, 1] * anchors[:, 2] + anchors[:, 1]
        r = torch.exp(deltas[:, 2]) * anchors[:, 2]

        detections = torch.stack((x, y, r), 1)

    elif geometry == "rectangles":
        x = deltas[:, 0] * anchors[:, 2] + anchors[:, 0]
        y = deltas[:, 1] * anchors[:, 3] + anchors[:, 1]
        l = torch.exp(deltas[:, 2]) * anchors[:, 2]
        w = torch.exp(deltas[:, 3]) * anchors[:, 3]

        detections = torch.stack((x, y, l, w), 1)
        
    elif geometry == "cylinders":
        x = deltas[:, 0] * anchors[:, 3] + anchors[:, 0]
        y = deltas[:, 1] * anchors[:, 3] + anchors[:, 1]
        z = deltas[:, 2] * anchors[:, 4] + anchors[:, 2]
        r = torch.exp(deltas[:, 3]) * anchors[:, 3]
        h = torch.exp(deltas[:, 4]) * anchors[:, 4]

        detections = torch.stack((x, y, z, r, h), 1)

    elif geometry == "cuboids":
        x = deltas[:, 0] * anchors[:, 3] + anchors[:, 0]
        y = deltas[:, 1] * anchors[:, 4] + anchors[:, 1]
        z = deltas[:, 2] * anchors[:, 5] + anchors[:, 2]
        l = torch.exp(deltas[:, 3]) * anchors[:, 3]
        w = torch.exp(deltas[:, 4]) * anchors[:, 4]
        h = torch.exp(deltas[:, 5]) * anchors[:, 5]

        detections = torch.stack((x, y, z, l, w, h), 1)

    return detections

def bdetection_transform(detections, anchors, geometry):
    """ Transform bounding geometry into residual encoding:
    Faster-RCNN / VoxelNet-style.

    The supported geometries are:
    Circles     (x, y, radius)
    Rectangles  (x, y, length, width)
    Cylinders   (x, y, z, radius, height)
    Cuboids     (x, y, z, length, width, height)

    detections: [[x, y, ...]] (e.g. cylinder [[x, y, z, r, h]])
    anchors:    [[x, y, ...]] <- x,y from locations
    """
    if geometry == "circles":
        d_x = (detections[:, 0] - anchors[:, 0]) / anchors[:, 2]  # norm by radius
        d_y = (detections[:, 1] - anchors[:, 1]) / anchors[:, 2]  # norm by radius
        d_r = torch.log(detections[:, 2] / anchors[:, 2])

        delta = torch.stack((d_x, d_y, d_r), 1)

    elif geometry == "rectangles":
        d_x = (detections[:, 0] - anchors[:, 0]) / anchors[:, 2]  # norm by length
        d_y = (detections[:, 1] - anchors[:, 1]) / anchors[:, 3]  # norm by width
        d_l = torch.log(detections[:, 2] / anchors[:, 2])
        d_w = torch.log(detections[:, 3] / anchors[:, 3])

        delta = torch.stack((d_x, d_y, d_l, d_w), 1)

    elif geometry == "cylinders":
        d_x = (detections[:, 0] - anchors[:, 0]) / anchors[:, 3]  # norm by radius
        d_y = (detections[:, 1] - anchors[:, 1]) / anchors[:, 3]  # norm by radius
        d_z = (detections[:, 2] - anchors[:, 2]) / anchors[:, 4]  # norm by height
        d_r = torch.log(detections[:, 3] / anchors[:, 3])
        d_h = torch.log(detections[:, 4] / anchors[:, 4])

        delta = torch.stack((d_x, d_y, d_z, d_r, d_h), 1)

    elif geometry == "cuboids":
        d_x = (detections[:, 0] - anchors[:, 0]) / anchors[:, 3]  # norm by length
        d_y = (detections[:, 1] - anchors[:, 1]) / anchors[:, 4]  # norm by width
        d_z = (detections[:, 2] - anchors[:, 2]) / anchors[:, 5]  # norm by height
        d_l = torch.log(detections[:, 3] / anchors[:, 3])
        d_w = torch.log(detections[:, 4] / anchors[:, 4])
        d_h = torch.log(detections[:, 5] / anchors[:, 5])
        
        delta = torch.stack((d_x, d_y, d_z, d_l, d_w, d_h), 1)

    return delta



def circle_area(radia):
    """
    Computes the area of a set of circles by their radius.

    Args:
        radia (Tensor[N, 1]): Circle radia for which the area will be computed.
    Returns:
        areas (Tensor[N]):    Area for each circle.
    """
    return torch.pow(radia, 2) * math.pi



###############################################################################
# Misc                                                                        #
#                                                                             #
###############################################################################



def project_z_torch(coords, features, default=0):
    ''' Projects point cloud into horizontal 2D grid.

    Resolution is one coordinate unit.

    Warning: Undefined behaviour for coord duplicates on parallel processes
             (like on cuda!)
             Solution for this would be to sort by z and unique(x,y), but
             in PyTorch unique currently does not return unique indices!

    Args:
        coords:     Nx3 pytorch tensor x,y,z.
        features:   Nxf pytorch tensor.
        default:    To fill empty locations.
    Returns:
        value-grid
        height-grid
    '''

    h_init = float("NaN")

    min_x = torch.min(coords[:, 0].long())
    min_y = torch.min(coords[:, 1].long())

    shape = (torch.max(coords[:, 0]).long() - min_x + 1,
             torch.max(coords[:, 1]).long() - min_y + 1)

    x = coords[:, 0].long() - min_x
    y = coords[:, 1].long() - min_y  # make sure, no negative indices
    z = coords[:, 2].long()

    values = features.new_full(shape + (features.shape[1],), default)
    height = torch.full(shape, h_init)

    values[x, y, :] = features
    height[x, y] = z.type(height.dtype)

    return values, height



def project_z_numpy(coords, features, mode='max', default=0, shape=None):
    ''' Projects point cloud into horizontal 2D grid.

    Resolution is one coordinate unit.

    In the numpy version, duplicate labels are allowed.

    Args:
        coords:     Nx3 numpy array x,y,z.
        features:   Nxf pytorch tensor.
        mode:       'max' or 'min'. 'max' = take value of highest point over
                    location.
        default:    To fill empty locations.
        shape:      List or tuple for shape in x and y.
    Returns:
        value-grid
        height-grid
    '''

    if mode not in ['min', 'max']:
        raise ValueError ('"mode" neither "min" nor "max"')

    if mode == 'min':
        h_init = float("Inf")
    elif mode == 'max':
        h_init = -float("Inf")


    min_x = np.min(coords[:, 0]).astype('long')
    min_y = np.min(coords[:, 1]).astype('long')

    x = coords[:, 0].astype('long') - min_x
    y = coords[:, 1].astype('long') - min_y  # make sure, no negative indices
    z = coords[:, 2].astype('long')
    f = features.copy()

    if shape is None:
        shape = (np.max(coords[:, 0]).astype('long') - min_x + 1,
                 np.max(coords[:, 1]).astype('long') - min_y + 1)
        ind = torch.ones(x.shape, dtype=bool)
    else:
        shape = tuple(shape)
        # Fight my bug in sample with might result in slighty to large inputs.
        # This will result in some data loss and is a quick and dirty fix!
        # TODO!
        ind = x < shape[0]
        ind *= y < shape[1]
        x = x[ind]
        y = y[ind]
        z = z[ind]
        f = f[ind]


    # Duplicate indices result in undefined behaviour. Therefore, first sort by
    # z and then remove duplicates (will keep first! occurances)
    indices = np.argsort(z)
    if mode == 'max':
        indices = np.flip(indices)

    x = x[indices]
    y = y[indices]
    z = z[indices]
    f = f[indices]
    _, unique_indices = np.unique(np.vstack((x, y)), return_index=True, axis=1)

    x = x[unique_indices]
    y = y[unique_indices]
    z = z[unique_indices]
    f = f[unique_indices]

    values = np.full(shape + (features.shape[1],), default, dtype=features.dtype)
    height = np.full(shape, h_init)

    values[x, y, :] = f
    height[x, y] = z.astype(height.dtype)

    height[height==h_init] = float("NaN")

    return values, height



# TODO: hier lassen?
# TODO: Doku
def to_one_hot(labels, nClasses):
    n, h, w = labels.size()
    one_hot = torch.zeros(n, nClasses, h, w).scatter_(1, labels.view(n, 1, h, w), 1)
    return one_hot
