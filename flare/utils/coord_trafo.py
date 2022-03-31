"""  Homogeneuos coordinate transformations for 3D cartesian coords

by Stefan Schmohl, 2020
"""

import math
import numpy as np


def transform(coords, tmat, reverse=False):
    """ Applies homogeneous transformation matrix to 3D coordinates

    Args:
        coords (ndarray):
            Nx3 or Nx4 input coordinates, where the forth dim is the
            homogeneuos scale.
        tmat (ndarray):
            Transformation matrix 4x4.
        reverse:
            If True, tmat = tmat^-1.

    Returns:
        coords: Nx3 transformed coordinates.
        tmat:   Transformation matrix actually used (may be inversed to input).
    """

    if reverse:
        tmat = np.linalg.inv(tmat)

    if coords.shape[1] == 3:
        coords = np.hstack((coords, np.ones((coords.shape[0], 1))))

    coords = np.matmul(tmat, coords.transpose()).transpose()
    coords = np.divide(coords[:, 0:-1], coords[:, [-1]])

    return coords, tmat


def T(vector):
    return np.array([[1, 0, 0, vector[0]],
                     [0, 1, 0, vector[1]],
                     [0, 0, 1, vector[2]],
                     [0, 0, 0, 1]])


def R_x(angle):
    angle = math.radians(angle)
    return np.array([[1, 0, 0, 0],
                     [0,  math.cos(angle), -math.sin(angle), 0],
                     [0,  math.sin(angle), math.cos(angle), 0],
                     [0, 0, 0, 1]])


def R_y(angle):
    angle = math.radians(angle)
    return np.array([[math.cos(angle), 0, math.sin(angle), 0],
                     [0, 1, 0, 0],
                     [-math.sin(angle), 0, math.cos(angle), 0],
                     [0, 0, 0, 1]])


def R_z(angle):
    angle = math.radians(angle)
    return np.array([[math.cos(angle), -math.sin(angle), 0, 0],
                     [math.sin(angle), math.cos(angle), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def S(scale):
    s = scale
    return np.array([[s, 0, 0, 0],
                     [0, s, 0, 0],
                     [0, 0, s, 0],
                     [0, 0, 0, 1]])
