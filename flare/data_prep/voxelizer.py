""" Voxelizer for point clouds to voxel clouds

by Stefan Schmohl, 2020 
"""

from globals import *
import numpy as np
import flare.utils.coord_trafo as ct

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},language_level=3)
from . import voxelizer_chelper as chelper




def voxelize(coords, features, labels, vsize, ignore_label=-100):
    """ Sparsely voxelizes point cloud to voxel cloud.
    
    Args:
        coords:     
            Nx3 ndarray with x,y,z per point.
        features:   
            Nxf ndarray with f features per point. Will be averaged over all
            points per voxel. May be an empty ndarray.
        labels:     
            Nxl ndarray with l label fields per point. Will be voted over all
            points per voxel. May be an empty ndarray.
        vsize:      
            Voxelsize in same unit as coords (meters mostly).
        ignore_labels:
            iterable of length l or scalar to be ignored in label vote.
                   
    Returns:
        coords_v:    nx3 discrete voxel coords, where n is the number of voxels.
        features_v:  nxf averaged features in ndarray.
        labels_v:    nxl voted labels in ndarray.
        map:         Nx index array to map each point to its voxel: v = map[p].
        nop:         nx number of points per voxel.
        acc:         lx accuracy per label field.
    """
    
    try:
        # works, if ignore_label is iterable:
        ignore_label = np.asarray(list(ignore_label), dtype=np.int32)
    except:
        # works, if ignore_label is scalar:
        ignore_label = np.zeros(labels.shape[1]) + ignore_label
        ignore_label = ignore_label.astype(np.int32)
        
    
    # Transform point coords to voxel-space:
    tmat = ct.S(1/vsize)
    coords_v, _ = ct.transform(coords, tmat)
    # Point coords in voxel-space -> voxel coords (of points):
    coords_v = np.floor(coords_v.astype(dtype=IDT))

    # Voxel-Filter: remove duplicates => results in all non empty voxels.
    # (reconstruction = rch = rch_sparse[map])
    coords_v, map = np.unique(coords_v, axis=0, return_inverse=True)
    map = map.astype(np.uint64)


    # Features and labels:
    f_v, l_v, nop = chelper.voxelize_fields(features, labels, map, 
                                            coords_v.shape[0], ignore_label)


    # Estimate quality meassure of voxelization:
    acc = []
    if l_v.ndim > 1:   # it might be empty => ndims = 1
        for i in range(l_v.shape[1]):
            l_p = l_v[:,i][map]
            acc_ = np.sum(l_p == labels[:,i]) / len(l_p)
            acc.append(acc_)
            del l_p

    return coords_v, f_v, l_v, map, nop, acc



