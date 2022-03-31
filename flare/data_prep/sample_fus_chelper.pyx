""" Cython helper functions for the heavy lifting of sampler.py

How to use:

    import pyximport
    pyximport.install(setup_args={"include_dirs":np.get_include()},language_level=3)
    import voxelizer_chelper as chelper

    ...
    chelper.tiling_kdTree(x, y, data, overlap, t_size, leafsize)
    

by Stefan Schmohl, 2020 
"""

import math
import warnings
import numpy as np

from scipy.spatial import cKDTree as KDTree

cimport cython
cimport numpy as cnp







def tiling_kdTree_int(cnp.ndarray[cnp.int32_t, ndim=1] x, 
                      cnp.ndarray[cnp.int32_t, ndim=1] y, 
                      cnp.ndarray[cnp.float32_t, ndim=2] data, 
                      overlap, t_size, leafsize=10, offset=(0,0)):
    
    """ Divides input point cloud into horizontal tiles.
    
    Args:
        x:         N vector of x coordinates per point.
        y:         N vector of y coordinates per point.
        data:      Nxf 2d array of point features, usually including the coords.
        overlap:   2 elements iterable with overlap in x and y direction.
        t_size:    2 elements iterable with tile size in x and y direction.
        leafsize:  Parameter for the underlyingly used kdTree. See scipy docu.
        offset:    Used to align tile borders with voxel edges.
    
    Returns:
        sample:    List of data 3d arrays per sample.
        indices:   List of index arrays, which indicates which point are in 
                   each sample relative to the original point cloud.
    """
    
    
    cdef cnp.ndarray[cnp.int32_t, ndim=2] coords = np.vstack((x,y)).transpose()
 
    cdef int y_, x_, 
    cdef float r
    cdef int x_inkrement, y_inkrement

    cdef cnp.ndarray[cnp.int32_t, ndim=1] x__
    cdef cnp.ndarray[cnp.int32_t, ndim=1] y__
    cdef cnp.ndarray[cnp.int64_t, ndim=1] point_indices
    cdef cnp.ndarray[cnp.int32_t, ndim=2] sample_coords
    cdef cnp.ndarray[cnp.float32_t, ndim=2] sample_data
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, cast=True] valid_indices # cast allows boolean
    
    cdef int t_size_x = int(t_size[0] / 2)
    cdef int t_size_y = int(t_size[1] / 2)
    
    
    ## Lay a grid over the horizontal plane:
    # +1 to also include highest value as well.
    x_inkrement = int(t_size[0] * (1-overlap[0]))
    y_inkrement = int(t_size[1] * (1-overlap[1]))
    
    lower_x = int(np.min(x) + t_size_x - offset[0])
    lower_y = int(np.min(y) + t_size_y - offset[1])
    
    upper_x = int(np.max(x) - t_size_x + x_inkrement) + 1
    upper_y = int(np.max(y) - t_size_y + y_inkrement) + 1
    
    x__ = np.arange(lower_x, upper_x, x_inkrement, dtype=np.int32)
    y__ = np.arange(lower_y, upper_y, y_inkrement, dtype=np.int32)

       
    ## Extract samples and trim them to a cube: 
    kdtree =  KDTree(coords, compact_nodes=False, 
                     balanced_tree=False, leafsize=leafsize)
    # +1 for error margin (better get some more. It's cut to size later anyway.
    r = math.sqrt((t_size_x +1) ** 2 + (t_size_y +1) ** 2) 

    samples = []
    indices = []
    p0 =  []

    for x_ in x__:
        for y_ in y__:

            p0 = [x_, y_]         
  
            point_indices = np.asarray(kdtree.query_ball_point(p0, r), dtype=np.int64)

            sample_coords = coords[point_indices, :]
            sample_data = data[point_indices]

            valid_indices = np.asarray(
                            (sample_coords[:, 0] >= p0[0] - t_size_x) & \
                            (sample_coords[:, 0] <  p0[0] + t_size_x) & \
                            (sample_coords[:, 1] >= p0[1] - t_size_y) & \
                            (sample_coords[:, 1] <  p0[1] + t_size_y), dtype=bool)

            if not np.any(valid_indices):
                continue
                
            sample_data = sample_data[valid_indices]
            point_indices = point_indices[valid_indices]

            samples.append(sample_data)
            indices.append(point_indices)

              
    # self test:
    if overlap[0] == 0 and overlap[1] == 0:
        n_points = [s.shape[0] for s in samples]
        if sum(n_points) != data.shape[0]:
            print("    Warning: sum(n_points) != data.shape[0]:")
            print("      sum(n_points) ==", sum(n_points))
            print("      data.shape[0] ==", data.shape[0])


    return samples, indices







def tiling_kdTree(cnp.ndarray[cnp.float32_t, ndim=1] x, 
                  cnp.ndarray[cnp.float32_t, ndim=1] y, 
                  cnp.ndarray[cnp.float32_t, ndim=2] data, 
                  overlap, t_size, leafsize=10, offset=(0,0)):
    
    """ Divides input point cloud into horizontal tiles.
    
    Args:
        x:         N vector of x coordinates per point.
        y:         N vector of y coordinates per point.
        data:      Nxf 2d array of point features, usually including the coords.
        overlap:   2 elements iterable with overlap in x and y direction.
        t_size:    2 elements iterable with tile size in x and y direction.
        leafsize:  Parameter for the underlyingly used kdTree. See scipy docu.
        offset:    Used to align tile borders with voxel edges.
    
    Returns:
        sample:    List of data 3d arrays per sample.
        indices:   List of index arrays, which indicates which point are in 
                   each sample relative to the original point cloud.
    """
    
    
    cdef cnp.ndarray[cnp.float32_t, ndim=2] coords = np.vstack((x,y)).transpose()
 
    cdef float y_, x_, r
    cdef float x_inkrement, y_inkrement
    cdef cnp.ndarray[cnp.float32_t, ndim=1] x__
    cdef cnp.ndarray[cnp.float32_t, ndim=1] y__
    cdef cnp.ndarray[cnp.int64_t, ndim=1] point_indices
    cdef cnp.ndarray[cnp.float32_t, ndim=2] sample_coords
    cdef cnp.ndarray[cnp.float32_t, ndim=2] sample_data
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, cast=True] valid_indices # cast allows boolean
    
    
    ## Lay a grid over the horizontal plane:
    # +.01 to also include highest value as well.
    x_inkrement = t_size[0] * (1-overlap[0])
    y_inkrement = t_size[1] * (1-overlap[1])
    
    lower_x = np.min(x) + t_size[0]/2 - offset[0]
    lower_y = np.min(y) + t_size[1]/2 - offset[1]
    
    upper_x = np.max(x) - t_size[0]/2 + x_inkrement + 0.01
    upper_y = np.max(y) - t_size[1]/2 + y_inkrement + 0.01
    
    x__ = np.arange(lower_x, upper_x, x_inkrement, dtype=np.float32)
    y__ = np.arange(lower_y, upper_y, y_inkrement, dtype=np.float32)

       
    ## Extract samples and trim them to a cube: 
    kdtree =  KDTree(coords, compact_nodes=False, 
                     balanced_tree=False, leafsize=leafsize)
    # +1 for error margin (better get some more. It's cut to size later anyway.
    r = math.sqrt((t_size[0] / 2 +1) ** 2 + (t_size[1] / 2 +1) ** 2) 

    samples = []
    indices = []
    p0 =  []

    for x_ in x__:
        for y_ in y__:

            p0 = [x_, y_]         
  
            point_indices = np.asarray(kdtree.query_ball_point(p0, r), dtype=np.int64)

            sample_coords = coords[point_indices, :]
            sample_data = data[point_indices]

            valid_indices = np.asarray(
                            (sample_coords[:, 0] >= p0[0] - t_size[0] / 2) & \
                            (sample_coords[:, 0] <  p0[0] + t_size[0] / 2) & \
                            (sample_coords[:, 1] >= p0[1] - t_size[1] / 2) & \
                            (sample_coords[:, 1] <  p0[1] + t_size[1] / 2), dtype=bool)

            if not np.any(valid_indices):
                continue
                
            sample_data = sample_data[valid_indices]
            point_indices = point_indices[valid_indices]

            samples.append(sample_data)
            indices.append(point_indices)

              
    # self test:
    if overlap[0] == 0 and overlap[1] == 0:
        n_points = [s.shape[0] for s in samples]
        if sum(n_points) != data.shape[0]:
            print("    Warning: sum(n_points) != data.shape[0]:")
            print("      sum(n_points) ==", sum(n_points))
            print("      data.shape[0] ==", data.shape[0])


    return samples, indices


