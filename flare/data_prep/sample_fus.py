"""
by Stefan Schmohl, 2020
"""

from globals import *

import os
import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},language_level=3)

from flare.utils import las
from flare.utils import pts
from flare.utils import PointCloud
from . import sample_fus_chelper as chelper



def samples_dir(voxels_dir, sample_shape, overlap):
    shape_str = 'x'.join(str(a) for a in sample_shape)
    overlap_str = '_'.join('{:02d}'.format(int(o*100)) for o in overlap)
    fs = '{}__{}__o{}'
    return fs.format(voxels_dir, shape_str, overlap_str)



def corefilename(f_name):
    fn = os.path.splitext(f_name)[0]
    fn = fn.replace('_' + VOX_SUFFIX, '')
    fn = fn.replace('_' + PNT_SUFFIX, '')
    fn = fn.replace('_' + MAP_SUFFIX, '')
    fn = fn.replace('_' + TRS_SUFFIX, '')
    fn = fn.replace('_' + IDT, '')
    fn = fn.replace('_' + 'float32', '')
    fn = fn.replace('_' + 'float64', '')
    return fn



def read_pc_from_file(f_path):
    
    f_name = os.path.basename(f_path)
    reduce = True
    
    if 'float32' in f_name:
        data, fields, precision, offset = las.read_from_float(f_path,
            reduced=reduce, dtype='float32')
            
    elif 'float64' in f_name:
        data, fields, precision, offset = las.read_from_float(f_path,
            reduced=reduce, dtype='float64')
            
    else:
        data, fields, precision, offset = las.read(f_path, reduced=reduce, IDT=IDT)

    pc = PointCloud(data, fields)
    pc.import_offset = offset
    pc.import_precision = precision
    pc.import_reduced = reduce

    return pc



def tiling(x, y, data, overlap, t_size, leafsize=10, offset=(0, 0)):
    """ Divides input point cloud into horizontal tiles.
    
    Just a wrapper for the cython function.
    """
    # Only ca. 25% performance gain via cython, but still...
    return chelper.tiling_kdTree(x, y, data, overlap, t_size, leafsize, offset)



def tiling_int(x, y, data, overlap, t_size, leafsize=10, offset=(0, 0)):
    """ Divides input point cloud into horizontal tiles.
    
    Just a wrapper for the cython function.
    """
    # Only ca. 25% performance gain via cython, but still...
    return chelper.tiling_kdTree_int(x, y, data, overlap, t_size, leafsize, offset)
