""" Includes functions used in import_script.py

by Stefan Schmohl, 2020
"""

from globals import *

from flare.utils import las
from flare.utils import pts
from flare.utils import PointCloud



def voxelized_dir(raw_dir, v_size, angles):
    angles_str = '_'.join(str(a) for a in angles)
    fs = '{}__vs{:03d}__a{}'
    return fs.format(raw_dir, int(v_size*100), angles_str)



def read_pc_from_file(f_path):

    ext = f_path.split('.')[-1]
    reduce = True
    
    if ext.lower() in ['pts']:
        data, fields, offset = pts.read(f_path, reduced=reduce, IDT=IDT)
        precision = None
        pass
    elif ext.lower() in ['las', 'laz']:
        data, fields, precision, offset = las.read(f_path, reduced=reduce, IDT=IDT)
        pass
    else:
        return None # ignore

    pc = PointCloud(data, fields)
    pc.import_offset = offset
    pc.import_precision = precision
    pc.import_reduced = reduce

    return pc


