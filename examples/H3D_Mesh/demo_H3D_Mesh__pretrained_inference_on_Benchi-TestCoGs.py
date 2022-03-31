""" Demo script to infer from two models on Hessigheim3D_Mesh test set.

Execute this script from its directory.
"""

from globals import *

import os
import numpy as np
from flare.utils import pts
from flare.utils import las
from flare import data_import
from flare import data_sampling
from flare import inference__classifier as inference


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device         = 'cuda'
voxel_size     = 0.25
label_fields   = ['classification']
sample_shape   = (32, 32)
sample_fformat = 'pkl'
precision      = 0.01
feat_standard  = False




#=#############################################################################
## Prepare Test data Center-of-Gravity Pointcloud                             #
#                                                                             #
#=#############################################################################

# Add pseudo-labels (const=11, because thats the ignore-class) and add
# pseudo-feature (const=1) and write as .las to designated folder:
fpath_in = './data/COGs_test.txt'
data, _ = pts.read(fpath_in, col_names=['x', 'y', 'z'], header=0, IDT='float64')
labels = np.zeros((len(data), 1)) + 11
featrs = np.ones((len(data), 1))
data = np.hstack((data, featrs, labels))
fields = ['x', 'y', 'z', 'constant_ones', 'classification']
fpath_out = PATH_2_RAW + '/H3D_Mesh__Test_geom/COGs_test.las'
las.write(fpath_out, data, fields, precision=0.001, offset=[513824.0, 5426480.0, 222.579])




#=#############################################################################
## Import Data for Inference                                                  #
#                                                                             #
#=#############################################################################

raw_dir        = 'H3D_Mesh__Test_geom'
angles         = [0]
voxels_dir     = data_import.main(raw_dir, voxel_size, angles, label_fields,
                                  precision=precision,
                                  feature_standardizing=feat_standard)
overlap        = (0.0, 0.0)
data_sampling.main(voxels_dir, voxel_size, sample_shape, overlap, sample_fformat)




#=#############################################################################
## Inference                                                                  #
#                                                                             #
#=#############################################################################

model_names          = ['H3D_Mesh__small_25cm_Geom #0',
                        'H3D_Mesh__small_25cm_Geom #1',
                        'H3D_Mesh__small_25cm_Geom #2',
                        'H3D_Mesh__small_25cm_Geom #3',
                        'H3D_Mesh__small_25cm_Geom #4',
                        'H3D_Mesh__small_25cm_Geom #5',
                        'H3D_Mesh__small_25cm_Geom #6',
                        'H3D_Mesh__small_25cm_Geom #7',
                        'H3D_Mesh__small_25cm_Geom #8',
                        'H3D_Mesh__small_25cm_Geom #9']

test_name            = 'H3D_Mesh__small_25cm_Geom'
epochs               = None
data_config          = 'config_data__H3D_Mesh_Test_geom__25cm_unsampled.json'
path_2_data          = PATH_2_VOXELIZED   # PATH_2_SAMPLES to use sampled data
from_las             = True               # False to use sampled data.
#------------------------------------------------------------------------------
# data_config          = 'config_data__H3D_Mesh_Test_geom__25cm.json'
# path_2_data          = PATH_2_SAMPLES
# from_las             = False
#------------------------------------------------------------------------------
evaluate             = False              # True if GT in test file.
save_only_prediction = True

inference.main(test_name, model_names, epochs, data_config, path_2_data, from_las,
               evaluate=evaluate, device=device, show=False,
               save_only_prediction=save_only_prediction)