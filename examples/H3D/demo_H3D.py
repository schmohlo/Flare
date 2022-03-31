""" Demo script to train two models on Hessigheim and test on their ensemble.


Execute this script from its directory.
"""

from globals import *

import os
from flare import data_import
from flare import data_sampling
from flare import training
from flare import inference__classifier as inference


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device         = 'cuda'
voxel_size     = 0.25
label_fields   = ['classification']
sample_shape   = (32, 32)
sample_fformat = 'pkl'
import_points  = False
precision      = 0.01


n_models             = 3  # better 10
verbose_step         = 10



#=#############################################################################
## Import Data for Training                                                   #
#                                                                             #
#=#############################################################################

raw_dir        = 'H3D__Train'
angles         = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
voxels_dir     = data_import.main(raw_dir, voxel_size, angles, label_fields,
                                  precision=precision, import_points=import_points)
overlap        = (0.3, 0.3)
data_sampling.main(voxels_dir, voxel_size, sample_shape, overlap, sample_fformat, 
                   include_points=import_points)


raw_dir        = 'H3D__Valid'
angles         = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
voxels_dir     = data_import.main(raw_dir, voxel_size, angles, label_fields, 
                                  precision=precision, import_points=import_points)                          
overlap        = (0.0, 0.0)
data_sampling.main(voxels_dir, voxel_size, sample_shape, overlap, sample_fformat, 
                   include_points=import_points)


#=#############################################################################
##  Training                                                                  #
#                                                                             #
#=#############################################################################

model_name           = 'H3D__small_25cm'

config = {'data_train': 'config_data__H3D_Train__25cm.json',
          'data_val'  : 'config_data__H3D_Valid__25cm.json',
          'model'     : 'config_model__H3D__small.json',
          'train'     : 'config_train__H3D.json'}

model_names, _ = training.main(model_name, config, False, n_models, verbose_step, device, show=False)

# OR USE PRE-TRAINED MODELS:
# model_names          = ['H3D__small_25cm_TrainNew_inv #0',
                        # 'H3D__small_25cm_TrainNew_inv #1',
                        # 'H3D__small_25cm_TrainNew_inv #2',
                        # 'H3D__small_25cm_TrainNew_inv #3',
                        # 'H3D__small_25cm_TrainNew_inv #4',
                        # 'H3D__small_25cm_TrainNew_inv #5',
                        # 'H3D__small_25cm_TrainNew_inv #6',
                        # 'H3D__small_25cm_TrainNew_inv #7',
                        # 'H3D__small_25cm_TrainNew_inv #8',
                        # 'H3D__small_25cm_TrainNew_inv #9']



#=#############################################################################
## Import Data for Inference                                                  #
#                                                                             #
#=#############################################################################

raw_dir        = 'H3D__Test'
angles         = [0]
voxels_dir     = data_import.main(raw_dir, voxel_size, angles, [], precision=precision)
overlap        = (0.0, 0.0)
data_sampling.main(voxels_dir, voxel_size, sample_shape, overlap, sample_fformat)



#=#############################################################################
## Inference                                                                  #
#                                                                             #
#=#############################################################################

test_name            = 'DEMO__H3D__small_25m'
data_config          = 'config_data__H3D_Test__25cm_unsampled.json'
path_2_data          = PATH_2_VOXELIZED   # PATH_2_SAMPLES to use sampled data
from_las             = True               # False to use sampled data.
#------------------------------------------------------------------------------
# data_config          = 'config_data__H3D_Test__25cm.json'
# path_2_data          = PATH_2_SAMPLES
# from_las             = False
#------------------------------------------------------------------------------
evaluate             = False   # False because no real GT.

inference.main(test_name, model_names, None, data_config, path_2_data, from_las,
               evaluate=evaluate, device=device, show=False)