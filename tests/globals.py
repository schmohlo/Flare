""" Global variables defining the project structure etc. """



## Internal datatype for point cloud data arrays.
IDT = 'float32'  # or 'float64' (not tested nor recomended!).  



## Project Structure
# All paths are relative to execution directory if not stated otherwise.

PATH_2_CONFIGS     = '../configs/'
PATH_2_DATA        = '../data/'
PATH_2_MODELS      = '../models/'
PATH_2_PREDICTIONS = '../predictions/'
# PATH_2_MODELS      = '/data/Schmohl/flare/models/'
# PATH_2_DATA        = '/data/Schmohl/flare/data/'

PATH_2_RAW         = PATH_2_DATA + 'raw/'
PATH_2_VOXELIZED   = PATH_2_DATA + 'voxelized/'
PATH_2_SAMPLES     = PATH_2_DATA + 'sampled/'

DIR_2_CHECKPOINTS  = 'checkpoints/'  # 'PATH_2_MODELS/modelname/DIR_2_CHECKPOINTS'


## Internal files:
VOX_SUFFIX  = 'voxels'
PNT_SUFFIX  = 'points'
MAP_SUFFIX  = 'map'