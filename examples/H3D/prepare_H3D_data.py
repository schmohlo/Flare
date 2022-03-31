"""
Run this file ahead of demo_H3D.py.

Based on the original benchmark files:
- adds pseudo ground truth to test set.
- moves files into ./data/raw/
"""

from globals import *
import os
import shutil
import numpy as np
from flare.utils import pts
from flare.utils import las


fpath_test  = './data/PC_Mar18_test.las'
fpath_train = './data/PC_Mar18_train.las'
fpath_valid = './data/PC_Mar18_val.las'



#=#############################################################################

# Test-Set:
print('Test file...')
data, fields, precision, offset = las.read(fpath_test, IDT='float64')
labels = np.ones((len(data), 1))  # dont use only zeros, wouldn't be read later.
data = np.hstack((data, labels))
fields = fields + ['classification']
breakpoint()
las.write(PATH_2_RAW + '/H3D__Test/H3D_test.las', data, fields, precision, offset)


# Training-Set:
print('Train file...')
shutil.copyfile(fpath_train, PATH_2_RAW + '/H3D__Train/H3D_train.las')
print('Valid file...')
shutil.copyfile(fpath_valid, PATH_2_RAW + '/H3D__Valid/H3D_valid.las')
