"""
Run this file ahead of demo_V3D.py.

Based on the original benchmark files:
- Adds column names and copies files into right directories.
- Splits training in train und val subsets.
"""

from flare.utils import pts
from flare.utils import las

fpath_test  = './data/Vaihingen3D_EVAL_WITH_REF.pts'
fpath_train = './data/Vaihingen3D_Traininig.pts'
val_split_y = 5419293.5


#=#############################################################################

fields = ['x', 'y', 'z', 'intensity', 'return_num', 'num_returns', 'classification']

# Test-Set:
data, _ = pts.read(fpath_test, col_names=fields, IDT='float64')
las.write('./data/raw/V3D__Test/V3D_test.las', data, fields)


# Training-Set:
data, _ = pts.read(fpath_train, col_names=fields, IDT='float64')

val_idx = data[:, 1] < val_split_y

data_train = data[~val_idx, :]
data_valid = data[val_idx, :]

las.write('./data/raw/V3D__Train/V3D_train.las', data_train, fields)
las.write('./data/raw/V3D__Valid/V3D_valid.las', data_valid, fields)
