""" Cython helper functions for the heavy lifting of voxelizer.py

How to use:

    import pyximport
    pyximport.install(setup_args={"include_dirs":np.get_include()},language_level=3)
    import voxelizer_chelper as chelper

    ...
    f_v, l_v, nop = chelper.voxelize_fields(features, labels, map, n)


by Stefan Schmohl, 2020 
"""


cimport cython
import numpy as np
cimport numpy as cnp



def vote(cnp.ndarray[cnp.uint16_t, ndim=1] labels,
         cnp.ndarray[cnp.uint64_t, ndim=1] map, 
         unsigned long n, long ignore_label):
    """ Summarizes point labels per voxel by voting.
    
    Args:
        labels:       Nx1 label vector.
        map:          Nx1 vector holding the voxel index for every point.
        n:            Number of voxels.
        ignore_label: lx1 vector of labels. One value per label channel. Will
                      be ignored, if in a voxel together with other classes.
        
    Returns:
        labels_v:     nxl voted label vector.
    """
    
    assert labels.shape[0] == map.shape[0]

    cdef unsigned long i, v, label_pos 
    cdef cnp.ndarray[cnp.uint16_t, ndim=1] classes
    cdef cnp.ndarray[cnp.uint16_t, ndim=2] counter 
    cdef cnp.ndarray[cnp.uint16_t, ndim=1] label_map
    cdef cnp.ndarray[cnp.uint16_t, ndim=1] label_map_inv
    cdef cnp.ndarray[cnp.uint16_t, ndim=1] labels_v
    
    # Find all the different values (=classes) in the label channel:
    classes = np.unique(labels, return_inverse=False).astype(np.uint16)
    # Buffer to count each class per voxel:
    counter = np.zeros((n, len(classes)), dtype=np.uint16)
    
    # Map that converts label values to a dense 0...number_of_classes range 
    # (saves memory compared to cover the whole original value range.
    label_map = np.zeros(np.max(classes + 1), dtype=np.uint16)
    label_map[classes] = range(len(classes))
    label_map_inv = classes

    for i in range(labels.shape[0]):
        v = map[i]     # voxel index
        label_pos = label_map[labels[i]]
        counter[v, label_pos] += 1

    # Test, to make sure, every voxel has at least one point label:
    assert np.min(np.sum(counter, axis=1)) != 0

    if ignore_label in classes:
        # "Delete" counter column of ignore index.
        # Do not realy delete, because argmax still needs correct indices
        ignore_label_pos = label_map[ignore_label]
        counter[:, ignore_label_pos] = 0
        # Argmax without ignore_index:
        labels_v = np.argmax(counter, axis=1).astype(np.uint16)
        # All voxels with sum=0 (i.e. voxels that only had ignore_index)
        # => ignore_index
        labels_v[np.sum(counter, axis=1) == 0] = ignore_label_pos
    else:
        labels_v = np.argmax(counter, axis=1).astype(np.uint16)
    
    labels_v = label_map_inv[labels_v]
    
    return labels_v



def average(cnp.ndarray[cnp.float32_t, ndim=2] features,
            cnp.ndarray[cnp.uint64_t, ndim=1] map, unsigned long n):
    """ Summarizes point labels per voxel by averaging.
    
    Args:
        features:     Nxf feature vector with f feature channels.
        map:          Nx1 vector holding the voxel index for every point.
        n:            Number of voxels.
        
    Returns:
        features_v:   nxf averaged feature vector.
        nop:          nx1 number of points per voxel.
    """
    
    assert features.shape[0] == map.shape[0]

    cdef unsigned long i, v, nof
    cdef cnp.ndarray[cnp.float32_t, ndim=2] features_v
    cdef cnp.ndarray[cnp.uint16_t, ndim=1] nop
   
    nof        = features.shape[1]
    features_v = np.zeros((n, nof), dtype=np.float32)   
    nop        = np.zeros(n, dtype=np.uint16) 

    for i in range(features.shape[0]):
        v = map[i]     # voxel index
        nop[v] += 1
        for j in range(nof):
            features_v[v, j] += features[i, j]

    for i in range(nof):
        features_v[:, i] = np.divide(features_v[:, i], nop)
    
    return features_v, nop




def voxelize_fields(cnp.ndarray[cnp.float32_t, ndim=2] features,
                    cnp.ndarray[cnp.uint16_t, ndim=2] labels,
                    cnp.ndarray[cnp.uint64_t, ndim=1] map, long n,
                    cnp.ndarray[cnp.int32_t, ndim=1] ignore_label):
              
    """ Summarizes point features and labels per voxel by averaging / voting.
   
    Memory usage and runtime is optimized for the expected application of
    voxelizing ALS point clouds.
    
    Be aware that due to data types, there are some implicit restrictions. 
    For example the number of classes or max points per voxel is limited to
    65535 !! (because using uint16)

    Args:
        feautes:      Nxf feature vector with f feature channels.
        labels:       Nxl label vector with l label channels.
        map:          Nx1 vector holding the voxel index for every point.
        n:            Number of voxels.
        ignore_label: lx1 vector of labels. One value per label channel. Will
                      be ignored, if in a voxel together with other classes.
    
    Returns:
        features_v:   nxf averaged feature vector.
        labels_v:     nxl voted label vector.
        nop:          nx1 number of points per voxel.
    """

    cdef long nolf = labels.shape[1]
    cdef cnp.ndarray[cnp.float32_t, ndim=2] features_v 
    cdef cnp.ndarray[cnp.uint16_t, ndim=2] labels_v 
    cdef cnp.ndarray[cnp.uint16_t, ndim=1] nop 
    
    
    if labels.size != 0:
        labels_v = np.zeros((n, nolf), dtype=np.uint16) 
        for i in range(nolf):
            labels_v[:,i] = vote(labels[:,i], map, n, ignore_label[i])
    else:
        labels_v = np.asarray([[]], dtype=np.uint16)
    
    
    if features.size != 0:
        features_v, nop = average(features, map, n)
    else:
        features_v = np.asarray([])
        nop = np.asarray([])

    return features_v, labels_v, nop









def voxelize_fields_old(cnp.ndarray[cnp.float32_t, ndim=2] features,
                        cnp.ndarray[cnp.uint16_t, ndim=1] labels,
                        cnp.ndarray[cnp.uint64_t, ndim=1] map, long numbOfVoxels):
    """ Old version with only one label channel.
    
    Keep for (runtime) comparison.
    """

    cdef long i, j, v, label_pos
    cdef int nof = features.shape[1]   # number of features
    cdef long n = features.shape[0]      # number of points total
    cdef long m = numbOfVoxels         # number of voxels total


    # Find all different class labels:
    cdef cnp.ndarray[cnp.uint16_t, ndim=1] classes = np.unique(labels, return_inverse=False).astype(np.uint16)
    cdef int nocl = len(classes)

    cdef cnp.ndarray[cnp.float32_t, ndim=2] feat  = np.zeros((m, nof), dtype=np.float32)   # averaged features per voxel
    cdef cnp.ndarray[cnp.uint16_t, ndim=1]  nop   = np.zeros(m, dtype=np.uint16)  # number of points per voxel
    cdef cnp.ndarray[cnp.uint16_t, ndim=2]  nopcl = np.zeros((m, nocl), dtype=np.uint16)  # nop per voxel and class
    cdef cnp.ndarray[cnp.uint16_t, ndim=1]  lab   = np.zeros(m, dtype=np.uint16)  # result labels per voxel

    # Build mapping from true class label to a 0...nocl set. This is to save memory compared to v1 of this script:
    cdef cnp.ndarray[cnp.int32_t, ndim=1] label_map = np.zeros(np.max(classes + 1), dtype=np.int32)
    label_map[classes] = range(nocl)

    for i in range(n):
        v = map[i]     # voxel index
        nop[v] += 1
        label_pos = label_map[labels[i]]
        nopcl[v, label_pos] += 1
        for j in range(nof):
            feat[v, j] += features[i, j]

    for i in range(nof):
        feat[:, i] = np.divide(feat[:, i], nop)

    lab = np.argmax(nopcl, axis=1).astype(np.uint16)
    lab = classes[lab]

    return feat, lab, nop