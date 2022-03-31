""" Script for sampling point / voxel cloud pairs into smaller tiles

The file structure looks like the following:

    PATH_2_VOXELIZED/
        Dataset1__vs050__a0_120_240/
            Dataset1.json
            pointcloud1_vs050__a0_float32.las
            pointcloud1_vs050__a0_map.pkl
            pointcloud1_vs050__a0_vox_float32.las
            pointcloud2_vs050__a0_float32.las
            pointcloud2_vs050__a0_map.pkl
            pointcloud2_vs050__a0_vox_float32.las
            pointcloud1_vs050__a120_float32.las
            ...

    PATH_2_SAMPLES
        Dataset1__vs050__a0_60_120_180_240__32x32__o01_01
            Dataset1.json
            pointcloud1__vs050__a0__32x32__o01_01
            pointcloud2__vs050__a0__32x32__o01_01
            pointcloud1__vs050__a120__32x32__o01_01
            ...
    or
    PATH_2_SAMPLES
        Dataset1__vs050__a0_60_120_180_240__32x32__o01_01
            Dataset1.json
            pointcloud1__vs050__a0__32x32__o01_01__s0000_map.pkl
            pointcloud1__vs050__a0__32x32__o01_01__s0000_points_float32.las
            pointcloud1__vs050__a0__32x32__o01_01__s0000_voxels_float32.las
            pointcloud1__vs050__a0__32x32__o01_01__s0001_map.pkl
            ...


by Stefan Schmohl, 2020
"""

from globals import *

import os
import shutil
import numpy as np
import _pickle as cPickle

from scipy.spatial import cKDTree

from flare.utils import makedirs_save
from flare.utils import Timer
from flare.utils import las
from flare.utils import PointCloud
from flare.data_prep.sample_fus import samples_dir
from flare.data_prep.sample_fus import corefilename
from flare.data_prep.sample_fus import read_pc_from_file
from flare.data_prep.sample_fus import tiling
from flare.data_prep.sample_fus import tiling_int



def main(voxels_dir,
         voxel_size,
         sample_shape=(32, 32),
         overlap=(0.0, 0.0),
         sample_fformat='pkl',
         leafsize=10,
         offset_margin=0.0,
         include_trees=False,
         tree_margin=0.0,
         include_points=True):
    """ Sampling point / voxel cloud pairs into smaller tiles (+ tree labels)


    For more details, see module docstring.

    Args:
        voxels_dir:         Direcotry to voxelezied (=imported) input data
                            inside PATH_2_VOXELIZED.
        voxel_size:         Just as info, not a parameter! (still important)
        sample_shape:       Sample shape (x,y) in meters.
        overlap:            Overlap (x, y) between samples, 0..1.
        sample_fformat.     File format to save samples in. Possible values are
                            'pkl' or 'las' ('las' only supposed for debugging).
        leafsize:           Optimize runtime for kdTree for sampling.
                            Set higher for bigger point clouds (e.g. 100).
        offset_margin:      For numeric edge cases of point lying on sample
                            edges. Adjust if warnings regarding to large
                            samples. (may help, may not).
        include_trees:      True to include tree labels.
        tree_margin:        Margin of tree labels to include in sample. In
                            'meters'. E.g. =0.5 includes also tree labels,
                            which are at maximum 0.5 meters outside of sample.
        include_points      If True, loads also the point cloud (and map) and
                            samples based on the point cloud. The voxels are
                            indirectly sampled via the map. (Old, default).
                            If False, load only voxels and sample based on them.
    """



    ## Create output path
    out_dir = samples_dir(voxels_dir, sample_shape, overlap)
    if tree_margin == 'enclosed':
        out_dir += '__enclosed'
    out_path = os.path.join(PATH_2_SAMPLES, out_dir)
    if not makedirs_save(out_path, "save sampled data"):
        return 0


    ## Collect all files from folder with the voxelized point cloud:
    in_path = os.path.join(PATH_2_VOXELIZED, voxels_dir)
    files = os.scandir(in_path)
    f_names = [f.name for f in files if f.is_file()]


    ## Sort f_names in 3 buckets for each type:
    voxel_files = [f for f in f_names if VOX_SUFFIX in f]
    voxel_files.sort()

    if include_points:
        point_files = [f for f in f_names if PNT_SUFFIX in f]
        map_files   = [f for f in f_names if MAP_SUFFIX in f]
        point_files.sort()
        map_files.sort()
        assert len(voxel_files) == len(point_files) == len(map_files), \
            ("Dismatch of input files!")

    if include_trees:
        tree_files  = [f for f in f_names if TRS_SUFFIX in f]
        tree_files.sort()
        assert len(voxel_files) == len(tree_files), ("Dismatch of input files!")


    print('\nSampling from ', in_path,
          '\n         to   ', out_path, ' :')
    global_timer = Timer()


    ## Copy json files, which include meta data:
    files = os.scandir(in_path)
    json_files = [f.name for f in files if os.path.splitext(f)[-1] == '.json']
    for f in json_files:
        source = os.path.join(in_path, f)
        destin = os.path.join(out_path, f)
        shutil.copyfile(source, destin)


    ## Check if overlap satisfies sample size
    # The bounds of samples should coincide with voxel bounds
    # "Number" of voxels per sample:
    voxel_num = np.asarray(sample_shape) / voxel_size
    # Find closest overlap that satisfies sample width and length
    overlapFit = tuple((np.round(voxel_num * overlap) / voxel_num))
    if not overlapFit == overlap:
    	print('\n  Overlap was resized to ({:.3f} x {:.3f}) to satisfy sample size'.format(
        	overlapFit[0],overlapFit[1]))
    	overlap = overlapFit


    ## Go through all input file tripplets:
    for i,_ in enumerate(voxel_files):


        ## Make sure that voxel, point and map file match:
        cfn_v = corefilename(voxel_files[i])
        if include_points:
            cfn_p = corefilename(point_files[i])
            cfn_m = corefilename(map_files[i])
            assert cfn_v == cfn_p == cfn_m, ("Dismatch of input files!")
        if include_trees:
            cfn_t = corefilename(tree_files[i])
            assert cfn_v == cfn_t, ("Dismatch of input files!")

        print('\n  Input file: ', cfn_v, ' ...')
        file_timer = Timer()


        ## Read input file group:
        vc = read_pc_from_file(os.path.join(in_path, voxel_files[i]))
        if include_points:
            pc = read_pc_from_file(os.path.join(in_path, point_files[i]))
            mapfile = open(os.path.join(in_path, map_files[i]), 'rb')
            map = cPickle.load(mapfile)
            mapfile.close()
        if include_trees:
            tr = read_pc_from_file(os.path.join(in_path, tree_files[i]))
            tr_kdTree = cKDTree(tr.get_xyz()[:,0:2],
                                compact_nodes=False,
                                balanced_tree=False)


        ## Sampling, first just only the current point cloud:
        # Convert pc coords so the sample grid alligns with voxel borders.
        # integern => eindeutige koords ohne float precision problems.
        x = np.squeeze(vc.get_fdata(['x'])).astype(np.int32)
        y = np.squeeze(vc.get_fdata(['y'])).astype(np.int32)
        data = vc.data
        if include_points:
            x = np.floor(np.squeeze(pc.get_fdata(['x'])) / voxel_size).astype(np.int32)
            y = np.floor(np.squeeze(pc.get_fdata(['y'])) / voxel_size).astype(np.int32)
            data = pc.data

        sample_shape_v = (np.asarray(sample_shape) / voxel_size).astype(np.int32)

        samples_data, indices = tiling_int(x, y, data, overlap,
                                           sample_shape_v,
                                           leafsize)


        ## Sampling voxels from indices and Saving to file:
        samples = []
        # Sum counters for quality control. Introduced after to large voxels
        # samples bug.
        sum_sample_points = 0
        sum_sample_voxels = 0
        sum_sample_trees  = 0

        for i, sample_data in enumerate(samples_data):
            sample_p_data = []
            sample_t_data = []

            if include_points:
                sample_p_data = sample_data

                # Sample voxel cloud based on the sampling indices and the map.
                # Sub_map includes the global voxel indices of all sample points
                sub_map = map[indices[i]]

                # Sample_voxel_indices are the global voxel indices of the sample
                # voxels. Reverse holds for each sample point the position of its
                # sample voxel.
                sample_voxel_indices, reverse = np.unique(sub_map, return_inverse=True)
                sample_v_data = vc.data[sample_voxel_indices]
                sample_map = reverse
            else:
                sample_v_data = sample_data


            ## Search all trees inside the sample:
            if include_trees:
                center = np.mean(sample_v_data[:, 0:2], 0) * voxel_size
                min_   = np.min(sample_v_data[:, 0:2], 0)  * voxel_size
                max_   = np.max(sample_v_data[:, 0:2], 0)  * voxel_size
                r = np.sqrt(np.sum(np.power(max_ - min_, 2))) / 2
                if tree_margin != 'enclosed':
                    r += tree_margin
                sample_tree_indices = tr_kdTree.query_ball_point(center, r)
                sample_t_data = tr.data[sample_tree_indices]
                if sample_t_data.size != 0:
                    tree_x = sample_t_data[:, 0]
                    tree_y = sample_t_data[:, 1]
                    tree_r = sample_t_data[:, 3]
                    if tree_margin != 'enclosed':
                        tree_r = np.zeros_like(tree_r)
                    ind_outside = np.logical_or.reduce((tree_x - tree_r < min_[0],
                                                        tree_y - tree_r < min_[1],
                                                        tree_x + tree_r > max_[0],
                                                        tree_y + tree_r > max_[1]))
                    sample_t_data = sample_t_data[~ind_outside]


            ## Quality controll:
            sum_sample_points += len(sample_p_data)
            sum_sample_voxels += len(sample_v_data)
            sum_sample_trees  += len(sample_t_data)

            sample_shape_v = (np.max(sample_v_data[:, 0]) -
                              np.min(sample_v_data[:, 0]) + 1,
                              np.max(sample_v_data[:, 1]) -
                              np.min(sample_v_data[:, 1]) + 1)
            if (sample_shape_v[0] - sample_shape[0] / voxel_size > 1e-7 or
                sample_shape_v[1] - sample_shape[1] / voxel_size > 1e-7):
                print('    Warning: shape of sample #{:d} to large: {:.0f}x{:.0f}'.format(
                      i,
                      sample_shape_v[0],
                      sample_shape_v[1]))


            ## Save output to file(s):
            sample_name = samples_dir(cfn_v, sample_shape, overlap) + '__s{:04d}'.format(i)

            if sample_fformat == 'las':
                out_fpath_raw = os.path.join(out_path, sample_name)

                # Voxel cloud:
                fpath = out_fpath_raw + '_' + VOX_SUFFIX + '_' + IDT + '.las'
                las.write_to_float(fpath, sample_v_data, vc.fields,
                                   vc.import_precision, vc.import_offset,
                                   vc.import_reduced, IDT=IDT)

                if include_points:
                    # Point cloud:
                    fpath = out_fpath_raw + '_' + PNT_SUFFIX + '_' + IDT + '.las'
                    las.write_to_float(fpath, sample_p_data, pc.fields,
                                       pc.import_precision, pc.import_offset,
                                       pc.import_reduced, IDT=IDT)

                    # Map:
                    fpath = out_fpath_raw + '_' + MAP_SUFFIX + '.pkl'
                    pckl_file = open(fpath, 'wb')
                    cPickle.dump(sample_map, pckl_file)
                    pckl_file.close()

                # Tree set:
                if len(sample_t_data) > 0:  # in case of no tree in sample
                    fpath = out_fpath_raw + '_' + TRS_SUFFIX + '_' + IDT + '.las'
                    las.write_to_float(fpath, sample_t_data, tr.fields,
                                       tr.import_precision, tr.import_offset,
                                       tr.import_reduced)


            elif sample_fformat == 'pkl':
                # TODO: das allgemein machen und in pointcloud eine to_las_float()
                sample = {'name': sample_name}

                sample_vc                  = PointCloud(sample_v_data, vc.fields)
                sample_vc.import_offset    = vc.import_offset
                sample_vc.import_precision = vc.import_precision
                sample_vc.import_reduced   = vc.import_reduced
                sample['voxel_cloud']      = sample_vc

                if include_points:
                    sample_pc                  = PointCloud(sample_p_data, pc.fields)
                    sample_pc.import_offset    = pc.import_offset
                    sample_pc.import_precision = pc.import_precision
                    sample_pc.import_reduced   = pc.import_reduced
                    sample['point_cloud']      = sample_pc
                    sample['map']              = sample_map

                if include_trees:
                    sample_tr                  = PointCloud(sample_t_data, tr.fields)
                    sample_tr.import_offset    = tr.import_offset
                    sample_tr.import_precision = tr.import_precision
                    sample_tr.import_reduced   = tr.import_reduced
                    sample['tree_set']         = sample_tr

                samples.append(sample)


        if sample_fformat == 'pkl':
            out_fname = samples_dir(cfn_v, sample_shape, overlap)
            fpath = os.path.join(out_path, out_fname + '.pkl')
            pckl_file = open(fpath, 'wb')
            cPickle.dump(samples, pckl_file)
            pckl_file.close()

        print('    Sum samples: {:,d}'.format(len(samples_data)))
        print('    Sum input   voxels: {:,d}'.format(len(vc)))
        print('    Sum sampled voxels: {:,d}'.format(sum_sample_voxels))
        if include_points:
            print('    Sum input   points: {:,d}'.format(len(pc)))
            print('    Sum sampled points: {:,d}'.format(sum_sample_points))
        if include_trees:
            print('    Sum input   trees : {:,d}'.format(len(tr)))
            print('    Sum sampled trees : {:,d}'.format(sum_sample_trees))

        print('    Time: {:.2f}s'.format(file_timer.restart()))

    print('\n  Total time: ', Timer.string(global_timer.time()), '\n')
    return 1
