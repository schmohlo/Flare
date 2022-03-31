""" Script for importing raw point cloud data into project pipeline

This script will, for all point cloud files (.las, .pts) from a specified
directory inside PATH_2_RAW:
    - read file to a point cloud object.
    - rotate for augmentation (must be done before voxelization)
    - VOXELIZE
    - store point cloud and voxel cloud for internal usage in custom las files
      with IDT precision (usually float32). Additionaly store map (lookup-table
      from point cloud to voxel cloud) as a pickle file.

The file structure looks like the following:

    PATH_2_RAW/
        Dataset1/
            Dataset1.json
            pointcloud1.las
           (pointcloud1__trees.txt)
            pointcloud2.las
           (pointcloud2__trees.txt)
            ...
    PATH_2_VOXELIZED/
        Dataset1__vs050__a0_120_240/
            Dataset1.json
            pointcloud1_vs050__a000_map.pkl
            pointcloud1_vs050__a000_points_float32.las
            pointcloud1_vs050__a000_voxels_float32.las
           (pointcloud1_vs050__a000_trees_float32.las)
            pointcloud2_vs050__a000_map.pkl
            pointcloud2_vs050__a000_points_float32.las
            pointcloud2_vs050__a000_voxels_float32.las
           (pointcloud2_vs050__a000_trees_float32.las)
            pointcloud1_vs050__a120_points_float32.las
            ...


by Stefan Schmohl, 2020
"""

from globals import *

import os
import shutil
import numpy as np
import _pickle as cPickle

import flare.utils as utils
import flare.utils.coord_trafo as trafo
from flare.utils import makedirs_save
from flare.utils import Timer
from flare.utils import las
from flare.utils import pts
from flare.utils import PointCloud
from flare.utils import print_dict
from flare.data_prep import voxelizer as voxer
from flare.data_prep.import_fus import read_pc_from_file
from flare.data_prep.import_fus import voxelized_dir



def main(raw_dir, v_size, angles, label_fields=['classification'],
         ignore_label=-100, precision=0.01, include_trees=False,
         import_points=True, feature_standardizing=False,
         global_vox_coords=True):
    """ Import ALS point cloud (+ tree labels) into pipeline.

    For more details, see module docstring.

    Args:
        raw_dir:        Directory inside PATH_2_RAW, includes input files.
        v_size:         Voxel size in meters.
        angles:         List of angles for rotation augmentation in degree.
        label_fields:   List of point cloud scalar fields which will by merged
                        by voting during voxelization. Rest will be averaged.s
        ignore_label:   Vale to ignore during voting. Single value or one for
                        each label_field.
        precision:      Floating point precision of internal storage, relative
                        to coordinate Unit. E.g. 0.01 for centimeter, if input
                        is in meters. None for function default or original
                        input  precision (only for .las input files)
        include_trees:  True to import also tree labels.
        import_points:  True to NOT export the rotated point cloud.
        feature_standardizing:  True standardize distribution of each point
                                feature.
    """

    ## Read raw point cloud files
    in_path = os.path.join(PATH_2_RAW, raw_dir)
    files = os.scandir(in_path)
    f_paths = [f.path for f in files if f.is_file()]


    ## Create output path
    out_dir = voxelized_dir(raw_dir, v_size, angles)
    out_path = os.path.join(PATH_2_VOXELIZED, out_dir)
    if not makedirs_save(out_path, "save sampled data"):
        return out_dir


    print('\nImporting from ', in_path,
          '\n          to   ', out_path, ' :')
    global_timer = Timer()


    ## Copy json files, which include meta data:
    files = os.scandir(in_path)
    json_files = [f.name for f in files if os.path.splitext(f)[-1] == '.json']
    for f in json_files:
        source = os.path.join(in_path, f)
        destin = os.path.join(out_path, f)
        shutil.copyfile(source, destin)


    ## Import each point cloud seperately (to save memory)
    for file in f_paths:

        file_timer = Timer()
        print('\n  Input file: ', file, '...')
        print('    Reading file ...')
        pc = read_pc_from_file(file)
        if pc is None:
            continue

        print("    Point Cloud fields: ")
        print("     ", pc.fields)
        
        print([f for f in pc.fields if len(f+'_float32') > 32])
        print([len(f+'_float32') for f in pc.fields if len(f+'_float32') > 32])

        # Remove invalid data:
        print('    Setting invalid data (nan & inf) to 0 ...')
        pc.data[np.isnan(pc.data)] = 0
        pc.data[np.isinf(pc.data)] = 0


        # Standardize features:
        if feature_standardizing:
            print('    Standardizing feature distributions ...')
            for field in pc.fields:
                if field in label_fields + ['x', 'y', 'z']:
                    continue
                print('      field: ', field)
                # Use an index instead of get/set_fdata() to
                # save memory and increase speed:
                ind = pc.field_indices([field])
                # Calc mean and standard devation:
                mean = np.nanmean(pc.data[:,ind])
                std = np.nansum(np.power(pc.data[:,ind] - mean, 2))
                std = np.sqrt(std / (len(pc) - 1))
                # To new dist:
                if std > 1e-6:
                    pc.data[:,ind] = (pc.data[:,ind] - mean) / std
                else:
                    pc.data[:,ind] = (pc.data[:,ind] - mean)


        ## Read tree data:
        if include_trees:
            file_path    = os.path.splitext(file)[0] + '__trees.txt'
            data, fields = pts.read(file_path, reduced=False, IDT='float64')
            tree_set     = PointCloud(data, fields)
            print("\n    trees:", data.shape[0])
            print("    tree fields: ")
            print_dict(fields, indent=6)


        ## Temporarily reduce to center of gravity:
        # Don't use np.mean(), because sum might overflow float32, if coords to
        # big! (e.g. when no offset had been in set in las file)
        print("\n")
        xyz = pc.get_xyz().copy()
        xyz_min = np.min(xyz, 0)
        xyz_max = np.max(xyz, 0)
        center = xyz_min + (xyz_max - xyz_min) / 2
        xyz -= center

        if include_trees:
            xyz_trees = tree_set.get_xyz().copy()
            xyz_trees -= pc.import_offset
            xyz_trees -= center


        ## Rotatation augmentation:
        for angle in angles:

            print('    a = {:3d}°: '.format(angle), end=' ')
            angle_timer = Timer()

            tmat = trafo.R_z(angle)
            xyz_rot, _ = trafo.transform(xyz.copy(), tmat)
            xyz_rot += center

            # Rounding rotated coords to same precision as (original) data.
            # Setting p = 0.01 / 0.0001 /... is also the workaround to fix sampling
            # shape errors.
            p = pc.import_precision if precision == None else [precision] * 3
            xyz_rot = (xyz_rot / p).astype(int) * p

            pc.set_xyz(xyz_rot)

            if include_trees:
                xyz_trees_rot, _ = trafo.transform(xyz_trees, tmat)
                if global_vox_coords:
                    xyz_trees_rot += center
                tree_set.set_xyz(xyz_trees_rot)

            ## Voxelization
            non_feature_fields = label_fields + ['x', 'y', 'z']
            feature_fields = [f for f in pc.fields if f not in non_feature_fields]
            features = pc.get_fdata(feature_fields)
            labels = pc.get_fdata(label_fields).astype(np.uint16)

            if global_vox_coords:
                xyz_to_vox = xyz_rot
            else:
                xyz_to_vox = xyz_rot - center

            coords_v, f_v, l_v, map, _, acc = voxer.voxelize(xyz_to_vox, features,
                                                             labels, v_size,
                                                             ignore_label)

            data_v = np.zeros((coords_v.shape[0], pc.data.shape[1]))
            data_v[:,pc.field_indices(['x', 'y', 'z'])] = coords_v
            data_v[:,pc.field_indices(feature_fields)] = f_v
            data_v[:,pc.field_indices(label_fields)] = l_v

            print('voxelization acc [%] =',
                  ', '.join(['{:2.8f}'.format(a*100) for a in acc]), end=', ')


            ## Saving Results:
            in_fname = os.path.splitext(os.path.basename(file))[0]
            out_fname_raw = in_fname + '__vs{:03d}__a{:03d}'.format(int(v_size*100), angle)
            out_fpath_raw = os.path.join(out_path, out_fname_raw)

            p = pc.import_precision if precision == None else [precision] * 3

            # Voxel cloud:
            fpath = out_fpath_raw + '_' + VOX_SUFFIX + '_' + IDT + '.las'
            las.write_to_float(fpath, data_v, pc.fields, p,
                               pc.import_offset, pc.import_reduced, IDT=IDT)

            if import_points:
                # Point cloud:
                fpath = out_fpath_raw + '_' + PNT_SUFFIX + '_' + IDT + '.las'
                las.write_to_float(fpath, pc.data, pc.fields, p,
                pc.import_offset, pc.import_reduced, IDT=IDT)

                # Map:
                fpath = out_fpath_raw + '_' + MAP_SUFFIX + '.pkl'
                pckl_file = open(fpath, 'wb')
                cPickle.dump(map, pckl_file)
                pckl_file.close()

            # Tree set:
            if include_trees:
                fpath = out_fpath_raw + '_' + TRS_SUFFIX + '_' + IDT + '.las'
                las.write_to_float(fpath, tree_set.data, tree_set.fields, p,
                                   pc.import_offset, pc.import_reduced)


            print('  Time = {:.2f}s'.format(angle_timer.restart()))

        print('    Time: {:.2f}s'.format(file_timer.restart()))

    print('\n  Total time: ', Timer.string(global_timer.time()), '\n')
    return out_dir
