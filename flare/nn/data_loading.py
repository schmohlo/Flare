""" Functions for loading data (samples, configs) to train or test a model.

by Stefan Schmohl, 2020
"""

from globals import *

import os
import json
import numpy as np
import pandas as pd
import _pickle as cPickle

from math import isnan

from flare.utils import Timer
from flare.utils import PointCloud
from flare.utils import las
from flare.utils import pts
from flare.utils import print_dict
from flare.data_prep.sample_fus import read_pc_from_file




def load_data_for_model(data_configs, model_config, sample_items,
                        from_las=False, path_2_samples=PATH_2_SAMPLES):
    """
    TODO:
    """
    print("\n\nLoading data for model ..."); timer=Timer()
    print("  Model input fields: ")
    print('   ', model_config['input_features'])
    print("  Model classes:      ")
    print_dict(model_config['classes'], indent=4)
    print('  ' + '-'*77)

    samples = {}
    label_maps = {}
    label_maps_inv = {}

    for key, data_config in data_configs.items():
        d = data_config
        m = model_config
        sample_path = os.path.join(path_2_samples, d['dir'])
        field_norming = d['field_norming'] if 'field_norming' in d else None
        field_renaming = d['field_renaming'] if 'field_renaming' in d else None
        samples[key], classes = load_samples(sample_path,
                                             sample_items,
                                             field_renaming,
                                             m['input_features']
                                             + [d['gt_field']],
                                             field_norming,
                                             from_las)
        d['classes'] = classes

        label_maps[key], label_maps_inv[key] = create_label_map(d['classes'],
                                                                m['classes'],
                                                                d['class_mapping'])

        print(' ', key, "samples:", len(samples[key]))

        if 'tree_set' in samples[key][0].keys():
            n_trees = [len(s['tree_set']) for s in samples[key]]
            print(' ', key, "trees:  ",  sum(n_trees), "  (including duplicates)")
        
        print(' ', key, "sample #0:")
        for item in sample_items:
            try:
                print('   ', str(item), "fields after loading:\n",
                      '    ', samples[key][0][item].fields)
            except:
                pass
        print(' ', key, "sample classes (data json): ")
        print_dict(classes, indent=4)
        print(' ', key, "sample labels (new):\n",
              '  ', label_maps[key][list(classes.keys())])
        print('  ' + '-'*77)
    print("  time = {:.2f}s\n".format(timer.time()))

    return samples, label_maps, label_maps_inv





def load_samples(sample_path, keys=None, field_renaming=None, fields=None,
                 field_norming=None, from_las=False):
    """ Loads samples from pickle files (or from las files).

    Since I opted to retain the data as close to it's original as long as
    possible, and also because I need to be as memory efficient as possible (and
    this function is pretty early in the training / inference pipeline and loads
    the samples bit by bit depending on the how many had been stored together),
    this function does some changes to the samples (more precicely its items
    that are point cloud objects):
        - norming scalar fields as specified in the data json.
        - renaming the fields compatible to the model they're used for.
        - deleting fields not used by the model.

    Args:
        sample_path:
            Path to folder were all samples lie in form of pickle files ('.pkl')
            Each pkl-file may inlcude a single sample or a list of samples.
        keys (list):
            Assuming the samples are dictionary, keep only these keys in memory,
            and delete the others as soon as possible. This may be helpfull for
            very large data.
        field_renaming (dict):
            Assuming some of the sample items are point clouds objects.
            {oldname: newname} for renaming the sample fields according to the
            network or another "standard". Does not have to include trivial
            renamings (same old/new name).
        fields (list[str]):
            Assuming some of the sample items are point clouds objects. Fields
            to load per sample. Other fields (except 'x', 'y', 'z') are deleted.
            Order is ['x', 'y', 'z'] + fields.
        field_norming (dict):
            {fieldname (old): {"old_min", "old_max", "new_min", "new_max"}
        from_las:
            If True, than, instead of loading pkl-files, searches for 2 las-
            files and one map-pkl per sample.
    Returns:
        samples (list):
            List of samples, each sample as an dict.
        classes (dict):
            Class dict from data json.
    """

    files = os.scandir(sample_path)
    json_files = [f for f in files if f.name.endswith('.json')]

    classes = None
    for file in json_files:
        f = open(file.path, mode='r')
        meta = json.load(f, object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()})
        f.close()
        classes = meta['classes']


    files = os.scandir(sample_path)
    if not from_las:
        in_files = [f for f in files if f.name.endswith('.pkl')]
    else:
        in_files = [f for f in files if f.name.endswith('_map.pkl')]


    samples = []
    for f in in_files:

        file = open(f.path, 'rb')
        smpl_data = cPickle.load(file)
        file.close()

        # Load single sample from las:
        if from_las:
            smpl_name = f.name[:-len('_map.pkl')]
            smpl_path = os.path.join(sample_path, smpl_name)
            ending = '_' + IDT + '.las'
            pc = read_pc_from_file(smpl_path + '_' + PNT_SUFFIX + ending)
            vc = read_pc_from_file(smpl_path + '_' + VOX_SUFFIX + ending)
            # Check, if tree file even exists (may not, if no trees in sample)
            try:
                tr = read_pc_from_file(smpl_path + '_' + TRS_SUFFIX + ending)
                smpl_data = {'name': smpl_name, 'map': smpl_data,
                             'voxel_cloud': vc, 'point_cloud': pc,
                             'tree_set': tr}
            except OSError:
                smpl_data = {'name': smpl_name, 'map': smpl_data,
                             'voxel_cloud': vc, 'point_cloud': pc}

        if type(smpl_data) is not list:  # falls nur ein sample geladen.
            smpl_data = [smpl_data]


        if keys is not None:
            smpl_data = [{k: sample[k] for k in keys} for sample in smpl_data]


        for sample in smpl_data:
            for item, name in zip(sample.values(), sample.keys()):
                if isinstance(item, PointCloud):

                    if name == 'tree_set': continue

                    # Norm fields:
                    if field_norming is not None:
                        for field, p in field_norming.items():
                            # Use an index instead of get/set_fdata() to
                            # save memory and increase speed:
                            ind = item.field_indices([field])
                            # Bring it to 0..1:
                            item.data[:,ind] = ((item.data[:,ind]
                                               - p['old_min'])
                                               / (p['old_max']-p['old_min']))
                            # To new range:
                            item.data[:,ind] = (item.data[:,ind]
                                               * (p['new_max']-p['new_min'])
                                               + p['new_min'])
                                               
                    # Rename fields:  (after norm. und standard.!)
                    if field_renaming is not None:
                        for n in field_renaming.items():
                            item.fields = [n[1] if f == n[0] else f for f in item.fields]
                
                    # Remove & resort fields (the later not realy necessary):
                    if fields is not None:
                        kp = ['x', 'y', 'z'] + fields
                        rm = [f for f in item.fields if f not in kp]
                        item.delete_fields(rm)
                        data = item.get_fdata(kp)
                        item.data = data
                        item.fields = kp

        samples.extend(smpl_data)

    # Sort samples alphabetically by name to ensure reproducibility:
    samples.sort(key=lambda sample: sample['name'])

    return samples, classes




def create_label_map(data_classes, model_classes, name_mapping):
    """ Creates a look up table for converting class labels

    Data_classes are assumed to be the input classes which are to be mapped to
    the model_classes. Name_mapping defines their relationship.

    The inverse map will, if multiple data classes are map into a single model
    class, convert to the first occurance in name_mapping.

    Args:
        data_classes:    {class_label(int): class_name(str)}
        model_classes:   {class_label(int): class_name(str)}
        name_mapping:    {data_class_name(str): model_class_name(str)}
    Returns:
        map:             Lookup nparray where map[data_class_label] =
                         model_class_label
        inv:             Lookup nparray where map[model_class_label] =
                         data_class_label
    """

    map_dict = {}

    for data_name, model_name in name_mapping.items():
        # ignore the multi-model-classes option and simply use the first m class:
        if isinstance(model_name, list):
            model_name = model_name[0]
        i = list(data_classes.values()).index(data_name)
        j = list(model_classes.values()).index(model_name)

        old_label = list(data_classes.keys())[i]
        new_label = list(model_classes.keys())[j]
        map_dict[old_label] = new_label

    map = np.full(max(map_dict.keys())   + 1, np.nan)
    inv = np.full(max(map_dict.values()) + 1, np.nan)

    k = list(map_dict.keys())
    v = list(map_dict.values())
    map[k] = v

    # Reverse so that first occurance in name_mapping overrides later ones:
    k = list(reversed(list(map_dict.values())))
    v = list(reversed(list(map_dict.keys())))
    inv[k] = v

    return map, inv
