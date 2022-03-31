""" Fnctions for reading and writing config dicts from/to json.

by Stefan Schmohl, 2020
""" 

from globals import *

import os
import json



def to_file(config, name, dir):
    """ Print config dict into json file.
    """
    f = open(os.path.join(dir, 'config_{}.json'.format(name)), 'w')
    json.dump(config, f, indent=4)
    f.close()



def to_files(configs, dir):
    """ Print configs into json files.
    configs = dict of dicts.
    first dict hirarchy level results in indivudall json files.
    """
    for key, value in configs.items():
        to_file(value, key, dir)



def from_file(f_name, path_2_configs=PATH_2_CONFIGS):
    """ Load single config from json file.
    """

    fpath = os.path.join(path_2_configs, f_name)
    f = open(fpath, mode='r')
    config = json.load(f)
    f.close()

    # Make string keys in class dicts to int:
    if 'classes' in config.keys():
        classes = config['classes']
        config['classes'] = {int(k):v for k, v in classes.items()}

    return config



def from_files(f_names, path_2_configs=PATH_2_CONFIGS):
    """ Load multiple configs from json files.
    TODO:
        f_names:  Dict{'config_name': 'file_name'}
    """
    configs = {}

    # Replace file names in config with respective json content:
    for key, value in f_names.items():
        configs[key] = from_file(value, path_2_configs)

    return configs



def from_model_dir(model_name, path_2_models=PATH_2_MODELS):
    """ Load configs from model folder saved by training script.
    """
    path_2_configs = os.path.join(path_2_models, model_name)
    config = {}
    config['data_train'] = 'config_data_train.json'
    config['data_val']   = 'config_data_val.json'
    config['model']      = 'config_model.json'
    config['train']      = 'config_train.json'
      
    return from_files(config, path_2_configs)
