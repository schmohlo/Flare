""" For training a 'generic' model

by Stefan Schmohl, 2020
"""

from globals import *

import sys
import os
import shutil
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import flare
import flare.nn
import flare.utils.config as cfg
import flare.nn.callbacks as cbs
from flare.nn.datasets import dataset_from_config
from flare.nn.weight_fus import *
from flare.nn.data_loading import load_data_for_model
from flare.utils import Timer
from flare.utils import pts
from flare.utils import import_class_from_string
from flare.utils.metricsHistory import MetricsHistory
from flare.utils import Logger





def main(model_name, config, resume=False,
         n_models=1, verbose_step=1, device='cuda',
         voxel_size=None, include_trees=False, use_config_as_is=False,
         show=True, init_checkpoint=None):
    """

    Args:
        model_name:
            Just the 'description-suffix' if creating new model. Full folder
            name if you want to resume training from checkpoint.
        config:
            Config file names. Ignored if resume = True.
        resume:
            Resume training from checkpoint. In case training had previously
            been interrupted.
        n_models:
            Number of models. Runs the same training prodecure n times, results
            in n 'independend' models. Usefull for ensembling. Ignored when
            resume = True.
        verbose_step:
            Verbose step within training loop. Verbose every ith epoch.
        device:
            'cuda' or 'cpu'. If you have two gpus, select one with '0' or '1'
            via os.environ["CUDA_VISIBLE_DEVICES"] = '0'.
        voxel_size:
            May be needed by some models / datasets.
        include_trees:
            If true, loads also tree object labels per sample.
        use_config_as_is:
            is True, asume config is ready config dict, not file names.
        show:
            If True, display end-of-training result plots.
    """


    ## Load configs:
    if not use_config_as_is:
        if resume:
            config = cfg.from_model_dir(model_name)
        else:
            config = cfg.from_files(config)


    ## Load data:
    data_configs = {'train': config['data_train'], 'val': config['data_val']}

    sample_items = ['voxel_cloud', 'name']
    if include_trees:
        sample_items += ['tree_set']

    samples, label_maps, label_maps_inv = load_data_for_model(data_configs,
                                                              config['model'],
                                                              sample_items)

    datasets = {'train': dataset_from_config(config, samples['train'],
                                             label_maps['train'],
                                             'train', voxel_size),
                'val':   dataset_from_config(config, samples['val'],
                                             label_maps['val'],
                                             'val', voxel_size)}

    dataloaders = {'train': DataLoader(datasets['train'],
                                       **config['train']['dataloader_train'],
                                       collate_fn=datasets['train'].collate),
                   'val'  : DataLoader(datasets['val'],
                                       **config['train']['dataloader_val'],
                                       collate_fn=datasets['val'].collate)}


    ## Trigger Training:
    # Prevent multiple resumes:
    if resume == True:
        n_models = 1

    # Train individual networks:
    model_names = []
    test_metrics = []
    for i in range(n_models):
        plt.close('all')
        mn, tm = train_single(config, model_name, device, dataloaders, label_maps,
                              label_maps_inv, verbose_step, resume, show, voxel_size,
                              init_checkpoint)
        model_names.append(mn)
        test_metrics.append(tm)
    return model_names, test_metrics



###############################################################################
# Train (single) network                                                      #
#                                                                             #
###############################################################################


def train_single(config, model_name, device, dataloaders, label_maps,
                 label_maps_inv, verbose_step=1, resume=False, show=True,
                 voxel_size=None, init_checkpoint=None):
    """
    TODO:
    config:
        must contain items named 'model', 'train', 'data_train' and 'data_model'
        To see what those configs must include, see the examplary json files in
        the config colder.
    wie müssen die samles aussehen?
    """

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if not resume:
        model_name = create_model_name(config, model_name, timestamp)

    path2model = os.path.join(PATH_2_MODELS, model_name)
    path2mcps  = os.path.join(path2model, DIR_2_CHECKPOINTS)
    path2log   = os.path.join(path2model, 'log.txt')

    if not resume:
        os.makedirs(path2model)

    # Mirror output into log file:
    sys.stdout = Logger(os.path.join(path2model, 'log_console.txt'))

    print('\n\n' + '='*79)
    print('Model:  \'{}\''.format(model_name))
    timer = Timer()


    ###########################################################################

    ## Network Setup
    net = import_class_from_string(config['model']['network']['architecture'])
    net = net.from_config(config)


    ## Optimizer Setup
    optimizer = import_class_from_string(config['train']['optimizer'])
    optimizer = optimizer(net.parameters(), **config['train'][config['train']['optimizer']])


    ## Loss Function Setup
    loss_fu = import_class_from_string(config['train']['loss_fu'])
    # Cross-Entropy Loss:
    if loss_fu == torch.nn.CrossEntropyLoss:
        weights = globals()[config['train'][config['train']['loss_fu']]['weight_fu']]
        weights = weights(dataloaders['train'].dataset, list(config['model']['classes'].keys()))
        ignore_label = config['model'].get('ignore_label', -100)
        loss_fu = loss_fu(weight=weights, ignore_index=ignore_label)
    elif 'weight_lookup' in config['train'][config['train']['loss_fu']].keys():
            fname = config['train'][config['train']['loss_fu']]['weight_lookup']
            f = open(os.path.join(PATH_2_CONFIGS, fname), 'rb')
            counts = pickle.load(f)['counts']
            weights = 1 / torch.tensor(counts, device=device)
            weights[torch.isinf(weights)] = 0
            #weights = weights / torch.sum(weights)
            f.close()
            weight_lookup = WeightLookup(weights, voxel_size)
            config_loss = config['train'][config['train']['loss_fu']].copy()
            config_loss['weight_lookup'] = weight_lookup
            loss_fu = loss_fu(**config_loss)
    # Other Losses:
    else:
        loss_fu = loss_fu(**config['train'][config['train']['loss_fu']])


    ## Callback Setup
    callbacks = create_callbacks(config['train'], optimizer, path2log, path2mcps)


    ## Model Setup
    model = import_class_from_string(config['model']['model'])
    model = model.from_config(config, net, optimizer, loss_fu, callbacks.values(), device)


    ## Load previous state or create new model folder
    if resume:
        cp, le = model.load_net_from_checkpoint(path2mcps)
        print("\n\nLoaded net state from checkpoint:", cp)

        mh = MetricsHistory.from_logfile(path2log)
        prev_epoch = mh.last_epoch
        print("Resuming training history from epoch {:d},".format(prev_epoch),
              "next one will be {:d}.".format(prev_epoch + 1))

        idx_le = np.where(mh['epoch'] == le)[0].item()

        # Initialize callbacks with previous monotor values:
        for name, cb in callbacks.items():
            try:
                cb.best = mh[cb.monitor][idx_le]
                cb.last_saved_epoch = le
                print("Setting Callback {} .best to {}".format(name, cb.best))
            except AttributeError:
                pass

        # Manually setting optimizer to previous lr:
        for i, param_group in enumerate(optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = mh['lr'][idx_le]
            param_group['lr'] = new_lr
            print("Setting Learning Rate from {} to {}".format(old_lr, new_lr))
    else:
        prev_epoch = -1
        os.makedirs(path2mcps)
        if init_checkpoint is not None:
            print("\n\nLoading net state from checkpoint:", init_checkpoint)
            model.net.load_state_dict(torch.load(init_checkpoint))
            shutil.copy(init_checkpoint, path2model + '/init_checkpoint')
            with open(path2model + '/init_checkpoint.txt', 'w') as f:
                f.write(init_checkpoint)


    ## Save configs
    # Done before training, in case you want to resume training later in case
    # it was aborted by some error.
    cfg.to_files(config, path2model)


    ###########################################################################

    ## Training
    print('\n\nStarting training ...')
    timer = Timer()
    model.train(dataloaders['train'],
                dataloaders['val'],
                config['train']['max_epochs'],
                start_epoch=prev_epoch + 1,
                verbose_step=verbose_step)
    print('\n  Finished training after', timer.string(timer.stop()))


    ###########################################################################

    ## Evaluation
    print("\n\nPost-training evaluation ...")
    test_metrics = model.post_training_eval(dataloaders, config,
                                            path2model, path2mcps,
                                            label_maps, label_maps_inv, show)

    # Load history from logfile, which is more complete in case of resume:
    history = MetricsHistory.from_logfile(path2log)
    history.plot_all_compressed(path2model)

    print_training_report(model, path2model, timestamp, timer.time(),
                          history, test_metrics,
                          callbacks['ModelCheckpoint'].last_saved_epoch)

    print('='*79)
    print("Finished after", timer.time_string(), '\n')
    return model_name, test_metrics




###############################################################################
# Auxiliaries                                                                 #
#                                                                             #
###############################################################################



def create_model_name(config, raw_name, timestamp):
    net_name = os.path.splitext(config['model']['network']['architecture'])[1][1:]
    model_name = "{}__{}__{}".format(timestamp, net_name, raw_name)
    return model_name



def create_callbacks(config_train, optimizer, path2log, path2mcps):
    config = config_train

    CSVLogger         = cbs.CSVLogger(path2log)
    EarlyStopping     = cbs.EarlyStopping(**config['EarlyStopping'])
    ReduceLROnPlateau = cbs.ReduceLROnPlateau(optimizer,
                                              **config['ReduceLROnPlateau'])
    ModelCheckpoint   = cbs.ModelCheckpoint(path2mcps, save_weights_only=True,
                                            **config['ModelCheckpoint'])

    callbacks = {'CSVLogger':         CSVLogger,
                 'EarlyStopping':     EarlyStopping,
                 'ReduceLROnPlateau': ReduceLROnPlateau,
                 'ModelCheckpoint':   ModelCheckpoint}

    for key, value in config_train.items():
        if 'ModelAttributeScheduler' in key:
            callbacks[key] = cbs.ModelAttributeScheduler(**value)

    return callbacks



def print_training_report(model, path, start_time, train_duration, history,
                          metrics, last_saved_epoch):

    smhd = Timer.s2smhd(train_duration)
    train_time = '%d days, %2dh %2d\' %2d\'\'' % (smhd['d'],
                                                  smhd['h'],
                                                  smhd['m'],
                                                  smhd['s'])

    f = open(os.path.join(path, 'training_report.txt'), 'w')

    ## Training Report:
    f.write('\n\nTraining report:\n')
    f.write('    date            : %s\n' % start_time)
    f.write('    duration        : %s\n' % train_time)
    f.write('    epochs          : %d\n' % len(history['epoch']))
    f.write('    last saved epoch: %d\n' % last_saved_epoch)
    f.write('         results from eval: (may be different from history)\n')
    for metric, value in metrics.items():
        f.write('             %s: %5.4f\n' % (metric.ljust(10), value))
    f.write('    history[-1]:\n')
    for metric, values in history.items():
        f.write('        ' + metric.ljust(12) + ': %5.4f\n' % values.iloc[-1])

    ## Network-Details:
    f.write('\n\nNetwork details:\n')
    nop = model.num_of_paras()
    f.write('    Total number of network parameters: {:,d} \n'.format(nop))
    f.write('    ' + str(model.net).replace('\n', '\n' + '  ' * 2))
    f.close()
