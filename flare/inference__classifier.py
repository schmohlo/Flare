""" Script for testing a voxel cloud classifier (sem seg) with SparseConvNet.

You can compare multiple models / epochs or use them as an "ensemble" to
classify the input. Will save results. May also do an analysis (conf matrix).

When using multiple models, make sure they all predict the same output classes
and have the same ignore_label!

Input allways needs a ground truth scalar field. At a pinch, generate pseudo-gt
externally.

by Stefan Schmohl, 2020
"""

from globals import *

import os
import time
import json
import numpy as np
import torch

from torch.utils.data import DataLoader

from flare.nn.datasets import dataset_from_config
from flare.nn.weight_fus import *
from flare.nn.data_loading import load_samples
from flare.nn.data_loading import load_data_for_model
from flare.nn.data_loading import create_label_map
from flare.utils import las
from flare.utils import config as cfg
from flare.utils import Timer
from flare.utils import PointCloud
from flare.utils import clean_accuracy
from flare.utils import conf_matrix
from flare.utils import import_class_from_string
from flare.utils import plot_metric_over_models



###############################################################################
# Setup                                                                       #
#                                                                             #
# test_name:                                                                  #
#       Just the 'description-suffix' of the test folder.                     #
# model_names:                                                                #
#       List of model names (i.e. their folder names).                        #
# epochs:                                                                     #
#       None (for respective last saved epoch per model) or list of epoch     #
#       number or None (for last) to load for each model.                     #
#       If only only one model_name but multiple epochs, will compare epochs  #
#       of same model.                                                        #
# data_config:                                                                #
#       Data config file name.                                                #
# path_2_data:                                                                #
#       Path to direcotory where the data folder lies specified by data       #
#       config. Usually PATH_2_SAMPLES or PATH_2_VOXELIZED (for when using    #
#       the whole input cloud at once, which removes sample border artifacts. #
# from_las:                                                                   #
#       True if sample files are in .LAS format (usually, if                  #
#       path_2_data = PATH_2_VOXELIZED. If False, samples musst be in .PKL    #
# include_points:                                                             #
#       Run the prediction / test also for the respective point cloud(s) by   #
#       mapping the voxel labels & probabilitities back to their points.      #
# mode:                                                                       #
#       'ensemble' or 'comparison'.                                           #
#       Comparison will only run a prediciton on each model and evaluate the  #
#       results (overall accuracy).                                           #
#       Ensemble averages the predicted probabilities over all models per     #
#       point. Result will be saved added to the original input and saved to  #
#       file.                                                                 #
# evaluate:                                                                   #
#       Creates confusion matrix for ensemble result.                         #
#       May be too expensive (or useless, if no groun  truth).                #
# save_as_data_classes:                                                       #
#       If True, predicted ensemble labels (which are in "model classes")     #
#       will be tranformed to match the "data classes".                       #
# save_probs:                                                                 #
#       If True, will save the predicted ensemble probabilities also into     #
#       the ouput file.                                                       #
# batch_size:                                                                 #
#       batch_size.                                                           #
# verbose_step:                                                               #
#       Verbose step within training loop. Verbose every ith epoch.           #
# device:                                                                     #
#       'cuda' or 'cpu'. If you have two gpus, select one with '0', or '1'.   #
# output_labeling:                                                            #
#       {model_class_label: output_prediction_label}                          #
#                                                                             #
###############################################################################




def main(test_name, model_names, epochs,
         data_config, path_2_data=PATH_2_SAMPLES, from_las=False, include_points=True,
         mode='ensemble', evaluate=True, save_as_data_classes=False, save_probs=True,
         batch_size=32, device='cuda', show=True, save_only_prediction=False,
         output_labeling=None, eval_model_classes=False, them_pixels=False, voxel_size=None,
         save_prediction_output=True):


    display_accs = (mode == 'comparison' or (mode =='ensemble' and evaluate))

    timer = Timer()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_name = '{}__{}'.format(timestamp, test_name)
    path_2_results = os.path.join(PATH_2_PREDICTIONS, test_name)

    data_config = cfg.from_file(data_config)

    if len(model_names) == 1:
        if not isinstance(epochs, list):
            epochs = [epochs]
        model_names = [model_names[0]] * len(epochs)
    else:
        if epochs is None:
            epochs = [None] * len(model_names)
    assert len(model_names) == len(epochs)

    ## Save / copy infos into results folder:
    os.makedirs(path_2_results)

    f = open(os.path.join(path_2_results, 'models.json'), 'w')
    json.dump({'model_name': model_names, 'epochs': epochs}, f, indent=4)
    f.close()

    f = open(os.path.join(path_2_results, 'data.json'), 'w')
    json.dump(data_config, f, indent=4)
    f.close()

    if them_pixels:
        include_points = False

    identifiers = {'v': 'voxels'}
    if include_points :
        identifiers['p'] = 'points'



    #=#########################################################################
    # Loop over models / epochs                                               #
    #                                                                         #
    #=#########################################################################

    accs          = {'p': [], 'v': []}
    probs         = []
    milfs         = []
    coords        = None
    label_map     = None
    label_map_inv = None
    map           = None
    ignore_label  = None
    ignore_label_real = None
    model_classes = None
    gts           = None

    for model_name, epoch in zip(model_names, epochs):

        result = test_single(model_name, epoch, data_config, path_2_data,
                             from_las, include_points, mode, evaluate,
                             batch_size, device, display_accs,
                             them_pixels)

        if display_accs:
            accs['v'].append(result['acc'])
            if include_points:
                accs['p'].append(result['acc_p'])

        if mode == 'ensemble':
            probs.append(result['probs'])
            milfs.append(result['milfs'])
            label_map_inv = result['label_map_inv'].astype(int)
            label_map     = result['label_map'].astype(int)

            if include_points:
                map = result['map']

            if evaluate:
                ignore_label  = result['ignore_label']
                ignore_label_real  = result['ignore_label_real']
                model_classes = result['model_classes']
                data_classes  = data_config['classes']
                gts = {'v': result['gt']}
                if include_points:
                    gts['p'] = result['gt_p'].astype('int')
                    gts['p_org'] = result['gt_p_org'].astype('int')
                    
            if them_pixels:
                coords = result['coords']

    print('\n\n' + '#'*79)


    ###########################################################################
    # Comparison                                                              #
    #                                                                         #
    # - Graph for accuarcies over models / epochs                             #
    ###########################################################################

    if display_accs:
        print("\nEvaluation ...")

        plot_metric_over_models(list(accs.values()), list(accs.keys()), model_names,
                                epochs, 'Test Accuracy', path_2_results)

        for i in identifiers.items():
            acc = accs[i[0]]
            mean_acc = np.mean(acc)*100
            std_acc = np.std(acc)*100
            message = "    Mean acc ({}):     {:6.2f} %  {} {:.2f}"
            print(message.format(i[1], mean_acc, u'\u00B1', std_acc))

    if mode == 'comparison':
        # rest is only for ensembles
        print('\n')
        return


    ###########################################################################
    # Ensemble-Evaluation                                                     #
    #                                                                         #
    # - Confusion matrices                                                    #
    ###########################################################################

    probs_raw  = {'v': np.stack(probs, axis=2)}
    probs_mean = {'v': np.sum(probs_raw['v'], axis=2) / probs_raw['v'].shape[2]}
    labels = {'v': np.argmax(probs_mean['v'], axis=1)}

    milfs_raw  = {'v': np.stack(milfs, axis=2)}
    milfs_mean = {'v': np.sum(milfs_raw['v'], axis=2) / milfs_raw['v'].shape[2]}

    if not save_probs:
        del probs_mean
        del milfs_mean
    del probs_raw   # dont want them right now
    del milfs_raw

    if include_points:
        labels['p'] = labels['v'][map]
        if save_probs:
            milfs_mean['p'] = milfs_mean['v'][map]
            probs_mean['p'] = probs_mean['v'][map]
            # probs_raw['p']  = probs_raw['v'][map]

    if evaluate:

        for key, name in identifiers.items():

            acc_ensemble = clean_accuracy(labels[key], gts[key], ignore_label)
            print("    Ensemble acc (%s): %6.2f %%" % (name, acc_ensemble * 100))

            if eval_model_classes:
                data_classes = model_classes.copy()
                cm = conf_matrix.create(gts[key], labels[key],
                                        list(data_classes.keys()),
                                        list(model_classes.keys()))
            else:
                if key == 'p':
                    cm = conf_matrix.create(gts['p_org'], labels[key],
                                            list(data_classes.keys()),
                                            list(model_classes.keys()))
                else:
                    cm = conf_matrix.create(label_map_inv[gts[key]], labels[key],
                            list(data_classes.keys()),
                            list(model_classes.keys()))

            if ignore_label_real is None:
                ignore_labels_gt = None
            else:
                ignore_labels_gt = [i for i,v in enumerate(label_map) if v == ignore_label]
                if them_pixels:
                    ignore_labels_gt.append(ignore_label)

            if output_labeling is not None:
                model_classes_ = {output_labeling[l]:n for l,n in model_classes.items()}
            else:
                model_classes_ = model_classes

            # Redo class mapping that allows multiple data classes per model class:
            if eval_model_classes:
                l_map = None
            else:
                l_map = {}
                for data_cname, model_cname in data_config['class_mapping'].items():
                    i = list(data_classes.values()).index(data_cname)
                    old_label = list(data_classes.keys())[i]
                    if not isinstance(model_cname , list):
                        j = list(model_classes_.values()).index(model_cname)
                        new_label = list(model_classes_.keys())[j]
                    else:
                        new_label = []
                        for mcn in model_cname:
                            j = list(model_classes_.values()).index(mcn)
                            new_label += [list(model_classes_.keys())[j]]
                    l_map[old_label] = new_label

            conf_matrix.plot(cm, data_classes, path_2_results,
                             file_suffix='_'+name,
                             classes_pred=model_classes_,
                             label_map=l_map,
                             ignore_labels=ignore_labels_gt,
                             show=show, iou=True)

            f = open(os.path.join(path_2_results, 'conf_matrix__%s.txt' % (name)), 'w')
            conf_matrix.print_to_file(cm, f,
                                      data_classes,
                                      model_classes_,
                                      l_map, indent=4,
                                      ignore_labels=ignore_labels_gt)
            f.close()
        del gts


    ###########################################################################
    # Save prediction of Ensemble into voxel (& point) cloud                  #
    #                                                                         #
    ###########################################################################

    if save_prediction_output:
        
        ## Load samples again, but without modifying them for a model:
        sample_items = ['name', 'voxel_cloud']
        if include_points:
            sample_items += ['point_cloud', 'map']
        path_2_samples = os.path.join(path_2_data, data_config['dir'])
        samples, _ = load_samples(path_2_samples, keys=sample_items, from_las=from_las)


        ## Merge samples into single big cloud and add predictions
        for key, name in identifiers.items():

            if save_as_data_classes:
                labels[key] = label_map_inv[labels[key]]
            elif output_labeling is not None:
                class_mapping = {n:n for n in model_classes.values()}
                mapping, _= create_label_map(model_classes, model_classes_, class_mapping)
                labels[key] = mapping[labels[key]]

            extra_data = labels[key][:,None]
            extra_fields = ['prediction']
            if save_probs:
                extra_data = np.hstack((extra_data, probs_mean[key]))
                extra_fields += ['p%d' % (i) for i in range(probs_mean[key].shape[1])]

            if save_probs and not np.isnan(np.sum(milfs_mean['v'][0])):
                extra_data = np.hstack((extra_data, milfs_mean[key]))
                extra_fields += ['milf%d' % (i) for i in range(milfs_mean[key].shape[1])]

            item = name[:-1] + '_cloud'
            _merge_and_save_samples(samples, item, path_2_results,
                                    data_config['gt_field'],
                                    extra_data, extra_fields,
                                    save_only_prediction,
                                    them_pixels, coords, voxel_size)


    print('Finished prediction (including I/O and evaluation) in', timer.time_string())
    print('\n')











###############################################################################
# Test on single network                                                      #
#                                                                             #
###############################################################################


def test_single(model_name, epoch, data_config, path_2_data, from_las,
                include_points, mode, evaluate, batch_size, device, display_accs,
                them_pixels):


    timer = Timer()
    print('\n\n' + '='*79)
    print('Model: \'{}\',  epoch: {} ...'.format(model_name, epoch))
    path_2_model       = os.path.join(PATH_2_MODELS, model_name)
    path_2_checkpoints = os.path.join(path_2_model, DIR_2_CHECKPOINTS)


    ## Load configs from model:
    config = cfg.from_model_dir(model_name)
    config = {**config, 'data_test': data_config}


    ###########################################################################
    # Data Setup                                                              #
    #                                                                         #
    # Load samples / the whole input cloud for inference                      #
    # (Dataloader can differ from model to model)                             #
    ###########################################################################

    sample_items = ['name', 'voxel_cloud']
    if include_points:
        sample_items += ['point_cloud', 'map']
    s, lm, lmi = load_data_for_model({'test': data_config}, config['model'],
                                     sample_items, from_las, path_2_data)
    samples       = s['test']
    label_map     = lm['test']
    label_map_inv = lmi['test']

    dataset = dataset_from_config(config, samples, label_map, 'test')

    # num_woker=1, because parallel processes result in "shuffle"
    dataloader = DataLoader(dataset, batch_size, num_workers=1, shuffle=False,
                            collate_fn=dataset.collate)


    ###########################################################################
    # Model Setup                                                             #
    #                                                                         #
    ###########################################################################

    print("\n\nPrediction ...")
    net = import_class_from_string(config['model']['network']['architecture'])
    net = net.from_config(config)

    loss_fu = import_class_from_string(config['train']['loss_fu'])
    weights = globals()[config['train'][config['train']['loss_fu']]['weight_fu']]
    weights = weights(dataloader.dataset)
    ignore_label = config['model'].get('ignore_label', -100)
    ignore_label_real = config['model'].get('ignore_label', None)
    loss_fu = loss_fu(weight=weights, ignore_index=ignore_label)

    classifier = import_class_from_string(config['model']['model'])
    classifier = classifier.from_config(config, net, None, loss_fu, None, device)

    cp, _ = classifier.load_net_from_checkpoint(path_2_checkpoints, epoch)
    print("  Loaded net state from checkpoint:", cp)


    ###########################################################################
    # Predict on voxel samples                                                #
    #                                                                         #
    ###########################################################################

    timer2 = Timer()
    results = classifier.test(dataloader, gt=True, acc=True, numpy=True, merge=True, coords=them_pixels)

    if hasattr(classifier.net, 'milf'):
        print("  Extracted Milfs")
        labels, probs, gt, acc, milfs = results
    else:
        if them_pixels:
            labels, probs, gt, acc, coords = results
        else:
            labels, probs, gt, acc = results
        milfs = np.zeros((len(labels),1,1)) + np.nan

    print("  Number of elements              : {:,d} ".format(len(labels)))
    print("  Prediction time                 : {}".format(timer2.time_string()))

    if display_accs:
        print('')
        misclassified = (labels != gt)
        misc_sum = np.sum(misclassified)
        misc_rel = misc_sum / misclassified.size * 100
        print("  Sum \'misclassified\'             : {:,d} = {:.2f}%".format(misc_sum, misc_rel))
        print("  Accuracy (from test-function)   : {:.2f} %".format(acc*100))

        acc = clean_accuracy(labels, gt, ignore_label)
        print("  Number of voxels                : {:,d}".format(labels.size))
        print("  Overall acc on voxels           : {:.2f} %".format(acc * 100))


    ###########################################################################
    # Map labels to point cloud                                               #
    #                                                                         #
    ###########################################################################

    if include_points:
        map = _merged_map_from_samples(samples)

    if display_accs and include_points:
        gt_field = [data_config['gt_field']]
        gt_p_org = np.vstack([s['point_cloud'].get_fdata(gt_field) for s in samples])
        gt_p     = label_map[gt_p_org.squeeze().astype('long')]

        labels_p = labels[map]
        acc_p = clean_accuracy(labels_p, gt_p, ignore_label)
        print("  Number of points                : {:,d}".format(len(gt_p)))
        print("  Overall acc on points           : {:.2f} %".format(acc_p * 100))


    print('\n' + '='*79)
    print("Finished after", timer.time_string(), '\n')

    output = {}
    if display_accs:
        output['acc'] = acc
        if include_points:
            output['acc_p'] = acc_p
    if mode == 'ensemble':
        output['probs']         = probs
        output['milfs']         = milfs
        output['label_map']     = label_map
        output['label_map_inv'] = label_map_inv
        if include_points:
             output['map'] = map
    if mode == 'ensemble' and evaluate:
        output['gt'] = gt
        output['ignore_label'] = ignore_label
        output['ignore_label_real'] = ignore_label_real
        output['model_classes'] = config['model']['classes']
        if include_points:
            output['gt_p'] = gt_p
            output['gt_p_org'] = gt_p_org
        if them_pixels:
            output['coords'] = coords
    return output




###############################################################################
# Auxiliaries                                                                 #
#                                                                             #
###############################################################################

def _merged_map_from_samples(samples):
    sample_sizes     = [len(s['voxel_cloud']) for s in samples]
    sample_sizes_cum = [sum(sample_sizes[0:i]) for i, _ in enumerate(sample_sizes)]
    merged_map = [s['map'] + ssc for s, ssc in zip(samples, sample_sizes_cum)]
    merged_map = np.hstack(merged_map)
    return merged_map



def _merge_and_save_samples(samples, item, path, gt_field, extra_data, extra_fields,
                            only_pred, them_pixels=False, coords=None, voxel_size=None):
    sample0 = samples[0][item]

    if them_pixels:
        only_pred = False

        # coords are merged together. only know which to which sample by iteration 
        # through sample lengths.

        sl = [len(s[item]) for s in samples]
        sl = sl
        slc = np.cumsum([0] + sl)

        coords_local = [coords[slc[i]:slc[i+1], 0:3] for i in range(len(slc)-1)]

        offsets = [np.min(s[item].get_xyz()[:,0:3], 0) for s in samples]

        coords_global = [c + o for c,o in zip(coords_local, offsets)]
        coords_global = np.vstack(coords_global) * voxel_size
        
        merged_pc = PointCloud(coords_global, ['x', 'y', 'z'])

        idx = [c.astype('int') for c in coords_local]
        extra_data = [d[:,i[:,0],i[:,1]] for d,i in zip(extra_data, idx)]
        extra_data = np.hstack(extra_data)
        extra_data = np.moveaxis(extra_data, 0, 1)
        
    else:
        merged_data = np.vstack([s[item].data for s in samples])
        merged_pc = PointCloud(merged_data, sample0.fields)


    if only_pred:
        rm_fields = merged_pc.fields.copy()
        for field in ['x', 'y', 'z'] + [gt_field]:
            rm_fields.remove(field)
        merged_pc.delete_fields(rm_fields)

    merged_pc.add_fields(extra_data, extra_fields)

    fpath = os.path.join(path, 'prediction_'+item+'.las')
    las.write(fpath,
              merged_pc.data,
              merged_pc.fields,
              sample0.import_precision,
              sample0.import_offset,
              sample0.import_reduced, IDT=IDT)
