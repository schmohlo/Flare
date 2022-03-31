"""

by Stefan Schmohl, Johannes Ernst, 2021
"""

from globals import *

import os
import time
import pickle
import json
import numpy as np
import _pickle as cPickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

from PIL import Image as PImage
from torch.utils.data import DataLoader

from flare.nn.datasets import dataset_from_config
from flare.nn.data_loading import load_samples
from flare.nn.data_loading import load_data_for_model
from flare.utils import average_precision as ap_tools
from flare.utils import nms as nms_fu
from flare.utils import config as cfg
from flare.utils import las
from flare.utils import pts
from flare.utils import Timer
from flare.utils import PointCloud
from flare.utils import import_class_from_string
from flare.utils import plot_metric_over_models


###############################################################################
# Setup                                                                       #
#
# TODO
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
#       path_2_data = PATH_2_VOXELIZED. If False, samples are stored in .PKL  #
# mode:                                                                       #
#       ('ensemble') or 'comparison'.                                         #
#       Comparison will only run a prediciton on each model and evaluate the  #
#       results (overall accuracy).                                           #
#       Ensemble averages the predicted probabilities over all models per     #
#       point. Result will be saved added to the original input and saved to  #
#       file.                                                                 #
# evaluate:                                                                   #
#       If True, computes average precision etc.                              #
#       May be too expensive (or useless, if no groun  truth).                #
# batch_size:                                                                 #
#       batch_size.                                                           #
# verbose_step:                                                               #
#       Verbose step within training loop. Verbose every ith epoch.           #
# device:                                                                     #
#       'cuda' or 'cpu'. If you have two gpus, select one with '0', or '1'.   #
#                                                                             #
###############################################################################





###############################################################################
# Main Funktion                                                               #
#                                                                             #
###############################################################################

def main(test_name, model_names, epochs,
         data_config, path_2_data=PATH_2_SAMPLES, from_las=False,
         mode='ensemble', evaluate=True,
         batch_size=32, device='cuda', voxel_size=None, 
         iou_thresholds=None, nms=None, show=True):

               
               
    display_accs = (mode == 'comparison' or (mode =='ensemble' and evaluate))

    timer = Timer()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    #test_name = '{}__{}'.format(timestamp, test_name)
    test_name = test_name.replace('SUNet3D2DTreeDetection', 'SUN3D2DTD')
    test_name = test_name.replace('SUNet2DTreeDetection', 'SUN2DTD')
    path_2_results = os.path.join(PATH_2_PREDICTIONS, test_name)

    data_config = cfg.from_file(data_config)

    ## Set geometry (must be the same for all models!)
    geometry = cfg.from_model_dir(model_names[0])['model']['geometry']
    for model_name in model_names:
        assert cfg.from_model_dir(model_name)['model']['geometry'] == geometry

    # Assemble pairs of model_names and their epochs to load checkpoint from:
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


    ###########################################################################
    # Inference in loop over models / epochs                                  #
    #                                                                         #
    ###########################################################################

    m_aps = []
    aps   = []
    trees = []
    probs = []

    for model_name, epoch in zip(model_names, epochs):

        result = test_single(model_name, epoch, data_config, path_2_data,
                             from_las, mode, evaluate, geometry,
                             batch_size, device, display_accs, voxel_size,
                             path_2_results=path_2_results, 
                             iou_thresholds=iou_thresholds, nms=nms,
                             show=show)
        trees.append(result['trees'])
        probs.append(result['probs'])
        if display_accs:
            m_aps.append(result['m_ap'])
            aps.append(result['ap'])
        

    print('\n\n' + '#'*79)


    ###########################################################################
    # Evaluation                                                              #
    #                                                                         #
    # - Graph for mAP over models / epochs                                    #
    ###########################################################################

    if display_accs:
        print("\nEvaluation ...")
        
        stacked_metrics = [m_aps] + np.asarray(aps).T.tolist()
        labls = ['mAP'] + ['AP_{{IoU={:.2f}}}'.format(t) for t in iou_thresholds]
        plot_metric_over_models(stacked_metrics, labls , model_names, 
                                epochs, 'Test Accuracy', path_2_results)

        mean_m_aps = np.mean(m_aps)
        std_m_aps  = np.std(m_aps)
        message = "    Mean mAP:     {:.2f} %  {} {:.2f}"
        print(message.format(mean_m_aps, u'\u00B1', std_m_aps))

    if mode == 'comparison':
        # rest is only for ensembles
        print('\n')
        return


    ###########################################################################
    # Save merged predictions into file                                       #
    #                                                                         #
    ###########################################################################

    print("\nSaving merged predictions into file ...")

    # Merge all tree predictions into same file with their model-nr:
    trees_merged = np.vstack(trees)
    probs_merged = np.hstack(probs)[:, None]

    m_ids = np.vstack([i * np.ones((trees[i].shape[0], 1)) for i, _ in enumerate(trees)])
    data_merged = np.hstack((trees_merged, probs_merged, m_ids))

    fpath = os.path.join(path_2_results, 'predictioned_trees_merged.pts')

    if geometry == 'circles':
        # Add pseudo z-coordinates (0-vector) for point cloud creation
        data_merged = np.hstack((data_merged[:,[0,1]], np.zeros((len(data_merged[:,1]),1)),
                                 data_merged[:,[2,3,4]]))
        fields = ['x', 'y', 'z', 'radius', 'score', 'model_number']
    elif geometry == 'rectangles':
        # Add pseudo z-coordinates (0-vector) for point cloud creation
        data_merged = np.hstack((data_merged[:,[0,1]], np.zeros((len(data_merged[:,1]),1)),
                                 data_merged[:,[2,3,4,5]]))
        fields = ['x', 'y', 'z', 'length', 'width', 'score', 'model_number']
    elif geometry == 'cylinders':
        fields = ['x', 'y', 'z', 'radius', 'height', 'score', 'model_number']
    elif geometry == 'cuboids':
        fields = ['x', 'y', 'z', 'length', 'width', 'height', 'score', 'model_number']
    pts.write(fpath, data_merged, fields, precision=0.000001)


    print('Finished prediction (including I/O and evaluation) in', timer.time_string())
    print('\n')




###############################################################################
# Test on single network                                                      #
#                                                                             #
###############################################################################


def test_single(model_name, epoch, data_config, path_2_data, from_las,
                mode, evaluate, geometry, batch_size, device, display_accs, 
                voxel_size, path_2_results, iou_thresholds, nms, show=True):


    timer = Timer()
    print('\n\n' + '='*79)
    print('Model: \'{}\',  epoch: {} ...'.format(model_name, epoch))
    path_2_model       = os.path.join(PATH_2_MODELS, model_name)
    path_2_checkpoints = os.path.join(path_2_model, DIR_2_CHECKPOINTS)


    ## Load configs from model:
    config = cfg.from_model_dir(model_name)
    config = {**config, 'data_test': data_config}
    if iou_thresholds is None:
        iou_thresholds = config['train']['ap_iou_thresholds']    
    if nms is not None:
        config['train']['nms'] = nms
        

    ###########################################################################
    # Data Setup                                                              #
    #                                                                         #
    # Load samples / the whole input cloud for inference                      #
    # (Dataloader can differ from model to model)                             #
    ###########################################################################

    sample_items = ['name', 'voxel_cloud', 'tree_set']

    s, lm, _ = load_data_for_model({'test': data_config}, config['model'],
                                     sample_items, from_las, path_2_data)
    samples       = s['test']
    label_map     = lm['test']

    dataset = dataset_from_config(config, samples, label_map, 'test', voxel_size)

    # num_worker=1, because parallel processes result in "shuffle"
    dataloader = DataLoader(dataset, batch_size, num_workers=1, shuffle=False,
                            collate_fn=dataset.collate)


    ###########################################################################
    # Model Setup                                                             #
    #                                                                         #
    ###########################################################################

    print("\n\nPrediction ...")
    net = import_class_from_string(config['model']['network']['architecture'])
    net = net.from_config(config)

    try:   # vermeide weight lookup zu laden (bei class. auch ohne Gewichtung)
        config['train'][config['train']['loss_fu']]['weight_lookup'] = None
    except:
        pass
    loss_fu = import_class_from_string(config['train']['loss_fu'])
    loss_fu = loss_fu(**config['train'][config['train']['loss_fu']])
            

    detector = import_class_from_string(config['model']['model'])
    detector = detector.from_config(config, net, None, loss_fu, None, device)

    cp_name, epoch = detector.load_net_from_checkpoint(path_2_checkpoints, epoch)
    print("  Loaded net state from checkpoint:", cp_name)


    ###########################################################################
    # Predict on voxel input / tress output samples                           #
    #                                                                         #
    ###########################################################################

    timer2 = Timer()
    if display_accs:
        trees, probs, m_ap, _, _, _, _ = detector.test(dataloader,ap=True,
                                                       iou_thresholds=iou_thresholds)
    else:
        trees, probs = detector.predict(dataloader)

    print("  Number of detections (samples)  : {:,d} ".format(len(trees)))
    print("  Number of detections (trees)    : {:,d} ".format(sum([len(t) for t in trees])))
    print("  Prediction time                 : {}".format(timer2.time_string()))
    if display_accs:
        print("  mAP (from test-function)        : {:.2f} %".format(m_ap))


    ###########################################################################
    # Merge samples together by making them global and performing second NMS  #
    #                                                                         #
    ###########################################################################
    
        
    
    from matplotlib.patches import Circle as mplcircle
    from matplotlib.patches import Rectangle as mplrectangle
    from matplotlib.patheffects import withStroke
    
    def print_detcts_list(detcts_list, titles, reduction=True, f_path=None, show=True):

        # Get minimum and maximum (x,y only)
        all_detcts = np.vstack(detcts_list)
        if len(all_detcts[0]) == 3:                             # for circles
            mi = np.r_[(np.min(all_detcts, 0))[0:2], 0]
            ma = np.r_[(np.max(all_detcts, 0))[0:2], 0]
        elif len(all_detcts[0]) == 4:                           # for rectangles
            mi = np.r_[(np.min(all_detcts, 0))[0:2], 0, 0]
            ma = np.r_[(np.max(all_detcts, 0))[0:2], 0, 0]
        elif len(all_detcts[0]) == 5:                           # for cylinders
            mi = np.r_[(np.min(all_detcts, 0))[0:2], 0, 0, 0]
            ma = np.r_[(np.max(all_detcts, 0))[0:2], 0, 0, 0]
        elif len(all_detcts[0]) == 6:                           # for cuboids
            mi = np.r_[(np.min(all_detcts, 0))[0:2], 0, 0, 0, 0]
            ma = np.r_[(np.max(all_detcts, 0))[0:2], 0, 0, 0, 0]
        di = ma - mi
        
        if reduction:
            for c in detcts_list:
                c -= mi
            mi = [0,0]
            ma = di

        for detcts, title in zip(detcts_list, titles):
            fig, ax = plt.subplots()
            plt.grid(True)
            ax.set_aspect('equal')
            ax.set(xlabel='x [m]'.format(offset[0]), 
                   ylabel='y [m]'.format(offset[1]),
                   title=title)
            plt.xlim((mi[0]-20, ma[0]+20))
            plt.ylim((mi[1]-20, ma[1]+20))
        
            for detct in detcts: 
                if geometry == 'circles':
                    ax.add_artist(mplcircle((detct[0], detct[1]), radius=detct[2], 
                        edgecolor='green', linewidth=1, facecolor=(0, 1, 0, .125),
                        path_effects=[withStroke(linewidth=1, foreground='w')]))
                elif geometry == 'rectangles':
                    ax.add_artist(mplrectangle((detct[0]-detct[2]/2, detct[1]-detct[3]/2), 
                        detct[2], detct[3], edgecolor='green', linewidth=1, 
                        facecolor=(0, 1, 0, .125), 
                        path_effects=[withStroke(linewidth=1, foreground='w')]))
                elif geometry == 'cylinders':
                    ax.add_artist(mplcircle((detct[0], detct[1]), radius=detct[3],
                        edgecolor='green', linewidth=1, facecolor=(0, 1, 0, .125),
                        path_effects=[withStroke(linewidth=1, foreground='w')]))
                elif geometry == 'cuboids':
                    ax.add_artist(mplrectangle((detct[0]-detct[3]/2, detct[1]-detct[4]/2), 
                        detct[3], detct[4], edgecolor='green', linewidth=1, 
                        facecolor=(0, 1, 0, .125), 
                        path_effects=[withStroke(linewidth=1, foreground='w')]))

            if f_path is not None:
                fig.savefig(f_path + title + '.png', dpi=600)
                fig.savefig(f_path + title + '.pdf', dpi=600)
                
            if show:
                plt.show(block = False) 

            if f_path is not None:
                f = open(f_path + title + '.pkl', 'wb')
                data_to_save = {'detcts_list': detcts_list, 'titles': titles}
                pickle.dump(data_to_save, f)
                f.close()                                     
        
        return 1
                              
                              

    ## Load samples again, but without modifying them for a model:
    sample_items = ['name', 'voxel_cloud']
    path_2_samples = os.path.join(path_2_data, config['data_test']['dir'])
    samples, _ = load_samples(path_2_samples, keys=sample_items, from_las=from_las)

    ## Transform trees into global metric space:
    # From float32 to float64 because of large global coords.
    trees = [t.astype('float64') for t in trees]
    for i, sample in enumerate(samples):
        if len(trees[i]) == 0:
            continue
        # Undo coord reduction from dataset:
        offset = np.min(sample['voxel_cloud'].get_xyz(), axis=0)
        if geometry == 'circles' or geometry == 'rectangles':
            trees[i][:, 0:2] += offset[0:2]
        else:
            trees[i][:, 0:3] += offset
        
        # Undo tranform to voxel space, back into metric space:
        trees[i] *= voxel_size
        # Undo coord reduction of import:
        offset = sample['voxel_cloud'].import_offset
        if geometry == 'circles' or geometry == 'rectangles':
            trees[i][:, 0:2] += offset[0:2]
        else:
            trees[i][:, 0:3] += offset
        

    ## Merge samples together:
    trees = np.vstack(trees)
    probs = np.hstack(probs)
    
    ## Perform global NMS to remove duplicates on sample borders:
    # Do a coord reduction again (for x,y,z), because nms/iou expects float32:
    if geometry == 'circles':
        offset = np.r_[np.min(trees[:, [0,1]], axis=0), 0]
    elif geometry == 'rectangles':
        offset = np.r_[np.min(trees[:, [0,1]], axis=0), 0, 0]
    elif geometry == 'cylinders':
        offset = np.r_[np.min(trees[:, [0,1,2]], axis=0), 0, 0]
    elif geometry == 'cuboids':
        offset = np.r_[np.min(trees[:, [0,1,2]], axis=0), 0, 0, 0]

    timer2 = Timer()
    detcts = (trees[:, :] - offset).astype('float32')
    nms_indices = nms_fu(detcts, probs,
                         config['train']['nms']['iou_threshold'],
                         config['train']['nms']['score_min'])
    trees = trees[nms_indices]
    probs = probs[nms_indices]
    print("  Global NMS time                 : {}".format(timer2.time_string()))


    ## Compute global mAP
    if display_accs:
    
        model_e = '{}__e={:d}'.format(model_name, epoch)

        # Get original tree gt (folder name includes geometry)
        dataset_name = config['data_test']['dir']

        dataset_name = dataset_name.split('__')
        dataset_name = dataset_name[0] + '__' + dataset_name[1]
        path_2_tree_gt = os.path.join(PATH_2_RAW, dataset_name)
        files = os.listdir(path_2_tree_gt)
        tree_gt_file = [f for f in files if '__trees.txt' in f][0]
        path_2_tree_gt = os.path.join(path_2_tree_gt, tree_gt_file)

        # name_split   = dataset_name.split('__')
        # dataset_name = name_split[0] + '__' + name_split[1]
        # dataset_name_geom = dataset_name + '__' + name_split[2]
        # tree_gt_file = dataset_name + '__trees.txt'
        # path_2_tree_gt = os.path.join(PATH_2_RAW, dataset_name_geom, tree_gt_file)
        
        data, fields = pts.read(path_2_tree_gt, reduced=False, IDT='float64')
        # trees_gt = PointCloud(data, fields)
        trees_gt = data
        
        ## Remove z-coordinate if handling circles or rectangles
        if geometry == 'circles':
            if len(trees_gt[0,:]) > 3:
                trees_gt = trees_gt[:,[0,1,3]]
        elif geometry == 'rectangles':
            if len(trees_gt[0,:]) > 4:
                trees_gt = trees_gt[:,[0,1,3,4]]

        f_path = os.path.join(path_2_results, geometry + '_' + model_e + '_')
        detcts_list = [np.copy(trees_gt), np.copy(trees)]
        titles      = ['gt', 'prediction']
        print_detcts_list(detcts_list, titles, f_path=f_path, show=show)

        # Compute global mAP
        m_ap, ap, p, r, t  = ap_tools.mean_ap(trees[:, :], 
                                              probs, 
                                              trees_gt[:, :],
                                              iou_thresholds,  
                                              return_extras=True)

        # Plot and save pr-curves
        labels = []
        labels_ = []
        for a, iou_t in zip(ap, iou_thresholds):
            labels.append('IoU={:.2f} (AP={:.2f})'.format(iou_t, a))
            labels_.append('IoU={:.2f}'.format(iou_t))

        # Plot P(R)-curves:
        f_path = os.path.join(path_2_results, 'p(r)__' + model_e)
        title = 'P(R) curve,  mAP={:4.2f}'.format(m_ap)
        ap_tools.plot_pr_curve(p, r, labels, title, f_path, show=show)

        # Plot R(T)-curves:
        f_path = os.path.join(path_2_results, 'r(t)__' + model_e)
        title = 'R(T) curve'
        ap_tools.plot_rt_curve(r, t, labels_, title, f_path, show=show)
        
        # Plot P(T)-curves:
        f_path = os.path.join(path_2_results, 'p(t)__' + model_e)
        title = 'P(T) curve'
        ap_tools.plot_pt_curve(p, t, labels_, title, f_path, show=show)

        len_trees = np.vstack(trees).shape[0]
        len_gt    = np.vstack(trees_gt).shape[0]

        print('')
        print('  mAP (after 2. NMS)              : {:4.2f}'.format(m_ap))
        print('  gt_trees                        : {:10d}'.format(len_gt))
        print('  detections                      : {:10d}'.format(len_trees))


    print('\n' + '='*79)
    print("Finished after", timer.time_string(), '\n')

    output = {}
    output['trees']    = trees
    output['probs']    = probs
    if display_accs:
        output['m_ap'] = m_ap
        output['ap'] = ap
    return output




###############################################################################
# Auxiliaries                                                                 #
#                                                                             #
###############################################################################


def _plot2D(array, colormap, title, fname, path_2_results):
    colormap_lookup = np.zeros((len(colormap), 3), dtype='long')
    for i, rgb in colormap.items():
        colormap_lookup[i] = rgb

    array_RGB = colormap_lookup[array.astype('long')]

    fpath = os.path.join(path_2_results, fname)
    img = PImage.fromarray(array_RGB.astype('uint8'), 'RGB')
    img.save(fpath + '.png')
