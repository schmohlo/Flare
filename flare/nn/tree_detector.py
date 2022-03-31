""" TODO

by Stefan Schmohl, Johannes Ernst 2021
"""

from globals import *

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sparseconvnet as scn

from matplotlib.patches import Circle as mplcircle
from matplotlib.patches import Rectangle as mplrectangle
from matplotlib.patheffects import withStroke
from torch.nn.functional import avg_pool2d as avgp2


from .model import Model
from flare.nn._utils import bdetection_transform
from flare.nn._utils import bdetection_transform_inv
from flare.utils import config as cfg
from flare.utils import iou as iou_fu
from flare.utils import nms
from flare.utils import average_precision as ap_tools
from flare.utils.average_precision import plot_pr_curve
from flare.utils.average_precision import plot_rt_curve
from flare.utils import colormapping
from flare.utils import Timer
from flare.utils.metricsHistory import MetricsHistory
from flare.nn.lr_scheduler import LinearLR




class TreeDetector(Model):
    """ TODO
    """


    @classmethod
    def from_config(cls, config, net, optimizer, loss_fu, callbacks, device):
        instance = cls(
            net,
            config['model']['input_features'],
            optimizer,
            loss_fu,
            config['model']['geometry'],
            config['model']['anchors'],
            config['train']['iou_pos'],
            config['train']['iou_neg'],
            config['train']['nms']['iou_threshold'],
            config['train']['nms']['score_min'],
            config['train']['ap_iou_thresholds'],
            callbacks,
            device)
        return instance



    def __init__(self, net, input_features, optimizer, loss_fu,
                 geometry, anchors, iou_pos, iou_neg,
                 nms_iou_threshold, nms_score_min, ap_iou_thresholds,
                 callbacks=[], device=None, colormap2Dsem=None):
        """
        TODO:
        Args:
            net:            PyTorch Module = the network to train / test.
                            Already initalized (with previous state or rand)
            input_features: List of strings that define the names of the input
                            channels of the model.
            optimizer:      PyTorch optimizer for training. May be None for
                            testing.
            loss_fu:        PyTorch loss function for training and test. In
                            test needed for loss, if required, but in any case
                            for its ignore_index.
            callbacks:      List of callback objects for training. See
                            flare.nn.callbacks.
            device:         PyTorch device. 'cuda' or 'cpu'
        """

        super().__init__(net, input_features, device)

        ## Properties for training and evaluation:
        self.loss_fu = loss_fu.to(self.device)
        self.geometry          = geometry
        self.anchors           = torch.tensor(anchors)
        self.iou_pos           = iou_pos
        self.iou_neg           = iou_neg
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_score_min     = nms_score_min
        self.ap_iou_thresholds = ap_iou_thresholds

        # Properties used during training (class members for easier exchange
        # between methods):
        self.optimizer         = optimizer
        self.callbacks         = callbacks
        self.history           = MetricsHistory()
        self.stop_training     = False  # Triggered by EarlyStopping

        ## For Debugging:
        colormap = cfg.from_file('config_colormap__V3D_labels.json', PATH_2_CONFIGS)
        self.colormap2Dsem     = colormap = colormap['classes']
        self.debugging_epoch   = 2220


    ###########################################################################
    # Training method                                                         #
    #                                                                         #
    ###########################################################################


    def train(self, data_loader, val_data_loader=None, max_epochs=100,
              start_epoch=0, verbose_step=1):
        """ Train this network.

        Network state checkpoints will be stored according to callbacks.

        Args:
            data_loader:        PyTorch data_loader for training data.
            val_data_loader:    PyTorch data_loader for validation data.
            max_epochs:         Maximum number of epochs to train. Default=100.
            start_epoch:        Number of the first epoch. Used to abort
                                training with max_epochs and correct logs, e.g.
                                if training is resumed from previous state.
            verbose_step:       Number of mini-batches between console status
                                updates. 0 for none.
        Returns:
            history:            MetricsHistory object.
            time:               Training time in seconds.
        """

        if self.geometry == 'circles':
            losses_names = ['oloss', 'rloss',
                            'rloss_x', 'rloss_y',
                            'rloss_r']
        if self.geometry == 'rectangles':
            losses_names = ['oloss', 'rloss',
                            'rloss_x', 'rloss_y',
                            'rloss_r', 'rloss_w']
        if self.geometry == 'cylinders':
            losses_names = ['oloss', 'rloss',
                            'rloss_x', 'rloss_y', 'rloss_z',
                            'rloss_r', 'rloss_h']
        if self.geometry == 'cuboids':
            losses_names = ['oloss', 'rloss',
                            'rloss_x', 'rloss_y', 'rloss_z',
                            'rloss_l', 'rloss_w', 'rloss_h']


        history_entries = ['epoch', 'lr', 'running_loss', 'running_m_ap']
        if val_data_loader is not None:
            history_entries += ['val_loss', 'val_m_ap']
        history_entries += losses_names
        self.history = self.history.reindex(columns=history_entries)

        for cb in self.callbacks:
            cb.on_train_begin(self)
        self.stop_training = False  # Triggered by EarlyStopping

        lr_scheduler = None  # local lr_scheduler for warmup in first epoch.

        timer = Timer()

        for epoch in range(start_epoch, start_epoch + max_epochs):

            if epoch == 0:
                warmup_factor = 1.0 / 1000
                warmup_iters = min(1000, len(data_loader) - 1)
                lr_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=warmup_factor,
                    total_iters=warmup_iters
                )

            time_epoch = timer.time()
            # Init running metrics:
            loss_verbose = []
            loss_epoch   = []
            m_ap_verbose = []
            m_ap_epoch   = []
            losses_verbose = []
            losses_epoch   = []

            # In case a SCN is used. From the scn repo example for sem seg:
            scn.forward_pass_multiplyAdd_count = 0
            scn.forward_pass_hidden_states = 0

            for i_batch, batch in enumerate(data_loader, 0):

                ## Forward pass
                data = self._batch_to_device(batch['data'], data_loader)
                gt = [t.to(self.device) for t in batch['trees']]

                ## Remove z-coordinate if handling circles or rectangles
                if self.geometry == 'circles':
                    if list(gt[0].shape)[1] > 3:
                        gt = [t.data[:,[0,1,3]] for t in gt]
                elif self.geometry == 'rectangles':
                    if list(gt[0].shape)[1] > 4:
                        gt = [t.data[:,[0,1,3,4]] for t in gt]

                # Set training mode to true. Certain modules have different
                # behaviors in train/eval mode.
                self.net.train()
                # Zero gradients buffer to prevent accumulation:
                self.optimizer.zero_grad()
                # Forward pass: output = (locations, regression, objectness)
                outputs = self.net(data)


                ## Make output useable for loss function:
                temp = self.pos_neg_mining(outputs, gt, return_pos=True)
                #plot_debug = epoch >= self.debugging_epoch
                #if plot_debug:
                    #temp = self.pos_neg_mining(outputs, gt, return_pos=True)
                     #detcts_pos, obj_pos, iou_pos    = temp[5:8]
                #else:
                    #temp = self.pos_neg_mining(outputs, gt)

                res_p, res_g, obj_p, obj_g, obj_w = temp[0:5]
                detcts_pos, obj_pos, iou_pos      = temp[5:8]
                detcts_pos_gt                     = temp[8]
                iou_neg                           = temp[9]

                ### ich braucht die GT fürs weighting!!!!
                
                
                if self.loss_fu.obj_IoU_target:
                    obj_g_ = [torch.cat((p,g),0) for p,g in zip(iou_pos, iou_neg)]


                ## Loss and Parameter-Update
                losses = self.loss_fu(res_p, res_g, obj_p, obj_g, obj_w, True, detcts_pos_gt)
                loss = losses[0]
                losses = losses[1:]
                # Skip, if loss has no meaning, e.g. if no gts in mini-batch:
                if torch.isnan(loss) or loss.requires_grad == False:
                    continue

                # Backpropagation:
                loss.backward()
                # Parameter update:
                self.optimizer.step()


                ## Monitoring:
                _, _, _, m_ap, ap = self.test_batch(outputs, gt,
                                                    self.ap_iou_thresholds)
                m_ap_epoch.append(m_ap)
                loss_epoch.append(loss.item())
                losses_epoch.append([l.item() for l in losses])

                # Print the training status every some mini-batches:
                if verbose_step:
                    losses_verbose.append([l.item() for l in losses])
                    loss_verbose.append(loss.item())
                    m_ap_verbose.append(m_ap)
                    if i_batch % verbose_step == verbose_step - 1:
                        print('')
                        print('  Epoch %5d, %5d:   lr: %8.6f  m_ap: %.4f  loss: %.4f' %
                              (epoch, i_batch + 1, self.optimizer.param_groups[0]['lr'],
                               np.mean(m_ap_verbose).item(),
                               np.mean(loss_verbose).item()), end='')
                        for i in range(len(losses)):
                            print('  %s: %.4f' % (losses_names[i],
                                                  np.mean(losses_verbose,0)[i]),
                                                  end='')
                        loss_verbose = []
                        m_ap_verbose = []


                ## Debugging Plots:
                plot_debug = epoch >= self.debugging_epoch
                if plot_debug:
                    _, _, _, m_ap, ap, pre, rec, thr = self.test_batch(outputs, gt,
                                                                       self.ap_iou_thresholds,
                                                                       return_extras=True)
                    label_str = 'IoU={:.2f} (AP={:.2f})'
                    labels = [label_str.format(t, ap_t) for t, ap_t in zip(self.ap_iou_thresholds, ap)]
                    plot_pr_curve(pre, rec, labels, title='P(R) curve')

                    label_str = 'IoU={:.2f}'
                    labels = [label_str.format(t) for t in self.ap_iou_thresholds]
                    plot_rt_curve(rec, thr, labels, title='R(T) curve')

                    # Search for first sample with at least one gt tree in it:
                    # => this will be index=0 in outputs
                    s_i = 0
                    for j in range(len(batch['trees'])):
                        if batch['trees'][j].shape[0] != 0:
                            s_i = j
                            break

                    try:
                        self.plot_sample_in_training(batch['labels_2Dsem'][s_i],
                                                     batch['height'][s_i],
                                                     batch['trees'][s_i],
                                                     outputs[0].detach().cpu(),
                                                     outputs[1][s_i].detach().cpu(),
                                                     outputs[2][s_i].detach().cpu(),
                                                     detcts_pos[s_i], obj_pos[s_i], iou_pos[s_i])
                    except:
                        breakpoint()
                        pass

                if lr_scheduler is not None:
                    lr_scheduler.step()

            ## End and evaluate epoch:
            print('\n  ' + '-'*77)
            loss_epoch = np.mean(loss_epoch).item()
            m_ap_epoch = np.mean(m_ap_epoch).item()
            losses_epoch = np.mean(losses_epoch,0)
            losses_dict = {n:l for n, l in zip(losses_names, losses_epoch)}

            self._update_history(epoch, loss_epoch, m_ap_epoch, val_data_loader, losses_dict)
            self._verbose_epoch(epoch, timer.time() - time_epoch, timer.time())

            for cb in self.callbacks:
                cb.on_epoch_end(self)
            print('  ' + '-'*77)

            # Supposed to help against memory runout:
            torch.cuda.empty_cache()

            if self.stop_training:
                # Triggered by EarlyStopping callback
                break


        ## End Training:
        for cb in self.callbacks:
            cb.on_train_end(self)
        return self.history




    ###########################################################################
    # Inference methods                                                       #
    #                                                                         #
    ###########################################################################


    def test_batch(self, outputs, gt_trees=None, iou_thresholds=None,
                   return_extras=False, nan_locs=None):
        """ Tests TreeDetector on one of its single mini-batch outputs.

        Will return lists of detections & their probabilities, ground truth
        and loss, one item each per sample in the data_loader.

        Will return mean average precision and lists of overall average_
        precision, recall and precision, one item per iou-threshold.

        Usage:
            outputs = tree_detector.net(data)
            stuff   = tree_detector.test_batch(outputs, gt_trees)

        Args:
            output:         List[locations, objectness, regression] output of
                            TreeDetector for one minibatch.
                            TODO: Details
            gt_trees:       List[Tensor] of ground truth samples per tree.
                            If None, will not return loss or AP etc.
            return_extras:  Whether to return P(R) & score thresholds.
            iou_thresholds: List of iou thresholds for ap, recall & precision.
                            If None, will not return AP etc.
            nan_locs:       List[Tensor] of boolean maks to restrict
                            predictions to non-nan locations.

        Returns:
            detcts:         List[ndarray] of detected trees per sample.
            probs:          List[ndarray] of probabilities / objectness per detection.
            loss:           Overall loss (average of mini-batch). Returns None,
                            if gt_trees is None.
            m_ap:           Mean average precision per iou threshold.
            ap:             List of overall average precision per iou threshold.
            recall:         List of overall recall-curve per iou threshold.
            precision:      List of overall precision-curve per iou threshold.
            thresholds:     List of objectness thresholds / scores of P(R)-
                            curve per iou threshold.
        """
        probs = torch.sigmoid(outputs[2]).cpu().detach().numpy().squeeze()
        detcts, _ = self.output_to_geometry(outputs)

        if gt_trees is not None:
            temp = self.pos_neg_mining(outputs, gt_trees, return_pos=True)
            loss_input  = temp[0:5]
            circ_pos_gt = temp[8]
            loss_output = self.loss_fu(*loss_input, True, circ_pos_gt)
            loss = loss_output[0].detach().cpu().numpy()
        else:
            loss = None

        ## Non-Maximum-Supression for each sample in mini-batch:
        probs_   = []
        detcts_  = []

        if nan_locs is None:
            nan_locs = [None] * len(detcts) # ugly but training-compatible.

        for detct, prob, nl in zip(detcts, probs, nan_locs):
            if nl is not None:
                detct = detct[~nl]
                prob  = prob[~nl]

            detct = detct.cpu().numpy()
            nms_indices = nms(detct[:, :], prob,
                              self.nms_iou_threshold,
                              self.nms_score_min)

            detcts_.append(detct[nms_indices])
            probs_.append(prob[nms_indices])

        if iou_thresholds is None or gt_trees is None:
            return detcts_, probs_, loss

        # Extract object arrays from detection-tensors
        obj    = [t[:, :] for t in detcts_]
        obj_gt = [t[:, :].cpu().numpy() for t in gt_trees]
        # Returns (m_ap, ap). If return_extras: +(precision, recall, thresholds)
        map_ap__p_r_t  = ap_tools.mean_ap(obj, probs_, obj_gt, iou_thresholds, return_extras)

        return( detcts_, probs_, loss) + map_ap__p_r_t




    def test(self, data_loader, gt=False, loss=False, ap=False, iou_thresholds=[0.5]):
        """ Tests TreeDetector on data_loader.

        Will return lists of detections & their probabilities, ground truth
        and loss, one item each per sample in the data_loader.

        Will return lists of overall average_precision, recall and precision,
        one item per iou-threshold.

        Args:
            data_loader:    PyTorch data_loader for data.
            gt:             Whether to return ground truth trees per sample.
            loss:           Whether to return overall loss.
            ap:             Whether to return overall ap, P(R) & score thresholds.
            iou_thresholds: List of iou thresholds for ap, recall & precision.

        Returns:
            detcts:         List[ndarray] of detected trees per sample.
            probs:          List[ndarray] of probabilities / objectness per detection.
            gt:             List[ndarray] of ground truth trees per sample.
            loss:           Overall loss (average per mini-batch).
            m_ap:           Mean average precision per iou threshold.
            ap:             List of overall average precision per iou threshold.
            precision:      List of overall precision-curve per iou threshold.
            recall:         List of overall recall-curve per iou threshold.
            thresholds:     List of objectness thresholds / scores of P(R)-
                            curve per iou threshold.
        """

        if not isinstance(iou_thresholds, list):
            iou_thresholds = [iou_thresholds]

        detcts_    = []
        probs_    = []
        gt_trees_ = []
        loss_     = []

        # Set training mode to false = set to eval mode. Certain modules like
        # Dropout have different behaviors in training/evaluation mode.
        self.net.eval()
        with torch.no_grad():  # prevent gradient calculation

            for batch in data_loader:
                data = self._batch_to_device(batch['data'], data_loader)
                gt_trees = [t.to(self.device) for t in batch['trees']]

                ## Remove z-coordinate if handling circles or rectangles
                if self.geometry == 'circles':
                    if list(gt_trees[0].shape)[1] > 3:
                        gt_trees = [t.data[:,[0,1,3]] for t in gt_trees]
                elif self.geometry == 'rectangles':
                    if list(gt_trees[0].shape)[1] > 4:
                        gt_trees = [t.data[:,[0,1,3,4]] for t in gt_trees]

                outputs = self.net(data)
                # outputs[0]     # locations n_a*64*64
                # outputs[1]     # y_r nxn_a*64*64*5
                # outputs[2]     # y_o nxn_a*64*64*1

                # Restrict to locations non-empty in input:
                nan_locs = torch.isnan(batch['height']) # nx128x128
                n   = nan_locs.shape[0]
                n_a = outputs[0].shape[0] // (nan_locs.shape[1]//2)**2
                # Downsampling to output-resolution (/2) in favour of nan.
                nan_locs = nan_locs.to(torch.float)     # avgpool2d needs float
                nan_locs_d = avgp2(nan_locs,2,2)  # avgpool is only torch 2d downsmpl-fu
                nan_locs_d = nan_locs_d > 0
                # Stretching to n_a anchors
                nan_locs_d = nan_locs_d.reshape(n,-1)
                nan_locs_d = nan_locs_d.repeat_interleave(n_a, dim=1)

                t, p, l = self.test_batch(outputs, gt_trees, nan_locs=nan_locs_d)

                detcts_.extend(t)
                probs_.extend(p)
                if loss:
                    loss_.append(l)
                if gt or ap:
                    gt_trees_.extend([t.cpu().numpy() for t in gt_trees])

        output = (detcts_, probs_)

        if gt:
            output += (gt_trees_,)
        if loss:
            output += (np.nanmean(loss_),)

        if ap:
            # Extract object arrays from detection-tensors
            obj    = [t[:, :] for t in detcts_]
            obj_gt = [t[:, :] for t in gt_trees_]
            m_ap, ap, p, r, t = ap_tools.mean_ap(obj, probs_, obj_gt,
                                                 iou_thresholds,
                                                 return_extras=True)
            output += (m_ap, ap, p, r, t)

        return output



    def predict(self, data_loader):
        """ Shortcut for self.test without any from of evaluation.

        Args:
            data_loader:    PyTorch data_loader for data.
            gt:             Whether to return ground truth trees per sample.
        Returns:
            detcts:         List[ndarray] of detected trees per sample.
            probs:          List[ndarray] of probabilities / objectness per detection.
            gt:             List[ndarray] of ground truth trees per sample.
        """
        return self.test(data_loader, gt=False, loss=False, ap=False)



    def evaluate(self, data_loader, iou_thresholds):
        """ Shortcut for self.test for pure evaluation result.

        Args:
            data_loader:    PyTorch data_loader for data.
            iou_thresholds: List of iou thresholds for ap, recall & precision.

        Returns:
            loss:           Overall loss (average per mini-batch).
            ap:             List of overall average precision per iou threshold.
            precision:      List of overall precision-curve per iou threshold.
            recall:         List of overall recall-curve per iou threshold.
            thresholds:     List of objectness thresholds / scores of P(R)-
                            curve per iou threshold.
        """
        _, _, loss, m_ap, ap, p, r, t = self.test(data_loader,
                                                  gt=False,
                                                  loss=True,
                                                  ap=True,
                                                  iou_thresholds=iou_thresholds)
        return loss, m_ap, ap, p, r, t



    def post_training_eval(self, dataloaders, config, path2model, path2mcps,
                           label_maps, label_maps_inv, show=True):
        # Load last saved epoch (which is the best epoch if specified in config):
        self.load_net_from_checkpoint(path2mcps)

        test_metrics = {}
        iou_ts = config['train']['ap_iou_thresholds']

        for key, dataloader in dataloaders.items():

            labels = []
            labels_ = []

            detcts, _, gt, loss, m_ap, ap, p, r, t = self.test(dataloader,
                                                              gt=True,
                                                              loss=True,
                                                              ap=True,
                                                              iou_thresholds=iou_ts)
            test_metrics[key + '_mAP'] = m_ap
            test_metrics[key + '_loss'] = loss

            for a, iou_t in zip(ap, iou_ts):
                test_metrics[key + '_AP@{:.2f}'.format(iou_t)] = a
                labels.append('IoU={:.2f} (AP={:.2f})'.format(iou_t, a))
                labels_.append('IoU={:.2f}'.format(iou_t))

            # Plot P(R)-curves:
            f_path = os.path.join(path2model, 'p(r)__{}'.format(key))
            title = 'P(R) curve {},  mAP={:4.2f}'.format(key, m_ap)
            ap_tools.plot_pr_curve(p, r, labels, title, f_path, show=show)

            # Plot R(T)-curves:
            f_path = os.path.join(path2model, 'r(t)__{}'.format(key))
            title = 'R(T) curve {}'.format(key)
            ap_tools.plot_rt_curve(r, t, labels_, title, f_path, show=show)

            # Plot P(T)-curves:
            f_path = os.path.join(path2model, 'p(t)__{}'.format(key))
            title = 'P(T) curve {}'.format(key)
            ap_tools.plot_pt_curve(p, t, labels_, title, f_path, show=show)

            len_detcts = np.vstack(detcts).shape[0]
            len_gt     = np.vstack(gt).shape[0]

            print('  Overall {:5s} mAP on {:10d} gt_trees & {:10d} detections: {:4.2f}'.format(
                  key, len_gt, len_detcts, m_ap))

        return test_metrics


    ###########################################################################
    # Utility methods                                                         #
    #                                                                         #
    ###########################################################################



    def output_to_geometry(self, output):
        """ Combines differential regression outputs with their anchors.

        For each detection, the differential regression outputs are combined
        with its anchor for the absolute values. Each anchor consists of its 2D
        position x,y (given by the network ouput) and the default anchor values
        (e.g. for cylinders z,r & h).

        The supported geometries are:
        Circles     (x, y, radius)                      --> p = 3
        Rectangles  (x, y, length, width)               --> p = 4
        Cylinders   (x, y, z, radius, height)           --> p = 5
        Cuboids     (x, y, z, length, width, height)    --> p = 6

        It's assumed, that the detections per location for multiple anchors are
        sorted like: (3 default anchors in this case)
            [x0, y0], [x0, y0], [x0, y0], [x1, y1], [x1, y1], [x1, y1], ...

        For details, see flare.nn._utils.bdetection_transform_inv

        Args:
            output:         List[locations, regression, objectness] network
                            output of one mini-batch.
                            - locations = tensor[n,2]:
                                - All n 2D anchor locations.
                            - regression = List[tensor[n,5+]]:
                                - [dx, dy, dz, dr, dh, ...] (depending on geometry)
                                - One list item per sample in mini-batch.
                                - n anchors per sample (may differ from sample
                                  to sample). Must be a multiple of the default
                                  anchor shapes in self.anchors.
                            - objectness = List[tensor[n]]
                                - Unused.
        Returns:
            detections:     List[tensor[n, p]] per sample an item containing
                            n detections with p geometry parameters
                            (e.g. for cylinders p = 5 [x, y, z, r, h])
            anchors:        List[tensor[n, p]] per sample an item containing
                            n anchors in with p geometry parameters
                            (e.g. for cylinders p = 5 [x, y, z, r, h])
        """

        locations  = output[0]
        regression = output[1]

        ## Assemble full anchors [x,y,z,r,h] for each detection:
        # Default anchor sizes [z,r,h]:
        anchors = self.anchors.to(locations.device).float()
        # Repeat default anchors sizes to match their locations:
        assert locations.shape[0] % anchors.shape[0] == 0
        anchors = anchors.repeat(int(locations.shape[0] / anchors.shape[0]), 1)
        # Combine anchor locations [x,y] with their default sizes:
        anchors = torch.cat((locations.float(), anchors), 1)

        detections = []

        ## For each sample in mini-batch:
        for reg in regression:
            diff_detection = reg[:, 0:].detach()
            detection = bdetection_transform_inv(diff_detection, anchors, self.geometry)
            detections.append(detection)

        return detections, anchors



    def pos_neg_mining(self, output, targets, return_pos=False):
        """ TODO
        output = output of one mini-batch.
        """

        locations  = output[0]  # [tensor([[x0, y0], [x0, y0] ... [x1, y1] ...)]
        regression = output[1]  # [tensor([[dx dy dz dr dh, iou], ...])]
        objectness = output[2]  # [tensor([c])]

        iou_pos = self.iou_pos
        iou_neg = self.iou_neg

        detcts_all, anchors = self.output_to_geometry(output)

        res_p = []
        res_g = []
        obj_p = []
        obj_g = []
        obj_w = []

        detcts_pos_ = []
        detcts_pos_gt_ = []

        obj_pos_ = []
        iou_pos_ = []
        iou_neg_ = []

        ## For each sample in mini-batch:
        for detcts, reg, obj, detcts_gt in zip(detcts_all, regression, objectness, targets):

            # Skip in case of no tree gt in sample:
            if detcts_gt.shape[0] == 0:
                res_p.append(torch.tensor([], device=reg.device))
                res_g.append(torch.tensor([], device=reg.device))
                obj_p.append(torch.tensor([], device=reg.device))
                obj_g.append(torch.tensor([], device=reg.device))
                obj_w.append(torch.tensor([], device=reg.device))

                if return_pos:
                    detcts_pos_.append(torch.tensor([], device=reg.device))
                    detcts_pos_gt_.append(torch.tensor([], device=reg.device))
                    obj_pos_.append(torch.tensor([], device=reg.device))
                    iou_pos_.append(torch.tensor([], device=reg.device))
                continue

            ## Compute overlap between "detections" and ground truth:
            iou = iou_fu(detcts   [:, :].cpu().numpy(),
                         detcts_gt[:, :].cpu().numpy())
            iou = torch.tensor(iou, device=reg.device)

            ## Positive detections:
            mask_pos = torch.zeros_like(iou, dtype=torch.bool)

            # Positive = per target detection the prediction with highest iou:
            ind_pos = torch.argmax(iou, dim=0)
            mask_pos[ind_pos, torch.arange(0, len(ind_pos))] = True

            # Positive += all predictions with an iou > iou_true:
            mask_pos += iou >= iou_pos
            pos_pairs = mask_pos.nonzero()
            pos_pred  = pos_pairs[:, 0]
            pos_gt    = pos_pairs[:, 1]

            ## Negative detections:
            # Negative = all non-positive predictions without any iou > iou_false:
            mask_neg = ~mask_pos & (iou <= iou_neg)
            ind_neg = torch.sum(mask_neg, dim=1) == iou.shape[1]
            neg_pred = ind_neg.nonzero().squeeze()

            ## Compute residual vector of positive candidates for regression:
            res_gt = bdetection_transform(detcts_gt[pos_gt], anchors[pos_pred], self.geometry)
            res_pr = reg[pos_pred]

            ## Objectness prediction and target vector for pos and neg detections:
            obj_pos = obj[pos_pred]
            obj_neg = obj[neg_pred]
            obj_pos_weights = iou[mask_pos]
            obj_neg_weights = 1 - torch.mean(iou[ind_neg, :], 1)

            # In case no positive or negative detections, create empty for cat:
            if obj_neg.numel() == 0:
                obj_neg         = torch.Tensor()
                obj_neg_weights = torch.Tensor()
            if obj_pos.numel() == 0:
                obj_pos         = torch.Tensor()
                obj_pos_weights = torch.Tensor()

            obj_pr = torch.cat((obj_pos, obj_neg), 0).squeeze()
            obj_gt = torch.cat((torch.ones( obj_pos.shape),
                                torch.zeros(obj_neg.shape)), 0).squeeze()
            obj_gt = obj_gt.to(obj_pr.device)
            obj_weights = torch.cat((obj_pos_weights, obj_neg_weights), 0)

            res_p.append(res_pr)
            res_g.append(res_gt)
            obj_p.append(obj_pr)
            obj_g.append(obj_gt)
            obj_w.append(obj_weights)

            if return_pos:
                detcts_pos_.append(detcts[pos_pred])
                detcts_pos_gt_.append(detcts_gt[pos_gt])
                obj_pos_.append(obj_pos)
                iou_pos_.append(iou[mask_pos])
                iou_neg_.append(torch.max(iou[ind_neg, :], 1)[0])

        if return_pos:
            return res_p, res_g, obj_p, obj_g, obj_w, detcts_pos_, obj_pos_, iou_pos_, detcts_pos_gt_, iou_neg_
        else:
            return res_p, res_g, obj_p, obj_g, obj_w



    def _update_history(self, epoch, loss, m_ap, val_data_loader=None, other=None):
        """ Add metrics from current epoch to metrics history. """

        lr = self.optimizer.param_groups[0]['lr']
        new_entry = {'epoch': epoch,
                     'lr': lr,
                     'running_loss': loss,
                     'running_m_ap': m_ap}

        if val_data_loader is not None:
            l, a, _,_,_,_  = self.evaluate(val_data_loader, self.ap_iou_thresholds)
            new_entry.update({'val_loss': l, 'val_m_ap': a})

        if other is not None:
            new_entry.update(other)

        self.history = self.history.append(new_entry, ignore_index=True)


    ###########################################################################
    # Plots and Prints                                                        #
    #                                                                         #
    ###########################################################################



    def plot_sample_in_training(self, semGT2D, height_map, trees_gt,
                                locations, regression, objectness,
                                detcts_pos, obj_pos, iou_pos):

        ## Make stuff numpy:
        semGT2D    = semGT2D.long().numpy()
        height_map = height_map.numpy()
        trees_gt   = trees_gt.numpy()
        rad_reg    = regression[:,3].detach().cpu().numpy() # (CHANGE for cylinder)
        objectness = objectness.sigmoid().numpy().squeeze()


        ## Refine network output sample to final predictions:
        detcts, _  = self.output_to_geometry((locations, [regression], [objectness]))
        detcts = detcts[0].cpu().numpy()

        nms_indices = nms(detcts[:, :],
                          objectness,
                          self.nms_iou_threshold,
                          self.nms_score_min)
        final_pred_c = detcts[nms_indices, :]
        final_pred_o = objectness[nms_indices]

        ## Generate images to plot:
        input_shape = semGT2D.shape
        otput_shape = (int(np.sqrt(regression.shape[0] / self.anchors.shape[0])),)*2
        background  = np.zeros(input_shape, dtype='long')
        background  = colormapping(background, {0: [255]*3})
        semGT2D_RGB = colormapping(semGT2D, self.colormap2Dsem)

        xticks = range(0, input_shape[1], 10)
        yticks = range(0, input_shape[0], 10)
        xlim = (0, input_shape[1]-1)
        ylim = (0, input_shape[0]-1)


        ## Per-Anchor-Plots:
        detcts_per_anchor = detcts.reshape(otput_shape+(self.anchors.shape[0],-1))
        n_anchors = self.anchors.shape[0]

        ## Per-Anchor Objectness:
        objectness2D = objectness.reshape(otput_shape+(-1,))
        fig = plt.figure(1)

        for i, a in enumerate(self.anchors):
            ax = plt.subplot(2, n_anchors, i + 1)
            ax.set(title='Objectness (raw), anchor='+str([str(e.item()) for e in a]))
            plt.imshow(objectness2D[:,:,i])
            plt.grid(True)
            plt.clim(0,1)
            plt.colorbar()
            # plt.xticks(xticks)
            # plt.yticks(yticks)
            plt.xlim((0, otput_shape[0]-1))
            plt.ylim((0, otput_shape[1]-1))

            detcts_for_anchor = detcts_per_anchor[:,:,i,:].reshape([-1,5])
            objec_for_anchor = objectness2D[:,:,i].flatten()

            objectness2D_relocated = np.zeros(input_shape)
            for detct, obj in zip(detcts_for_anchor, objec_for_anchor):
                loc = tuple(detct[0:2].astype('long'))
                if loc[0] < 0 or loc[1] < 0:
                    continue
                if loc[0] >= input_shape[0] or loc[1] >= input_shape[1]:
                    continue
                old = objectness2D_relocated[loc]
                new = obj.squeeze()
                if new > old:
                    objectness2D_relocated[loc] = obj

            ax = plt.subplot(2, n_anchors, i + 1 + n_anchors)
            ax.set(title='Objectness (regressed locations)')
            plt.imshow(objectness2D_relocated)
            plt.grid(True)
            plt.clim(0,1)
            plt.colorbar()
            plt.xticks(xticks)
            plt.yticks(yticks)
            plt.xlim(xlim)
            plt.ylim(ylim)


        ## Per-Anchor Radius (ONLY WORKS FOR CIRCLES!!):
        rad_reg2D = rad_reg.reshape(otput_shape+(-1,))
        fig = plt.figure(2)

        for i, a in enumerate(self.anchors):
            ax = plt.subplot(2, n_anchors, i + 1)
            ax.set(title='Radius [-] (raw), anchor='+str([str(e.item()) for e in a]))
            plt.imshow(rad_reg2D[:,:,i])
            plt.grid(True)
            plt.clim(-1,1)
            plt.colorbar()
            # plt.xticks(xticks)
            # plt.yticks(yticks)
            plt.xlim((0, otput_shape[0]-1))
            plt.ylim((0, otput_shape[1]-1))

            detcts_for_anchor = detcts_per_anchor[:,:,i,:].reshape([-1,5])
            objec_for_anchor = objectness2D[:,:,i].flatten()

            objectness2D_relocated = np.zeros(input_shape)
            radius2D_relocated = np.zeros(input_shape)
            for detct, obj in zip(detcts_for_anchor, objec_for_anchor):
                loc = tuple(detct[0:2].astype('long'))
                if loc[0] < 0 or loc[1] < 0:
                    continue
                if loc[0] >= input_shape[0] or loc[1] >= input_shape[1]:
                    continue
                old = objectness2D_relocated[loc]
                new = obj.squeeze()
                if new > old:
                    objectness2D_relocated[loc] = obj
                    radius2D_relocated[loc] = detct[3]

            ax = plt.subplot(2, n_anchors, i + 1 + n_anchors)
            ax.set(title='Radius [m/vs] (regressed locations)')
            plt.imshow(radius2D_relocated)
            plt.grid(True)
            plt.clim(0,10)
            plt.colorbar()
            plt.xticks(xticks)
            plt.yticks(yticks)
            plt.xlim(xlim)
            plt.ylim(ylim)


        ## Plotting
        fig = plt.figure(3)

        ## Plotting GT
        ax = plt.subplot(3, 3, 1)
        ax.set(title='2D Sem GT')
        plt.imshow(semGT2D_RGB)
        plt.grid(True)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.xlim(xlim)
        plt.ylim(ylim)

        ax = plt.subplot(3, 3, 4)
        ax.set(title='2D Height Map (DSM)')
        plt.imshow(height_map)
        plt.grid(True)
        plt.colorbar(label='m / voxel_size')
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.xlim(xlim)
        plt.ylim(ylim)

        ax = plt.subplot(3, 3, 7)
        ax.set(title='2D Tree GT')
        plt.imshow(background)
        plt.grid(True)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.xlim(xlim)
        plt.ylim(ylim)
        if self.geometry == "circles":
            for tree in trees_gt:
                ax.add_artist(mplcircle((tree[1], tree[0]), radius=tree[2],
                           edgecolor='green', linewidth=3, facecolor=(0, 1, 0, .125),
                           path_effects=[withStroke(linewidth=5, foreground='w')]))
        elif self.geometry == "rectangles":
            for tree in trees_gt:
                ax.add_artist(mplrectangle((tree[1]-tree[3]/2, tree[0]-tree[2]/2),
                              tree[3], tree[2], edgecolor='green', linewidth=3,
                              facecolor=(0, 1, 0, .125),
                              path_effects=[withStroke(linewidth=5, foreground='w')]))
        elif self.geometry == "cylinders":
            for tree in trees_gt:
                ax.add_artist(mplcircle((tree[1], tree[0]), radius=tree[3],
                           edgecolor='green', linewidth=3, facecolor=(0, 1, 0, .125),
                           path_effects=[withStroke(linewidth=5, foreground='w')]))
        elif self.geometry == "cuboids":
            for tree in trees_gt:
                ax.add_artist(mplrectangle((tree[1]-tree[4]/2, tree[0]-tree[3]/2),
                              tree[4], tree[3], edgecolor='green', linewidth=3,
                              facecolor=(0, 1, 0, .125),
                              path_effects=[withStroke(linewidth=5, foreground='w')]))


        ## Plotting Predictions
        ax = plt.subplot(3, 3, 5)
        ax.set(title='Positive Predictions (IoU)')
        plt.imshow(background)
        plt.grid(True)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.xlim(xlim)
        plt.ylim(ylim)
        if self.geometry == "circles":
            for detct, iou in zip(detcts_pos, iou_pos):
                ax.add_artist(mplcircle((detct[1], detct[0]), radius=detct[2],
                           edgecolor='green', linewidth=3, facecolor=(0, 1, 0, .125),
                           path_effects=[withStroke(linewidth=5, foreground='w')]))
            text = '{:.2f}'.format(iou.item())
            ax.text(detct[1], detct[0], text, ha='center', va='center')
        elif self.geometry == "rectangles":
            for detct, iou in zip(detcts_pos, iou_pos):
                ax.add_artist(mplrectangle((detct[1]-detct[3]/2, detct[0]-detct[2]/2),
                              detct[3], detct[2], edgecolor='green', linewidth=3,
                              facecolor=(0, 1, 0, .125),
                              path_effects=[withStroke(linewidth=5, foreground='w')]))
            text = '{:.2f}'.format(iou.item())
            ax.text(detct[1], detct[0], text, ha='center', va='center')
        elif self.geometry == "cylinders":
            for detct, iou in zip(detcts_pos, iou_pos):
                ax.add_artist(mplcircle((detct[1], detct[0]), radius=detct[3],
                           edgecolor='green', linewidth=3, facecolor=(0, 1, 0, .125),
                           path_effects=[withStroke(linewidth=5, foreground='w')]))
            text = '{:.2f}'.format(iou.item())
            ax.text(detct[1], detct[0], text, ha='center', va='center')
        elif self.geometry == "cuboids":
            for detct, iou in zip(detcts_pos, iou_pos):
                ax.add_artist(mplrectangle((detct[1]-detct[4]/2, detct[0]-detct[3]/2),
                              detct[4], detct[3], edgecolor='green', linewidth=3,
                              facecolor=(0, 1, 0, .125),
                              path_effects=[withStroke(linewidth=5, foreground='w')]))
            text = '{:.2f}'.format(iou.item())
            ax.text(detct[1], detct[0], text, ha='center', va='center')

        ax = plt.subplot(3, 3, 8)
        ax.set(title='Positive Predictions (objectness)')
        plt.imshow(background)
        plt.grid(True)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.xlim(xlim)
        plt.ylim(ylim)
        if self.geometry == "circles":
            for detct, score in zip(detcts_pos, obj_pos):
                ax.add_artist(mplcircle((detct[1], detct[0]), radius=detct[2],
                           edgecolor='green', linewidth=3, facecolor=(0, 1, 0, .125),
                           path_effects=[withStroke(linewidth=5, foreground='w')]))
            text = '{:.2f}'.format(torch.sigmoid(score).item())
            ax.text(detct[1], detct[0], text, ha='center', va='center')
        elif self.geometry == "rectangles":
            for detct, score in zip(detcts_pos, obj_pos):
                ax.add_artist(mplrectangle((detct[1]-detct[3]/2, detct[0]-detct[2]/2),
                              detct[3], detct[2], edgecolor='green', linewidth=3,
                              facecolor=(0, 1, 0, .125),
                              path_effects=[withStroke(linewidth=5, foreground='w')]))
            text = '{:.2f}'.format(torch.sigmoid(score).item())
            ax.text(detct[1], detct[0], text, ha='center', va='center')
        elif self.geometry == "cylinders":
            for detct, score in zip(detcts_pos, obj_pos):
                ax.add_artist(mplcircle((detct[1], detct[0]), radius=detct[3],
                           edgecolor='green', linewidth=3, facecolor=(0, 1, 0, .125),
                           path_effects=[withStroke(linewidth=5, foreground='w')]))
            text = '{:.2f}'.format(torch.sigmoid(score).item())
            ax.text(detct[1], detct[0], text, ha='center', va='center')
        elif self.geometry == "cuboids":
            for detct, score in zip(detcts_pos, obj_pos):
                ax.add_artist(mplrectangle((detct[1]-detct[4]/2, detct[0]-detct[3]/2),
                              detct[4], detct[3], edgecolor='green', linewidth=3,
                              facecolor=(0, 1, 0, .125),
                              path_effects=[withStroke(linewidth=5, foreground='w')]))
            text = '{:.2f}'.format(torch.sigmoid(score).item())
            ax.text(detct[1], detct[0], text, ha='center', va='center')

        ## Plotting Results:
        ax = plt.subplot(3, 3, 9)
        ax.set(title='Final Predictions (score)')
        plt.imshow(background)
        plt.grid(True)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.xlim(xlim)
        plt.ylim(ylim)
        if self.geometry == "circles":
            for detct, score in zip(final_pred_c, final_pred_o):
                ax.add_artist(mplcircle((detct[1], detct[0]), radius=detct[2],
                           edgecolor='green', linewidth=3, facecolor=(0, 1, 0, .125),
                           path_effects=[withStroke(linewidth=5, foreground='w')]))
            text = '{:.2f}'.format(score.item())
            ax.text(detct[1], detct[0], text, ha='center', va='center')
        elif self.geometry == "rectangles":
            for detct, score in zip(final_pred_c, final_pred_o):
                ax.add_artist(mplrectangle((detct[1]-detct[3]/2, detct[0]-detct[2]/2),
                              detct[3], detct[2], edgecolor='green', linewidth=3,
                              facecolor=(0, 1, 0, .125),
                              path_effects=[withStroke(linewidth=5, foreground='w')]))
            text = '{:.2f}'.format(score.item())
            ax.text(detct[1], detct[0], text, ha='center', va='center')
        elif self.geometry == "cylinders":
            for detct, score in zip(final_pred_c, final_pred_o):
                ax.add_artist(mplcircle((detct[1], detct[0]), radius=detct[3],
                           edgecolor='green', linewidth=3, facecolor=(0, 1, 0, .125),
                           path_effects=[withStroke(linewidth=5, foreground='w')]))
            text = '{:.2f}'.format(score.item())
            ax.text(detct[1], detct[0], text, ha='center', va='center')
        elif self.geometry == "cuboids":
            for detct, score in zip(final_pred_c, final_pred_o):
                ax.add_artist(mplrectangle((detct[1]-detct[4]/2, detct[0]-detct[3]/2),
                              detct[4], detct[3], edgecolor='green', linewidth=3,
                              facecolor=(0, 1, 0, .125),
                              path_effects=[withStroke(linewidth=5, foreground='w')]))
            text = '{:.2f}'.format(score.item())
            ax.text(detct[1], detct[0], text, ha='center', va='center')

        plt.show()
