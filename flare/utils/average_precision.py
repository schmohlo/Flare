""" Functions for creating and plotting PR-Curves, AP, mAP.

by Stefan Schmohl, Johannes Ernst 2021
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

from .nms_iou import iou as iou_fu



def mean_ap(detcts, scores, gt, iou_thresholds, return_extras=False):
    """ Mean average precision over different IoU thresholds.

    Args:
        detcts:         predicted objects with a fixed geometry (circle, cylinder...)
        scores:         ndarray[N] score value for each prediction.
        gt:             ground truth objects
        iou_thresholds: List[m] of thresholds that define true positives.
    Returns:
        m_ap:           Mean average precision over all iou_thresholds.
        ap:             Average Precision per iou_threshold.
    """

    ap = []
    precision = []
    recall = []
    thresholds = []

    for t in iou_thresholds:
        ap_t, p_t, r_t, t_t = ap_pr(detcts, scores, gt, t)

        ap.append(ap_t)
        if return_extras:
            precision.append(p_t)
            recall.append(r_t)
            thresholds.append(t_t)

    m_ap = np.mean(ap).item()

    if return_extras:
        return m_ap, ap, precision, recall, thresholds
    else:
        return m_ap, ap




def ap_pr(detcts, scores, gt, iou_threshold, plot=False):
    """ Average precision (interpolated) & precision-recall-curve.

    Calculation of P(R)-curve follows algorithm from Pascal VOC 2012 devkit's,
    'VOCevaldet.m'  @ http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit

    Args:
        detcts:     list(ndarray[N, P]) len=S, one list item per sample.
        scores:     list(ndarray[N])    len=S, i.e. confidence.
        gt:         list(ndarray[M, P]) len=S, same format as detcts.

    """
    detcts_org = detcts

    if not isinstance(detcts, list):
        detcts = [detcts]
    if not isinstance(scores, list):
        scores = [scores]
    if not isinstance(gt, list):
        gt = [gt]
    # Squeeze to 1D, but prevent 0 dims in case only one element in 1D:
    scores = [s.squeeze() if len(s.shape) > 1 else s for s in scores]

    assert len(detcts) == len(scores) == len(gt)

    # IoU functions expect float32 input:
    # Also, remove samples with 0 detections (causes error in np.concatenate)
    detcts = [c.astype('float32') for c in detcts if np.prod(c.shape) != 0]
    gt     = [g.astype('float32') for g in gt]

    if len(detcts) == 0:
        return 0.0, np.asarray([]), np.asarray([]), np.asarray([])

    ## Convert detections into single array. Keep track of original sample id:
    ids = [i * np.ones(x.shape[0], dtype=int) for i, x in enumerate(scores)]
    ids = np.concatenate(ids)
    try:
        detcts = np.concatenate(detcts)
    except:
        breakpoint()
        pass
    scores  = np.concatenate(scores)

    ## Sort by decreasing confidence scores:
    ind_sort   = np.argsort(-scores)
    detcts     = detcts[ind_sort]
    ids        = ids[ind_sort]
    thresholds = scores[ind_sort] # i.e. the thresholds one would use to filter
                                  # out detections based on score value.

    ## Assign detections to ground truth objects & count tp / fp at threshold:
    tp = np.zeros(len(detcts))
    fp = np.zeros(len(detcts))
    has_match = [[False] * len(g) for g in gt]

    for i, detection in enumerate(detcts):

        # id = sample id to look for gts
        id = ids[i]

        # In case no gt in sample: (works both for torch.tensor & np.ndarray)
        if np.prod(gt[id].shape) == 0:
            fp[i] = 1
            continue

        # Find gt with highest overlap = iou to current prediction:
        iou = iou_fu(detection[None, :], gt[id])
        iou = iou.squeeze(axis=0)
        gt_ind = np.argmax(iou)
        iou = iou[gt_ind]

        # Assign detection as true positive / false positive:
        if iou >= iou_threshold:
            if not has_match[id][gt_ind]:
                tp[i] = 1                 # true positive
                has_match[id][gt_ind] = True
            else:
                fp[i] = 1                 # false positive (multiple detection)
        else:
            fp[i] = 1                     # false positive

    ## Compute precision/recall & average precision:
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    ngt = sum([len(g) for g in gt])
    #==========================================================================
    # Padding with 0 & 1 leads to over estimation of mAP
    # recall    = np.r_[0, tp / ngt, 1]
    # precision = np.r_[1, tp / (fp + tp), 0]
    #==========================================================================
    recall     = np.r_[tp / ngt]
    precision  = np.r_[tp / (fp + tp)]
    recall     = np.r_[0, recall, recall[-1]]
    precision  = np.r_[precision[0], precision, 0]
    thresholds = np.r_[thresholds[0], thresholds, thresholds[-1]]
    ap         = ap_interp(precision, recall)

    if plot:
        label = 'IoU={:.2f} (AP={:.2f})'.format(iou_threshold, ap)
        plot_pr_curve([precision], [recall], [label], title='P(R) curve')
        label = 'IoU={:.2f}'.format(iou_threshold)
        plot_rt_curve([recall], [thresholds], [label], title='R(T) curve')
    return ap, precision, recall, thresholds




def ap_interp(precision, recall):
    """ Interpolated average precision (AP) = area under curve (AUC)

    Calculation of P(R)-curve follows algorithm from Pascal VOC 2012 devkit's
    'VOCap.m'  @ http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit
    """

    prec = precision.copy()
    for i in range(len(precision)-2, 0, -1):
        prec[i] = max(prec[i], prec[i+1])

    ap = np.sum(np.diff(recall) * prec[:-1])
    return ap




def plot_pr_curve(precision, recall, labels, title='', f_path=None, show=True):
    """ Plot n P(R) curves into one graph.

    Args:
        precision:  List(ndarray[m]), len() = n.
        recall:     List(ndarray[m]), len() = n.
        labels:     List(str), len() = n.
        title:      Title of figure.
        f_path:     Path to save figures at.
        show:       Whether or not to show (alternative would be only to save).
    """

    assert len(precision) == len(recall) == len(labels)

    fig, ax = plt.subplots()

    for i in range(len(precision)):
        x = recall[i]
        y = precision[i]
        ax.plot(x, y, label=labels[i])

    ax.set(xlabel='recall', ylabel='precision', title=title)
    ax.legend(loc='lower left')
    plt.xticks(np.arange(0, 1.1, step=0.1))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax.grid(True)
    ax.set_aspect('equal')
    
    ax.grid(True)

    if f_path is not None:
        fig.savefig(f_path + '.png', dpi=600)
        fig.savefig(f_path + '.pdf', dpi=600)
        f = open(f_path + '.pkl', 'wb')
        data_to_save = {'precision': precision, 'recall':    recall,
                        'labels':    labels,    'title':     title}
        pickle.dump(data_to_save, f)
        f.close()

    if show:
        plt.show(block = False)




def plot_rt_curve(recall, threshold, labels, title='', f_path=None, show=True):
    """ Plot n R(T) curves into one graph.

    Args:
        recall:     List(ndarray[m]), len() = n.
        threshold:  List(ndarray[m]), len() = n.
        labels:     List(str), len() = n.
        title:      Title of figure.
        f_path:     Path to save figures at.
        show:       Whether or not to show (alternative would be only to save).
    """

    assert len(recall) == len(threshold) == len(labels)

    fig, ax = plt.subplots()

    for i in range(len(threshold)):
        x = threshold[i]
        y = recall[i]
        ax.plot(x, y, label=labels[i])

    ax.set(xlabel='threshold', ylabel='recall', title=title)
    ax.legend(loc='lower left')
    ax.grid(True)
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    if f_path is not None:
        fig.savefig(f_path + '.png', dpi=600)
        fig.savefig(f_path + '.pdf', dpi=600)
        f = open(f_path + '.pkl', 'wb')
        data_to_save = {'threshold': threshold, 'recall':    recall,
                        'labels':    labels,    'title':     title}
        pickle.dump(data_to_save, f)
        f.close()

    if show:
        plt.show(block = False)



def plot_pt_curve(precision, threshold, labels, title='', f_path=None, show=True):
    """ Plot n P(T) curves into one graph.

    Args:
        precision   List(ndarray[m]), len() = n.
        threshold:  List(ndarray[m]), len() = n.
        labels:     List(str), len() = n.
        title:      Title of figure.
        f_path:     Path to save figures at.
        show:       Whether or not to show (alternative would be only to save).
    """

    assert len(precision) == len(threshold) == len(labels)

    fig, ax = plt.subplots()

    for i in range(len(threshold)):
        x = threshold[i][0:-1] # remove last element with y=0
        y = precision[i][0:-1] # remove last element with y=0
        ax.plot(x, y, label=labels[i])

    ax.set(xlabel='threshold', ylabel='precision', title=title)
    ax.legend(loc='lower left')
    ax.grid(True)
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    if f_path is not None:
        fig.savefig(f_path + '.png', dpi=600)
        fig.savefig(f_path + '.pdf', dpi=600)
        f = open(f_path + '.pkl', 'wb')
        data_to_save = {'threshold': threshold, 'precision': precision,
                        'labels':    labels,    'title':     title}
        pickle.dump(data_to_save, f)
        f.close()

    if show:
        plt.show(block = False)