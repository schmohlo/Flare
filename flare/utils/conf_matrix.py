""" Functions for creating and dealing with cunfion matrices.

by Stefan Schmohl, 2020
"""

import os
import math
import warnings
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import coo_matrix



def create(ground_truth, prediction, labels_gt=None, labels_pr=None):
    """ Creates confusion matrix.

    Other than the sklearn equivalent, it is slithly faster and allows the
    creation of unbalanced matrizes.

    Labels contain the class labels to look for. If given, matrix contains
    entries also for those classes not appearing in the given data. Also speeds
    up runtime of this function a bit.

    Args:
        ground_truth:   ndarray(int) of ground truth labels of size n.
        prediction:     ndarray(int) of prediction labels of size n.
        labels_gt:      list(int) classes labels of ground truth.
        labels_pr:      list(int) class labels of prediction.

    Returns:
        confusion_matrix:  ndarray(int), size = len(labels_gt) x len(labels_pr)
    """


    def make_continous(data, labels):
        if labels != list(range(0, max(labels)+1)):
            label_map = np.full(max(labels)+1, np.nan)
            label_map[labels] = np.arange(0, len(labels))
            data = label_map[data]
        return data

    gt = np.asarray(ground_truth).copy().flatten()
    pr = np.asarray(prediction).copy().flatten()

    if labels_gt is None:
        labels_gt = np.unique(ground_truth).tolist()
    else:
        # Sometimes random label outliers. Maybe memory fault???
        o_i = np.isin(ground_truth, labels_gt, invert=True)
        if np.sum(o_i) > 0:
            breakpoint()
            warnings.warn("  Invalid label in ground_truth in CM creation")
            ground_truth[o_i] = labels_gt[0]

    if labels_pr is None:
        labels_pr = np.unique(prediction).tolist()

    gt = make_continous(gt, labels_gt)
    pr = make_continous(pr, labels_pr)

    try:
        cm = coo_matrix((np.ones(gt.shape[0], dtype=int), (gt, pr)),
                        shape=(len(labels_gt), len(labels_pr)),
                        dtype=np.int64).toarray()
    except:
        gt_min = np.min(gt)
        gt_max = np.max(gt)
        gt_unique = np.unique(gt)
        
        pr_min = np.min(pr)
        pr_max = np.max(pr)
        pr_unique = np.unique(pr)
        
        breakpoint()
        a=1
    
    return cm



def analize(cm, labels_gt=None, labels_pr=None, label_map=None,
            detailed=False, ignore_labels=None, return_iou=False):
    """ Returns various metrics calculated from a confusion matrix.

    Allows conf matrix to be unsymmetric (more classes in ground truth than in
    prediction). In that case, lists of labels for ground truth and predictions
    have to be given, together with a mapping dict that maps gt classes to pr
    classes, This mapping tells, which gt classes are to be considered correct
    for a given pr class.

    Args:
        cm:             NxM consuion matrix
        labels_gt:      Nx list of class labels in ground truth (optional).
        labels_pr:      Mx list of class labels in prediction (optional).
        label_map:      Nx dict{gt_label:pr_label} (optional)
        details:        Whether to return also some interim results (optional).
        ignore_labels:  List of ground truth labels to ignore in precision and
                        acc. Requires also labels_gt as input.

    Returns:
        acc:        Overall accuracy.
        recall:     Nx list of recall values (metric of ground truth).
        precision:  Mx list of precision values (metric of prediction).
        f1:         Nx list of f1 values (metric of prediction).
        ...
    """

    # TODO: return dict, drop "detailed" input arg.

    def save_divide(a, b):
        return np.divide(a, b, out=np.zeros_like(b), where=b!=0)

    if labels_pr is None:
        matches_mask = np.eye(cm.shape[0], dtype=bool)
        if ignore_labels is not None:
            labels_pr = labels_gt
            label_map = {l:l for l in labels_gt}
    else:
        if cm.shape[0] != len(labels_gt) or cm.shape[1] != len(labels_pr):
            raise ValueError('conf_marix shape does not match classes!')
        if len(label_map) != len(labels_gt):
            raise ValueError('Lenghts of classes and label_map do not match!')

        matches_mask = np.full((len(labels_gt), len(labels_pr)), False)
        for g, p in label_map.items():
            i = labels_gt.index(g)
            # allow multiple prediction classes per gt class
            if not isinstance(p, list):
                p = [p]
            for p_ in p:
                j = labels_pr.index(p_)
                matches_mask[i,j] = True

    ignore_mask = np.zeros(cm.shape, dtype=bool)
    ignore_mask_pr = np.zeros(cm.shape, dtype=bool)
    if ignore_labels is not None:
        for i in ignore_labels:
            ind = labels_gt.index(i)
            ignore_mask[ind, :] = True
            # But do not ignore them in their own prediction class:
            ind_pr = labels_pr.index(label_map[i])
            ignore_mask_pr = ignore_mask.copy()
            ignore_mask_pr[ind, ind_pr] = False

    sum_pr_correct = np.sum(cm * matches_mask, 0)
    sum_gt_correct = np.sum(cm * matches_mask, 1)
    sum_correct = np.sum(cm * matches_mask * ~ignore_mask)

    sum_pr_all = np.sum(cm * ~ignore_mask_pr, axis=0, dtype=float)
    sum_gt_all = np.sum(cm, axis=1, dtype=float)
    sum_all = np.sum(cm * ~ignore_mask)

    precision = save_divide(sum_pr_correct, sum_pr_all).squeeze()
    recall    = save_divide(sum_gt_correct, sum_gt_all).squeeze()
    acc = float(sum_correct / sum_all)

    sum_gt_all_pr = np.zeros_like(precision)
    for i in range(len(precision)):
        sum_gt_all_pr[i]  = np.sum(sum_gt_all[matches_mask[:,i]])

    f1_pr  = np.zeros_like(precision)
    iou_pr = np.zeros_like(precision)
    for i in range(len(f1_pr)):
        tp = sum_pr_correct[i]
        fp = sum_pr_all[i] - sum_pr_correct[i]
        fn = (np.sum(sum_gt_all[matches_mask[:,i]]) -
              np.sum(sum_gt_correct[matches_mask[:,i]]))
        if (sum_pr_all[i] == 0) or (np.sum(sum_gt_all[matches_mask[:,i]]) == 0):
            f1_pr[i]  = np.nan
            iou_pr[i] = np.nan
        else:
            f1_pr[i]  = save_divide(tp, tp + 0.5 * (fp + fn))
            iou_pr[i] = save_divide(tp, tp + fp + fn)

    f1_gt = np.zeros_like(recall)
    for i in range(len(f1_gt)):
        tp = sum_gt_correct[i]
        fp = sum_gt_all[i] - sum_gt_correct[i]
        fn = (np.sum(sum_pr_all[matches_mask[i,:]]) -
              np.sum(sum_pr_correct[matches_mask[i,:]]))
        if (sum_gt_all[i] == 0) or (np.sum(sum_pr_all[matches_mask[i,:]]) == 0):
            f1_gt[i] = np.nan
        else:
            f1_gt[i] = save_divide(tp, tp + 0.5 * (fp + fn))

    f1 = f1_pr
    iou = iou_pr

    precision[sum_pr_all==0] = np.nan
    recall[sum_gt_all==0] = np.nan
    if ignore_labels is not None:
        recall[np.sum(ignore_mask,1) == ignore_mask.shape[1]] = np.nan

    if detailed:
        if return_iou:
            return acc, recall, precision, f1, \
               sum_correct, sum_gt_correct, sum_pr_correct, \
               sum_gt_all, sum_pr_all, \
               matches_mask, sum_gt_all_pr, f1_pr, f1_gt, iou
        else:
            return acc, recall, precision, f1, \
                   sum_correct, sum_gt_correct, sum_pr_correct, \
                   sum_gt_all, sum_pr_all, \
                   matches_mask, sum_gt_all_pr, f1_pr, f1_gt
    else:
        return acc, recall, precision, f1




def plot(cm, classes, path='.', file_suffix='', rel_vals=True,
         abs_vals=True, rel_precision=1, abs_max=99999, F1=True,
         class_freq=True, classes_pred=None, label_map=None,
         ignore_labels=None, show=True, iou=False):
    """ Plots a confusion matrix and saves to png and pdf.

    Allows conf matrix to be unsymmetric (more classes in ground truth than in
    prediction). In that case, a additional list of labels in the predictions
    have to be given, together with a mapping dict that maps gt classes to pr
    classes, This mapping tells, which gt classes are to be considered correct
    for a given pr class.

    For the effects of most parameters, see the examples in this repositories
    test directory.

    Args:
        cm:             NxM confusion matrix
        classes:        {label:name} for (gt) clases. Also for pred if symm.
        path:           Path to save results.
        file_suffix:    File name suffix for unique naming.
        rel_vals:       Whether to plot relative occrances.
        abs_vals:       Whether to plot absolute occrances.
        rel_precision:  Decimal places of relative valiues.
        abs_max:        Maximum absolute value to plot.
        F1:             Whether to plot row and column for F1 scores.
        class_freq:     Whether to plot the class occurances in the data.
        classes_pred:   {label:name} for prediction clases.
        label_map:      Nx dict{gt_label:pr_label} (optional).
        ignore_labels:  List of ground truth labels to ignore in precision and
                        acc. Requires also labels_gt as input.
    """

    def save_divide(a, b):
        return np.divide(a, b, out=np.zeros_like(b), where=b!=0)

    red   = np.asarray([220,  60,  30]) / 255
    green = np.asarray([ 30, 180,  60]) / 255
    white = np.asarray([255, 255, 255]) / 255


    if ignore_labels is not None:
        classes = classes.copy()
        for i in ignore_labels:
            classes[i] = classes[i] + '*'

    class_names = list(classes.values())
    class_labels = list(classes.keys())

    symmetric = True
    if isinstance(classes_pred, dict) and isinstance(label_map, dict):
        if not classes_pred == classes:
            symmetric = False
            class_names_pred = list(classes_pred.values())
            class_labels_pred = list(classes_pred.keys())


    ## Calc user, producer and overall accuracy:
    if symmetric:
        analysis = analize(cm, detailed=True, labels_gt=class_labels,
                           ignore_labels=ignore_labels, return_iou=True)
    else:
        analysis = analize(cm,
                           labels_gt=class_labels,
                           labels_pr=class_labels_pred,
                           label_map=label_map, detailed=True,
                           ignore_labels=ignore_labels, return_iou=True)
    acc            = analysis[0]
    recall         = analysis[1][:, None]
    precision      = analysis[2][None, :]
    f1_pr          = analysis[11][None, :]
    f1_gt          = analysis[12][:, None]
    sum_correct    = analysis[4]
    sum_gt_correct = analysis[5][:, None]
    sum_pr_correct = analysis[6][None, :]
    sum_gt_all     = analysis[7].squeeze()
    sum_pr_all     = analysis[8].squeeze()
    matches_mask   = analysis[9]
    iou_per_class  = analysis[13][None, :]

    sum_all = np.sum(cm)
    sum_gt_all_rel = sum_gt_all / sum_all
    sum_pr_all_rel = sum_pr_all/ sum_all

    # Concatenating accuracies to the confusion matrix:
    cm_rel = save_divide(cm, np.repeat(sum_gt_all[:,None], cm.shape[1], axis=1))
    cm_abs = np.hstack((cm,     sum_gt_correct))
    cm_rel = np.hstack((cm_rel, recall))
    t_abs = np.hstack((sum_pr_correct, np.asarray(sum_correct)[None, None]))
    t_rel = np.hstack((precision,      np.asarray(acc)[None, None]))
    cm_abs = np.vstack((cm_abs, t_abs))
    cm_rel = np.vstack((cm_rel, t_rel))


    ## Color:
    cm_rgb = np.ones((cm_rel.shape + (4,)))
    cm_rgb[:, :, 3] = cm_rel     # alpha
    cm_rgb[:, :, 0:3] = red      # everything
    # Append precision row & recall coloumn:
    m = np.vstack((matches_mask, np.zeros((1, matches_mask.shape[1] ))))
    m = np.hstack((m, np.zeros( (m.shape[0], 1) ) )).astype(bool)
    cm_rgb[m, 0:3] = green           # correct
    cm_rgb[ -1,   :, 0:3] = green    # precision
    cm_rgb[  :,  -1, 0:3] = green    # recall
    if ignore_labels is not None:
        for i, cl in enumerate(class_labels):
            if cl in ignore_labels:
                cm_rgb[i, :, 0:3] = white    # ignore gt classes


    ## Tick-Labels:
    fstring = '{{}}  ({{}}) \n {{:,.0f}} / {{:.{:d}f}}%'.format(rel_precision)
    if symmetric:
        x_labels = ['({})'.format(cl) for cl in class_labels]
    else:
        if class_freq:
            x_labels = [fstring.format(cn, cl, nb, nbr * 100)
                        for cn, cl, nb, nbr in
                        zip(class_names_pred, class_labels_pred,
                        sum_pr_all.tolist(), sum_pr_all_rel.tolist())]
        else:
            x_labels = ['{} ({})'.format(cn, cl) for cn, cl in
                        zip(class_names_pred, class_labels_pred)]
    if class_freq:
        y_labels = [fstring.format(cn, cl, nb, nbr * 100)
                    for cn, cl, nb, nbr in
                    zip(class_names, class_labels,
                    sum_gt_all.tolist(), sum_gt_all_rel.tolist())]
    else:
        y_labels = ['{} ({})'.format(cn, cl) for cn, cl in
                    zip(class_names, class_labels)]
    x_labels += ['recall']
    y_labels += ['precision']


    F1_gt = False
    if F1:
        f1_ = np.hstack((f1_pr, np.asarray([[0]])))
        cm_abs = np.vstack((cm_abs, np.ones(f1_.shape, dtype=int)))
        cm_rel = np.vstack((cm_rel, f1_))
        cm_rgb = np.vstack((cm_rgb, np.ones((f1_.shape + (4,)))))
        cm_rgb[-1, :, 3] = f1_
        cm_rgb[-1, :, 0:3] = green
        f1_label = 'F1 \n '+r'$\mu$'+'={{:.{:d}f}}%'.format(rel_precision)
        y_labels += [f1_label.format(np.nanmean(f1_pr)*100)]

        F1_gt = (f1_gt.size != f1_pr.size) or (not symmetric and not np.allclose(f1_gt.squeeze(), f1_pr.squeeze(), equal_nan=True))

        if F1_gt:
            f1_ = np.vstack((f1_gt, np.asarray([[0],[0]])))
            cm_abs = np.hstack((cm_abs, np.ones(f1_.shape, dtype=int)))
            cm_rel = np.hstack((cm_rel, f1_))
            cm_rgb = np.hstack((cm_rgb, np.ones((f1_.shape + (4,)))))
            cm_rgb[:, -1, 3] = f1_.squeeze()
            cm_rgb[:, -1, 0:3] = green
            f1_label = 'F1 \n '+r'$\mu$'+'={{:.{:d}f}}%'.format(rel_precision)
            x_labels += [f1_label.format(np.nanmean(f1_gt)*100)]

    if iou:
        iou_ = np.hstack((iou_per_class, np.asarray([[0]])))
        cm_abs = np.vstack((cm_abs, np.ones(iou_.shape, dtype=int)))
        cm_rel = np.vstack((cm_rel, iou_))
        cm_rgb = np.vstack((cm_rgb, np.ones((iou_.shape + (4,)))))
        cm_rgb[-1, :, 3] = iou_
        cm_rgb[-1, :, 0:3] = green
        iou_label = 'IoU \n '+r'$\mu$'+'={{:.{:d}f}}%'.format(rel_precision)
        y_labels += [iou_label.format(np.nanmean(iou_per_class)*100)]




    ## Plot the conf matrix:
    fig, ax = plt.subplots()
    ax.set(xlabel='prediction', ylabel='ground truth', title='Confusion Matrix')
    ax.imshow(cm_rgb)
    fontsize = min(8, 70 / (len(classes) + 1 + F1 + (0 if symmetric else 2)))
    plt.yticks(range(cm_rel.shape[0]), y_labels, fontsize=fontsize)
    plt.xticks(range(cm_rel.shape[1]), x_labels, fontsize=fontsize,
               rotation=0 if symmetric else 45,
               horizontalalignment='center' if symmetric else 'right')

    ## Plot line grid:
    n_x = cm.shape[1]
    n_y = cm.shape[0]
    x_ticks = np.arange(-0.5, cm_rel.shape[1] + .5)
    y_ticks = np.arange(-0.5, cm_rel.shape[0] + .5)

    ax.plot(np.repeat(x_ticks[None, :n_x], len(y_ticks), axis=0),
            np.repeat(y_ticks[   :, None], n_x         , axis=1),
            color='k', linewidth=1)
    ax.plot(np.repeat(x_ticks[   :, None], n_y         , axis=1),
            np.repeat(y_ticks[None, :n_y], len(x_ticks), axis=0),
            color='k', linewidth=1)

    # Border from data to metrics:
    ax.plot(x_ticks[:n_x+1],
            np.repeat(y_ticks[n_y], n_x+1, axis=0),
            color='k', linewidth=3)
    ax.plot(np.repeat(x_ticks[n_x], n_y+1, axis=0),
            y_ticks[:n_y+1],
            color='k', linewidth=3)

    # Border at lower right end of reacll & precision (around OA):
    ax.plot(x_ticks[n_x:],
            np.repeat(y_ticks[n_y], len(x_ticks)-n_x, axis=0),
            color='k', linewidth=2)
    ax.plot(np.repeat(x_ticks[n_x], len(y_ticks)-n_y, axis=0),
            y_ticks[n_y:],
            color='k', linewidth=2)

    # Border for F1:
    if F1:
        ax.plot(x_ticks[:n_x+2],
                np.repeat(y_ticks[n_y+1], n_x+2, axis=0),
                color='k', linewidth=2)
        if F1_gt:
            ax.plot(np.repeat(x_ticks[n_x+1], n_y+2, axis=0),
                    y_ticks[:n_y+2],
                    color='k', linewidth=2)

    if iou:
        if F1:
            ax.plot(x_ticks[:n_x+1],
                    np.repeat(y_ticks[n_y+2], n_x+1, axis=0),
                    color='k', linewidth=2)
        else:
            ax.plot(x_ticks[:n_x+2],
                np.repeat(y_ticks[n_y+1], n_x+2, axis=0),
                color='k', linewidth=2)

    ## Text Inside matrix fields:
    for r in range(cm_rel.shape[0]):
        for c in range(cm_rel.shape[1]):
            value_rel = [cm_rel[r, c] * 100] if rel_vals else []
            value_abs = [cm_abs[r, c]] if abs_vals else []
            value = value_rel + value_abs

            textcolor = 'black'
            fstring_rel = '{{:.{:d}f}}%'.format(rel_precision)
            fstring_rel = fstring_rel if rel_vals else ''
            fstring_abs = '{:,d}' if abs_vals and value_abs[0] <= abs_max else ''

            if len(value_rel) == 0 or np.isnan(value_rel):
                fstring_rel = '{:.0f}'
                fstring_abs = ''

            fstring = fstring_rel + '\n' + fstring_abs

            if F1 and r == n_y+1:
                if c >= n_x:
                    fstring = ''
                else:
                    fstring = fstring_rel

            if F1_gt and c == n_x+1:
                if r >= n_y:
                    fstring = ''
                else:
                    fstring = fstring_rel

            if (iou and not F1 and r == n_y+1) or (iou and F1 and r == n_y+2):
                if c >= n_x:
                    fstring = ''
                else:
                    fstring = fstring_rel

            fstring = fstring.strip() # remove '\n' if only rel or abs.

            fontsize = min(9, 65 / (len(classes) + 1 + F1 + (0 if symmetric else 2)))
            ax.text(c, r, fstring.format(*value), color=textcolor,
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=fontsize)


    # Call tight_layout() twice for the ylabel to be fully in the figure:
    plt.tight_layout()
    plt.tight_layout()

    if show:
        plt.show()

    ## save to file:
    path = os.path.join(path, 'conf_matrix_' + file_suffix)
    f = open(path + '.pkl', 'wb')
    to_save = {'conf_matrix'   : cm,             'classes': classes,
               'path': path, 'file_suffix': file_suffix,
               'rel_vals'      : rel_vals,       'abs_vals':abs_vals,
               'rel_precision' : rel_precision,  'abs_max': abs_max,
               'F1'            : F1,   'class_freq': class_freq,
               'classes_pred'  : classes_pred,   'label_map': label_map,
               'iou'           : iou}
    pickle.dump(to_save, f)
    f.close()
    fig.savefig(path + '.png', dpi=600)
    fig.savefig(path + '.pdf', dpi=600)




def print_to_file(cm, file, classes_gt, classes_pr=None, label_map=None,
                  indent=0, ignore_labels=None):
    """ Prints a confusion matrix into an ASCII file object.

    Allows conf matrix to be unsymmetric (more classes in ground truth than in
    prediction). In that case, a additional list of labels in the predictions
    have to be given, together with a mapping dict that maps gt classes to pr
    classes, This mapping tells, which gt classes are to be considered correct
    for a given pr class.

    Args:
        cm:             NxM confusion matrix.
        file:           File object to write into.
        classes_gt:     {label:name} for (gt) clases. Also for pred if symm.
        classes_pr:     {label:name} for prediction classes (optional).
        label_map:      Nx dict{gt_label:pr_label} (optional).
        indent:         Indent of output.
        ignore_labels:  List of ground truth labels to ignore in precision and
                        acc. Requires also labels_gt as input.
    """

    if ignore_labels is not None:
        classes_gt = classes_gt.copy()
        for i in ignore_labels:
            classes_gt[i] = classes_gt[i] + '*'

    labels_gt = list(classes_gt.keys())
    names_gt = list(classes_gt.values())

    if classes_pr is None:
        labels_pr = labels_gt
        names_pr = names_gt
        analysis = analize(cm, labels_gt, detailed=True,
                           ignore_labels=ignore_labels, return_iou=True)
    else:
        labels_pr = list(classes_pr.keys())
        names_pr  = list(classes_pr.values())
        if label_map is None:
            label_map = {l:l for l in labels_gt}
        analysis = analize(cm, labels_gt, labels_pr, label_map, True,
                           ignore_labels=ignore_labels, return_iou=True)

    acc            = analysis[0]
    recall         = analysis[1]
    precision      = analysis[2]
    f1             = analysis[3]
    sum_gt_all     = analysis[7].squeeze()
    sum_pr_all     = analysis[10].squeeze()   #  technically sum_gt_all_pr
    iou_per_class  = analysis[13].squeeze()


    ## Find larges class name / label / cm value for nicer printing:
    maxl_names = 0
    for cn in names_gt:
        maxl_names = max(maxl_names, len(cn))

    maxl_labels = max(len('%d' % np.max(labels_gt)), 1)
    maxl_labels = max(len('%d' % np.max(labels_pr)), maxl_labels)
    maxl_labels += 2  # '()'

    maxl_values = max(len('%d' % np.max(cm)), 1)
    n_of_commas = math.floor((maxl_values - 1) / 3)
    maxl_values += n_of_commas

    column_width = max(maxl_labels, maxl_values)
    column_width = max(column_width, 6)  # '000.00'%
    column_width += 3  # margin

    first_row_width = max(len('precision '), maxl_labels + maxl_names + 3)

    h_line = (' ' * indent +
              '-' * first_row_width +
              '|' +
              '-' * (column_width * cm.shape[1] + 2) +
              '|' +
              '-' * 20 + '\n')

    ## Header Line
    file.write('\n')
    file.write(' ' * (indent + first_row_width) + '|')
    for l in labels_pr:
        file.write(('(%d)' % l).rjust(column_width))
    file.write('  |   recall \n')
    file.write(h_line)


    ## Content
    for i, (l, n) in enumerate(zip(labels_gt, names_gt)):
        # Gt class name and label:
        file.write(' ' * indent +
                   ('%s  ' % n.rjust(maxl_names) +
                    '({:,d}) '.format(l).rjust(maxl_labels)
                   ).rjust(first_row_width) +
                   '|')
        # Data:
        for j in range(cm.shape[1]):
            file.write('{:,d}'.format(cm[i, j]).rjust(column_width))
        # Recall:
        file.write('  |   %6.2f\n' % (recall[i]*100))


    ## Fooder Lines
    file.write(h_line)
    # Precision & OA:
    file.write(' ' * indent +
               'precision '.rjust(first_row_width) +
               '|')
    for p in precision:
        file.write(('%6.2f' % (p*100)).rjust(column_width))
    file.write('  |   oval_acc = %6.2f\n' % (acc*100))
    # F1:
    file.write(' ' * indent +
               'F1 '.rjust(first_row_width) +
               '|')
    for f in f1:
        file.write(('%6.2f' % (f*100)).rjust(column_width))
    file.write('  |   mean(F1) = %6.2f\n' % (np.nanmean(f1).item()*100))
    # IoU:
    file.write(' ' * indent +
               'IoU '.rjust(first_row_width) +
               '|')
    for i in iou_per_class:
        file.write(('%6.2f' % (i*100)).rjust(column_width))
    file.write('  |   mean(IoU)= %6.2f\n' % (np.nanmean(iou_per_class).item()*100))


    ## Data stats:
    for cn in names_pr:
        maxl_names = max(maxl_names, len(cn))
    maxl_values = max(len('%d' % np.max(sum_pr_all)), 1)
    maxl_values = max(len('%d' % np.max(sum_gt_all)), maxl_values)
    n_of_commas = math.floor((maxl_values - 1) / 3)
    maxl_values += n_of_commas

    file.write('\n')
    file.write(' ' * indent + 'Ground-Truth Data Statistics:\n')
    for l, cn, n in zip(labels_gt, names_gt, sum_gt_all):
        file.write(' ' * (indent+4) + 'Number of elements in class ' +
                   cn.ljust(maxl_names) +
                   ('(%d):' % l).rjust(maxl_labels + 3) + '  ' +
                   '{:,d}'.format(int(n)).rjust(maxl_values) +
                   '\n')
    file.write(' ' * (indent+4) + '=> total: %d\n' % np.sum(sum_gt_all))
    file.write('\n')

    file.write(' ' * indent + 'Ground-Truth Data Statistics (as "Prediction Classes"):\n')
    for l, cn, n in zip(labels_pr, names_pr, sum_pr_all):
        file.write(' ' * (indent+4) + 'Number of elements in class ' +
                   cn.ljust(maxl_names) +
                   ('(%d):' % l).rjust(maxl_labels + 3) + '  ' +
                   '{:,d}'.format(int(n)).rjust(maxl_values) +
                   '\n')
    file.write(' ' * (indent+4) + '=> total: %d\n' % np.sum(sum_pr_all))
    file.write('\n')
