"""

by Stefan Schmohl, 2020
"""

import os
import shutil
import numpy as np
import json
import _pickle as cPickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch



###############################################################################
# Machine Learning Metrics                                                    #
#                                                                             #
###############################################################################



def clean_accuracy(pred, gt, ignore_label):
    """ Calc accuracy, ignoring ignore_label.

    Args:
        pred:           Tensor or ndarray of prediction labels.
        gt:             Tensor or ndarray of ground truth labels (same size as
                        pred).
        ignore_label:   Label in ground truth to ignore. I.e. wrong classifi-
                        cations on this gt label are not considered false nor
                        true.

    Return
    """

    # Prefer pytorch, in case data is on gpu. If its on cpu(), then they
    # will share memory, so no memory or computation overhead.
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt)

    ind = gt != ignore_label
    num_matches = torch.sum(pred[ind].long() == gt[ind].long()).float()
    num_total = torch.sum(ind)
    if num_total == 0:
        acc = torch.tensor(0.0)
    else:
        acc = num_matches / num_total
    return acc.item()



def cohen_kappa_score(confusion, weights=None):
    """Cohen's kappa: a statistic that measures inter-annotator agreement.

    This function is a copy from sklearn, however saves the step of computing a conf matrix.
    """
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    if weights is None:
        w_mat = np.ones([n_classes, n_classes], dtype=np.int)
        w_mat.flat[:: n_classes + 1] = 0
    elif weights == "linear" or weights == "quadratic":
        w_mat = np.zeros([n_classes, n_classes], dtype=np.int)
        w_mat += np.arange(n_classes)
        if weights == "linear":
            w_mat = np.abs(w_mat - w_mat.T)
        else:
            w_mat = (w_mat - w_mat.T) ** 2
    else:
        raise ValueError("Unknown kappa weighting type.")

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return 1 - k



###############################################################################
# Various Functions                                                           #
#                                                                             #
###############################################################################



def colormapping(img, colormap):
    """ Returns RGB version of greyscale image, color coded by colormap.

    Args:
        img:       ndarray
        colormap:  dict[integer value: [R,G,B]]

    Returns:
        img_RGB:   ndarray, shape = img.shape + [3]
    """

    colormap_lookup = np.zeros((len(colormap), 3), dtype='long')
    for i, rgb in colormap.items():
        colormap_lookup[i] = rgb

    img_RGB = colormap_lookup[img.astype('long')]
    return img_RGB



def makedirs_save(path, purpose_str):
    if os.path.exists(path):
        print("\nPath to", purpose_str, "already exists:")
        print("  {}".format(path))
        print("  Overwrite? [y/n]")

        while True:
            inpt = input()
            if inpt == 'n':
                return 0
            elif inpt == 'y':
                shutil.rmtree(path)
                break
    os.makedirs(path)
    return 1




def import_class_from_string(path):
    """ For example: will return class object for "'torch.optim.SGD'"
    """
    from importlib import import_module
    module_path, _, class_name = path.rpartition('.')
    mod = import_module(module_path)
    klass = getattr(mod, class_name)
    return klass



###############################################################################
# Plots and Prints                                                            #
#                                                                             #
###############################################################################



def print_dict(dictionary, indent=0, recursive=False):
    t_width = os.get_terminal_size().columns
    d_width = len(str(dictionary))
    indent_str = ' ' * indent

    if d_width + indent <= t_width:
        print(indent_str + str(dictionary))
    else:
        for key, value in dictionary.items():
            if recursive and isinstance(value, dict):
                print(indent_str + str(key), ':')
                print_dict(value, indent+4, recursive=True)
            else:
                print(indent_str + str(key), ':', value)



def plot_metric_over_models(values, labels, model_names, epochs, title, fpath):
    """
    Args:
        values:         List[ndarray] or List[List]] or 2D ndarray
        labels:         List[str]
        model_names:    List[str]
    """

    same_model = [m == model_names[0] for m in model_names]
    same_model = sum(same_model) == len(model_names)

    if same_model:
        xlabels = ['epoch={}'.format(e) if e is not None else '' for e in epochs]
        fontsize = None
    else:
        xlabels = ['{}, e={}'.format(m,e) for m,e in zip(model_names, epochs)]
        fontsize = 4

    fig, ax = plt.subplots()
    for v, l in zip(values, labels):
        ax.plot(v, label=l)

    plt.xticks(range(len(xlabels)), xlabels, rotation=90, fontsize=fontsize)
    ax.set(xlabel='networks', ylabel='accuracy [%]', title=title)
    ax.legend()
    ax.grid()

    # Call tight_layout() for the ylabel to be fully in the figure:
    plt.tight_layout()

    ## save to file:
    fpath = os.path.join(fpath, title)

    to_save = {'values': values, 'labels': labels, 'model_names': model_names,
               'epochs': epochs, 'fpath':  fpath}

    f = open(fpath + '.pkl', 'wb')
    cPickle.dump(to_save, f)
    f.close()

    f = open(fpath + '.json', 'w')
    json.dump(to_save, f, indent=4)
    f.close()

    fig.savefig(fpath + '.png', dpi=600)
    fig.savefig(fpath + '.pdf', dpi=600)
