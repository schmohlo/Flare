""" Generic base class for Machine Learning Models

by Stefan Schmohl, 2020
"""

import os
import warnings
import torch
import re
from collections import OrderedDict

from flare.nn import datasets
from flare.utils import Timer


class Model():
    """ Base class for ML models, wrapping neural networks.

    It's main purpose is to ensure compatibility of input data and the network
    trough the definition of input field names and order.
    It also keeps track of what the output classes of the network mean by
    storing a class name to each label in a dict.
    """

    def __init__(self, net, input_features, device):
        """
        Args:
            net:            Pytorch module.
            input_feature:  List[str] of input feature names the networks
                            expects as input. Order is impoortant!
            device:         PyTorch device. 'cuda' or 'cpu'
        """

        self.net = net
        self.input_features = input_features

        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

        self.net = self.net.to(device)
        self.device = device


    def load_net_from_checkpoint(self, path2mcps, epoch=None):
        """ Loads pytorch network state (weights) from checkpoint file.

        Note: this only works, if network has been defined before and matches to
        the weights stored in the checkpoint file!

        If epoch != None, it asumes the file names to include "epochXXXXX",
        where XXXXX is the epoch number.
        """

        digits = 5
        cp_files = sorted(os.listdir(path2mcps))
        if epoch is None:
            cp_file = cp_files[-1]
            s = cp_file.find('epoch') + len('epoch')
            epoch = int(cp_file[s:s + digits])
        else:
            s = [c.find('epoch') for c in cp_files]
            s = [s_ + len('epoch') for s_ in s]
            epochs = [int(cp_files[i][s_:s_ + digits]) for i, s_ in enumerate(s)]
            if epoch in epochs:
                cp_file = cp_files[epochs.index(epoch)]
            else:
                warnings.warn('epoch %d not found.' % epoch)
                return None
        cp_path = os.path.join(path2mcps, cp_file)
        
        # Quick and dirty fix for layer counter not reseted on original creation:
        cp = torch.load(cp_path)
        if re.search('Model.1.left.batchnorm2d_\d+.weight', list(cp.keys())[2]):
            m = re.search('batchnorm2d_\d+', list(cp.keys())[2])
            c0 = int(list(cp.keys())[2][m.span()[0]+12: m.span()[1]])
            for i,k in enumerate(cp.keys()):
                m = re.search('batchnorm2d_\d+', k)
                if m:
                    c = int(k[m.span()[0]+12: m.span()[1]])
                    c_ = c - c0
                    k_ = re.sub('batchnorm2d_\d+', 'batchnorm2d_%d' % c_, k)
                    cp = OrderedDict([(k_, v) if kk == k else (kk, v) for kk, v in cp.items()])
                    continue
                m = re.search('Conv2d_\d+', k)
                if m:
                    c = int(k[m.span()[0]+7: m.span()[1]])
                    c_ = c - c0
                    k_ = re.sub('Conv2d_\d+', 'Conv2d_%d' % c_, k)
                    cp = OrderedDict([(k_, v) if kk == k else (kk, v) for kk, v in cp.items()])      
        
        self.net.load_state_dict(cp)
        return cp_file, epoch



    def num_of_paras(self):
        """ Number of network parameters. """
        nop = 0
        for parameter in self.net.parameters():
            nop += parameter.numel()
        return nop



    def _batch_to_device(self, data, data_loader):
        """ Move batch from dataloader to the device of the network. """
        if isinstance(data_loader.dataset, datasets.VoxelCloudDataset):
            # for SCN, only features need to be transfered to gpu.
            data[1] = data[1].to(self.device)
        else:
            data = data.to(self.device)
        return data



    ###########################################################################
    # Plots and Prints                                                        #
    #                                                                         #
    ###########################################################################



    def _verbose_epoch(self, epoch, dt, t):
        """ Print epoch metrics and split time to stdout.

        Use this during training. Subclass has to have a MetricsHistory object
        in self.history.ssss
        """

        line = '  Epoch %5d: ' % epoch
        for metric in self.history.columns:
            value = self.history[metric].tolist()[-1]
            if 'acc' in metric:
                line += metric + ': %5.2f%%,    ' % (value * 100)
            elif 'm_ap' in metric:
                line += metric + ': %2.4f,    ' % value
            elif 'loss' in metric:
                line += metric + ': %2.4f,    ' % value
        line = line[0:-5]
        print(line)
        print('  Epoch %5d: finished in %.1f sec (%.1f sec since start).' %
              (epoch, dt, t))
