""" This module holds a torch nn.module sublass for sparse UNets for tree deteciton.

by Stefan Schmohl, 2020
"""

import math
import torch
import torch.nn as nn
import sparseconvnet as scn
import flare.nn._utils as nnutils
from .sunet_2d import UNet2D

from flare.utils import Timer


class SUNet2DTreeDetection(nn.Module):
    """ Class for a neural network in sparse U-Net architecture for tree detection

    __init__() provides convenient setup utilizing the SCN U-Net generator.
    """

    @classmethod
    def from_config(cls, config):
        nIn       = len(config['model']['input_features'])
        n_anchors = len(config['model']['anchors'])
        geometry  = config['model']['geometry']
        net = cls(**config['model']['network']['net-params'],
                  geometry = geometry, nIn=nIn, n_anchors=n_anchors)
        return net


    def __init__(self, geometry, nIn, input_size, filter_size, bias,
                 residual_blocks, downsample, reps, nPlanes,
                 downsample_flatten_fsize, downsample_flatten_stride, n_anchors,
                 head_kernel_size=1, n_regression_layers=1, leakiness=0):
        """
        Args:
            nIn:
                Number of input features.
            input_size:
                Spatial input size of samples. Note: only important for the
                framework to check, wheather there is to much downsampling
                happening (apparently).
            filter_size:
                Size of conv-filters, except for down- and upconv layers. One
                value, for all dims.
            bias:
                True / False for neurons to have bias weight.
            residual_block:
                True / False if conv-blocks should be redisual. My experience
                tells to better set to False.
            downsample:
                [filter_size, stride] for down- and upconv layers.
            reps:
                Number of conv layers per Unet block.
            nPlanes:
                Filter number per Unet level. The length defines the depth of
                the network, while the values define the depth.
            n_anchors:
                Number of anchors.
        """

        super().__init__()

        # Set number of parameters depending on geometry
        self.geometry = geometry
        if self.geometry == "circles":
            noR = 3
        elif self.geometry == "rectangles":
            noR = 4
        elif geometry == "cylinders":
            noR = 5
        elif self.geometry == "cuboids":
            noR = 6

        noC = 1

        self.noC = noC
        self.noR = noR
        self.n_anchors = n_anchors
        self.input_size = input_size

        noC = noC * n_anchors
        noR = noR * n_anchors

        nOut = noC + noR

        m = nPlanes[0]  # Base filterbank size.

        self.Model = nn.Sequential(
            nn.Conv2d(nIn, m, filter_size, bias=bias, padding=filter_size//2),
            UNet2D(reps, nPlanes, residual_blocks, downsample, filter_size,
                   bias, leakiness),
            nn.BatchNorm2d(m),
            nn.LeakyReLU(leakiness))

        self.FlattenNet = nn.Sequential(nn.Conv2d(m, nOut, 1, bias=bias),
                                        nn.BatchNorm2d(nOut),
                                        nn.LeakyReLU(leakiness))

        self.linear = nn.Sequential(nn.Dropout(p=0.5))


        p = head_kernel_size // 2
        hks = head_kernel_size

        if n_regression_layers == 1:
            # for downwards compatibily no sequential:
            self.regression_layer = nn.Conv2d(nOut, noR, kernel_size=hks, padding=p)
        else:
            modules = [nn.Conv2d(nOut, noR, kernel_size=hks, padding=p)]
            for i in range(n_regression_layers-1):
                modules.append(nn.BatchNorm2d(noR))
                modules.append(nn.ReLU())
                modules.append(nn.Conv2d(noR, noR, kernel_size=hks, padding=p))
            self.regression_layer = nn.Sequential(*modules)

        self.objectness_layer = nn.Conv2d(nOut, noC, kernel_size=hks, padding=p)



    def forward(self, x_org):

        coords = x_org[0]

        ## Feature Extration:
        y_2D = self.Model(x_org[1])
        x = self.FlattenNet(y_2D)
        batch_size = x.shape[0]

        ## Detection head:
        x = self.linear(x)
        y_regression = self.regression_layer(x)
        y_objectness = self.objectness_layer(x)

        ## Reshape to lists of "detections" / "rois" per sample:
        height = x.shape[2]
        width  = x.shape[3]
        y_regression = y_regression.permute(0,2,3,1)    # place "feature"-dim at the end.
        y_objectness = y_objectness.permute(0,2,3,1)    # place "feature"-dim at the end.
        y_regression = y_regression.reshape(batch_size, width * height * self.n_anchors, self.noR)
        y_objectness = y_objectness.reshape(batch_size, width * height * self.n_anchors, self.noC)
        # Include 2D location per detection:
        grid_x, grid_y = torch.meshgrid(torch.arange(height), torch.arange(width))
        # Undo xy downsampling:
        grid_x = grid_x * self.input_size[0] / height
        grid_y = grid_y * self.input_size[1] / width
        locations = torch.stack((grid_x, grid_y), 2).reshape(-1, 2)
        locations = locations.repeat_interleave(self.n_anchors, 0)
        locations = locations.to(y_regression.device)


        ### Debugging:
        #si = 0  # sample index
        #i_s = coords[:,3] == si # indices of sample 0
        #import matplotlib.pyplot as plt
        #from mpl_toolkits.mplot3d import Axes3D
        ## Input coords:
        ##fig = plt.figure()
        ##ax = fig.add_subplot(111, projection='3d')
        ##ax.scatter(coords[i_s,0], coords[i_s,1], coords[i_s,2]); plt.show()
        ## 2D output form FlattenNet:
        #fig = plt.figure()
        #plt.imshow(x[si,0,:,:].cpu().detach()); plt.show()


        return locations, y_regression, y_objectness



