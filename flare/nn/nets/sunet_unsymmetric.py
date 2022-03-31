""" This module holds a torch nn.module sublass for sparse UNets.

by Stefan Schmohl, 2020
"""

import torch.nn as nn
import sparseconvnet as scn



class SUNet_Unsymmetric(nn.Module):
    """ Class for a neural network in sparse U-Net architecture.
        
    __init__() provides convenient setup utilizing the SCN U-Net generator.
    
    Instead of nPlanes = [16, 32, 48, 64] for standard SUNet, 
    here use   nPlanes = [16, 16,16 32,32, 48,48, 64,64, 48,48, 32,32, 16,16].

    Especially usefull, if you have a lot of input features. For Example:
    nPlanes = [128,  64,48,  48,32,  48,48,  64,64,  48,48,  32,32,  16,16]
    """

    @classmethod
    def from_config(cls, config):
        nIn = len(config['model']['input_features'])
        noC = len(config['model']['classes'])
        net = cls(**config['model']['network']['net-params'], nIn=nIn, noC=noC)
        return net


    def __init__(self, nIn, noC, input_size, filter_size, bias,
                 residual_blocks, downsample, reps, nPlanes):
        """
        Args:
            nIn:   
                Number of input features.
            noC:    
                Number of output classes.
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
        """

        super().__init__()

        dims = 3        # 3D, because designed for 3D point clouds.

        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(dims, input_size, mode=2)).add(

            scn.SubmanifoldConvolution(dims, nIn, nPlanes[0], filter_size, bias)).add(

            UNet(dims, reps, nPlanes, residual_blocks, downsample, filter_size,
                 bias)).add(

            scn.BatchNormReLU(nPlanes[-1])).add(
            scn.OutputLayer(dims))
            
        print(self.sparseModel)

        # 1x1x1 Convolutions in Semantic Segmentation,
        # = here modeled as FC on each "sample" = Voxel
        self.linear = nn.Sequential(nn.Linear(nPlanes[-1], nPlanes[-1]*3),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(nPlanes[-1]*3, noC))

    def forward(self, x):
        x = self.sparseModel(x)
        x = self.linear(x)
        return x



def UNet(dimension, reps, nPlanes, residual_blocks=False, downsample=[2, 2], 
         filter_size=3, bias=False, leakiness=0, n_input_planes=-1):
    """
    Modification of:
    https://github.com/facebookresearch/SparseConvNet/blob/master/sparseconvnet/networkArchitectures.py
    Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
    
    To also include filter_size and bias as input.
    """
    def block(m, a, b):
        if residual_blocks: #ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, a, b, filter_size, bias))
                    .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, b, b, filter_size, bias)))
             ).add(scn.AddTable())
        else: #VGG style blocks
            m.add(scn.Sequential()
                 .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                 .add(scn.SubmanifoldConvolution(dimension, a, b, filter_size, bias)))
    def U(nPlanes,n_input_planes=-1): #Recursive function
        m = scn.Sequential()
        for i in range(reps):
            block(m, n_input_planes if n_input_planes!=-1 else nPlanes[i], nPlanes[i+1])
            n_input_planes=-1
        if len(nPlanes) > reps+1:
            m.add(
                scn.ConcatTable().add(
                    scn.Identity()).add(
                    scn.Sequential().add(
                        scn.BatchNormLeakyReLU(nPlanes[i+1],leakiness=leakiness)).add(
                        scn.Convolution(dimension, nPlanes[i+1], nPlanes[i+2],
                            downsample[0], downsample[1], bias)).add(
                        U([nPlanes[i+2]]+nPlanes[i+2:-i-1])).add(
                        scn.BatchNormLeakyReLU(nPlanes[-i-2],leakiness=leakiness)).add(
                        scn.Deconvolution(dimension, nPlanes[-i-2], nPlanes[-i-2],
                            downsample[0], downsample[1], bias))))
            m.add(scn.JoinTable())
            for i in range(reps):
                block(m, (nPlanes[reps]+nPlanes[-reps-1]) if i == 0 else nPlanes[-3+i], nPlanes[-2+i])
        return m
    m = U(nPlanes,n_input_planes)
    return m
