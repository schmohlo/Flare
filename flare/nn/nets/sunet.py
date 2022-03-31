""" This module holds a torch nn.module sublass for sparse UNets.

by Stefan Schmohl, 2020
"""

import torch.nn as nn
import sparseconvnet as scn



class SUNet(nn.Module):
    """ Class for a neural network in sparse U-Net architecture.
        
    __init__() provides convenient setup utilizing the SCN U-Net generator.
    """

    @classmethod
    def from_config(cls, config):
        nIn = len(config['model']['input_features'])
        noC = len(config['model']['classes'])
        net = cls(**config['model']['network']['net-params'], nIn=nIn, noC=noC)
        return net


    def __init__(self, nIn, noC, input_size, filter_size, bias,
                 residual_blocks, downsample, reps, nPlanes, leakiness=0):
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
        m = nPlanes[0]  # Base filterbank size. 

        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(dims, input_size, mode=2)).add(

            scn.SubmanifoldConvolution(dims, nIn, m, filter_size, bias)).add(

            UNet(dims, reps, nPlanes, residual_blocks, downsample, filter_size,
                 bias, leakiness)).add(

            scn.BatchNormReLU(m)).add(
            scn.OutputLayer(dims))

        # 1x1x1 Convolutions in Semantic Segmentation,
        # = here modeled as FC on each "sample" = Voxel
        self.linear = nn.Sequential(nn.Linear(m, m*3),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(m*3, noC))

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
            block(m, n_input_planes if n_input_planes!=-1 else nPlanes[0], nPlanes[0])
            n_input_planes=-1
        if len(nPlanes) > 1:
            m.add(
                scn.ConcatTable().add(
                    scn.Identity()).add(
                    scn.Sequential().add(
                        scn.BatchNormLeakyReLU(nPlanes[0],leakiness=leakiness)).add(
                        scn.Convolution(dimension, nPlanes[0], nPlanes[1],
                            downsample[0], downsample[1], bias)).add(
                        U(nPlanes[1:])).add(
                        scn.BatchNormLeakyReLU(nPlanes[1],leakiness=leakiness)).add(
                        scn.Deconvolution(dimension, nPlanes[1], nPlanes[0],
                            downsample[0], downsample[1], bias))))
            m.add(scn.JoinTable())
            for i in range(reps):
                block(m, nPlanes[0] * (2 if i == 0 else 1), nPlanes[0])
        return m
    m = U(nPlanes,n_input_planes)
    return m


# BSD License

# For SparseConvNet software

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

 # * Redistributions of source code must retain the above copyright notice, this
   # list of conditions and the following disclaimer.

 # * Redistributions in binary form must reproduce the above copyright notice,
   # this list of conditions and the following disclaimer in the documentation
   # and/or other materials provided with the distribution.

 # * Neither the name Facebook nor the names of its contributors may be used to
   # endorse or promote products derived from this software without specific
   # prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
