""" This module holds a torch nn.module sublass for sparse UNets with 2D results.

by Stefan Schmohl, 2020
"""

import math
import torch
import torch.nn as nn
import sparseconvnet as scn
import flare.nn._utils as nnutils
from .sunet import UNet

from flare.utils import Timer


class SUNet3D2D(nn.Module):
    """ Class for a neural network in sparse U-Net architecture with 2D results
        
    __init__() provides convenient setup utilizing the SCN U-Net generator.
    """

    def __init__(self, nIn, noC, input_size, filter_size, bias,
                 residual_blocks, downsample, reps, nPlanes, downsample_flatten):
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

        self.noC = noC
        self.input_size = input_size
         
        # FlattenNet has no padding, so ouput would be to small.
        # => Cheat by adding padding beforehand.
        # Since U-Net needs devide-by two (more or less), double input size.
        input_size = input_size.copy()
        input_size[0] *= 2
        input_size[1] *= 2
    
    
        dims = 3        # 3D, because designed for 3D point clouds.
        m = nPlanes[0]  # Base filterbank size. 
    
        self.sparseUnet = scn.Sequential().add(
            scn.InputLayer(dims, input_size, mode=2)).add(
            scn.SubmanifoldConvolution(dims, nIn, m, filter_size, bias)).add(
            UNet(dims, reps, nPlanes, residual_blocks, downsample, filter_size, bias)).add(
            scn.BatchNormReLU(m))
            
        self.flattenNet, nOut = FlattenNet(dims, m, input_size, 
                                               [filter_size, filter_size],
                                               downsample_flatten, bias)  

        nInter = int(nOut / 2)
        self.linear = nn.Sequential(nn.Conv2d(nOut, nInter, kernel_size=1),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Conv2d(nInter, noC, kernel_size=1))



    def forward(self, x_org):

        coords = x_org[0]
        coords_padded = coords.clone()
        coords_padded[:, [0,1]] += (torch.tensor(self.input_size[0:2])/2).long()
        x_org[0] = coords_padded


        x = self.sparseUnet(x_org)
        x = self.flattenNet(x).squeeze()  # squeeze to remove empty z-dim.


        # Undo padding: extract center.
        shape_diff = torch.tensor(x.shape[-2:]) - torch.tensor(self.input_size[0:2])
        lower = (shape_diff/2).long()
        upper = shape_diff - lower       
        x = x[:, :, lower[0]:-upper[0], lower[1]:-upper[1]]  
        
        
        x = self.linear(x)
        
        
        # Make output into a list of probs per batch-pixel:
        shape = tuple(x.shape)    # in case I want to return it, also.
        x = x.permute(0,2,3,1)    # place "feature"-dim at the end.
        x = x.reshape((-1, self.noC))
        
        
        # TODO maybe do the whole centering-thing in dataset?:
        x_org[0] = coords  # just to be sure they are unchanged for next iteration. 

        return x





def FlattenNet(dims, nIn, input_size, filter_size=[3, 3], downsample=[2,2], 
               bias=False, leakiness=0):

    def add_downsampleLayer(m, nIn, nOut):
        m.add(scn.Convolution(dims, nIn, nOut,
                              [filter_size[0], filter_size[1], downsample[0]], 
                              [1, 1, downsample[1]], 
                              bias))
        m.add(scn.BatchNormLeakyReLU(nOut, leakiness=leakiness))


    height   = input_size[2]
    stride   = downsample[1]
    n_layers = math.log(height, stride)
    
    if not n_layers.is_integer():
        raise ValueError("Input height is no power of stride=", stride, "!")

    m = scn.Sequential()
    
    for i in range(int(n_layers)):
        add_downsampleLayer(m, nIn*(i+1), nIn*(i+2)) 
        
    m.add(scn.SparseToDense(3, nIn*(i+2)))

    return m, nIn*(i+2)
            
    