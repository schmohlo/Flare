""" This module holds a torch nn.module sublass for sparse UNets.

+ Output of all mid-level features (MLF), concated at the end.

by Stefan Schmohl, 2021
"""

import torch
import torch.nn as nn
import sparseconvnet as scn

from sparseconvnet.sparseConvNetTensor import SparseConvNetTensor



class SUNetMlf(nn.Module):
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
        m = nPlanes[0]  # Base filterbank size.

        self.sparseModel_Input = scn.Sequential().add(
            scn.InputLayer(dims, input_size, mode=2)).add(
            scn.SubmanifoldConvolution(dims, nIn, m, filter_size, bias))

        self.sparseModel_UNet = UNet(dims, reps, nPlanes, residual_blocks, downsample, filter_size, bias)

        self.sparseModel_Output = scn.Sequential().add(
            scn.BatchNormReLU(m)).add(
            scn.OutputLayer(dims))
            
        self.mlf_OutputLayer = scn.OutputLayer(dims)


        # 1x1x1 Convolutions in Semantic Segmentation,
        # = here modeled as FC on each "sample" = Voxel
        self.linear = nn.Sequential(nn.Linear(m, m*3),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(m*3, noC))
                                    
        self.mlf = None

    def forward(self, x):
        x = self.sparseModel_Input(x)
        x, mlf = self.sparseModel_UNet(x)
        x = self.sparseModel_Output(x)
        x = self.linear(x)
        
        self.mlf = self.mlf_OutputLayer(mlf)
        return x



def UNet(dimension, reps, nPlanes, residual_blocks=False, downsample=[2, 2],
         filter_size=3, bias=False, leakiness=0, n_input_planes=-1):

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
            m.add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
            m.add(scn.SubmanifoldConvolution(dimension, a, b, filter_size, bias))

    class U(nn.Module):
        def __init__(self,nPlanes,n_input_planes=-1): #Recursive function
            super(U, self).__init__()

            left = scn.Sequential()
            for i in range(reps):
                block(left, n_input_planes if n_input_planes!=-1 else nPlanes[0], nPlanes[0])
                n_input_planes=-1

            if len(nPlanes) > 1:
                mid = U(nPlanes[1:])

                down = scn.Sequential().add(
                            scn.BatchNormLeakyReLU(nPlanes[0],leakiness=leakiness)).add(
                            scn.Convolution(dimension, nPlanes[0], nPlanes[1],
                                downsample[0], downsample[1], bias))

                up = scn.Sequential().add(
                            scn.BatchNormLeakyReLU(nPlanes[1],leakiness=leakiness)).add(
                            scn.Deconvolution(dimension, nPlanes[1], nPlanes[0],
                                downsample[0], downsample[1], bias))


                unpool =  scn.UnPooling(dimension, downsample[0], downsample[1])

                right = scn.Sequential()
                for i in range(reps):
                    block(right, nPlanes[0] * (2 if i == 0 else 1), nPlanes[0])

            else:
                down   = None
                mid    = None
                up     = None
                unpool = None
                right  = None
            self.left   = left
            self.down   = down
            self.mid    = mid
            self.up     = up
            self.unpool = unpool
            self.right  = right


        def join(self, input):
            output = SparseConvNetTensor()
            output.metadata = input[0].metadata
            output.spatial_size = input[0].spatial_size
            output.features = torch.cat([i.features for i in input], 1) if input[0].features.numel() else input[0].features
            return output


        def forward(self, inputs):
            l = self.left(inputs)
            if self.right is None:
                return l

            d    = self.down(l)
            m    = self.mid(d)
            if isinstance(m, tuple):   # gilfs only if not lowest layer
                m, g = m
            else:
                g = None
            u    = self.up(m)
            j    = self.join([l,u])
            r    = self.right(j)

            if g is None:            
                mlf = j
            else:
                mlf = self.join([j, self.unpool(g)])
            # print('----\n', mlf)
            # breakpoint()
            
            return r, mlf


    m = U(nPlanes,n_input_planes)
    return m

#[]
