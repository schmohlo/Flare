""" This module holds a torch nn.module sublass for 2D UNets.

by Stefan Schmohl, 2020
"""

import torch
import torch.nn as nn
import sparseconvnet as scn

#class Conv2dSamePadding(nn.Conv2d):
    #"""
    #https://gist.github.com/sumanmichael/4de9dee93f972d47c80c4ade8e149ea6
    #"""
    #def __init__(self,*args,**kwargs):
        #super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        #self.zero_pad_2d = nn.ZeroPad2d[(k // 2) for k in self.kernel_size[::-1]]))

    #def forward(self, input):
        #return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)
    
    
    

class SUNet2D(nn.Module):
    """ Class for a neural network in  U-Net architecture.
        
    __init__() provides convenient setup utilizing the U-Net generator.
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

        m = nPlanes[0]  # Base filterbank size. 

        self.Model = nn.Sequential(
            nn.Conv2d(nIn, m, filter_size, bias=bias, padding=filter_size//2),
            UNet2D(reps, nPlanes, residual_blocks, downsample, filter_size,
                   bias, leakiness),
            nn.BatchNorm2d(m),
            nn.LeakyReLU(leakiness))

        # 1x1x1 Convolutions in Semantic Segmentation,
        self.linear = nn.Sequential(nn.Conv2d(m, m*3, kernel_size=1),
                                    nn.BatchNorm2d(m*3),
                                    nn.LeakyReLU(leakiness),
                                    nn.Dropout(p=0.5),
                                    nn.Conv2d(m*3, noC, kernel_size=1))
    def forward(self, x):
        x = x[1]  # [1] is data, [0] are coords
        x = self.Model(x)
        x = self.linear(x)
        return x




def UNet2D(reps, nPlanes, residual_blocks=False, downsample=[2, 2],
           filter_size=3, bias=False, leakiness=0, n_input_planes=-1):

    global c
    c = 0
    
    def block(m, a, b):
        global c
        if residual_blocks: #ResNet style blocks
            raise Exception("not implemented yet")
        else: #VGG style blocks
            m.add_module('batchnorm2d_{}'.format(c), nn.BatchNorm2d(a)); c+=1;
            m.add_module('leakyRelu_{}'.format(c), nn.LeakyReLU(leakiness)); c+=1;
            m.add_module('Conv2d_{}'.format(c), nn.Conv2d(a, b, filter_size, bias=bias, padding=filter_size//2)); c+=1;

    class U(nn.Module):
        def __init__(self,nPlanes,n_input_planes=-1): #Recursive function
            super(U, self).__init__()

            left = nn.Sequential()
            for i in range(reps):
                block(left, n_input_planes if n_input_planes!=-1 else nPlanes[0], nPlanes[0])
                n_input_planes=-1

            if len(nPlanes) > 1:
                mid = U(nPlanes[1:])

                down = nn.Sequential(
                            nn.BatchNorm2d(nPlanes[0]),
                            nn.LeakyReLU(leakiness),
                            nn.Conv2d(nPlanes[0], nPlanes[1],
                                downsample[0], downsample[1], bias=bias))

                up = nn.Sequential(
                            nn.BatchNorm2d(nPlanes[1]),
                            nn.LeakyReLU(leakiness),
                            nn.ConvTranspose2d(nPlanes[1], nPlanes[0],
                                downsample[0], downsample[1], bias=bias))

                right = nn.Sequential()
                for i in range(reps):
                    block(right, nPlanes[0] * (2 if i == 0 else 1), nPlanes[0])

            else:
                down   = None
                mid    = None
                up     = None
                right  = None
            self.left   = left
            self.down   = down
            self.mid    = mid
            self.up     = up
            self.right  = right


        def join(self, input):
            return torch.cat(input, dim=1)


        def forward(self, inputs):            
            l = self.left(inputs)
            if self.right is None:
                return l

            d    = self.down(l)
            m    = self.mid(d)
            u    = self.up(m)
            j    = self.join([l,u])
            r    = self.right(j)
            return r

    m = U(nPlanes,n_input_planes)
    return m




