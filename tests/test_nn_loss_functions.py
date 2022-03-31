
import unittest
import numpy as np
import torch
import numpy.testing as nptest
import torch.testing as totest

from math import pi

from flare.nn._utils import bdetection_transform
from flare.nn.loss_functions import TreeDetectionLoss



inf = float("Inf")
NaN = float("NaN")


# =============================================================================
# 
# 
# class Test_TreeDetectionLoss(unittest.TestCase):
# 
#     # anchors = [z, radius, height]
# 
#     anchors = torch.tensor([[270, 4, 11], [10, 3.3, 9]])
# 
#     loss_fu = TreeDetectionLoss(anchors, iou_pos=0.7, iou_neg=0.4, regression_weight=1)
#     
#     
# 
#     def test_ResultNotChanged(self):
#     
#         #TODO: iou schwellwerte benutzen lassen
# 
#         # targets = [x,y,z,r,h]
#         targets = torch.tensor([[  0,   0, 255, 4, 18], 
#                                 [5.5, 9.2, 240, 2,  7],
#                                 [  6,   6, 260, 5,  8]])
# 
#         locations = torch.tensor([[0, 0], 
#                                   [1, 0],
#                                   [1, 1],
#                                   [1, 2],
#                                   [6, 9]])
# 
#         regression = torch.tensor([[ 0,  0, 255, 4, 18], [  0,   0, 255, 4, 18],  # complete match with first target with both anchors
#                                    [ 0,  0, 255, 4, 18], [  0,   0, 255, 4, 18],  # complete match with first target with both anchors, different location
#                                    [ 0,  1, 265, 3, 19], [ -1,   0, 250, 5, 17],  # good match with first target with both anchors
#                                    [ 2,  4, 245, 9, 11], [ -1,   0, 250, 9, 11],  # partly match with targets 1 & 2
#                                    [15, 15, 245, 1, 11], [-10, -10, 250, 2, 11]]) # no match
# 
#         anchor_locations = locations.repeat_interleave(self.anchors.shape[0], 0)
#         
#         anchors = torch.cat((anchor_locations.float(), self.anchors.repeat(locations.shape[0], 1)), 1)
#         
#         ## Simulate network output with gradient:
#         regression = bdetection_transform(regression.float(), anchors)
#         regression.requires_grad = True
#         
#         objectness = torch.tensor([[.6], [.8],
#                                    [.9], [.7],
#                                    [.5], [.4],
#                                    [.2], [.2],
#                                    [.0], [.1]], requires_grad=True)
#                    
#         loss = self.loss_fu((anchor_locations, [regression], [objectness]), [targets])
#         
#         totest.assert_allclose(loss, 0.5379572510719299)  # empirical value
# 
# 
# 
#     def test_loss0(self):
# 
#         # targets = [x,y,z,r,h]
#         targets = torch.tensor([[  0,   0, 255, 4, 18], 
#                                 [5.5, 9.2, 240, 2,  7],
#                                 [  6,   6, 260, 5,  8]])
# 
#         locations = torch.tensor([[0, 0], 
#                                   [1, 0],
#                                   [1, 1],
#                                   [1, 2],
#                                   [6, 9]])
# 
#         regression = torch.tensor([[  0,   0, 255, 4, 18], [  0,   0, 255, 4, 18],
#                                    [  0,   0, 255, 4, 18], [  0,   0, 255, 4, 18],  
#                                    [  0,   0, 255, 4, 18], [  0,   0, 255, 4, 18],  
#                                    [  0,   0, 255, 4, 18], [5.5, 9.2, 240, 2,  7],  
#                                    [5.5, 9.2, 240, 2,  7], [  6,   6, 260, 5,  8]]) 
# 
#         
#         anchor_locations = locations.repeat_interleave(self.anchors.shape[0], 0)
#         
#         anchors = torch.cat((anchor_locations.float(), self.anchors.repeat(locations.shape[0], 1)), 1)
#         
#         
#         ## Simulate network output with gradient:
#         regression = bdetection_transform(regression.float(), anchors)
#         regression.requires_grad = True
#         
#         objectness = torch.tensor([[1.], [1.],
#                                    [1.], [1.],
#                                    [1.], [1.],
#                                    [1.], [1.],
#                                    [1.], [1.]], requires_grad=True) * 100 # => sigmoid(100)=1.0
#                    
#         loss = self.loss_fu((anchor_locations, [regression], [objectness]), [targets])
#         
#         
#         totest.assert_allclose(loss, 0.0)
# 
# 
# 
#     def test_rloss0_olossMax(self):
# 
#         # targets = [x,y,z,r,h]
#         targets = torch.tensor([[  0,   0, 255, 4, 18], 
#                                 [5.5, 9.2, 240, 2,  7],
#                                 [  6,   6, 260, 5,  8]])
# 
#         locations = torch.tensor([[0, 0], 
#                                   [1, 0],
#                                   [1, 1],
#                                   [1, 2],
#                                   [6, 9]])
# 
#         regression = torch.tensor([[  0,   0, 255, 4, 18], [  0,   0, 255, 4, 18],
#                                    [  0,   0, 255, 4, 18], [  0,   0, 255, 4, 18],  
#                                    [  0,   0, 255, 4, 18], [  0,   0, 255, 4, 18],  
#                                    [  0,   0, 255, 4, 18], [5.5, 9.2, 240, 2,  7],  
#                                    [5.5, 9.2, 240, 2,  7], [  6,   6, 260, 5,  8]]) 
# 
# 
#         anchor_locations = locations.repeat_interleave(self.anchors.shape[0], 0)
#         
#         anchors = torch.cat((anchor_locations.float(), self.anchors.repeat(locations.shape[0], 1)), 1)
#         
#         
#         ## Simulate network output with gradient:
#         regression = bdetection_transform(regression.float(), anchors)
#         regression.requires_grad = True
#         
#         objectness_factor = 100 # => sigmoid(-100)=0.0
#         objectness = torch.tensor([[1.], [1.],
#                                    [1.], [1.],
#                                    [1.], [1.],
#                                    [1.], [1.],           
#                                    [1.], [1.]], requires_grad=True) * -objectness_factor 
#                    
#         loss = self.loss_fu((anchor_locations, [regression], [objectness]), [targets])
#         
#         
#         totest.assert_allclose(loss, objectness_factor)  # due to the log-sum-exp 
#                                                          # trick, result is not 
#                                                          # infinity, but the 
#                                                          # logit objectness
# 
# 
# 
#     def test_iou0(self):
#         # It will still find 3 positives (one for each gt), since 0 is still
#         # the maximum among other 0s.
# 
#         # targets = [x,y,z,r,h]
#         targets = torch.tensor([[  0,   0, 255, 4, 18], 
#                                 [5.5, 9.2, 240, 2,  7],
#                                 [  6,   6, 260, 5,  8]])
# 
#         locations = torch.tensor([[0, 0], 
#                                   [1, 0],
#                                   [1, 1],
#                                   [1, 2],
#                                   [6, 9]])
# 
#         regression = torch.tensor([[100.0459, 0.2022, 0.0920, 0.9578, 0.6669],
#                                    [100.1304, 0.0490, 0.9617, 0.3295, 0.5588],
#                                    [100.0033, 0.4561, 0.8729, 0.7063, 0.9506],
#                                    [100.2913, 0.5923, 0.1175, 0.9095, 0.3353],
#                                    [100.5463, 0.4666, 0.4682, 0.4151, 0.8880],
#                                    [100.6694, 0.7379, 0.6582, 0.2545, 0.4105],
#                                    [100.1592, 0.1528, 0.0705, 0.4965, 0.7153],
#                                    [100.8718, 0.1154, 0.2118, 0.4528, 0.2247],
#                                    [100.4511, 0.2733, 0.6875, 0.0700, 0.5229],
#                                    [100.8752, 0.4638, 0.3760, 0.4217, 0.2836]])
# 
# 
#         anchor_locations = locations.repeat_interleave(self.anchors.shape[0], 0)
#         
#         anchors = torch.cat((anchor_locations.float(), self.anchors.repeat(locations.shape[0], 1)), 1)
#         
#         
#         ## Simulate network output with gradient:
#         regression = bdetection_transform(regression.float(), anchors)
#         regression.requires_grad = True
#         
#         objectness = torch.tensor([[1.], [1.],
#                                    [1.], [1.],
#                                    [1.], [1.],
#                                    [1.], [1.],
#                                    [1.], [-1.]], requires_grad=True) * 100 # => sigmoid(100)=1.0
#                    
#         loss = self.loss_fu((anchor_locations, [regression], [objectness]), [targets])
#         
#         
#         totest.assert_allclose(loss, 87.4195861)  # empirical
# =============================================================================
        
        
        





if __name__ == '__main__':
    unittest.main()
