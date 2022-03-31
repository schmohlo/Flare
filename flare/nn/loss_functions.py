""" Utility function for machine learning stuff

by Stefan Schmohl, 2020
"""

import torch
import os

from flare.utils import import_class_from_string



###############################################################################
# Loss functions                                                              #
#                                                                             #
###############################################################################

# class WeightedSmoothL1Loss():
    # def __init__(self, reduction='mean', beta=1.0)

    # def __call__(self, input, target, weights):
        # # type: (Tensor, Tensor, Tensor) -> Tensor
        # t = torch.abs(input - target)
        # b = self.beta
        # z = weights * torch.where(t < b, 0.5 * t ** 2 /b, t - 0.5*b)
        # if reduction == 'mean':
            # return torch.mean(z)
        # if reduction == 'sum':
            # return torch.sum(
    
    

class TreeDetectionLoss(torch.nn.Module):
    """ TODO
    """

    def __init__(self, regression_weight=1, objectness_weight=1,
                 objectness_pos_weight_factor=1, obj_weighting=True, 
                 focal_power=0, regression_loss='torch.nn.SmoothL1Loss',
                 regression_weights=None, weight_lookup=None,
                 weight_lookup_idx=None,
                 obj_IoU_target=False):
        """ TODO
        """
        torch.nn.Module.__init__(self)

        self.regression_weights           = regression_weights
        self.regression_weight            = regression_weight
        self.objectness_weight            = objectness_weight
        self.objectness_pos_weight_factor = objectness_pos_weight_factor
        
        self.weight_lookup     = weight_lookup
        self.weight_lookup_idx = weight_lookup_idx

        # self.regression_loss = torch.nn.SmoothL1Loss
        self.regression_loss = import_class_from_string(regression_loss)
        self.objectness_loss = torch.nn.BCEWithLogitsLoss
        
        self.obj_weighting = obj_weighting
        self.focal_power   = focal_power
        self.obj_IoU_target = obj_IoU_target


    def forward(self, residual_pred, residual_gt, objectness_pred,
                objectness_gt, objectness_weights=None, return_singles=False,
                regression_gt=None, obj_target=None):
        """ TODO
        """       

        # Find first non-empty sample to get some infos:
        for i in range(len(residual_pred)):
            if residual_pred[i].numel() != 0:
                n_paras = residual_pred[i].shape[1]
                device  = residual_pred[i].device
                break
        
        loss_n    = 0
        loss_sum  = 0
        oloss_sum = 0
        rloss_sum = 0
        rlosses_sum = torch.zeros(n_paras)

        for i in range(len(residual_pred)):  # for each sample in mini-batch

            # skip, if sample is empty:
            if residual_gt[i].numel() == 0:
                continue

            res_pr = residual_pred[i]
            res_gt = residual_gt[i]
            obj_pr = objectness_pred[i]
            obj_gt = objectness_gt[i]


            ## Sample-wise weighting for objectness loss
            obj_w = torch.ones_like(obj_pr)
            if self.obj_weighting:
                obj_w  *= objectness_weights[i]
            if self.focal_power != 0:
                p = torch.sigmoid(objectness_pred[i]).detach()
                obj_w  *= torch.pow(1 - p, self.focal_power)
                # TODO: should be p for negatives!

            N_pos = res_pr.shape[0]
            N_neg = obj_pr.shape[0] - N_pos
            
            
            ## Sample-wise weighting for regression loss
            if self.weight_lookup is not None: # and regression_gt is not None:
                rloss_fu = self.regression_loss(reduction='none')
            else:
                rloss_fu = self.regression_loss(reduction='mean')
            

            ## Loss for regression:
            if self.regression_weights is not None or return_singles:
                # rloss for each parameter:
                rlosses = []
                for j in range(n_paras):
                    rlosses.append(rloss_fu(res_pr[:,j], res_gt[:,j]))
                rlosses = torch.stack(rlosses) # => 5x1 or 5xn_treesw

                if self.weight_lookup is not None:                  
                    # replicate reduction='mean', but with weights:
                    lookups = regression_gt[i][:, self.weight_lookup_idx]
                    lookups = torch.floor(lookups).long()
                    weights = self.weight_lookup(lookups) 
                    rlosses = torch.sum(rlosses * weights, 1) / torch.sum(weights)
                    
                # mean instead of sum to stay consistent with rloss_fu("all"):
                if self.regression_weights is not None:
                    rlosses_w = torch.tensor(self.regression_weights, device=device)
                    rloss = torch.mean(rlosses * rlosses_w)  
                else:
                    rloss = torch.mean(rlosses)         
            else:
                rloss = rloss_fu(res_pr, res_gt)

                
            ## Loss for objectness:
            pos_weight = (N_neg / N_pos) if N_pos != 0 else 0
            pos_weight = 1 if N_neg == 0 else pos_weight
            pos_weight = torch.tensor(pos_weight, device=device)
            pos_weight *= self.objectness_pos_weight_factor
            oloss_fu = self.objectness_loss(weight=obj_w,
                                            pos_weight=pos_weight,
                                            reduction='mean')
            oloss = oloss_fu(obj_pr, obj_gt)


            ## Sum over mini-batch samples
            loss_sum += self.objectness_weight * oloss + \
                        self.regression_weight * rloss
            loss_n += 1
            oloss_sum += oloss
            rloss_sum += rloss
            rlosses_sum += rlosses.cpu()
            
            
        loss = torch.tensor(0) if loss_n == 0 else loss_sum / loss_n
        oloss = torch.tensor(0) if loss_n == 0 else oloss_sum / loss_n
        rloss = torch.tensor(0) if loss_n == 0 else rloss_sum / loss_n
        rlosses = torch.zeros(n_paras) if loss_n == 0 else rlosses_sum / loss_n
        rlosses = [l for l in rlosses]
        
        if return_singles:
            losses = [loss, oloss, rloss] + rlosses
            return losses
        else:
            return loss





class SoftDiceLoss(torch.nn.Module):
    """
    TODO: doc
    """
    def __init__(self, weight=None, size_average=True, ignore_index=[], eps=1e-8):
        super(SoftDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        smooth = self.eps

        num = input.shape[0] * input.shape[1]

        probs = torch.nn.functional.softmax(input, dim=1)
        gt = torch.zeros(len(target), probs.shape[1], device=target.device)
        gt.scatter_(1, target.unsqueeze(1), 1.)

        m1 = probs.view(num, -1)
        m2 = gt.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum() + smooth) / (m1.sum() + m2.sum() + smooth)
        loss = 1 - score
        return loss
