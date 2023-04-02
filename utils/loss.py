
"""
Loss for brain segmentaion (not used)
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from medpy import metric
import numpy as np



def entropy_loss(p, c=3):
    # p N*C*W*H*D
     p = F.softmax(p, dim=1)
     y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=0) / torch.tensor(np.log(c)).cuda()
     ent = torch.mean(y1)
     return ent

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def dice_coef(self, pred, gt):
        """ computational formula
        """
       
        softmax_pred = torch.nn.functional.softmax(pred, dim=1)
        seg_pred = torch.argmax(softmax_pred, dim=1)
        all_dice = 0
        gt = gt.squeeze(dim=1)
        batch_size = gt.shape[0]
        num_class = softmax_pred.shape[1]
        for i in range(num_class):

            each_pred = torch.zeros_like(seg_pred)
            each_pred[seg_pred==i] = 1

            each_gt = torch.zeros_like(gt)
            each_gt[gt==i] = 1            

        
            intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)
            
            union = each_pred.view(batch_size,-1).sum(1) + each_gt.view(batch_size,-1).sum(1)
            dice = (2. *  intersection )/ (union + 1e-5)
         
            all_dice += torch.mean(dice)
 
        return all_dice * 1.0 / num_class


    def forward(self, pred, gt):
        sigmoid_pred = F.softmax(pred,dim=1)
        

        batch_size = gt.shape[0]
        num_class = sigmoid_pred.shape[1]
        
        # conver label to one-hot
        bg = torch.zeros_like(gt)
        bg[gt==0] = 1
        label1 = torch.zeros_like(gt)
        label1[gt==1] = 1
        label2 = torch.zeros_like(gt)
        label2[gt == 2] = 1
        label = torch.cat([bg, label1, label2], dim=1)
        
        loss = 0
        smooth = 1e-5


        for i in range(num_class):
            intersect = torch.sum(sigmoid_pred[:, i, ...] * label[:, i, ...])
            z_sum = torch.sum(sigmoid_pred[:, i, ...] )
            y_sum = torch.sum(label[:, i, ...] )
            loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss * 1.0 / num_class
        return loss
        
