import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_fn_multiclass(pred, target, num_classes, eps=1e-5):
    """
    Dice loss function for multiclass classification
    
    Parameters:
    - pred (torch.Tensor): The predicted probabilities of shape (:, num_classes, ...).
    - target (torch.Tensor): The true classes of shape (:, ...).
    - num_classes (int): The number of classes
    - eps (float): epsilon value to avoid division by zero
    
    Returns:
    - dice_loss (torch.Tensor): The average Dice loss for all classes
    """
    pred = F.softmax(pred, dim=1)
    
    dice_loss = 0.0
    for class_idx in range(num_classes):
        class_true = (target == class_idx).float()
        class_pred = pred[:, class_idx, ...]
        
        intersection = torch.sum(class_true * class_pred)
        union = torch.sum(class_true) + torch.sum(class_pred)
        
        class_dice = (2.0 * intersection + eps) / (union + eps)
        dice_loss += 1.0 - class_dice
    
    return dice_loss / num_classes  # average over classes



class DiceLoss(nn.Module):
    """
    Dice loss module for multiclass classification
    """
    def __init__(self, num_classes, eps=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        
    def forward(self, pred, target):
        return dice_loss_fn_multiclass(pred, target, self.num_classes, self.eps)
    
    
FocalLoss = torch.hub.load(
                'adeelh/pytorch-multi-class-focal-loss',
                model='FocalLoss',
                alpha=None,
                gamma=2,
                reduction='mean',
                force_reload=False,
            )


class SumLoss(nn.Module):
    """
    Sum of several loss functions
    """
    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses=torch.nn.ModuleList(losses)
        self.weights=weights
        if weights is None:
            self.weights=[1. for _ in self.losses]
    
    def forward(self, pred, target):
        loss=0.0
        for l,w in zip(self.losses,self.weights):
            loss+=w*l(pred,target)
        return loss/sum(self.weights)