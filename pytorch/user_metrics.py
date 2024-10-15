import torch
from torch import nn
from torchmetrics.functional.classification import dice

class DiceLoss(nn.Module):
    def __init__(self, reduction='micro'):
        super(DiceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, preds, targets):
        dice_value = dice(preds, targets, average=self.reduction)

        return 1 - dice_value

class MMetric(nn.Module):
    def __init__(self, alpha_sen, alpha_spe):
        super(MMetric, self).__init__()
        self.alpha_sen = alpha_sen
        self.alpha_spe = alpha_spe

    def forward(self, sen, spe):
        return self.alpha_sen * sen + self.alpha_spe * spe
