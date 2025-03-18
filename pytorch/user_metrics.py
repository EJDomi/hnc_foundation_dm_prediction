import torch
from torch import nn
from torch import Tensor
from torchmetrics.functional.classification import dice
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from torchmtlr import mtlr_risk
from lifelines.utils import concordance_index

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


class ConcordanceIndex(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('preds', default=[], dist_reduce_fx='cat')
        self.add_state('times', default=[], dist_reduce_fx='cat')
        self.add_state('events', default=[], dist_reduce_fx='cat')

    def update(self, preds: Tensor, times: Tensor, events: Tensor) -> None:
        self.preds.append(preds)
        self.times.append(times)
        self.events.append(events)

    def compute(self):
        preds = dim_zero_cat(self.preds).cpu()
        times = dim_zero_cat(self.times).cpu()
        events = dim_zero_cat(self.events).cpu()

        pred_risk = mtlr_risk(preds).detach().numpy()
        return torch.tensor(concordance_index(times, -pred_risk, event_observed=events))

    full_state_update: bool = True
