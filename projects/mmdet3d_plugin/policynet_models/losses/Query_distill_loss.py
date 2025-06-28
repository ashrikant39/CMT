import torch
import torch.nn as nn
from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class QueryDistillLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(QueryDistillLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred_query, gt_query):
        # pred_query and gt_query are both (B, 5, 900, 256)
        
        # Compute a simple L2 (MSE) loss:
        diff = pred_query - gt_query
        loss = (diff ** 2).sum(dim=-1)  # sum over the last dimension (256)
        # now loss is (B, 5, 900)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        # else, 'none' would return the raw loss

        return self.loss_weight * loss
