import torch.nn as nn
import torch
from typing import Dict, Callable


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        bce = nn.BCELoss(reduction='none')(y_pred, y_true)
        p_t = torch.where(y_true == 1, y_pred, 1 - y_pred)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = self.alpha * modulating_factor * bce
        return loss.mean()
