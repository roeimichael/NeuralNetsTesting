import torch  # type: ignore
from torch import nn  # type: ignore
import numpy as np
import torch.nn.functional as F
import time


class CostSensitiveLoss(nn.Module):
    def __init__(self, weight, cost_matrix, reduction):
        super().__init__()
        self.cost_matrix = cost_matrix
        self.TP_weight = self.cost_matrix[1, 1]
        self.TN_weight = self.cost_matrix[0, 0]
        self.FP_weight = self.cost_matrix[0, 1]
        self.FN_weight = self.cost_matrix[1, 0]
        self.weight = weight
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        epsilon = 1e-8
        TP = -1 * (0 + y_true) * (torch.log(1 + ((0 + y_pred) - 1) * (
                1 - torch.tanh(self.weight * (torch.max(y_pred, 1 - y_pred) - y_pred))) + epsilon))
        FP = -1 * (1 - y_true) * (torch.log(1 + ((1 - y_pred) - 1) * (
                1 - torch.tanh(self.weight * (torch.max(y_pred, 1 - y_pred) - y_pred))) + epsilon))
        TN = -1 * (1 - y_true) * (torch.log(1 + ((1 - y_pred) - 1) * (
                0 + torch.tanh(self.weight * (torch.max(y_pred, 1 - y_pred) - y_pred))) + epsilon))
        FN = -1 * (0 + y_true) * (torch.log(1 + ((0 + y_pred) - 1) * (
                0 + torch.tanh(self.weight * (torch.max(y_pred, 1 - y_pred) - y_pred))) + epsilon))

        out = (TP * self.TP_weight + FP * self.FP_weight + TN * self.TN_weight + FN * self.FN_weight)

        if not self.reduction:
            return out

        if self.reduction == "mean":
            return out.mean()

        if self.reduction == "sum":
            return out.sum()
