import torch
from torch import nn


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_l2_norm(self, x1, x2):
        return torch.norm(x1 - x2, p=2, dim=1)

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ):
        distance_positive = self.calc_l2_norm(anchor, positive)
        distance_negative = self.calc_l2_norm(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
