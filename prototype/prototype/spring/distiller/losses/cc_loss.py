import torch
import torch.nn as nn


class CCLoss(nn.Module):
    """
    Correlation Congruence for Knowledge Distillation, ICCV 2019.
    https://arxiv.org/pdf/1904.01802.pdf
    """
    def __init__(self):
        super(CCLoss, self).__init__()

    def forward(self, s_features, t_features, **kwargs):
        loss = 0
        for f_s, f_t in zip(s_features, t_features):
            delta = torch.abs(f_s - f_t)
            loss += torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss
