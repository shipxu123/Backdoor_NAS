import torch
import torch.nn as nn


class ABLoss(nn.Module):
    """
    Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons, AAAI2019.
    https://arxiv.org/pdf/1811.03233.pdf
    """
    def __init__(self, margin=1.0):
        super(ABLoss, self).__init__()
        self.margin = margin

    def forward(self, s_features, t_features, **kwargs):
        feat_num = len(s_features)
        w = [2 ** (i - feat_num + 1) for i in range(feat_num)]
        bsz = s_features[0].shape[0]
        losses = [self.criterion_alternative_l2(s, t) for s, t in zip(s_features, t_features)]
        losses = [w * l for w, l in zip(w, losses)]
        losses = [l / bsz for l in losses]
        losses = [l / 1000 * 3 for l in losses]
        return sum(losses)

    def criterion_alternative_l2(self, source, target):
        loss = ((source + self.margin) ** 2 * ((source > -self.margin) & (target <= 0)).float()
                + (source - self.margin) ** 2 * ((source <= self.margin) & (target > 0)).float())
        return torch.abs(loss).sum()
