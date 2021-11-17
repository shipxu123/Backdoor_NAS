import torch
import torch.nn as nn


class NSTLoss(nn.Module):
    """
    Like What You Like: Knowledge Distill via Neuron Selectivity Transfer
    https://arxiv.org/pdf/1707.01219.pdf
    """
    def __init__(self):
        super(NSTLoss, self).__init__()

    def forward(self, s_features, t_features, **kwargs):
        return sum([self.nst(x, y) for x, y in zip(s_features, t_features)])

    def nst(self, x, y):
        x = x.view(x.shape[0], x.shape[1], -1)
        y = y.view(y.shape[0], y.shape[1], -1)
        xt = torch.transpose(x, 1, 2)
        yt = torch.transpose(y, 1, 2)
        xm = torch.bmm(xt, x).sum(0)
        ym = torch.bmm(yt, y).sum(0)

        return torch.norm(xm / torch.norm(xm) - ym / torch.norm(ym)) ** 2
