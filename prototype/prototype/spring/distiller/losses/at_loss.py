import torch.nn as nn
import torch.nn.functional as F


class ATLoss(nn.Module):
    """
    Paying More Attention to Attention:
    Improving the Performance of Convolutional Neural Networks via Attention Transfer, ICLR2017.
    https://arxiv.org/pdf/1612.03928.pdf
    """
    def __init__(self):
        super(ATLoss, self).__init__()

    def forward(self, s_features, t_features, **kwargs):
        return sum([(self.attention(x) - self.attention(y)).pow(2).mean() for x, y in zip(s_features, t_features)])

    def attention(self, x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
