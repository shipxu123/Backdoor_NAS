import torch
import torch.nn as nn


class SPLoss(nn.Module):
    """
    Similarity-Preserving Knowledge Distillation, ICCV2019
    https://arxiv.org/pdf/1907.09682.pdf
    """
    def __init__(self):
        super(SPLoss, self).__init__()

    def forward(self, s_features, t_features, **kwargs):
        return sum([self.similarity(f_s, f_t) for f_s, f_t in zip(s_features, t_features)])

    def similarity(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)
        G_s = torch.mm(f_s, torch.t(f_s))
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        G_t = torch.nn.functional.normalize(G_t)
        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss
