import torch
import torch.nn as nn
import copy
from queue import deque


class CRDLoss(nn.Module):
    """
    Variate CRD Loss, ICLR 2020.
    https://arxiv.org/abs/1910.10699
    """
    def __init__(self, s_trans, t_trans, output_dim=128, T=0.07,
                 momentum=0, neg_num=16384, max_deque_length=50000, use_labels=False):
        super().__init__()
        self.s_trans = s_trans
        self.t_trans = t_trans
        self.output_dim = output_dim
        self.tempreture = T
        self.momentum = momentum
        self.neg_num = neg_num
        self.deque = deque()
        self.max_deque_length = max_deque_length
        self.t_trans_memory = None
        self.use_labels = use_labels
        if use_labels:
            self.label_deque = deque()

    def forward(self, s_feats, t_feats, **kwargs):
        label = self.check_input(**kwargs)
        assert len(s_feats) == 1, 'you should register only one layer in a single crd mimicjob.'
        s_feats = s_feats[0]
        t_feats = t_feats[0]
        self.momentum_update()
        self.t_trans_memory = copy.deepcopy(self.t_trans.state_dict())
        DQ_EMPTY_FLAG = len(self.deque) == 0
        reserved_t_feats = t_feats.detach().clone()
        s_feats = self.norm_feats(self.s_trans(s_feats))
        t_feats = self.norm_feats(self.t_trans(t_feats))
        if not DQ_EMPTY_FLAG:
            n_feats = torch.cat(list(self.deque))
            selected_neg_idx = torch.randperm(n_feats.size(0))[:min(self.neg_num, n_feats.size(0))]
            n_feats = n_feats[selected_neg_idx]
            n_feats = self.norm_feats(self.t_trans(n_feats))
            sn_feats = torch.cat([s_feats, n_feats])
            tn_feats = torch.cat([t_feats, n_feats])
            if self.use_labels:
                all_labels = torch.cat(list(self.label_deque))[selected_neg_idx]
                all_labels = torch.cat([label, all_labels])
        else:
            sn_feats = s_feats
            tn_feats = t_feats
            if self.use_labels:
                all_labels = label

        if len(self.deque) >= self.max_deque_length // s_feats.size(0):
            self.deque.popleft()
            if self.use_labels:
                self.label_deque.popleft()

        self.deque.append(reserved_t_feats)
        if self.use_labels:
            self.label_deque.append(label)

        s_tn = torch.mm(s_feats, tn_feats.transpose(0, 1)) / self.tempreture
        t_sn = torch.mm(t_feats, sn_feats.transpose(0, 1)) / self.tempreture
        gts = torch.arange(s_tn.size(0)).long().to(device=s_tn.device)

        if self.use_labels:
            masks = ~(label.view(-1, 1) == all_labels.view(1, -1))
            masks[range(masks.size(0)), range(masks.size(0))] = 1
            s_tn += (masks.float() + 1e-45).log()
            t_sn += (masks.float() + 1e-45).log()

        loss_s = torch.nn.functional.cross_entropy(s_tn, gts)
        loss_t = torch.nn.functional.cross_entropy(t_sn, gts)
        loss = (loss_s + loss_t) / 2
        return loss

    def momentum_update(self):
        if self.t_trans_memory is None:
            return
        for p in self.t_trans.named_parameters():
            old_param = self.t_trans_memory[p[0]]
            p[1].data.mul_(1 - self.momentum).add_(old_param, alpha=self.momentum)

    def norm_feats(self, feats, norm=2):
        assert len(feats.shape) == 2, 'feats after transformation must \
        be a {}-dim Tensor.'.format(self.output_dim)
        normed_feats = feats / feats.norm(norm, dim=1, keepdim=True)
        return normed_feats

    def check_input(self, **kwargs):
        if self.use_labels:
            assert 'label' in kwargs, 'you should pass a dict with key \'label\' in mimic function.'
            return kwargs['label']
        else:
            return None
