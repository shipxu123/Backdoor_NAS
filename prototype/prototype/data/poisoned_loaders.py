import torch
import math
import random
import os
import argparse
from torch.utils.data import DataLoader

from prototype.data.marks import Watermark

class PoisonedDataLoader(DataLoader):
    def __init__(self, mark: Watermark = None, keep_org: bool = True, poison_label: bool=True, target_class: int = 0, poison_percent: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        if not Watermark:
            self.mark: Watermark = mark
        else:
            self.mark  = Watermark()
        self.keep_org = keep_org
        self.poison_label = poison_label
        self.target_class: int = target_class
        self.poison_percent: float = poison_percent
        self.poison_num = self.batch_size * self.poison_percent

    def __iter__(self):
        _iter = super().__iter__()
        return (self.get_data(data, self.keep_org, self.poison_label) for data in _iter)

    def add_mark(self, x: torch.Tensor, **kwargs):
        return self.mark.add_mark(x, **kwargs)

    def get_data(self, data, keep_org: bool = True, poison_label=True, **kwargs):
        _input, _label = data
        decimal, integer = math.modf(self.poison_num)
        integer = int(integer)
        if random.uniform(0, 1) < decimal:
            integer += 1
        if not keep_org:
            integer = len(_label)
        if not keep_org or integer:
            org_input, org_label = _input, _label
            _input = self.add_mark(org_input[:integer])
            _label = _label[:integer]
            if poison_label:
                _label = self.target_class * torch.ones_like(org_label[:integer])
            if keep_org:
                _input = torch.cat((_input, org_input))
                _label = torch.cat((_label, org_label))
        return _input, _label
