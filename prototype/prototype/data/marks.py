#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import torch
import argparse
import PIL.Image as Image
from collections.abc import Callable
from typing import Union

import os
import math
import torchvision.transforms.functional as F
from PIL import Image

byte2float = F.to_tensor

# ------------------- Format Transform --------------------------- #
def to_tensor(x: Union[torch.Tensor, np.ndarray, list, Image.Image],
              dtype: Union[str, torch.dtype] = None,
              device: Union[str, torch.device] = 'default',
              **kwargs) -> torch.Tensor:
    _map = {'int': torch.int, 'float': torch.float,
            'double': torch.double, 'long': torch.long}

    if x is None:
        return None
    if isinstance(dtype, str):
        dtype = _map[dtype]

    if device == 'default':
        device = env['device']

    if isinstance(x, (list, tuple)):
        try:
            x = torch.stack(x)
        except TypeError:
            pass
    elif isinstance(x, Image.Image):
        x = byte2float(x)
    try:
        x = torch.as_tensor(x, dtype=dtype).to(device=device, **kwargs)
    except Exception as e:
        print('tensor: ', x)
        if torch.is_tensor(x):
            print('shape: ', x.shape)
            print('device: ', x.device)
        raise e
    return x

def to_numpy(x, **kwargs) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x, **kwargs)

def to_pil_image(x: Union[torch.Tensor, np.ndarray, list, Image.Image], mode=None) -> Image.Image:
    # TODO: Linting for mode
    if isinstance(x, Image.Image):
        return x
    x = to_tensor(x, device='cpu')
    return F.to_pil_image(x, mode=mode)

def gray_img(x: Union[torch.Tensor, np.ndarray, Image.Image], num_output_channels: int = 1) -> Image.Image:
    if not isinstance(x, Image.Image):
        x = to_pil_image(x)
    return F.to_grayscale(x, num_output_channels=num_output_channels)

class Watermark:
    name: str = 'mark'

    def __init__(self, mark_path: str = '/data/home/scv3915/xupeng/codes/trojanzoo/trojanvision/marks/square_white.png',
                 data_shape = [3, 32, 32], edge_color = 'auto',
                 mark_alpha: float = 0.0, mark_height: int = 3, mark_width: int = 3,
                 height_offset: int = 0, width_offset: int = 0,
                 random_pos=False, random_init=False, mark_distributed=False,
                 **kwargs):
        self.param_list: dict[str, list[str]] = {}
        self.param_list['mark'] = ['mark_path', 'data_shape', 'edge_color',
                                   'mark_alpha', 'mark_height', 'mark_width']
        assert mark_height > 0 and mark_width > 0
        # --------------------------------------------------- #
        # WaterMark Image Parameters
        self.mark_alpha: float = mark_alpha
        self.data_shape: list[int] = data_shape
        self.mark_path: str = mark_path
        self.mark_height: int = mark_height
        self.mark_width: int = mark_width
        # --------------------------------------------------- #
        org_mark_img: Image.Image = self.load_img(img_path=mark_path,
                                                  height=mark_height, width=mark_width, channel=data_shape[0])
        self.org_mark: torch.Tensor = byte2float(org_mark_img)
        self.edge_color: torch.Tensor = self.get_edge_color(
            self.org_mark, data_shape, edge_color)
        self.org_mask, self.org_alpha_mask = self.org_mask_mark(self.org_mark, self.edge_color, self.mark_alpha)

        self.param_list['mark'].extend(['height_offset', 'width_offset'])
        self.height_offset: int = height_offset
        self.width_offset: int = width_offset
        self.mark, self.mask, self.alpha_mask = self.mask_mark()

    @staticmethod
    def get_edge_color(mark: torch.Tensor, data_shape, edge_color = 'auto'):

        assert data_shape[0] == mark.shape[0]
        t: torch.Tensor = torch.zeros(data_shape[0], dtype=torch.float)
        if isinstance(edge_color, str):
            if edge_color == 'black':
                pass
            elif edge_color == 'white':
                t += 1
            elif edge_color == 'auto':
                mark = mark.transpose(0, -1)
                if mark.flatten(start_dim=1).std(dim=1).max() < 1e-3:
                    t = -torch.ones_like(mark[0, 0])
                else:
                    _list = [mark[0, :, :], mark[-1, :, :],
                             mark[:, 0, :], mark[:, -1, :]]
                    _list = torch.cat(_list)
                    t = _list.mode(dim=0)[0]
            else:
                raise ValueError(edge_color)
        else:
            t = torch.as_tensor(edge_color)
            assert t.dim() == 1
            assert t.shape[0] == data_shape[0]
        return t

    @staticmethod
    def org_mask_mark(org_mark: torch.Tensor, edge_color: torch.Tensor, mark_alpha: float):
        height, width = org_mark.shape[-2:]
        mark = torch.zeros_like(org_mark, dtype=torch.float)
        mask = torch.zeros([height, width], dtype=torch.bool)
        for i in range(height):
            for j in range(width):
                if not org_mark[:, i, j].equal(edge_color):
                    mark[:, i, j] = org_mark[:, i, j]
                    mask[i, j] = 1
        alpha_mask = mask * (1 - mark_alpha)
        return mask, alpha_mask

    def mask_mark(self, org_mark: torch.Tensor = None, org_mask: torch.Tensor = None, org_alpha_mask: torch.Tensor = None,
                  height_offset: int = None, width_offset: int = None):
        if org_mark is None:
            org_mark = self.org_mark
        if org_mask is None:
            org_mask = self.org_mask
        if org_alpha_mask is None:
            org_alpha_mask = self.org_alpha_mask
        if height_offset is None:
            height_offset = self.height_offset
        if width_offset is None:
            width_offset = self.width_offset
        mark = -torch.ones(self.data_shape, dtype=torch.float)
        mask = torch.zeros(self.data_shape[-2:], dtype=torch.bool)
        alpha_mask = torch.zeros_like(mask, dtype=torch.float)

        start_h = height_offset
        start_w = width_offset
        end_h = height_offset + self.mark_height
        end_w = width_offset + self.mark_width

        mark[:, start_h:end_h, start_w:end_w] = org_mark
        mask[start_h:end_h, start_w:end_w] = org_mask
        alpha_mask[start_h:end_h, start_w:end_w] = org_alpha_mask
        return mark, mask, alpha_mask
    
    # add mark to the Image with mask.
    def add_mark(self, _input: torch.Tensor, alpha: float = None, **kwargs):
        mark, mask, alpha_mask = self.mark, self.mask, self.alpha_mask
        if alpha is not None:
            alpha_mask = torch.ones_like(self.alpha_mask) * (1 - alpha)
        _mask = mask * alpha_mask
        mark, _mask = mark.to(_input.device), _mask.to(_input.device)
        return _input + _mask * (mark - _input)

    @staticmethod
    def load_img(img_path: str, height: int, width: int, channel: int = 3):
        if not os.path.exists(img_path) and not os.path.exists(img_path):
            raise FileNotFoundError(img_path)
        mark: Image.Image = Image.open(img_path)
        mark = mark.resize((width, height), Image.ANTIALIAS)

        if channel == 1:
            mark = gray_img(mark, num_output_channels=1)
        elif channel == 3 and mark.mode in ['1', 'L']:
            mark = gray_img(mark, num_output_channels=3)
        return mark

    def save_img(self, img_path: str):
        img = self.org_mark * self.org_mask if self.random_pos else self.mark * self.mask
        save_tensor_as_img(img_path, img)
