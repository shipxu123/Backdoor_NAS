import os

import copy
import numpy as np
import random
import torch
import argparse
import logging

from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torchvision.transforms as transforms

from PIL import Image, ImageFilter
import matplotlib.cm as mpl_color_map

import pdb

class NaiveVisualizer(object):
    def __init__(self, mean, std):
        denorm_mean = [-m for m in mean]
        denorm_std = [(1.0 / s) for s in std]

        self.denorm = transforms.Compose(
            [transforms.Normalize(mean=[ 0., 0., 0. ],
                                  std=denorm_std),
             transforms.Normalize(mean=denorm_mean,
                                  std=[ 1., 1., 1. ]),
          ])


    def apply_heatmap(self, images, heatmaps):
        images = images.cpu()
        images = [self.denorm(image) for image in images]
        origin_images = [transforms.ToPILImage()(img) for img in images]

        img_imgheat = [self.apply_colormap_on_image(img, heatmap) for img, heatmap in zip(origin_images, heatmaps)]

        heat, imgheat = zip(*img_imgheat)

        return origin_images, heat, imgheat


    def apply_colormap_on_image(self, image, activation, colormap_name='jet'):
        """
            Apply heatmap on image
        Args:
            image (PIL img): Original image
            activation_map (numpy arr): Activation map (grayscale) 0-255
            colormap_name (str): Name of the colormap
        """
        # Get colormap
        color_map = mpl_color_map.get_cmap(colormap_name)

#         if activation.max() > 1:
        if True:
            top = activation.max()
            bottom = activation.min()
            activation = (activation - bottom) / (top - bottom + 1e-12)

        no_trans_heatmap = color_map(activation)
        heatmap = copy.copy(no_trans_heatmap)
        heatmap[:, :, 3] = 0.5
        heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
        no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))

        if heatmap.size != image.size:
            heatmap = heatmap.resize(image.size,Image.BILINEAR)
            no_trans_heatmap = no_trans_heatmap.resize(image.size,Image.BILINEAR)

        # Apply heatmap on image
        heatmap_on_image = Image.new("RGBA", image.size)
        heatmap_on_image = Image.alpha_composite(heatmap_on_image, image.convert('RGBA'))
        heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
        return no_trans_heatmap, heatmap_on_image


def naive_heatmap(target):
    """
    :param target: Tensor; N, C, H, W
    :return:
    """
    norms = torch.norm(target, p=2, dim=1, keepdim=False)
    return norms


def retrivel_heatmap(target, retrivel):
    """
    :param target: Tensor; N,C,H,W
    :param retrivel: Tensor; N,C
    :return:
    """
    N,C,H,W = target.size()
    activation_map = target * retrivel.unsqueeze(-1).unsqueeze(-1)
    # pdb.set_trace()
    activation_map = activation_map.mean(dim=1)

    return activation_map

# def retrivel_heatmap(target, retrivel):
#     """
#     :param target: Tensor; N1, C, H, W
#     :param retrivel: Tensor; N2, C
#     :return:
#     """
#     N1, C, H, W = target.size()
#     N2, _ = retrivel.size()

#     target_t = target.permute(0,2,3,1).reshape(-1, C)
#     activation = target_t.mm(retrivel.t())
#     activation_map = activation.reshape(N1, H, W, N2)
#     activation_map = activation_map.permute(3, 0, 1, 2)

#     return activation_map


# def scale_cat(big1, small4):
#     """
#     :param big1: C,H,W
#     :param small4: 4,C,H,W
#     :return:
#     """
#     big = F.interpolate(big1, (2,2), mode='bilinear')

#     small1 = torch.cat(small4[:2], dim=1)
#     small2 = torch.cat(small4[2:], dim=1)
#     small = torch.cat([small1,small2], dim=2)

#     target = torch.cat([big, small], dim=-2)

#     return target


def scale_cat(big1, small4):
    """

    :param big1: N,C,H,W
    :param small4: 4N,C,H,W
    :return:
    """
    N = big1.size(0)
    tran = transforms.Compose([
        transforms.Normalize(mean=[ 0., 0., 0. ],
        std=[1/0.2023, 1/0.1994, 1/0.2010]),
        transforms.Normalize(mean=[-0.4914, -0.4822,- 0.4465],
        std=[ 1., 1., 1. ]),
        transforms.ToPILImage(),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    big_processed = [tran(i) for i in big1]
    big = torch.stack(big_processed).contiguous()

    small1, small2, small3, small4 = small4.split(N, dim=0)

    small_up = torch.cat([small1, small2], dim=2)
    small_down = torch.cat([small3, small4], dim=2)
    small = torch.cat([small_up,small_down], dim=3)

    target = torch.cat([big, small], dim=-2)

    return target