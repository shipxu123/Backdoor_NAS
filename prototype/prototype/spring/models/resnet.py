import torch
import torch.nn as nn

from prototype.spring.models.utils.normalize import build_norm_layer
from prototype.spring.models.utils.initializer import initialize_from_cfg
from prototype.spring.models.utils.modify import modify_state_dict

__all__ = ['ResNet',
           'resnet18c_x0_125',
           'resnet18c_x0_25',
           'resnet18c_x0_5',
           'resnet18',
           'resnet34',
           'resnet50',
           'resnet101',
           'resnet152',
           'resnext50_32x4d',
           'resnext101_32x8d',
           'wide_resnet50_2',
           'wide_resnet101_2',
           'resnet5400',
           'resnet_custom']


model_urls = {
    'resnet18c_x0_125': 'http://spring.sensetime.com/drop/$/DwzIl.pth',
    'resnet18c_x0_25': 'http://spring.sensetime.com/drop/$/4R4IN.pth',
    'resnet18c_x0_5': 'http://spring.sensetime.com/drop/$/HVYjx.pth',
    'resnet18': 'http://spring.sensetime.com/drop/$/mXqXX.pth',
    'resnet34': 'http://spring.sensetime.com/drop/$/pzweP.pth',
    'resnet50': 'http://spring.sensetime.com/drop/$/jqBS1.pth',
    'resnet101': 'http://spring.sensetime.com/drop/$/WLxUz.pth',
    'resnet152': 'http://spring.sensetime.com/drop/$/HA6Om.pth',
    'wide_resnet50_2': 'http://spring.sensetime.com/drop/$/zUwMl.pth',
    'wide_resnet101_2': 'http://spring.sensetime.com/drop/$/FmLI7.pth',
    'resnext50_32x4d': 'http://spring.sensetime.com/drop/$/L1zLi.pth',
    'resnext101_32x8d': 'http://spring.sensetime.com/drop/$/HaZrP.pth',
    'resnet18_imagenet22k': 'http://spring.sensetime.com/drop/$/2XK1M.pth',
    'resnet50_imagenet22k': 'http://spring.sensetime.com/drop/$/UqmmY.pth',
    'resnet152_imagenet22k': 'http://spring.sensetime.com/drop/$/33Oi4.pth',
    'resnext101_32x8d_imagenet22k': 'http://spring.sensetime.com/drop/$/HyHjl.pth',
}

model_performances = {
    'resnet18c_x0_125': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 0.543, 'accuracy': 33.012, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 1.021, 'accuracy': 33.012, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 3.308, 'accuracy': 33.012, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 2.525, 'accuracy': 33.718, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 22.327, 'accuracy': 33.718, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 182.243, 'accuracy': 33.718, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': 1.606, 'accuracy': 32.916, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': 10.669, 'accuracy': 32.916, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': 86.082, 'accuracy': 32.916, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 0.725, 'accuracy': 33.704, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 2.154, 'accuracy': 33.704, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 99.396, 'accuracy': 33.704, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 1.813, 'accuracy': 27.41, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 13.368, 'accuracy': 27.41, 'input_size': (3, 224, 224)},
    ],
    'resnet18c_x0_25': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 0.585, 'accuracy': 48.84, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 1.150, 'accuracy': 48.84, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 4.200, 'accuracy': 48.84, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 5.111, 'accuracy': 49.068, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 44.710, 'accuracy': 49.068, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 355.919, 'accuracy': 49.068, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': 2.264, 'accuracy': 48.222, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': 13.620, 'accuracy': 48.222, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': 107.024, 'accuracy': 48.222, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 0.703, 'accuracy': 49.046, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 2.259, 'accuracy': 49.046, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 14.837, 'accuracy': 49.046, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 1.836, 'accuracy': 47.564, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 14.234, 'accuracy': 47.564, 'input_size': (3, 224, 224)},
    ],
    'resnet18c_x0_5': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 0.729, 'accuracy': 61.59, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 1.605, 'accuracy': 61.59, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 6.715, 'accuracy': 61.59, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 15.050, 'accuracy': 61.58, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 112.540, 'accuracy': 61.58, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 936.028, 'accuracy': 61.58, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': 2.575, 'accuracy': 61.274, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': 20.571, 'accuracy': 61.274, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': 164.294, 'accuracy': 61.274, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 0.742, 'accuracy': 61.552, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 2.898, 'accuracy': 61.552, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 22.151, 'accuracy': 61.552, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 2.090, 'accuracy': 60.772, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 15.778, 'accuracy': 60.772, 'input_size': (3, 224, 224)},
    ],
    'resnet18': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 1.114, 'accuracy': 70.22, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 3.240, 'accuracy': 70.22, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 15.472, 'accuracy': 70.22, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 46.584, 'accuracy': 70.338, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 358.089, 'accuracy': 70.338, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 2903.838, 'accuracy': 70.338, 'input_size': (3, 224, 224)},

        # {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
        #     'latency': 21.722, 'accuracy': 70.25, 'input_size': (3, 224, 224)},
        # {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
        #     'latency': 167.996, 'accuracy': 70.25, 'input_size': (3, 224, 224)},
        # {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
        #     'latency': 1316.073, 'accuracy': 70.25, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 1.308, 'accuracy': 70.332, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 5.690, 'accuracy': 70.332, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 61.739, 'accuracy': 70.332, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 3.083, 'accuracy': 69.82, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 23.505, 'accuracy': 69.82, 'input_size': (3, 224, 224)},
    ],
    'resnet34': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 1.975, 'accuracy': 74.1, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 5.434, 'accuracy': 74.1, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 28.868, 'accuracy': 74.1, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 88.349, 'accuracy': 74.072, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 685.120, 'accuracy': 74.072, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 5654.022, 'accuracy': 74.072, 'input_size': (3, 224, 224)},

        # {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
        #     'latency': 16.146, 'accuracy': 73.966, 'input_size': (3, 224, 224)},
        # {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
        #     'latency': 193.207, 'accuracy': 73.966, 'input_size': (3, 224, 224)},
        # {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
        #     'latency': 1008.771, 'accuracy': 73.966, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 1.913, 'accuracy': 74.05, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 8.966, 'accuracy': 74.05, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 98.594, 'accuracy': 74.05, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 4.374, 'accuracy': 73.712, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 33.194, 'accuracy': 73.712, 'input_size': (3, 224, 224)},
    ],
    'resnet50': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 2.221, 'accuracy': 76.64, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 6.932, 'accuracy': 76.64, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 38.035, 'accuracy': 76.64, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 138.828, 'accuracy': 76.714, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 1316.171, 'accuracy': 76.714, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 10976.451, 'accuracy': 76.714, 'input_size': (3, 224, 224)},

        # {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
        #     'latency': 23.305, 'accuracy': , 'input_size': (3, 224, 224)},
        # {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
        #     'latency': 225.535, 'accuracy': , 'input_size': (3, 224, 224)},
        # {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
        #     'latency': 1364.937, 'accuracy': , 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 2.553, 'accuracy': 76.714, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 13.998, 'accuracy': 76.714, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 149.903, 'accuracy': 76.714, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 4.564, 'accuracy': 76.41, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 35.474, 'accuracy': 76.41, 'input_size': (3, 224, 224)},
    ],
    'resnet101': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 3.590, 'accuracy': 76.992, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 11.814, 'accuracy': 76.992, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 65.848, 'accuracy': 76.992, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 327.537, 'accuracy': 78.164, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 2484.786, 'accuracy': 78.164, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 20961.507, 'accuracy': 78.164, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 4.021, 'accuracy': 78.292, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 21.733, 'accuracy': 78.292, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 213.137, 'accuracy': 78.292, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 6.749, 'accuracy': 77.968, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 52.704, 'accuracy': 77.968, 'input_size': (3, 224, 224)},
    ],
    'resnet152': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 4.487, 'accuracy': 78.726, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 16.616, 'accuracy': 78.726, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 95.278, 'accuracy': 78.726, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 440.293, 'accuracy': 78.78, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 3912.053, 'accuracy': 78.78, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 25572.144, 'accuracy': 78.78, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 5.513, 'accuracy': 78.762, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 29.639, 'accuracy': 78.762, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 284.153, 'accuracy': 78.762, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 8.827, 'accuracy': 78.342, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 70.347, 'accuracy': 78.342, 'input_size': (3, 224, 224)},
    ],
    'wide_resnet50_2': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 3.700, 'accuracy': 78.072, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 13.971, 'accuracy': 78.072, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 84.779, 'accuracy': 78.072, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 272.586, 'accuracy': 78.128, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 2228.988, 'accuracy': 78.128, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 17541.762, 'accuracy': 78.128, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 4.843, 'accuracy': 78.132, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 33.839, 'accuracy': 78.132, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 311.964, 'accuracy': 78.132, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
    ],
    'wide_resnet101_2': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 6.152, 'accuracy': 78.986, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 26.144, 'accuracy': 78.986, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 161.571, 'accuracy': 78.986, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 637.290, 'accuracy': 79.068, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 6166.510, 'accuracy': 79.068, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 34836.213, 'accuracy': 79.068, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 7.757, 'accuracy': 79.062, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 55.179, 'accuracy': 79.062, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 510.502, 'accuracy': 79.062, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 16.262, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
    ],
    'resnext101_32x8d': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 20.816, 'accuracy': 79.328, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 34.596, 'accuracy': 79.328, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 162.154, 'accuracy': 79.328, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 365.051, 'accuracy': 79.404, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 3091.069, 'accuracy': 79.404, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 24787.624, 'accuracy': 79.404, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 23.254, 'accuracy': 79.396, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 171.498, 'accuracy': 79.396, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 1516.825, 'accuracy': 79.396, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
    ],
    'resnext50_32x4d': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 7.305, 'accuracy': 77.918, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 15.582, 'accuracy': 77.918, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 59.254, 'accuracy': 77.918, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 102.540, 'accuracy': 77.996, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 871.686, 'accuracy': 77.996, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 7232.253, 'accuracy': 77.996, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 4.823, 'accuracy': 78.008, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 33.830, 'accuracy': 78.008, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 312.180, 'accuracy': 78.008, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': None, 'accuracy': 0.098, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 35.526, 'accuracy': 0.098, 'input_size': (3, 224, 224)},
    ],
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


class ResNet(nn.Module):
    """Redidual Networks class, based on
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_
    """
    def __init__(self,
                 block,
                 layers,
                 inplanes=64,
                 scale=1.0,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 normalize={'type': 'solo_bn'},
                 initializer={'method': 'msra'},
                 ceil_mode=False,
                 deep_stem=False,
                 avg_down=False,
                 frozen_layers=[],
                 out_layers=[],
                 out_strides=[],
                 task='classification',):
        r"""
        Arguments:

        - block (:obj:`torch.nn.Module`): block type
        - layers (:obj:`list` of 4 ints): how many layers in each stage
        - inplanes (:obj:`int`): number of channels for the first convolutional layer
        - scale (:obj:`float`): channel scale
        - num_classes (:obj:`int`): number of classification classes
        - zero_init_residual (:obj:`bool`): zero-initialization the weights of the last bn in each block
        - groups (:obj:`int`): number of groups for convolutional layers
        - width_per_group (:obj:`int`): number of channels each group for convolutional layers
        - replace_stride_with_dilation (:obj:`bool`): replace stride=2 with dilation convolution
        - normalize (:obj:`dict`): configurations for normalize
        - initializer (:obj:`dict`): initializer method
        - ceil_mode (:obj:`bool`): if `True`, set the `ceil_mode=True` in maxpooling
        - deep_stem (:obj:`bool`): whether to use deep_stem as the first conv
        - avg_down (:obj:`bool`): whether to use avg_down when spatial downsample
        - frozen_layers (:obj:`list` of :obj:`int`): index of frozen layers, [0,4]
        - out_layers (:obj:`list` of :obj:`int`): index of output layers, [0,4]
        - out_strides (:obj:`list` of :obj:`int`): stride of outputs
        - task (:obj:`str`): type of task, 'classification' or object 'detection'
        """

        super(ResNet, self).__init__()

        norm_layer = build_norm_layer(normalize)

        self._norm_layer = norm_layer

        self.inplanes = int(inplanes * scale)
        self.dilation = 1
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.num_classes = num_classes
        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        self.task = task
        self.performance = None

        layer_out_planes = [64 * scale] + [i * block.expansion * scale for i in [64, 128, 256, 512]]
        self.out_planes = [int(layer_out_planes[i]) for i in self.out_layers]

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if self.deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, self.inplanes // 2, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(self.inplanes // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inplanes // 2, self.inplanes // 2, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(self.inplanes // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inplanes // 2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if ceil_mode:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, int(64 * scale), layers[0])
        self.layer2 = self._make_layer(block, int(128 * scale), layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(256 * scale), layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(512 * scale), layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # classifier only for classification task
        if self.task == 'classification':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(int(512 * scale) * block.expansion, num_classes)

        if initializer is not None:
            initialize_from_cfg(self, initializer)

        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params to optimizer
        self.freeze_layer()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride=stride,
                                 ceil_mode=True, count_include_pad=False),
                    conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, input):
        # See note [TorchScript super()]
        x = input['image'] if isinstance(input, dict) else input

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c0 = self.maxpool(x)

        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        if self.task == 'classification':
            x = self.avgpool(c4)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

        outs = [c0, c1, c2, c3, c4]
        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}

    def forward(self, x):
        return self._forward_impl(x)

    def freeze_layer(self):
        """
        Freezing layers during training.
        """
        layers = [
            nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool),
            self.layer1, self.layer2, self.layer3, self.layer4
        ]
        for layer_idx in self.frozen_layers:
            layer = layers[layer_idx]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            - module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.freeze_layer()
        return self

    def get_outplanes(self):
        """
        Get dimensions of the output tensors.

        Returns:
            - out (:obj:`list` of :obj:`int`)
        """
        return self.out_planes

    def get_outstrides(self):
        """
        Get strides of output tensors w.r.t inputs.

        Returns:
            - out (:obj:`list` of :obj:`int`)
        """
        return self.out_strides


def resnet18c_x0_125(pretrained=False, **kwargs):
    kwargs['scale'] = 0.125
    kwargs['ceil_mode'] = True
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model.performance = model_performances['resnet18c_x0_125']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet18c_x0_125'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18c_x0_25(pretrained=False, **kwargs):
    kwargs['scale'] = 0.25
    kwargs['ceil_mode'] = True
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model.performance = model_performances['resnet18c_x0_25']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet18c_x0_25'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18c_x0_5(pretrained=False, **kwargs):
    kwargs['scale'] = 0.5
    kwargs['ceil_mode'] = True
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model.performance = model_performances['resnet18c_x0_5']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet18c_x0_5'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, pretrain_type='imagenet1k', **kwargs):
    """
    Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model.performance = model_performances['resnet18']
    if pretrained:
        if pretrain_type == 'imagenet1k':
            _model_url = model_urls['resnet18']
        elif pretrain_type == 'imagenet22k':
            _model_url = model_urls['resnet18_imagenet22k']
            print({'name': "spring.models", 'action': 'load_imagenet22k_pretrain'})
        else:
            raise Exception('Unsupported pretrain type.')
        state_dict = torch.hub.load_state_dict_from_url(_model_url, map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


# def resnet26(pretrained=False, **kwargs):
#     """
#     Constructs a ResNet-26 model.
#     """
#     model = ResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
#     return model


def resnet34(pretrained=False, **kwargs):
    """
    Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    model.performance = model_performances['resnet34']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet34'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet50(pretrained=False, pretrain_type='imagenet1k', **kwargs):
    """
    Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model.performance = model_performances['resnet50']
    if pretrained:
        if pretrain_type == 'imagenet1k':
            _model_url = model_urls['resnet50']
        elif pretrain_type == 'imagenet22k':
            _model_url = model_urls['resnet50_imagenet22k']
            print({'name': "spring.models", 'action': 'load_imagenet22k_pretrain'})
        else:
            raise Exception('Unsupported pretrain type.')
        state_dict = torch.hub.load_state_dict_from_url(_model_url, map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """
    Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    model.performance = model_performances['resnet101']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet101'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet152(pretrained=False, pretrain_type='imagenet1k', **kwargs):
    """
    Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    model.performance = model_performances['resnet152']
    if pretrained:
        if pretrain_type == 'imagenet1k':
            _model_url = model_urls['resnet152']
        elif pretrain_type == 'imagenet22k':
            _model_url = model_urls['resnet152_imagenet22k']
            if get_rank() == 0:
                print({'name': "spring.models", 'action': 'load_imagenet22k_pretrain'})
        else:
            raise Exception('Unsupported pretrain type.')
        state_dict = torch.hub.load_state_dict_from_url(_model_url, map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnext50_32x4d(pretrained=False, **kwargs):
    """
    Construct a ResNeXt-50 model with 32 groups (4 channels per group)
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model.performance = model_performances['resnext50_32x4d']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['resnext50_32x4d'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnext101_32x8d(pretrained=False, pretrain_type='imagenet1k', **kwargs):
    """
    Construct a ResNeXt-101 model with 32 groups (8 channels per group)
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    model.performance = model_performances['resnext101_32x8d']
    if pretrained:
        if pretrain_type == 'imagenet1k':
            _model_url = model_urls['resnext101_32x8d']
        elif pretrain_type == 'imagenet22k':
            _model_url = model_urls['resnext101_32x8d_imagenet22k']
            if get_rank() == 0:
                print({'name': "spring.models", 'action': 'load_imagenet22k_pretrain'})
        else:
            raise Exception('Unsupported pretrain type.')
        state_dict = torch.hub.load_state_dict_from_url(_model_url, map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def wide_resnet50_2(pretrained=False, **kwargs):
    """
    Construct a Wide-ResNet-50 model with double channels
    """
    kwargs['width_per_group'] = 64 * 2
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model.performance = model_performances['wide_resnet50_2']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['wide_resnet50_2'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def wide_resnet101_2(pretrained=False, **kwargs):
    """
    Construct a Wide-ResNet-101 model with double channels
    """
    kwargs['width_per_group'] = 64 * 2
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    model.performance = model_performances['wide_resnet101_2']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['wide_resnet101_2'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet_custom(**kwargs):
    """
    Constructs a custom ResNet model with custom block and depth.
    """
    assert 'block' in kwargs and 'layers' in kwargs, 'Require block and layers'
    block = kwargs.pop('block')
    layers = kwargs.pop('layers')
    if block == 'basic':
        block = BasicBlock
    elif block == 'bottleneck':
        block = Bottleneck
    else:
        raise Exception('Unsupported block type.')
    model = ResNet(block, layers, **kwargs)
    return model


def resnet5400(pretrained=False, **kwargs):
    layers = [3, 13, 30, 3]
    layers = [int(depth * 3.05) for depth in layers]
    kwargs['layers'] = layers
    kwargs['scale'] = 4.125
    model = ResNet(Bottleneck, **kwargs)
    return model


if __name__ == '__main__':
    det_model = resnet5400(
        pretrained=True,
        normalize={'type': 'freeze_bn'},
        frozen_layers=[0, 1],
        out_layers=[2, 3, 4],
        out_strides=[8, 16, 32],
        task='detection',
    )
    cls_model = resnet5400(
        pretrained=True,
        num_classes=10,
        task='classification',
    )
    input = torch.randn(4, 3, 224, 224)
    det_output = det_model({'image': input})
    cls_output = cls_model(input)
    # function test
    det_model.train()
    print(det_model.get_outstrides())
    print(det_model.get_outplanes())
    # output test
    print('detection output size: {}'.format(det_output['features'][0].size()))
    print('detection output size: {}'.format(det_output['features'][1].size()))
    print('detection output size: {}'.format(det_output['features'][2].size()))
    print('classification output size: {}'.format(cls_output.size()))
