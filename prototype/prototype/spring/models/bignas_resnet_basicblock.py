from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prototype.spring.models.utils.normalize import build_norm_layer
from prototype.spring.models.utils.initializer import initialize_from_cfg
from prototype.spring.models.utils.modify import modify_state_dict


# equal to ResNet18-1/8, ResNet18-1/4, ResNet18-1/2, ResNet18 respectively
__all__ = ['bignas_resnet18_9M', 'bignas_resnet18_37M', 'bignas_resnet18_50M',
           'bignas_resnet18_49M', 'bignas_resnet18_65M',
           'bignas_resnet18_107M', 'bignas_resnet18_125M', 'bignas_resnet18_150M',
           'bignas_resnet18_312M', 'bignas_resnet18_403M', 'bignas_resnet18_492M',
           'bignas_resnet18_1555M',
           'bignas_resnet18_37M_pooling', 'bignas_resnet18_107M_pooling', 'bignas_resnet18_492M_pooling',
           'bignas_det_resnet18_1930M']


model_urls = {
    'bignas_resnet18_9M': 'http://spring.sensetime.com/drop/$/UwaSe.pth',
    'bignas_resnet18_37M': 'http://spring.sensetime.com/drop/$/xCPzO.pth',
    'bignas_resnet18_50M': 'http://spring.sensetime.com/drop/$/ZrK9i.pth',
    'bignas_resnet18_49M': 'http://spring.sensetime.com/drop/$/21T73.pth',
    'bignas_resnet18_65M': 'http://spring.sensetime.com/drop/$/91kG9.pth',
    'bignas_resnet18_107M': 'http://spring.sensetime.com/drop/$/mAzno.pth',
    'bignas_resnet18_125M': 'http://spring.sensetime.com/drop/$/c9KzS.pth',
    'bignas_resnet18_150M': 'http://spring.sensetime.com/drop/$/GIGzn.pth',
    'bignas_resnet18_312M': 'http://spring.sensetime.com/drop/$/U6tnd.pth',
    'bignas_resnet18_403M': 'http://spring.sensetime.com/drop/$/dNMWc.pth',
    'bignas_resnet18_492M': 'http://spring.sensetime.com/drop/$/qfTvY.pth',
    'bignas_resnet18_1555M': 'http://spring.sensetime.com/drop/$/obpif.pth',
    'bignas_det_resnet18_1930M': 'http://spring.sensetime.com/drop/$/SdO8J.pth',
}


model_performances = {
    'bignas_resnet18_9M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
         'latency': 0.318, 'accuracy': 28.328, 'input_size': (3, 128, 128)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
         'latency': 0.556, 'accuracy': 28.328, 'input_size': (3, 128, 128)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
         'latency': 0.960, 'accuracy': 28.328, 'input_size': (3, 128, 128)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
         'latency': 0.381, 'accuracy': 29.136, 'input_size': (3, 128, 128)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
         'latency': 2.992, 'accuracy': 29.136, 'input_size': (3, 128, 128)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
         'latency': 26.831, 'accuracy': 29.136, 'input_size': (3, 128, 128)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
         'latency': 1.139, 'accuracy': 28.158, 'input_size': (3, 128, 128)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
         'latency': 3.107, 'accuracy': 28.158, 'input_size': (3, 128, 128)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
         'latency': 23.558, 'accuracy': 28.158, 'input_size': (3, 128, 128)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
         'latency': 0.475, 'accuracy': 29.108, 'input_size': (3, 128, 128)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
         'latency': 0.848, 'accuracy': 29.108, 'input_size': (3, 128, 128)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
         'latency': 3.513, 'accuracy': 29.108, 'input_size': (3, 128, 128)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
         'latency': 0.739, 'accuracy': 26.866, 'input_size': (3, 128, 128)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
         'latency': 5.011, 'accuracy': 26.866, 'input_size': (3, 128, 128)},
    ],
    'bignas_resnet18_37M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
         'latency': 0.475, 'accuracy': 38.502, 'input_size': (3, 188, 188)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
         'latency': 0.753, 'accuracy': 38.502, 'input_size': (3, 188, 188)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
         'latency': 2.271, 'accuracy': 38.502, 'input_size': (3, 188, 188)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
         'latency': 1.95, 'accuracy': 39.2, 'input_size': (3, 188, 188)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
         'latency': 15.122, 'accuracy': 39.2, 'input_size': (3, 188, 188)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
         'latency': 134.292, 'accuracy': 39.2, 'input_size': (3, 188, 188)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
         'latency': 2.396, 'accuracy': 38.476, 'input_size': (3, 188, 188)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
         'latency': 7.670, 'accuracy': 38.476, 'input_size': (3, 188, 188)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
         'latency': 66.395, 'accuracy': 38.476, 'input_size': (3, 188, 188)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
         'latency': 0.586, 'accuracy': 39.208, 'input_size': (3, 188, 188)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
         'latency': 1.625, 'accuracy': 39.208, 'input_size': (3, 188, 188)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
         'latency': 10.057, 'accuracy': 39.208, 'input_size': (3, 188, 188)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
         'latency': 1.397, 'accuracy': 36.398, 'input_size': (3, 188, 188)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
         'latency': 12.260, 'accuracy': 36.398, 'input_size': (3, 188, 188)},
    ],
    'bignas_resnet18_49M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
         'latency': 0.517, 'accuracy': 40.644, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
         'latency': 0.967, 'accuracy': 40.644, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
         'latency': 3.040, 'accuracy': 40.644, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
         'latency': 3.058, 'accuracy': 40.896, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
         'latency': 23.865, 'accuracy': 40.896, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
         'latency': 226.802, 'accuracy': 40.896, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
         'latency': 1.969, 'accuracy': 40.06, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
         'latency': 11.690, 'accuracy': 40.06, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
         'latency': 78.883, 'accuracy': 40.06, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
         'latency': 0.664, 'accuracy': 40.91, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
         'latency': 2.060, 'accuracy': 40.91, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
         'latency': 14.002, 'accuracy': 40.91, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
         'latency': 1.818, 'accuracy': 39.13, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
         'latency': 16.754, 'accuracy': 39.13, 'input_size': (3, 224, 224)},
    ],
    'bignas_resnet18_50M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
         'latency': 0.573, 'accuracy': 42.152, 'input_size': (3, 192, 192)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
         'latency': 0.895, 'accuracy': 42.152, 'input_size': (3, 192, 192)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
         'latency': 2.757, 'accuracy': 42.152, 'input_size': (3, 192, 192)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
         'latency': 2.254, 'accuracy': 42.842, 'input_size': (3, 192, 192)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
         'latency': 18.419, 'accuracy': 42.842, 'input_size': (3, 192, 192)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
         'latency': 154.017, 'accuracy': 42.842, 'input_size': (3, 192, 192)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
         'latency': 1.412, 'accuracy': 41.926, 'input_size': (3, 192, 192)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
         'latency': 7.477, 'accuracy': 41.926, 'input_size': (3, 192, 192)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
         'latency': 69.946, 'accuracy': 41.926, 'input_size': (3, 192, 192)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
         'latency': 0.639, 'accuracy': 42.828, 'input_size': (3, 192, 192)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
         'latency': 1.743, 'accuracy': 42.828, 'input_size': (3, 192, 192)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
         'latency': 11.010, 'accuracy': 42.828, 'input_size': (3, 192, 192)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
         'latency': 1.536, 'accuracy': 40.078, 'input_size': (3, 192, 192)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
         'latency': 12.748, 'accuracy': 40.078, 'input_size': (3, 192, 192)},
    ],
    'bignas_resnet18_65M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
         'latency': 0.510, 'accuracy': 43.582, 'input_size': (3, 220, 220)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
         'latency': 0.919, 'accuracy': 43.582, 'input_size': (3, 220, 220)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
         'latency': 3.121, 'accuracy': 43.582, 'input_size': (3, 220, 220)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
         'latency': 3.342, 'accuracy': 42.918, 'input_size': (3, 220, 220)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
         'latency': 25.491, 'accuracy': 43.918, 'input_size': (3, 220, 220)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
         'latency': 217.448, 'accuracy': 43.918, 'input_size': (3, 220, 220)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
         'latency': 1.411, 'accuracy': 43.082, 'input_size': (3, 220, 220)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
         'latency': 11.602, 'accuracy': 43.082, 'input_size': (3, 220, 220)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
         'latency': 90.953, 'accuracy': 43.082, 'input_size': (3, 220, 220)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
         'latency': 0.610, 'accuracy': 43.918, 'input_size': (3, 220, 220)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
         'latency': 2.096, 'accuracy': 43.918, 'input_size': (3, 220, 220)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
         'latency': 13.804, 'accuracy': 43.918, 'input_size': (3, 220, 220)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
         'latency': 1.782, 'accuracy': 41.014, 'input_size': (3, 220, 220)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
         'latency': 13.043, 'accuracy': 41.014, 'input_size': (3, 220, 220)},
    ],
    'bignas_resnet18_66M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
         'latency': 0.514, 'accuracy': 44.268, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
         'latency': 0.965, 'accuracy': 44.268, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
         'latency': 3.169, 'accuracy': 44.268, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
         'latency': 5.537, 'accuracy': 44.774, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
         'latency': 30.019, 'accuracy': 44.774, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
         'latency': 221.161, 'accuracy': 44.774, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
         'latency': 1.512, 'accuracy': 43.726, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
         'latency': 10.070, 'accuracy': 43.726, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
         'latency': 94.863, 'accuracy': 43.726, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
         'latency': 0.638, 'accuracy': 44.758, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
         'latency': 2.075, 'accuracy': 44.758, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
         'latency': 14.077, 'accuracy': 44.758, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
         'latency': 2.252, 'accuracy': 41.682, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
         'latency': 14.445, 'accuracy': 41.682, 'input_size': (3, 224, 224)},
    ],
    'bignas_resnet18_107M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 0.570, 'accuracy': 51.87, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 1.054, 'accuracy': 51.87, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 3.180, 'accuracy': 51.87, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 3.239, 'accuracy': 2.096, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 27.707, 'accuracy': 2.096, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 248.652, 'accuracy': 52.096, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': 2.064, 'accuracy': 51.398, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': 13.467, 'accuracy': 51.398, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': 103.497, 'accuracy': 51.398, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 0.607, 'accuracy': 52.136, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 1.696, 'accuracy': 52.136, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 11.310, 'accuracy': 52.136, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 1.747, 'accuracy': 50.838, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 13.275, 'accuracy': 50.838, 'input_size': (3, 224, 224)},
    ],
    'bignas_resnet18_125M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 1,
            'latency': 0.532, 'accuracy': 52.934, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 8,
            'latency': 0.989, 'accuracy': 52.934, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 64,
            'latency': 3.223, 'accuracy': 52.934, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 3.515, 'accuracy': 53.22, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 30.704, 'accuracy': 53.22, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 275.500, 'accuracy': 53.22, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': 2.404, 'accuracy': 52.864, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': 13.502, 'accuracy': 52.864, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': 105.669, 'accuracy': 52.864, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 0.572, 'accuracy': 53.212, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 1.749, 'accuracy': 53.212, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 11.802, 'accuracy': 53.212, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 1.748, 'accuracy': 51.516, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 14.027, 'accuracy': 51.516, 'input_size': (3, 224, 224)},
    ],
    'bignas_resnet18_150M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 1,
            'latency': 0.600, 'accuracy': 55.194, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 8,
            'latency': 1.159, 'accuracy': 55.194, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 64,
            'latency': 3.556, 'accuracy': 55.194, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 4.255, 'accuracy': 55.342, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 36.789, 'accuracy': 55.342, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 323.336, 'accuracy': 55.342, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': 2.421, 'accuracy': 54.994, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': 12.516, 'accuracy': 54.994, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': 112.359, 'accuracy': 54.994, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 0.613, 'accuracy': 55.348, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 1.809, 'accuracy': 55.348, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 12.579, 'accuracy': 55.348, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 1.793, 'accuracy': 53.548, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 16.979, 'accuracy': 53.548, 'input_size': (3, 224, 224)},
    ],
    'bignas_resnet18_312M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 1,
            'latency': 0.634, 'accuracy': 60.912, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 8,
            'latency': 1.302, 'accuracy': 60.912, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 64,
            'latency': 5.197, 'accuracy': 60.912, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 8.482, 'accuracy': 61.13, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 73.516, 'accuracy': 61.13, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 615.002, 'accuracy': 61.13, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': 2.869, 'accuracy': 60.836, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': 18.352, 'accuracy': 60.836, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': 130.668, 'accuracy': 60.836, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 0.714, 'accuracy': 61.124, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 2.507, 'accuracy': 61.124, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 18.393, 'accuracy': 61.124, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 1.929, 'accuracy': 59.904, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 14.919, 'accuracy': 59.904, 'input_size': (3, 224, 224)},
    ],
    'bignas_resnet18_403M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 1,
            'latency': 0.789, 'accuracy': 63.68, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 8,
            'latency': 1.555, 'accuracy': 63.68, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 64,
            'latency': 5.929, 'accuracy': 63.68, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 11.115, 'accuracy': 63.85, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 87.605, 'accuracy': 63.85, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 732.330, 'accuracy': 63.85, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': 2.767, 'accuracy': 63.61, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': 20.186, 'accuracy': 63.61, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': 160.851, 'accuracy': 63.61, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 0.751, 'accuracy': 63.848, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 2.602, 'accuracy': 63.848, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 18.098, 'accuracy': 63.848, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 2.047, 'accuracy': 62.952, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 15.565, 'accuracy': 62.952, 'input_size': (3, 224, 224)},
    ],
    'bignas_resnet18_492M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 1,
            'latency': 0.777, 'accuracy': 64.858, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 8,
            'latency': 1.577, 'accuracy': 64.858, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 64,
            'latency': 6.919, 'accuracy': 64.858, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 13.628, 'accuracy': 65.082, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 118.207, 'accuracy': 65.082, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 949.243, 'accuracy': 65.082, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': 3.036, 'accuracy': 64.79, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': 20.563, 'accuracy': 64.79, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': 176.521, 'accuracy': 64.79, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 0.794, 'accuracy': 65.066, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 2.908, 'accuracy': 65.066, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 20.629, 'accuracy': 65.066, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 2.044, 'accuracy': 64.184, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 15.922, 'accuracy': 64.184, 'input_size': (3, 224, 224)},
    ],
    'bignas_resnet18_1555M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 1,
            'latency': 1.937, 'accuracy': 68.79, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 8,
            'latency': 6.270, 'accuracy': 68.79, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 64,
            'latency': 35.148, 'accuracy': 68.79, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 41.445, 'accuracy': 68.79, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 319.413, 'accuracy': 68.79, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 2599.132, 'accuracy': 68.79, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': 5.332, 'accuracy': 68.478, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': 38.171, 'accuracy': 68.478, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': 312.130, 'accuracy': 68.478, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 1.739, 'accuracy': 68.782, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 7.669, 'accuracy': 68.782, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 61.131, 'accuracy': 68.782, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 2.925, 'accuracy': 68.234, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 22.371, 'accuracy': 68.234, 'input_size': (3, 224, 224)},
    ],
    'bignas_det_resnet18_1930M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 1,
            'latency': 1.621, 'accuracy': 72.064, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 8,
            'latency': 3.767, 'accuracy': 72.064, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4-int8', 'batch': 64,
            'latency': 16.563, 'accuracy': 72.064, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 49.878, 'accuracy': 72.09, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 423.592, 'accuracy': 72.09, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 3169.000, 'accuracy': 72.09, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': 6.253, 'accuracy': 72.134, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': 93.211, 'accuracy': 72.134, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': 375.173, 'accuracy': 72.134, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 1.542, 'accuracy': 72.094, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 8.620, 'accuracy': 72.094, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 47.443, 'accuracy': 72.094, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 3.075, 'accuracy': 71.59, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 23.673, 'accuracy': 71.59, 'input_size': (3, 224, 224)},
    ],

}


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'h_swish':
        return Hswish(inplace=inplace)
    elif act_func == 'swish':
        return swish()
    elif act_func == 'h_sigmoid':
        return Hsigmoid(inplace=inplace)
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, dropout_rate=0.):
        super(LinearBlock, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.dropout_rate = dropout_rate

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        self.linear = nn.Linear(
            in_features=self.in_features, out_features=self.out_features, bias=self.bias
        )

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x)


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, dilation=1,
                 use_bn=True, act_func='relu'):
        super(ConvBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func

        padding = get_same_padding(self.kernel_size)
        self.conv = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                              kernel_size=self.kernel_size, padding=padding, groups=1,
                              stride=self.stride, dilation=self.dilation)
        if self.use_bn:
            self.bn = NormLayer(self.out_channel)
        self.act = build_activation(self.act_func, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x


class BasicBlock(nn.Module):

    def __init__(self, in_channel, out_channel,
                 kernel_size=3, expand_ratio=1, stride=1, act_func='relu'):
        super(BasicBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio

        self.stride = stride
        self.act_func = act_func

        # build modules default is 1
        middle_channel = make_divisible(self.out_channel * self.expand_ratio, 8)
        padding = get_same_padding(self.kernel_size)
        self.normal_conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(self.in_channel, middle_channel, self.kernel_size,
                               stride=self.stride, groups=1, padding=padding)),
            ('bn', NormLayer(middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))

        self.normal_conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(middle_channel, self.out_channel, self.kernel_size, groups=1, padding=padding)),
            ('bn', NormLayer(self.out_channel)),
        ]))
        self.act2 = build_activation(self.act_func, inplace=True)

        if self.in_channel == self.out_channel and self.stride == 1:
            self.shortcut = None
        else:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, groups=1, stride=stride)),
                ('bn', NormLayer(self.out_channel)),
            ]))

    def forward(self, x):
        identity = x

        x = self.normal_conv1(x)
        x = self.normal_conv2(x)
        if self.shortcut is None:
            x += identity
        else:
            x += self.shortcut(identity)
        return self.act2(x)


def get_same_length(element, depth):
    if len(element) == len(depth):
        element_list = []
        for i, d in enumerate(depth):
            element_list += [element[i]] * d
    elif len(element) == sum(depth):
        element_list = element
    else:
        raise ValueError('we only need stage-wise or block wise settings')
    return element_list


class BigNAS_ResNet_Basic(nn.Module):

    def __init__(self,
                 num_classes=1000,
                 width=[8, 8, 16, 48, 224],
                 depth=[1, 2, 2, 1, 2],
                 stride_stages=[2, 2, 2, 2, 2],
                 kernel_size=[7, 3, 3, 3, 3, 3, 3, 3],
                 expand_ratio=[0, 1, 1, 1, 1, 0.5, 0.5, 0.5],
                 act_stages=['relu', 'relu', 'relu', 'relu', 'relu'],
                 dropout_rate=0.,
                 normalize={'type': 'solo_bn'},
                 initializer={'method': 'msra'},
                 frozen_layers=[],
                 out_layers=[],
                 out_strides=[],
                 use_maxpool=False,
                 task='classification'):
        r"""
        Arguments:

        - num_classes (:obj:`int`): number of classification classes
        - width (:obj:`list` of 5 (stages+1) ints): channel list
        - depth (:obj:`list` of 5 (stages+1) ints): depth list for stages
        - stride_stages (:obj:`list` of 5 (stages+1) ints): stride list for stages
        - kernel_size (:obj:`list` of 8 (blocks+1) ints): kernel size list for blocks
        - expand_ratio (:obj:`list` of 8 (blocks+1) ints): expand ratio list for blocks
        - act_stages(:obj:`list` of 8 (blocks+1) ints): activation list for blocks
        - dropout_rate (:obj:`float`): dropout rate
        - normalize (:obj:`dict`): configurations for normalize
        - initializer (:obj:`dict`): initializer method
        - frozen_layers (:obj:`list` of :obj:`int`): Index of frozen layers, [0,4]
        - out_layers (:obj:`list` of :obj:`int`): Index of output layers, [0,4]
        - out_strides (:obj:`list` of :obj:`int`): Stride of outputs
        - task (:obj:`str`): Type of task, 'classification' or object 'detection'
        """

        super(BigNAS_ResNet_Basic, self).__init__()

        global NormLayer

        NormLayer = build_norm_layer(normalize)

        self.num_classes = num_classes
        self.depth = depth
        self.width = width
        self.kernel_size = get_same_length(kernel_size, self.depth)
        self.expand_ratio = get_same_length(expand_ratio, self.depth)
        self.dropout_rate = dropout_rate
        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        self.task = task
        self.out_planes = [int(width[i]) for i in self.out_layers]
        self.performance = None
        self.use_maxpool = use_maxpool

        # first conv layer
        self.first_conv = ConvBlock(
            in_channel=3, out_channel=self.width[0], kernel_size=self.kernel_size[0],
            stride=stride_stages[0], act_func=act_stages[0])

        if self.use_maxpool:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        blocks = []
        input_channel = self.width[0]

        _block_index = 1
        self.stage_out_idx = []
        for s, act_func, n_block, output_channel in zip(stride_stages[1:], act_stages[1:], self.depth[1:],
                                                        self.width[1:]):
            for i in range(n_block):
                kernel_size = self.kernel_size[_block_index]
                expand_ratio = self.expand_ratio[_block_index]
                _block_index += 1
                if i == 0:
                    stride = s
                else:
                    stride = 1
                basic_block = BasicBlock(
                    in_channel=input_channel, out_channel=output_channel, kernel_size=kernel_size,
                    expand_ratio=expand_ratio, stride=stride, act_func=act_func)
                blocks.append(basic_block)
                input_channel = output_channel
            self.stage_out_idx.append(_block_index - 2)

        self.blocks = nn.ModuleList(blocks)

        if self.task == 'classification':
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self.classifier = LinearBlock(
                in_features=self.width[-1], out_features=num_classes, bias=True, dropout_rate=dropout_rate)

        if initializer is not None:
            initialize_from_cfg(self, initializer)

        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params to optimizer
        self.freeze_layer()

    def forward(self, input):
        x = input['image'] if isinstance(input, dict) else input
        outs = []
        # first conv
        x = self.first_conv(x)
        outs.append(x)
        if self.use_maxpool:
            x = self.max_pool(x)

        # blocks
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.stage_out_idx:
                outs.append(x)

        if self.task == 'classification':
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}

    def freeze_layer(self):
        """
        Freezing layers during training.
        """
        layers = [self.first_conv]
        start_idx = 0
        for stage_out_idx in self.stage_out_idx:
            end_idx = stage_out_idx + 1
            stage = [self.blocks[i] for i in range(start_idx, end_idx)]
            layers.append(nn.Sequential(*stage))
            start_idx = end_idx

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


def bignas_resnet18_9M(pretrained=False, **kwargs):
    """
    equal to ResNet18-1/8
    """
    kwargs['width'] = [8, 8, 24, 32, 112]
    kwargs['depth'] = [1, 1, 1, 2, 1]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    model = BigNAS_ResNet_Basic(**kwargs)
    model.performance = model_performances['bignas_resnet18_9M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['bignas_resnet18_9M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def bignas_resnet18_37M(pretrained=False, **kwargs):
    """
    equal to ResNet18-1/8
    """
    kwargs['width'] = [8, 8, 16, 48, 192]
    kwargs['depth'] = [1, 1, 1, 1, 2]
    kwargs['kernel_size'] = [7, 3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 0.5, 0.5, 0.5]
    model = BigNAS_ResNet_Basic(**kwargs)
    model.performance = model_performances['bignas_resnet18_37M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['bignas_resnet18_37M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def bignas_resnet18_37M_pooling(**kwargs):
    """
    equal to ResNet18-1/8
    """
    kwargs['width'] = [8, 8, 16, 48, 192]
    kwargs['depth'] = [1, 1, 1, 1, 2]
    kwargs['kernel_size'] = [7, 3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 0.5, 0.5, 0.5]
    kwargs['stride_stages'] = [2, 1, 2, 2, 2]
    kwargs['use_maxpool'] = True
    model = BigNAS_ResNet_Basic(**kwargs)
    return model


def bignas_resnet18_49M(pretrained=False, **kwargs):
    """
    equal to ResNet18-1/8
    """
    kwargs['width'] = [8, 8, 24, 32, 224]
    kwargs['depth'] = [1, 1, 1, 2, 2]
    kwargs['kernel_size'] = [7, 3, 3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1, 0.25, 0.25]
    model = BigNAS_ResNet_Basic(**kwargs)
    model.performance = model_performances['bignas_resnet18_49M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['bignas_resnet18_49M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def bignas_resnet18_50M(pretrained=False, **kwargs):
    """
    equal to ResNet18-1/8
    """
    kwargs['width'] = [8, 8, 16, 48, 224]
    kwargs['depth'] = [1, 2, 2, 1, 2]
    kwargs['kernel_size'] = [7, 3, 3, 3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1, 0.5, 0.5, 0.5]
    model = BigNAS_ResNet_Basic(**kwargs)
    model.performance = model_performances['bignas_resnet18_50M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['bignas_resnet18_50M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def bignas_resnet18_65M(pretrained=False, **kwargs):
    """
    equal to ResNet18-1/8
    """
    kwargs['width'] = [8, 8, 16, 48, 192]
    kwargs['depth'] = [1, 1, 1, 1, 2]
    kwargs['kernel_size'] = [7, 3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 0.75, 0.75, 0.75]
    model = BigNAS_ResNet_Basic(**kwargs)
    model.performance = model_performances['bignas_resnet18_65M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['bignas_resnet18_65M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def bignas_resnet18_107M(pretrained=False, **kwargs):
    """
    equal to ResNet18 1/4
    """
    kwargs['width'] = [16, 16, 32, 48, 160]
    kwargs['depth'] = [1, 1, 1, 2, 3]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    model = BigNAS_ResNet_Basic(**kwargs)
    model.performance = model_performances['bignas_resnet18_107M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['bignas_resnet18_107M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def bignas_resnet18_107M_pooling(**kwargs):
    """
    equal to ResNet18 1/4
    """
    kwargs['width'] = [16, 16, 32, 48, 160]
    kwargs['depth'] = [1, 1, 1, 2, 3]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    kwargs['stride_stages'] = [2, 1, 2, 2, 2]
    kwargs['use_maxpool'] = True
    model = BigNAS_ResNet_Basic(**kwargs)
    return model


def bignas_resnet18_125M(pretrained=False, **kwargs):
    """
    equal to ResNet18 1/4
    """
    kwargs['width'] = [16, 16, 48, 64, 192]
    kwargs['depth'] = [1, 1, 1, 2, 2]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    model = BigNAS_ResNet_Basic(**kwargs)
    model.performance = model_performances['bignas_resnet18_125M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['bignas_resnet18_125M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def bignas_resnet18_150M(pretrained=False, **kwargs):
    """
    equal to ResNet18 1/4
    """
    kwargs['width'] = [16, 16, 48, 80, 192]
    kwargs['depth'] = [1, 1, 1, 1, 3]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    model = BigNAS_ResNet_Basic(**kwargs)
    model.performance = model_performances['bignas_resnet18_150M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['bignas_resnet18_150M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def bignas_resnet18_312M(pretrained=False, **kwargs):
    """
    equal to ResNet18 1/2
    """
    kwargs['width'] = [24, 24, 48, 112, 320]
    kwargs['depth'] = [1, 1, 1, 2, 2]
    kwargs['kernel_size'] = [5, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    model = BigNAS_ResNet_Basic(**kwargs)
    model.performance = model_performances['bignas_resnet18_312M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['bignas_resnet18_312M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def bignas_resnet18_403M(pretrained=False, **kwargs):
    """
    equal to ResNet18 1/2
    """
    kwargs['width'] = [16, 24, 48, 128, 320]
    kwargs['depth'] = [1, 1, 1, 2, 3]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    model = BigNAS_ResNet_Basic(**kwargs)
    model.performance = model_performances['bignas_resnet18_403M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['bignas_resnet18_403M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def bignas_resnet18_492M(pretrained=False, **kwargs):
    """
    equal to ResNet18 1/2
    """
    kwargs['width'] = [32, 32, 64, 144, 320]
    kwargs['depth'] = [1, 1, 1, 2, 3]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    model = BigNAS_ResNet_Basic(**kwargs)
    model.performance = model_performances['bignas_resnet18_492M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['bignas_resnet18_492M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def bignas_resnet18_492M_pooling(**kwargs):
    """
    equal to ResNet18 1/2
    """
    kwargs['width'] = [32, 32, 64, 144, 320]
    kwargs['depth'] = [1, 1, 1, 2, 3]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    kwargs['stride_stages'] = [2, 1, 2, 2, 2]
    kwargs['use_maxpool'] = True
    model = BigNAS_ResNet_Basic(**kwargs)
    return model


def bignas_resnet18_1555M(pretrained=False, **kwargs):
    """
    equal to ResNet18
    """
    kwargs['width'] = [32, 64, 112, 256, 592]
    kwargs['depth'] = [1, 1, 1, 3, 2]
    kwargs['kernel_size'] = [7, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    model = BigNAS_ResNet_Basic(**kwargs)
    model.performance = model_performances['bignas_resnet18_1555M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['bignas_resnet18_1555M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def bignas_det_resnet18_1930M(pretrained=False, **kwargs):
    """
    equal to ResNet18
    """
    kwargs['width'] = [64, 64, 128, 198, 384]
    kwargs['depth'] = [1, 2, 2, 3, 5]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    model = BigNAS_ResNet_Basic(**kwargs)
    model.performance = model_performances['bignas_det_resnet18_1930M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['bignas_det_resnet18_1930M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


if __name__ == '__main__':
    from prototype.spring.models import SPRING_MODELS_REGISTRY
    SPRING_MODELS_REGISTRY.register('bignas_resnet18_1555M', bignas_resnet18_1555M)

    cls_model = SPRING_MODELS_REGISTRY['bignas_resnet18_1555M'](pretrained=True)
    det_model = SPRING_MODELS_REGISTRY['bignas_resnet18_1555M'](
        normalize={'type': 'freeze_bn'},
        frozen_layers=[0, 1],
        out_layers=[2, 3, 4],
        out_strides=[8, 16, 32],
        task='detection',
    )
    input = torch.randn(4, 3, 224, 224)
    det_output = det_model({'image': input})
    cls_output = cls_model(input)
    # function test
    det_model.train()
    print(det_model.get_outstrides())
    print(det_model.get_outplanes())
    print('detection output size: {}'.format(det_output['features'][0].size()))
    print('detection output size: {}'.format(det_output['features'][1].size()))
    print('detection output size: {}'.format(det_output['features'][2].size()))
    print('classification output size: {}'.format(cls_output.size()))
