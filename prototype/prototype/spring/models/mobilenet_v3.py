import torch
import torch.nn as nn

from prototype.spring.models.utils.normalize import build_norm_layer
from prototype.spring.models.utils.initializer import initialize_from_cfg
from prototype.spring.models.utils.modify import modify_state_dict

from collections import OrderedDict

__all__ = ['mobilenet_v3_small_x0_35', 'mobilenet_v3_small_x0_5', 'mobilenet_v3_small_x0_75',
           'mobilenet_v3_small_x1_0', 'mobilenet_v3_small_x1_4',
           'mobilenet_v3_large_x0_35', 'mobilenet_v3_large_x0_5', 'mobilenet_v3_large_x0_75',
           'mobilenet_v3_large_x1_0', 'mobilenet_v3_large_x1_4']


model_urls = {
    'mobilenet_v3_small_x0_35': 'http://spring.sensetime.com/drop/$/ZhxWO.pth',
    'mobilenet_v3_small_x0_5': 'http://spring.sensetime.com/drop/$/KXHDE.pth',
    'mobilenet_v3_small_x0_75': 'http://spring.sensetime.com/drop/$/HTIWU.pth',
    'mobilenet_v3_small_x1_0': 'http://spring.sensetime.com/drop/$/vL3M2.pth',
    'mobilenet_v3_small_x1_4': 'http://spring.sensetime.com/drop/$/rxgYp.pth',
    'mobilenet_v3_large_x0_35': 'http://spring.sensetime.com/drop/$/UZICn.pth',
    'mobilenet_v3_large_x0_5': 'http://spring.sensetime.com/drop/$/FnPSh.pth',
    'mobilenet_v3_large_x0_75': 'http://spring.sensetime.com/drop/$/EoaWn.pth',
    'mobilenet_v3_large_x1_0': 'http://spring.sensetime.com/drop/$/WpL5S.pth',
    'mobilenet_v3_large_x1_4': 'http://spring.sensetime.com/drop/$/KTKwx.pth'
}


model_performances = {
    'mobilenet_v3_small_x0_35': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 1.158, 'accuracy': 48.104, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 1.845, 'accuracy': 48.104, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 5.637, 'accuracy': 48.104, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 1.283, 'accuracy': 50.26, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 11.664, 'accuracy': 50.26, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 121.455, 'accuracy': 50.26, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 2.029, 'accuracy': 50.272, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 4.565, 'accuracy': 50.272, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 26.430, 'accuracy': 50.272, 'input_size': (3, 224, 224)},
    ],
    'mobilenet_v3_small_x0_5': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 1.083, 'accuracy': 52.836, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 1.936, 'accuracy': 52.836, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 6.636, 'accuracy': 52.836, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 1.687, 'accuracy': 56.866, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 16.207, 'accuracy': 56.866, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 162.683, 'accuracy': 56.866, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 1.774, 'accuracy': 56.854, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 3.982, 'accuracy': 56.854, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 23.795, 'accuracy': 56.854, 'input_size': (3, 224, 224)},
    ],
    'mobilenet_v3_small_x0_75': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 1.290, 'accuracy': 59.144, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 2.528, 'accuracy': 59.144, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 10.809, 'accuracy': 59.144, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 2.403, 'accuracy': 62.49, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 22.312, 'accuracy': 62.49, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 217.642, 'accuracy': 62.49, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 2.004, 'accuracy': 62.464, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 4.931, 'accuracy': 62.464, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 33.262, 'accuracy': 62.464, 'input_size': (3, 224, 224)},
    ],
    'mobilenet_v3_small_x1_0': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 1.264, 'accuracy': 63.832, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 5.035, 'accuracy': 63.832, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 13.435, 'accuracy': 63.832, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 3.298, 'accuracy': 66.0, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 29.372, 'accuracy': 66.0, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 289.879, 'accuracy': 66.0, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 1.718, 'accuracy': 66.028, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 4.086, 'accuracy': 66.028, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 25.653, 'accuracy': 66.028, 'input_size': (3, 224, 224)},
    ],
    'mobilenet_v3_small_x1_4': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 1.595, 'accuracy': 68.894, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 3.474, 'accuracy': 68.894, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 14.938, 'accuracy': 68.894, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 6.026, 'accuracy': 70.274, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 55.127, 'accuracy': 70.274, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 512.317, 'accuracy': 70.274, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 2.623, 'accuracy': 70.308, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 8.615, 'accuracy': 70.308, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 72.794, 'accuracy': 70.308, 'input_size': (3, 224, 224)},
    ],
    'mobilenet_v3_large_x0_35': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 1.372, 'accuracy': 61.882, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 4.442, 'accuracy': 61.882, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 8.310, 'accuracy': 61.882, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 2.489, 'accuracy': 63.406, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 25.728, 'accuracy': 63.406, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 286.23, 'accuracy': 63.406, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 2.097, 'accuracy': 63.412, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 5.406, 'accuracy': 63.412, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 39.763, 'accuracy': 63.412, 'input_size': (3, 224, 224)},
    ],
    'mobilenet_v3_large_x0_5': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 1.372, 'accuracy': 67.038, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 2.913, 'accuracy': 67.038, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 10.521, 'accuracy': 67.038, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 3.893, 'accuracy': 68.426, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 39.448, 'accuracy': 68.426, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 433.597, 'accuracy': 68.426, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 1.861, 'accuracy': 68.438, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 4.606, 'accuracy': 68.438, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 33.32, 'accuracy': 68.438, 'input_size': (3, 224, 224)},
    ],
    'mobilenet_v3_large_x0_75': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 1.598, 'accuracy': 69.036, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 3.544, 'accuracy': 69.036, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 13.689, 'accuracy': 69.036, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 6.844, 'accuracy': 71.596, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 65.855, 'accuracy': 71.596, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 686.259, 'accuracy': 71.596, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 2.508, 'accuracy': 71.624, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 8.52, 'accuracy': 71.624, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 72.014, 'accuracy': 71.624, 'input_size': (3, 224, 224)},
    ],
    'mobilenet_v3_large_x1_0': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 1.654, 'accuracy': 72.784, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 4.182, 'accuracy': 72.784, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 18.092, 'accuracy': 72.784, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 10.407, 'accuracy': 73.448, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 104.815, 'accuracy': 73.448, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 925.625, 'accuracy': 73.448, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 2.228, 'accuracy': 73.416, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 7.161, 'accuracy': 73.416, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 58.381, 'accuracy': 73.416, 'input_size': (3, 224, 224)},
    ],
    'mobilenet_v3_large_x1_4': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 2.065, 'accuracy': 75.21, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 5.282, 'accuracy': 75.21, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 26.665, 'accuracy': 75.21, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 20.47, 'accuracy': 75.548, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 177.058, 'accuracy': 75.548, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 1681.996, 'accuracy': 75.548, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 2.744, 'accuracy': 75.57, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 10.72, 'accuracy': 75.57, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 91.369, 'accuracy': 75.57, 'input_size': (3, 224, 224)},
    ]
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        NormLayer(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        NormLayer(oup),
        nn.ReLU(inplace=True)
    )


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()

        s = OrderedDict()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        s['conv1'] = nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0, bias=False)
        s['act1'] = nn.ReLU(inplace=True)
        s['conv2'] = nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0, bias=False)
        s['act2'] = nn.Sigmoid()
        self.fc = nn.Sequential(s)

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        SELayer = SEModule if se else Identity

        layers = []
        if inp != exp:
            # pw
            layers.extend([
                nn.Conv2d(inp, exp, 1, 1, 0, bias=False),
                NormLayer(exp),
                nn.ReLU(inplace=True),
            ])
        layers.extend([
            # dw
            nn.Conv2d(exp, exp, kernel, stride,
                      padding, groups=exp, bias=False),
            NormLayer(exp),
            SELayer(exp),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(exp, oup, 1, 1, 0, bias=False),
            NormLayer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    """
    MobileNet V3 main class, based on
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_
    """
    def __init__(self,
                 num_classes=1000,
                 scale=1.0,
                 dropout=0.8,
                 round_nearest=8,
                 mode='small',
                 normalize={'type': 'solo_bn'},
                 initializer={'method': 'msra'},
                 frozen_layers=[],
                 out_layers=[],
                 out_strides=[],
                 task='classification',):
        r"""
        Arguments:
            - num_classes (:obj:`int`): Number of classes
            - scale (:obj:`float`): Width multiplier, adjusts number of channels in each layer by this amount
            - round_nearest (:obj:`int`): Round the number of channels in each layer to be a multiple of this number
              Set to 1 to turn off rounding
            - mode: model type, 'samll' or 'large'
            - dropout (:obj:`float`): dropout rate
            - normalize (:obj:`dict`): configurations for normalize
            - initializer (:obj:`dict`): initializer method
            - frozen_layers (:obj:`list` of :obj:`int`): Index of frozen layers, [0,4]
            - out_layers (:obj:`list` of :obj:`int`): Index of output layers, [0,4]
            - out_planes (:obj:`list` of :obj:`int`): Output planes for features
            - out_strides (:obj:`list` of :obj:`int`): Stride of outputs
            - task (:obj:`str`): Type of task, 'classification' or object 'detection'
        """
        super(MobileNetV3, self).__init__()

        global NormLayer

        NormLayer = build_norm_layer(normalize)

        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        self.task = task
        self.num_classes = num_classes
        self.performance = None

        input_channel = 16
        last_channel = 1280

        if mode == 'large':
            mobile_setting = [
                [3, 16, 16, False, 1],
                [3, 64, 24, False, 2],
                [3, 72, 24, False, 1],
                [5, 72, 40, True, 2],
                [5, 120, 40, True, 1],
                [5, 120, 40, True, 1],
                [3, 240, 80, False, 2],
                [3, 200, 80, False, 1],
                [3, 184, 80, False, 1],
                [3, 184, 80, False, 1],
                [3, 480, 112, True, 1],
                [3, 672, 112, True, 1],
                [5, 672, 160, True, 2],
                [5, 960, 160, True, 1],
                [5, 960, 160, True, 1],
            ]
        elif mode == 'small':
            mobile_setting = [
                [3, 16, 16, True, 2],
                [3, 72, 24, False, 2],
                [3, 88, 24, False, 1],
                [5, 96, 40, True, 2],
                [5, 240, 40, True, 1],
                [5, 240, 40, True, 1],
                [5, 120, 48, True, 1],
                [5, 144, 48, True, 1],
                [5, 288, 96, True, 2],
                [5, 576, 96, True, 1],
                [5, 576, 96, True, 1],
            ]
        else:
            raise NotImplementedError

        out_planes = []
        # building first layer
        last_channel = _make_divisible(
            last_channel * scale, round_nearest) if scale > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        out_planes.append(input_channel)
        self.stage_out_idx = [0]
        self.classifier = []

        # building mobile blocksa
        _block_idx = 1
        for k, exp, c, se, s in mobile_setting:
            output_channel = _make_divisible(c * scale, round_nearest)
            exp_channel = _make_divisible(exp * scale, round_nearest)
            self.features.append(InvertedResidual(
                input_channel, output_channel, k, s, exp_channel, se))
            input_channel = output_channel
            _block_idx += 1
            out_planes.append(output_channel)
            self.stage_out_idx.append(_block_idx - 1)

        # building last several layers
        out_planes.append(last_channel)
        self.stage_out_idx.append(_block_idx)
        self.out_planes = [out_planes[i] for i in self.out_layers]
        if mode == 'large':
            last_conv = _make_divisible(960 * scale, round_nearest)
            self.features.append(conv_1x1_bn(
                input_channel, last_conv))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(nn.ReLU(inplace=True))
        elif mode == 'small':
            last_conv = _make_divisible(576 * scale, round_nearest)
            self.features.append(conv_1x1_bn(
                input_channel, last_conv))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError

        self.features = nn.Sequential(*self.features)

        # classifier only for classification task
        if self.task == 'classification':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(last_channel, num_classes),
            )

        # initialization
        if initializer is not None:
            initialize_from_cfg(self, initializer)

        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params to optimizer
        self.freeze_layer()

    def _forward_impl(self, input):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = input['image'] if isinstance(input, dict) else input
        outs = []
        # blocks
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx in self.stage_out_idx:
                outs.append(x)

        if self.task == 'classification':
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}

    def forward(self, x):
        return self._forward_impl(x)

    def freeze_layer(self):
        """
        Freezing layers during training.
        """
        layers = []

        start_idx = 0
        for stage_out_idx in self.stage_out_idx:
            end_idx = stage_out_idx + 1
            stage = [self.features[i] for i in range(start_idx, end_idx)]
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


def mobilenet_v3_small_x0_35(pretrained=False, **kwargs):
    """
    Constructs a MobileNet-V3 model.
    """
    kwargs['scale'] = 0.35
    kwargs['mode'] = 'small'
    model = MobileNetV3(**kwargs)
    model.performance = model_performances['mobilenet_v3_small_x0_35']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['mobilenet_v3_small_x0_35'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def mobilenet_v3_small_x0_5(pretrained=False, **kwargs):
    """
    Constructs a MobileNet-V3 model.
    """
    kwargs['scale'] = 0.5
    kwargs['mode'] = 'small'
    model = MobileNetV3(**kwargs)
    model.performance = model_performances['mobilenet_v3_small_x0_5']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['mobilenet_v3_small_x0_5'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def mobilenet_v3_small_x0_75(pretrained=False, **kwargs):
    """
    Constructs a MobileNet-V3 model.
    """
    kwargs['scale'] = 0.75
    kwargs['mode'] = 'small'
    model = MobileNetV3(**kwargs)
    model.performance = model_performances['mobilenet_v3_small_x0_75']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['mobilenet_v3_small_x0_75'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def mobilenet_v3_small_x1_0(pretrained=False, **kwargs):
    """
    Constructs a MobileNet-V3 model.
    """
    kwargs['scale'] = 1.0
    kwargs['mode'] = 'small'
    model = MobileNetV3(**kwargs)
    model.performance = model_performances['mobilenet_v3_small_x1_0']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['mobilenet_v3_small_x1_0'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def mobilenet_v3_small_x1_4(pretrained=False, **kwargs):
    """
    Constructs a MobileNet-V3 model.
    """
    kwargs['scale'] = 1.4
    kwargs['mode'] = 'small'
    model = MobileNetV3(**kwargs)
    model.performance = model_performances['mobilenet_v3_small_x1_4']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['mobilenet_v3_small_x1_4'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def mobilenet_v3_large_x0_35(pretrained=False, **kwargs):
    """
    Constructs a MobileNet-V3 model.
    """
    kwargs['scale'] = 0.35
    kwargs['mode'] = 'large'
    model = MobileNetV3(**kwargs)
    model.performance = model_performances['mobilenet_v3_large_x0_35']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['mobilenet_v3_large_x0_35'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def mobilenet_v3_large_x0_5(pretrained=False, **kwargs):
    """
    Constructs a MobileNet-V3 model.
    """
    kwargs['scale'] = 0.5
    kwargs['mode'] = 'large'
    model = MobileNetV3(**kwargs)
    model.performance = model_performances['mobilenet_v3_large_x0_5']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['mobilenet_v3_large_x0_5'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def mobilenet_v3_large_x0_75(pretrained=False, **kwargs):
    """
    Constructs a MobileNet-V3 model.
    """
    kwargs['scale'] = 0.75
    kwargs['mode'] = 'large'
    model = MobileNetV3(**kwargs)
    model.performance = model_performances['mobilenet_v3_large_x0_75']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['mobilenet_v3_large_x0_75'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def mobilenet_v3_large_x1_0(pretrained=False, **kwargs):
    """
    Constructs a MobileNet-V3 model.
    """
    kwargs['scale'] = 1.0
    kwargs['mode'] = 'large'
    model = MobileNetV3(**kwargs)
    model.performance = model_performances['mobilenet_v3_large_x1_0']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['mobilenet_v3_large_x1_0'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def mobilenet_v3_large_x1_4(pretrained=False, **kwargs):
    """
    Constructs a MobileNet-V3 model.
    """
    kwargs['scale'] = 1.4
    kwargs['mode'] = 'large'
    model = MobileNetV3(**kwargs)
    model.performance = model_performances['mobilenet_v3_large_x1_4']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['mobilenet_v3_large_x1_4'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


if __name__ == '__main__':

    from prototype.spring.models import SPRING_MODELS_REGISTRY
    for model_name in __all__:
        SPRING_MODELS_REGISTRY.register(model_name, locals()[model_name])

        cls_model = SPRING_MODELS_REGISTRY[model_name](pretrained=True)
        det_model = SPRING_MODELS_REGISTRY[model_name](
            normalize={'type': 'freeze_bn'},
            frozen_layers=[0, 1],
            out_layers=[i for i in range(13)],
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
        # output test
        print('detection output size: {}'.format(det_output['features'][0].size()))
        print('detection output size: {}'.format(det_output['features'][1].size()))
        print('detection output size: {}'.format(det_output['features'][2].size()))
        print('classification output size: {}'.format(cls_output.size()))
