import torch
import torch.nn as nn

from functools import partial

from prototype.spring.models.utils.normalize import build_norm_layer
from prototype.spring.models.utils.initializer import initialize_from_cfg
from prototype.spring.models.utils.modify import modify_state_dict


__all__ = [
    'oneshot_supcell_9M',
    'oneshot_supcell_27M',
    'oneshot_supcell_37M',
    'oneshot_supcell_55M',
    'oneshot_supcell_70M',
    'oneshot_supcell_91M',
    'oneshot_supcell_96M',
    'oneshot_supcell_113M',
    'oneshot_supcell_168M',
    'oneshot_supcell_304M',
    'oneshot_supcell_1710M',
    'oneshot_supcell_3072M'
]

model_urls = {
    'oneshot_supcell_9M': 'http://spring.sensetime.com/drop/$/zHvb6.pth',
    'oneshot_supcell_27M': 'http://spring.sensetime.com/drop/$/cGqXI.pth',
    'oneshot_supcell_37M': 'http://spring.sensetime.com/drop/$/8UXAW.pth',
    'oneshot_supcell_55M': 'http://spring.sensetime.com/drop/$/74wAm.pth',
    'oneshot_supcell_70M': 'http://spring.sensetime.com/drop/$/mgsaQ.pth',
    'oneshot_supcell_91M': 'http://spring.sensetime.com/drop/$/rQvfG.pth',
    'oneshot_supcell_96M': 'http://spring.sensetime.com/drop/$/J2U1I.pth',
    'oneshot_supcell_113M': 'http://spring.sensetime.com/drop/$/K99Wk.pth',
    'oneshot_supcell_168M': 'http://spring.sensetime.com/drop/$/1FRuF.pth',
    'oneshot_supcell_304M': 'http://spring.sensetime.com/drop/$/zRGri.pth',
    'oneshot_supcell_1710M': 'http://spring.sensetime.com/drop/$/mt7Sc.pth',
    'oneshot_supcell_3072M': 'http://spring.sensetime.com/drop/$/9Xlxt.pth',
}

model_performances = {
    'oneshot_supcell_9M': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 1.672,
            'input_size': (3, 224, 224), 'accuracy': 24.852},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 10.001,
            'input_size': (3, 224, 224), 'accuracy': 24.852},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 73.610,
            'input_size': (3, 224, 224), 'accuracy': 24.852},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 0.355,
            'input_size': (3, 224, 224), 'accuracy': 28.122},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 0.773,
            'input_size': (3, 224, 224), 'accuracy': 28.122},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 2.380,
            'input_size': (3, 224, 224), 'accuracy': 28.122},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 0.699,
            'input_size': (3, 224, 224), 'accuracy': 29.446},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 5.837,
            'input_size': (3, 224, 224), 'accuracy': 29.446},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 56.236,
            'input_size': (3, 224, 224), 'accuracy': 29.446},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 1.718,
            'input_size': (3, 224, 224), 'accuracy': 24.872},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 12.824,
            'input_size': (3, 224, 224), 'accuracy': 24.872},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 0.567,
            'input_size': (3, 224, 224), 'accuracy': 29.448},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 1.642,
            'input_size': (3, 224, 224), 'accuracy': 29.448},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 11.106,
            'input_size': (3, 224, 224), 'accuracy': 29.448},
    ],
    'oneshot_supcell_27M': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 1.915,
            'input_size': (3, 224, 224), 'accuracy': 32.59},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 10.656,
            'input_size': (3, 224, 224), 'accuracy': 32.59},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 80.458,
            'input_size': (3, 224, 224), 'accuracy': 32.59},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 0.460,
            'input_size': (3, 224, 224), 'accuracy': 41.694},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 0.961,
            'input_size': (3, 224, 224), 'accuracy': 41.694},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 2.870,
            'input_size': (3, 224, 224), 'accuracy': 41.694},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 2.280,
            'input_size': (3, 224, 224), 'accuracy': 43.336},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 14.922,
            'input_size': (3, 224, 224), 'accuracy': 43.336},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 170.525,
            'input_size': (3, 224, 224), 'accuracy': 43.336},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 1.834,
            'input_size': (3, 224, 224), 'accuracy': 40.014},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 13.917,
            'input_size': (3, 224, 224), 'accuracy': 40.014},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 0.674,
            'input_size': (3, 224, 224), 'accuracy': 43.314},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 1.815,
            'input_size': (3, 224, 224), 'accuracy': 43.314},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 12.021,
            'input_size': (3, 224, 224), 'accuracy': 43.314},
    ],
    'oneshot_supcell_37M': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 1.614,
            'input_size': (3, 224, 224), 'accuracy': 37.54},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 10.717,
            'input_size': (3, 224, 224), 'accuracy': 37.54},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 81.391,
            'input_size': (3, 224, 224), 'accuracy': 37.54},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 0.444,
            'input_size': (3, 224, 224), 'accuracy': 44.802},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 0.849,
            'input_size': (3, 224, 224), 'accuracy': 44.802},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 2.731,
            'input_size': (3, 224, 224), 'accuracy': 44.802},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 1.696,
            'input_size': (3, 224, 224), 'accuracy': 46.068},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 13.432,
            'input_size': (3, 224, 224), 'accuracy': 46.068},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 116.596,
            'input_size': (3, 224, 224), 'accuracy': 46.068},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 1.847,
            'input_size': (3, 224, 224), 'accuracy': 43.876},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 13.830,
            'input_size': (3, 224, 224), 'accuracy': 43.876},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 0.568,
            'input_size': (3, 224, 224), 'accuracy': 46.056},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 1.741,
            'input_size': (3, 224, 224), 'accuracy': 46.056},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 11.731,
            'input_size': (3, 224, 224), 'accuracy': 46.056},
    ],
    'oneshot_supcell_55M': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 1.942,
            'input_size': (3, 224, 224), 'accuracy': 45.832},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 9.331,
            'input_size': (3, 224, 224), 'accuracy': 45.832},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 78.196,
            'input_size': (3, 224, 224), 'accuracy': 45.832},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 0.462,
            'input_size': (3, 224, 224), 'accuracy': 47.694},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 0.943,
            'input_size': (3, 224, 224), 'accuracy': 47.694},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 3.136,
            'input_size': (3, 224, 224), 'accuracy': 47.694},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 2.836,
            'input_size': (3, 224, 224), 'accuracy': 48.1},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 24.133,
            'input_size': (3, 224, 224), 'accuracy': 48.1},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 209.498,
            'input_size': (3, 224, 224), 'accuracy': 48.1},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 1.753,
            'input_size': (3, 224, 224), 'accuracy': 45.302},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 13.401,
            'input_size': (3, 224, 224), 'accuracy': 45.302},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 0.586,
            'input_size': (3, 224, 224), 'accuracy': 48.134},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 1.779,
            'input_size': (3, 224, 224), 'accuracy': 48.134},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 12.124,
            'input_size': (3, 224, 224), 'accuracy': 48.134},
    ],
    'oneshot_supcell_70M': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 3.129,
            'input_size': (3, 224, 224), 'accuracy': 47.4},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 9.603,
            'input_size': (3, 224, 224), 'accuracy': 47.4},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 88.026,
            'input_size': (3, 224, 224), 'accuracy': 47.4},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 0.441,
            'input_size': (3, 224, 224), 'accuracy': 49.428},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 0.929,
            'input_size': (3, 224, 224), 'accuracy': 49.428},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 3.297,
            'input_size': (3, 224, 224), 'accuracy': 49.428},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 3.415,
            'input_size': (3, 224, 224), 'accuracy': 50.01},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 28.818,
            'input_size': (3, 224, 224), 'accuracy': 50.01},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 257.904,
            'input_size': (3, 224, 224), 'accuracy': 50.01},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 1.747,
            'input_size': (3, 224, 224), 'accuracy': 47.082},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 13.544,
            'input_size': (3, 224, 224), 'accuracy': 47.082},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 0.596,
            'input_size': (3, 224, 224), 'accuracy': 50.004},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 1.805,
            'input_size': (3, 224, 224), 'accuracy': 50.004},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 12.715,
            'input_size': (3, 224, 224), 'accuracy': 50.004},
    ],
    'oneshot_supcell_91M': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 1.867,
            'input_size': (3, 224, 224), 'accuracy': 48.4},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 13.011,
            'input_size': (3, 224, 224), 'accuracy': 48.4},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 100.089,
            'input_size': (3, 224, 224), 'accuracy': 48.4},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 0.617,
            'input_size': (3, 224, 224), 'accuracy': 52.886},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 1.098,
            'input_size': (3, 224, 224), 'accuracy': 52.886},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 3.679,
            'input_size': (3, 224, 224), 'accuracy': 52.886},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 4.244,
            'input_size': (3, 224, 224), 'accuracy': 53.612},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 39.197,
            'input_size': (3, 224, 224), 'accuracy': 53.612},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 299.968,
            'input_size': (3, 224, 224), 'accuracy': 53.612},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 1.887,
            'input_size': (3, 224, 224), 'accuracy': 0.098},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 14.150,
            'input_size': (3, 224, 224), 'accuracy': 0.098},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 0.680,
            'input_size': (3, 224, 224), 'accuracy': 53.608},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 1.944,
            'input_size': (3, 224, 224), 'accuracy': 53.608},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 13.467,
            'input_size': (3, 224, 224), 'accuracy': 53.608},
    ],
    'oneshot_supcell_96M': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 1.921,
            'input_size': (3, 224, 224), 'accuracy': 51.11},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 12.2,
            'input_size': (3, 224, 224), 'accuracy': 51.11},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 88.316,
            'input_size': (3, 224, 224), 'accuracy': 51.11},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 0.58500,
            'input_size': (3, 224, 224), 'accuracy': 54.312},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 1.093,
            'input_size': (3, 224, 224), 'accuracy': 54.312},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 3.68,
            'input_size': (3, 224, 224), 'accuracy': 54.312},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 4.403,
            'input_size': (3, 224, 224), 'accuracy': 55.036},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 37.642,
            'input_size': (3, 224, 224), 'accuracy': 55.036},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 306.170,
            'input_size': (3, 224, 224), 'accuracy': 55.036},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 1.808,
            'input_size': (3, 224, 224), 'accuracy': 52.19},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 13.533,
            'input_size': (3, 224, 224), 'accuracy': 52.19},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 0.724,
            'input_size': (3, 224, 224), 'accuracy': 55.06},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 1.966,
            'input_size': (3, 224, 224), 'accuracy': 55.06},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 13.014,
            'input_size': (3, 224, 224), 'accuracy': 55.06},
    ],
    'oneshot_supcell_113M': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 2.267,
            'input_size': (3, 224, 224), 'accuracy': 51.72},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 12.067,
            'input_size': (3, 224, 224), 'accuracy': 51.72},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 102.510,
            'input_size': (3, 224, 224), 'accuracy': 51.72},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 0.681,
            'input_size': (3, 224, 224), 'accuracy': 55.686},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 1.172,
            'input_size': (3, 224, 224), 'accuracy': 55.686},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 3.915,
            'input_size': (3, 224, 224), 'accuracy': 55.686},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 5.177,
            'input_size': (3, 224, 224), 'accuracy': 57.0},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 42.046,
            'input_size': (3, 224, 224), 'accuracy': 57.0},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 361.623,
            'input_size': (3, 224, 224), 'accuracy': 57.0},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 1.962,
            'input_size': (3, 224, 224), 'accuracy': 52.179},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 14.546,
            'input_size': (3, 224, 224), 'accuracy': 52.179},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 0.783,
            'input_size': (3, 224, 224), 'accuracy': 57.03},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 2.151,
            'input_size': (3, 224, 224), 'accuracy': 57.03},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 14.079,
            'input_size': (3, 224, 224), 'accuracy': 57.03},
    ],
    'oneshot_supcell_168M': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 1.853,
            'input_size': (3, 224, 224), 'accuracy': 56.336},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 13.480,
            'input_size': (3, 224, 224), 'accuracy': 56.336},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 95.302,
            'input_size': (3, 224, 224), 'accuracy': 56.336},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 0.804,
            'input_size': (3, 224, 224), 'accuracy': 58.594},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 1.356,
            'input_size': (3, 224, 224), 'accuracy': 58.594},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 4.545,
            'input_size': (3, 224, 224), 'accuracy': 58.594},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 6.763,
            'input_size': (3, 224, 224), 'accuracy': 59.28},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 57.994,
            'input_size': (3, 224, 224), 'accuracy': 59.28},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 498.215,
            'input_size': (3, 224, 224), 'accuracy': 59.28},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 2.026,
            'input_size': (3, 224, 224), 'accuracy': 57.478},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 14.782,
            'input_size': (3, 224, 224), 'accuracy': 57.478},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 0.845,
            'input_size': (3, 224, 224), 'accuracy': 59.27},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 2.236,
            'input_size': (3, 224, 224), 'accuracy': 59.27},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 15.099,
            'input_size': (3, 224, 224), 'accuracy': 59.27},
    ],
    'oneshot_supcell_304M': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 3.085,
            'input_size': (3, 224, 224), 'accuracy': 59.244},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 16.843,
            'input_size': (3, 224, 224), 'accuracy': 59.244},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 128.045,
            'input_size': (3, 224, 224), 'accuracy': 59.244},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 1.259,
            'input_size': (3, 224, 224), 'accuracy': 61.862},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 2.136,
            'input_size': (3, 224, 224), 'accuracy': 61.862},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 7.642,
            'input_size': (3, 224, 224), 'accuracy': 61.862},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 13.558,
            'input_size': (3, 224, 224), 'accuracy': 62.792},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 108.941,
            'input_size': (3, 224, 224), 'accuracy': 62.792},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 908.182,
            'input_size': (3, 224, 224), 'accuracy': 62.792},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 2.062,
            'input_size': (3, 224, 224), 'accuracy': 61.262},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 15.682,
            'input_size': (3, 224, 224), 'accuracy': 61.262},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 908.182,
            'input_size': (3, 224, 224), 'accuracy': 62.792},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 3.321,
            'input_size': (3, 224, 224), 'accuracy': 62.792},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 25.498,
            'input_size': (3, 224, 224), 'accuracy': 62.792},
    ],
    'oneshot_supcell_1710M': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 14.279,
            'input_size': (3, 224, 224), 'accuracy': 76.956},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 118.251,
            'input_size': (3, 224, 224), 'accuracy': 76.956},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 906.043,
            'input_size': (3, 224, 224), 'accuracy': 76.956},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 2.372,
            'input_size': (3, 224, 224), 'accuracy': 77.47},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 6.080,
            'input_size': (3, 224, 224), 'accuracy': 77.47},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 36.065,
            'input_size': (3, 224, 224), 'accuracy': 77.47},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 49.718,
            'input_size': (3, 224, 224), 'accuracy': 77.738},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 409.632,
            'input_size': (3, 224, 224), 'accuracy': 77.738},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 3443.275,
            'input_size': (3, 224, 224), 'accuracy': 77.738},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 2.828,
            'input_size': (3, 224, 224), 'accuracy': 0},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 21.454,
            'input_size': (3, 224, 224), 'accuracy': 0},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 3.031,
            'input_size': (3, 224, 224), 'accuracy': 77.748},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 11.694,
            'input_size': (3, 224, 224), 'accuracy': 77.748},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 96.855,
            'input_size': (3, 224, 224), 'accuracy': 77.748},
    ],
    'oneshot_supcell_3072M': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 15.526,
            'input_size': (3, 224, 224), 'accuracy': 76.914},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 123.070,
            'input_size': (3, 224, 224), 'accuracy': 76.914},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 994.328,
            'input_size': (3, 224, 224), 'accuracy': 76.914},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 3.112,
            'input_size': (3, 224, 224), 'accuracy': 77.254},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 7.7955,
            'input_size': (3, 224, 224), 'accuracy': 77.254},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 39.954,
            'input_size': (3, 224, 224), 'accuracy': 77.254},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 101.075,
            'input_size': (3, 224, 224), 'accuracy': 77.442},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 791.226,
            'input_size': (3, 224, 224), 'accuracy': 77.442},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 6578.816,
            'input_size': (3, 224, 224), 'accuracy': 77.442},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 4.436,
            'input_size': (3, 224, 224), 'accuracy': 0},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 34.346,
            'input_size': (3, 224, 224), 'accuracy': 0},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 3.510,
            'input_size': (3, 224, 224), 'accuracy': 77.434},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 15.611,
            'input_size': (3, 224, 224), 'accuracy': 77.434},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 136.772,
            'input_size': (3, 224, 224), 'accuracy': 77.434}
    ]}

BN = None


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


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.time = 0
        self.netpara = 0

    def forward(self, input, *args):
        return input


class Rec(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, k=3, t=6, forward=True):
        super(Rec, self).__init__()

        padding = k // 2
        self.time = 0

        if forward:
            self.conv1 = nn.Conv2d(inplanes,
                                   inplanes * t,
                                   kernel_size=1,
                                   bias=False)
            self.bn1 = BN(inplanes * t)

            self.conv2_1 = nn.Conv2d(inplanes * t,
                                     inplanes * t,
                                     kernel_size=(1, k),
                                     stride=(1, stride),
                                     padding=(0, padding),
                                     bias=False)
            self.bn2_1 = BN(inplanes * t)
            self.conv2_2 = nn.Conv2d(inplanes * t,
                                     inplanes * t,
                                     kernel_size=(k, 1),
                                     stride=(stride, 1),
                                     padding=(padding, 0),
                                     bias=False)
            self.bn2_2 = BN(inplanes * t)

            self.conv3 = nn.Conv2d(inplanes * t,
                                   outplanes,
                                   kernel_size=1,
                                   bias=False)
            self.bn3 = BN(outplanes)

            self.activation = nn.ReLU(inplace=True)

        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.activation(out)

        out = self.conv2_2(out)
        out = self.bn2_2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class DualConv(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, k=3, t=6, forward=True):
        super(DualConv, self).__init__()
        padding = k // 2
        self.time = 0

        if forward:
            self.conv1 = nn.Conv2d(inplanes,
                                   inplanes * t,
                                   kernel_size=1,
                                   bias=False)
            self.bn1 = BN(inplanes * t)

            self.conv2_1 = nn.Conv2d(inplanes * t,
                                     inplanes * t,
                                     kernel_size=k,
                                     stride=1,
                                     padding=padding,
                                     bias=False)
            self.bn2_1 = BN(inplanes * t)
            self.conv2_2 = nn.Conv2d(inplanes * t,
                                     inplanes * t,
                                     kernel_size=k,
                                     stride=stride,
                                     padding=padding,
                                     bias=False)
            self.bn2_2 = BN(inplanes * t)

            self.conv3 = nn.Conv2d(inplanes * t,
                                   outplanes,
                                   kernel_size=1,
                                   bias=False)
            self.bn3 = BN(outplanes)
            self.activation = nn.ReLU(inplace=True)

        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.activation(out)

        out = self.conv2_2(out)
        out = self.bn2_2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class NormalConv(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, k=3, t=6, forward=True):
        super(NormalConv, self).__init__()
        padding = k // 2
        self.time = 0

        if forward:
            self.conv1 = nn.Conv2d(inplanes,
                                   inplanes * t,
                                   kernel_size=1,
                                   bias=False)
            self.bn1 = BN(inplanes * t)

            self.conv2 = nn.Conv2d(inplanes * t,
                                   inplanes * t,
                                   kernel_size=k,
                                   stride=stride,
                                   padding=padding,
                                   bias=False)
            self.bn2 = BN(inplanes * t)

            self.conv3 = nn.Conv2d(inplanes * t,
                                   outplanes,
                                   kernel_size=1,
                                   bias=False)
            self.bn3 = BN(outplanes)
            self.activation = nn.ReLU(inplace=True)

        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class LinearBottleneck(nn.Module):
    def __init__(self,
                 inplanes,
                 outplanes,
                 stride=1,
                 k=3,
                 t=6,
                 activation=nn.ReLU,
                 forward=True,
                 group=1,
                 dilation=1):
        super(LinearBottleneck, self).__init__()
        dk = k + (dilation - 1) * 2
        padding = dk // 2
        self.time = 0

        if forward:
            self.conv1 = nn.Conv2d(inplanes,
                                   inplanes * t,
                                   kernel_size=1,
                                   bias=False,
                                   groups=group)
            self.bn1 = BN(inplanes * t)
            self.conv2 = nn.Conv2d(inplanes * t,
                                   inplanes * t,
                                   kernel_size=k,
                                   stride=stride,
                                   padding=padding,
                                   bias=False,
                                   groups=inplanes * t,
                                   dilation=dilation)
            self.bn2 = BN(inplanes * t)
            self.conv3 = nn.Conv2d(inplanes * t,
                                   outplanes,
                                   kernel_size=1,
                                   bias=False,
                                   groups=group)
            self.bn3 = BN(outplanes)
            self.activation = activation(inplace=True)

        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

        self.netpara = sum(p.numel() for p in self.parameters()) / 1e6

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class AlignedSupValCell(nn.Module):
    candidate_num = 19
    candidates = []
    for k in [3, 5, 7]:
        for t in [1, 3, 6]:
            candidates.append(partial(LinearBottleneck, k=k, t=t))
    for t in [1, 2]:
        candidates.append(partial(NormalConv, k=3, t=t))
    for t in [1, 2]:
        candidates.append(partial(DualConv, k=3, t=t))
    for k in [5, 7]:
        for t in [1, 2, 4]:
            candidates.append(partial(Rec, k=k, t=t))

    def __init__(self, cin, stride, cout, branch, keep_prob=-1):
        super(AlignedSupValCell, self).__init__()
        self.stride = stride
        self.cin = cin
        self.cout = cout

        if self.cin == self.cout and self.stride == 1 and \
                branch == AlignedSupValCell.candidate_num:
            self.path = Identity()
        else:
            self.path = \
                AlignedSupValCell.candidates[branch](inplanes=self.cin,
                                                     outplanes=self.cout,
                                                     stride=self.stride)

    def forward(self, curr_layer):
        """Runs the conv cell."""
        if self.cin == self.cout and self.stride == 1:
            return self.path(curr_layer) + curr_layer
        else:
            return self.path(curr_layer)


class Oneshot_SupCell(nn.Module):
    def __init__(self,
                 alloc_code,
                 scale=1.0,
                 channel_dist=[16, 32, 64, 128, 256],
                 alloc_space=[1, 4, 4, 8, 4],
                 num_classes=1000,
                 alloc_plan='NR',
                 cell_plan='aligned',
                 last_channel=1024,
                 normalize={'type': 'solo_bn'},
                 initializer={'method': 'msra'},
                 frozen_layers=[],
                 out_strides=[],
                 task='classification'):
        r"""
        Arguments:

        - alloc_code: (:obj:`list` of :obj:`int`): selected index of operators
        - scale (:obj:`float`): scale of channels
        - channel_dist: (:obj:`list` of :obj:`int`): number of channels of each stage
        - alloc_space: (:obj:`list` of :obj:`int`): number of normal cells of each stage
        - num_classes (:obj:`int`): number of classification classes
        - alloc_plan (:obj:`str`): cell type of stage switch, ['NR', 'NER']
        - cell_plan (:obj:`str`): search space type, ['aligned']
        - last_channel (:obj:`int`): channel of last conv; < 0 for no last conv
        - normalize (:obj:`dict`): configurations for normalize
        - initializer (:obj:`dict`): initializer method
        - frozen_layers (:obj:`list` of :obj:`int`): Index of frozen layers, [0,4]
        - out_layers (:obj:`list` of :obj:`int`): Index of output layers, [0,4]
        - out_strides (:obj:`list` of :obj:`int`): Stride of outputs
        - task (:obj:`str`): Type of task, 'classification' or object 'detection'
        """

        super(Oneshot_SupCell, self).__init__()

        global BN
        BN = build_norm_layer(normalize)

        self.frozen_layers = frozen_layers
        self.task = task

        cell_seq = {
            'NR':
            lambda x: "N" * x[0] + "R" + "N" * x[1] + "R" + "N" * x[2] + "R" + "N" * x[3] + "R" + "N" * x[4]
        }[alloc_plan](alloc_space)
        cell_seq = zip(cell_seq, alloc_code)
        self.cell_seq = cell_seq
        Cell = {
            'aligned': AlignedSupValCell,
        }[cell_plan]

        self.scale = scale
        self.c = list(channel_dist[:1]) + [
            _make_divisible(ch * self.scale, 8) for ch in channel_dist[1:]
        ]
        cell_seq = list(cell_seq)
        self.total_blocks = len(cell_seq)

        self._set_stem()

        self.cells = nn.ModuleList()

        self.stage = 0
        sstride = 2
        stride_map = {}
        for cell_idx, (c, branch) in enumerate(cell_seq):
            if c == 'N':
                stride = 1

                cout = cin = self.c[self.stage]
            elif c == 'E':
                self.stage += 1
                stride = 1

                cout = self.c[self.stage]
                cin = self.c[self.stage - 1]
            elif c == 'R':
                self.stage += 1
                stride = 2
                cout = self.c[self.stage]
                cin = self.c[self.stage - 1]

                stride_map[sstride] = (cell_idx, cin)
                sstride *= 2
            else:
                raise NotImplementedError(
                    'unimplemented cell type: {}'.format(c))

            self.cells.append(Cell(cin, stride, cout, branch))

        self._set_tail(last_channel=last_channel)
        tail_channel = last_channel if last_channel > 0 else self.c[-1]
        stride_map[sstride] = (cell_idx + 1, tail_channel)

        self.out_strides = out_strides
        self.out_idx = [stride_map[_][0] for _ in out_strides]
        self.out_planes = [stride_map[_][1] for _ in self.out_strides]

        self.num_classes = num_classes

        if self.task == 'classification':
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
            self.fc = nn.Linear(tail_channel, num_classes)

        if initializer is not None:
            initialize_from_cfg(self, initializer)

        self.freeze_layer()
        self.performance = None

    def _set_stem(self):
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.c[0], 3, stride=2, padding=1, bias=False),
            BN(self.c[0]), nn.ReLU(inplace=True))

    def _set_tail(self, last_channel=1024):
        if last_channel > 0:
            self.last_conv = nn.Sequential(
                nn.Conv2d(self.c[self.stage],
                          last_channel,
                          kernel_size=1,
                          bias=False), nn.ReLU(inplace=True))
        else:
            self.last_conv = Identity()

    def forward(self, input):
        x = input['image'] if isinstance(input, dict) else input
        outs = []
        curr_layer = self.stem(x)
        outs.append(curr_layer)
        for cell_idx, cell in enumerate(self.cells):
            curr_layer = cell(curr_layer)
            outs.append(curr_layer)

        curr_layer = self.last_conv(curr_layer)
        outs.append(curr_layer)

        if self.task == 'classification':
            x = self.avgpool(curr_layer)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        else:
            features = [outs[_] for _ in self.out_idx]
            return {'features': features, 'strides': self.get_outstrides()}

    def freeze_layer(self):
        """
        Freezing layers during training.
        """
        stage = 0
        if stage in self.frozen_layers:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for cell_idx, (c, branch) in enumerate(self.cell_seq):
            if c == 'R':
                stage += 1
            if stage in self.frozen_layers:
                layer = self.cells[cell_idx]
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False
        if stage in self.frozen_layers:
            self.tail.eval()
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


def oneshot_supcell_9M(pretrained=False, **kwargs):
    # search on nnie11 latency table; target=20
    model = Oneshot_SupCell([9, 17, 13, 9],
                            channel_dist=[4, 8, 16, 32, 64],
                            alloc_space=[0, 0, 0, 0, 0],
                            cell_plan='aligned',
                            alloc_plan='NR',
                            last_channel=256,
                            **kwargs)
    model.performance = model_performances['oneshot_supcell_9M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls['oneshot_supcell_9M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def oneshot_supcell_27M(pretrained=False, **kwargs):
    # search on nnie11 latency table; target=15; sample2
    model = Oneshot_SupCell([14, 17, 9, 9, 12],
                            channel_dist=[4, 8, 16, 32, 64],
                            alloc_space=[0, 0, 0, 0, 1],
                            cell_plan='aligned',
                            alloc_plan='NR',
                            last_channel=256,
                            **kwargs)
    model.performance = model_performances['oneshot_supcell_27M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls['oneshot_supcell_27M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def oneshot_supcell_37M(pretrained=False, **kwargs):
    # search on nnie11 latency table; target=20; sample2
    model = Oneshot_SupCell([9, 14, 15, 18],
                            channel_dist=[4, 8, 16, 32, 64],
                            alloc_space=[0, 0, 0, 0, 0],
                            cell_plan='aligned',
                            alloc_plan='NR',
                            last_channel=256,
                            **kwargs)
    model.performance = model_performances['oneshot_supcell_37M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls['oneshot_supcell_37M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def oneshot_supcell_55M(pretrained=False, **kwargs):
    # search on nnie11 latency table; target=10; sample2
    model = Oneshot_SupCell([18, 14, 15, 18],
                            channel_dist=[4, 8, 16, 32, 64],
                            alloc_space=[0, 0, 0, 0, 1],
                            cell_plan='aligned',
                            alloc_plan='NR',
                            last_channel=256,
                            **kwargs)
    model.performance = model_performances['oneshot_supcell_55M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls['oneshot_supcell_55M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def oneshot_supcell_70M(pretrained=False, **kwargs):
    # search on nnie11 latency table; target=10; sample3
    model = Oneshot_SupCell([18, 18, 15, 18],
                            channel_dist=[4, 8, 16, 32, 64],
                            alloc_space=[0, 0, 0, 0, 1],
                            cell_plan='aligned',
                            alloc_plan='NR',
                            last_channel=256,
                            **kwargs)
    model.performance = model_performances['oneshot_supcell_70M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls['oneshot_supcell_70M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def oneshot_supcell_91M(pretrained=False, **kwargs):
    # search on nnie11 latency table; target=15
    model = Oneshot_SupCell([18, 4, 17, 18, 18],
                            channel_dist=[4, 8, 16, 32, 64],
                            alloc_space=[0, 0, 0, 0, 1],
                            cell_plan='aligned',
                            alloc_plan='NR',
                            last_channel=256,
                            **kwargs)
    model.performance = model_performances['oneshot_supcell_91M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls['oneshot_supcell_91M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def oneshot_supcell_96M(pretrained=False, **kwargs):
    # search on nnie11 latency table; target=21
    model = Oneshot_SupCell([18, 14, 15, 15, 18],
                            channel_dist=[4, 8, 16, 32, 64],
                            alloc_space=[0, 0, 0, 0, 1],
                            cell_plan='aligned',
                            alloc_plan='NR',
                            last_channel=256,
                            **kwargs)
    model.performance = model_performances['oneshot_supcell_96M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls['oneshot_supcell_96M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def oneshot_supcell_113M(pretrained=False, **kwargs):
    # search on nnie11 latency table; target=30, sample1
    model = Oneshot_SupCell([15, 19, 15, 19, 18, 9, 18, 18],
                            channel_dist=[4, 8, 16, 32, 64],
                            alloc_space=[0, 1, 1, 1, 1],
                            cell_plan='aligned',
                            alloc_plan='NR',
                            last_channel=256,
                            **kwargs)
    model.performance = model_performances['oneshot_supcell_113M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls['oneshot_supcell_113M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def oneshot_supcell_168M(pretrained=False, **kwargs):
    # search on nnie11 latency table; target=30
    model = Oneshot_SupCell([15, 19, 15, 18, 15, 12, 18, 18],
                            channel_dist=[4, 8, 16, 32, 64],
                            alloc_space=[0, 1, 1, 1, 1],
                            cell_plan='aligned',
                            alloc_plan='NR',
                            last_channel=256,
                            **kwargs)
    model.performance = model_performances['oneshot_supcell_168M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls['oneshot_supcell_168M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def oneshot_supcell_304M(pretrained=False, **kwargs):
    # search on nnie11 latency table; target=45
    model = Oneshot_SupCell([14, 18, 19, 18, 17, 18, 11, 18, 13, 18, 15, 18, 18],
                            channel_dist=[4, 8, 16, 32, 64],
                            alloc_space=[1, 2, 2, 2, 2],
                            cell_plan='aligned',
                            alloc_plan='NR',
                            last_channel=256,
                            **kwargs)
    model.performance = model_performances['oneshot_supcell_304M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls['oneshot_supcell_304M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def oneshot_supcell_1710M(pretrained=False, **kwargs):
    # search on 1080TI latency table; target=66
    model = Oneshot_SupCell([
        12, 15, 10, 11, 11, 11, 7, 6, 3, 11, 6, 5, 11, 11,
        11, 6, 16, 11, 16, 11, 2, 7, 11, 7, 7],
        scale=1.11,
        channel_dist=[16, 32, 64, 128, 256],
        alloc_space=[1, 4, 4, 8, 4],
        cell_plan='aligned',
        alloc_plan='NR',
        **kwargs)
    model.performance = model_performances['oneshot_supcell_1710M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls['oneshot_supcell_1710M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def oneshot_supcell_3072M(pretrained=False, **kwargs):
    # search on nnie11 latency table; target=207
    model = Oneshot_SupCell([
        16, 15, 10, 19, 9, 9, 18, 4, 14, 16, 12, 8, 17,
        9, 6, 11, 1, 13, 12, 6, 5, 5, 12, 14, 18],
        channel_dist=[16, 32, 64, 128, 256],
        alloc_space=[1, 4, 4, 8, 4],
        cell_plan='aligned',
        alloc_plan='NR',
        **kwargs)
    model.performance = model_performances['oneshot_supcell_3072M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls['oneshot_supcell_3072M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


if __name__ == '__main__':
    from prototype.spring.models import SPRING_MODELS_REGISTRY
    SPRING_MODELS_REGISTRY.register('oneshot_supcell_1710M',
                                    oneshot_supcell_1710M)

    cls_model = SPRING_MODELS_REGISTRY['oneshot_supcell_1710M'](
        pretrained=True)
    det_model = SPRING_MODELS_REGISTRY['oneshot_supcell_1710M'](
        normalize={
            'type': 'freeze_bn'
        },
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
