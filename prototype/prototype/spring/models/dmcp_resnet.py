import torch
import torch.nn as nn

from prototype.spring.models.utils.normalize import build_norm_layer
from prototype.spring.models.utils.initializer import initialize_from_cfg
from prototype.spring.models.utils.modify import modify_state_dict

NormLayer = None

__all__ = ['dmcp_resnet18_45M', 'dmcp_resnet18_47M', 'dmcp_resnet18_1040M',
           'dmcp_resnet50_282M', 'dmcp_resnet50_1100M',
           'dmna_resnet18_1800M', 'dmna_resnet50_4100M']

model_urls = {
    'dmcp_resnet18_47M': 'http://spring.sensetime.com/drop/$/4l7nw.pth',
    'dmna_resnet18_1800M': 'http://spring.sensetime.com/drop/$/xUjTm.pth',
}


model_performances = {
    'dmcp_resnet18_47M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 0.479, 'accuracy': 40.394, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 0.867, 'accuracy': 40.394, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 2.963, 'accuracy': 40.394, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 2.689, 'accuracy': 41.096, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 22.782, 'accuracy': 41.096, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 186.842, 'accuracy': 41.096, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': 1.313, 'accuracy': 38.906, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': 11.391, 'accuracy': 38.906, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': 85.175, 'accuracy': 38.906, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 0.702, 'accuracy': 41.058, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 2.093, 'accuracy': 41.058, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 14.578, 'accuracy': 41.058, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 1.881, 'accuracy': 38.364, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 13.797, 'accuracy': 38.364, 'input_size': (3, 224, 224)},
    ],
    'dmna_resnet18_1800M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 1.376, 'accuracy': 71.79, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 3.549, 'accuracy': 71.79, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 16.559, 'accuracy': 71.79, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 50.376, 'accuracy': 71.802, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 378.340, 'accuracy': 71.802, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 3075.866, 'accuracy': 71.802, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': 5.742, 'accuracy': 71.634, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': 45.986, 'accuracy': 71.634, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': 354.831, 'accuracy': 71.634, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 1.532, 'accuracy': 71.79, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 7.189, 'accuracy': 71.79, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 67.601, 'accuracy': 71.79, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 3.254, 'accuracy': 71.22, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 25.003, 'accuracy': 71.22, 'input_size': (3, 224, 224)},
    ],

}


resnet18_45M = {
    'conv1': 8,
    'fc': 136,
    'layer1': {'0': {'conv1': [8, 8], 'conv2': [8, 8]}},
    'layer2': {'0': {'conv1': [8, 16], 'conv2': [16, 24]}},
    'layer3': {'0': {'conv1': [24, 40], 'conv2': [40, 40]}, '1': {'conv1': [40, 32], 'conv2': [32, 40]}},
    'layer4': {'0': {'conv1': [40, 88], 'conv2': [88, 136]}, '1': {'conv1': [136, 88], 'conv2': [88, 136]}}
}

resnet18_47M = {
    'conv1': 8,
    'layer1': {'0': {'conv1': [8, 8], 'conv2': [8, 8]}},
    'layer2': {'0': {'conv1': [8, 16], 'conv2': [16, 24]}},
    'layer3': {'0': {'conv1': [24, 40], 'conv2': [40, 48]}},
    'layer4': {'0': {'conv1': [48, 88], 'conv2': [88, 144]}, '1': {'conv1': [144, 96], 'conv2': [96, 144]}},
    'fc': 144
}

resnet18_51M = {
    'conv1': 8,
    'layer1': {'0': {'conv1': (8, 8), 'conv2': (8, 8)}, '1': {'conv1': (8, 8), 'conv2': (8, 8)}},
    'layer2': {'0': {'conv1': (8, 16), 'conv2': (16, 16)}, '1': {'conv1': (16, 16), 'conv2': (16, 16)}},
    'layer3': {'0': {'conv1': (16, 32), 'conv2': (32, 40)}, '1': {'conv1': (40, 24), 'conv2': (24, 40)}},
    'layer4': {'0': {'conv1': (40, 48), 'conv2': (48, 248)}, '1': {'conv1': (248, 40), 'conv2': (40, 248)}},
    'fc': 248,
}

resnet18_480M = {
    'conv1': 24,
    'layer1': {'0': {'conv1': (24, 16), 'conv2': (16, 24)}, '1': {'conv1': (24, 16), 'conv2': (16, 24)}},
    'layer2': {'0': {'conv1': (24, 40), 'conv2': (40, 72)}, '1': {'conv1': (72, 40), 'conv2': (40, 72)}},
    'layer3': {'0': {'conv1': (72, 120), 'conv2': (120, 192)}, '1': {'conv1': (192, 104), 'conv2': (104, 192)}},
    'layer4': {'0': {'conv1': (192, 264), 'conv2': (264, 496)}, '1': {'conv1': (496, 256), 'conv2': (256, 496)}},
    'fc': 496,
}

resnet18_1040M = {
    'conv1': 24,
    'layer1': {'0': {'conv1': (24, 16), 'conv2': (16, 24)}, '1': {'conv1': (24, 16), 'conv2': (16, 24)}},
    'layer2': {'0': {'conv1': (24, 40), 'conv2': (40, 72)}, '1': {'conv1': (72, 40), 'conv2': (40, 72)}},
    'layer3': {'0': {'conv1': (72, 120), 'conv2': (120, 192)}, '1': {'conv1': (192, 104), 'conv2': (104, 192)}},
    'layer4': {'0': {'conv1': (192, 264), 'conv2': (264, 496)}, '1': {'conv1': (496, 256), 'conv2': (256, 496)}},
    'fc': 496,
}

resnet18_1800M = {
    'conv1': 56,
    'layer1': {'0': {'conv1': [56, 56], 'conv2': [56, 56]}},
    'layer2': {'0': {'conv1': [56, 128], 'conv2': [128, 128]},
               '1': {'conv1': [128, 104], 'conv2': [104, 128]}},
    'layer3': {'0': {'conv1': [128, 256], 'conv2': [256, 256]},
               '1': {'conv1': [256, 248], 'conv2': [248, 256]},
               '2': {'conv1': [256, 232], 'conv2': [232, 256]}},
    'layer4': {'0': {'conv1': [256, 512], 'conv2': [512, 504]},
               '1': {'conv1': [504, 504], 'conv2': [504, 504]},
               '2': {'conv1': [504, 440], 'conv2': [440, 504]}},
    'fc': 504,
}

# 0.25x
resnet50_282M = {
    'conv1': 16,
    'layer1': {'0': {'conv1': [16, 8], 'conv2': [8, 8], 'conv3': [8, 32]},
               '1': {'conv1': [32, 8], 'conv2': [8, 8], 'conv3': [8, 32]},
               '2': {'conv1': [32, 8], 'conv2': [8, 8], 'conv3': [8, 32]}},
    'layer2': {'0': {'conv1': [32, 24], 'conv2': [24, 24], 'conv3': [24, 88]},
               '1': {'conv1': [88, 24], 'conv2': [24, 24], 'conv3': [24, 88]},
               '2': {'conv1': [88, 24], 'conv2': [24, 24], 'conv3': [24, 88]},
               '3': {'conv1': [88, 24], 'conv2': [24, 24], 'conv3': [24, 88]}},
    'layer3': {'0': {'conv1': [88, 64], 'conv2': [64, 72], 'conv3': [72, 248]},
               '1': {'conv1': [248, 32], 'conv2': [32, 56], 'conv3': [56, 248]},
               '2': {'conv1': [248, 40], 'conv2': [40, 64], 'conv3': [64, 248]},
               '3': {'conv1': [248, 48], 'conv2': [48, 72], 'conv3': [72, 248]},
               '4': {'conv1': [248, 56], 'conv2': [56, 80], 'conv3': [80, 248]},
               '5': {'conv1': [248, 64], 'conv2': [64, 64], 'conv3': [64, 248]}},
    'layer4': {'0': {'conv1': [248, 184], 'conv2': [184, 176], 'conv3': [176, 1304]},
               '1': {'conv1': [1304, 136], 'conv2': [136, 224], 'conv3': [224, 1304]},
               '2': {'conv1': [1304, 184], 'conv2': [184, 224], 'conv3': [224, 1304]}},
    'fc': 1304,
}

# 0.5x
resnet50_1100M = {
    'conv1': 48,
    'layer1': {'0': {'conv1': [48, 16], 'conv2': [16, 16], 'conv3': [16, 136]},
               '1': {'conv1': [136, 16], 'conv2': [16, 16], 'conv3': [16, 136]},
               '2': {'conv1': [136, 16], 'conv2': [16, 24], 'conv3': [24, 136]}},
    'layer2': {'0': {'conv1': [136, 24], 'conv2': [24, 56], 'conv3': [56, 288]},
               '1': {'conv1': [288, 40], 'conv2': [40, 48], 'conv3': [48, 288]},
               '2': {'conv1': [288, 32], 'conv2': [32, 40], 'conv3': [40, 288]},
               '3': {'conv1': [288, 40], 'conv2': [40, 56], 'conv3': [56, 288]}},
    'layer3': {'0': {'conv1': [288, 80], 'conv2': [80, 120], 'conv3': [120, 920]},
               '1': {'conv1': [920, 64], 'conv2': [64, 112], 'conv3': [112, 920]},
               '2': {'conv1': [920, 88], 'conv2': [88, 104], 'conv3': [104, 920]},
               '3': {'conv1': [920, 80], 'conv2': [80, 112], 'conv3': [112, 920]},
               '4': {'conv1': [920, 96], 'conv2': [96, 128], 'conv3': [128, 920]},
               '5': {'conv1': [920, 112], 'conv2': [112, 128], 'conv3': [128, 920]}},
    'layer4': {'0': {'conv1': [920, 256], 'conv2': [256, 392], 'conv3': [392, 1304]},
               '1': {'conv1': [1304, 312], 'conv2': [312, 392], 'conv3': [392, 1304]},
               '2': {'conv1': [1304, 400], 'conv2': [400, 288], 'conv3': [288, 1304]}},
    'fc': 1304
}

resnet50_4100M = {
    'conv1': 56,
    'fc': 2048,
    'layer1': {'0': {'conv1': [56, 40], 'conv2': [40, 40], 'conv3': [40, 176]},
               '1': {'conv1': [176, 56], 'conv2': [56, 40], 'conv3': [40, 176]}},
    'layer2': {'0': {'conv1': [176, 112], 'conv2': [112, 128], 'conv3': [128, 440]},
               '1': {'conv1': [440, 104], 'conv2': [104, 104], 'conv3': [104, 440]},
               '2': {'conv1': [440, 96], 'conv2': [96, 120], 'conv3': [120, 440]},
               '3': {'conv1': [440, 96], 'conv2': [96, 112], 'conv3': [112, 440]},
               '4': {'conv1': [440, 96], 'conv2': [96, 104], 'conv3': [104, 440]}},
    'layer3': {'0': {'conv1': [440, 248], 'conv2': [248, 256], 'conv3': [256, 1000]},
               '1': {'conv1': [1000, 176], 'conv2': [176, 208], 'conv3': [208, 1000]},
               '2': {'conv1': [1000, 176], 'conv2': [176, 208], 'conv3': [208, 1000]},
               '3': {'conv1': [1000, 224], 'conv2': [224, 256], 'conv3': [256, 1000]},
               '4': {'conv1': [1000, 232], 'conv2': [232, 224], 'conv3': [224, 1000]},
               '5': {'conv1': [1000, 232], 'conv2': [232, 232], 'conv3': [232, 1000]},
               '6': {'conv1': [1000, 224], 'conv2': [224, 232], 'conv3': [232, 1000]}},
    'layer4': {'0': {'conv1': [1000, 512], 'conv2': [512, 512], 'conv3': [512, 2048]},
               '1': {'conv1': [2048, 512], 'conv2': [512, 512], 'conv3': [512, 2048]},
               '2': {'conv1': [2048, 496], 'conv2': [496, 512], 'conv3': [512, 2048]},
               '3': {'conv1': [2048, 512], 'conv2': [512, 512], 'conv3': [512, 2048]},
               '4': {'conv1': [2048, 392], 'conv2': [392, 456], 'conv3': [456, 2048]},
               '5': {'conv1': [2048, 360], 'conv2': [360, 400], 'conv3': [400, 2048]},
               '6': {'conv1': [2048, 216], 'conv2': [216, 256], 'conv3': [256, 2048]},
               '7': {'conv1': [2048, 360], 'conv2': [360, 480], 'conv3': [480, 2048]}}}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, stride=1, bottleneck_settings=None):
        super(BasicBlock, self).__init__()
        conv1_in_ch, conv1_out_ch = bottleneck_settings['conv1']
        self.conv1 = conv3x3(conv1_in_ch, conv1_out_ch, stride)
        self.bn1 = NormLayer(conv1_out_ch)
        self.relu = nn.ReLU(inplace=True)

        conv2_in_ch, conv2_out_ch = bottleneck_settings['conv2']
        self.conv2 = conv3x3(conv2_in_ch, conv2_out_ch)
        self.bn2 = NormLayer(conv2_out_ch)
        self.downsample = None
        self.stride = stride

        if stride != 1 or conv1_in_ch != conv2_out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(conv1_in_ch, conv2_out_ch,
                          kernel_size=1, stride=stride, bias=False),
                NormLayer(conv2_out_ch)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    def __init__(self, stride=1, bottleneck_settings=None):
        super(Bottleneck, self).__init__()
        conv1_in_ch, conv1_out_ch = bottleneck_settings['conv1']
        self.conv1 = nn.Conv2d(conv1_in_ch, conv1_out_ch, kernel_size=1, bias=False)
        self.bn1 = NormLayer(conv1_out_ch)

        conv2_in_ch, conv2_out_ch = bottleneck_settings['conv2']
        self.conv2 = nn.Conv2d(conv2_in_ch, conv2_out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = NormLayer(conv2_out_ch)

        conv3_in_ch, conv3_out_ch = bottleneck_settings['conv3']
        self.conv3 = nn.Conv2d(conv3_in_ch, conv3_out_ch, kernel_size=1, bias=False)
        self.bn3 = NormLayer(conv3_out_ch)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride != 1 or conv1_in_ch != conv3_out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(conv1_in_ch, conv3_out_ch,
                          kernel_size=1, stride=stride, bias=False),
                NormLayer(conv3_out_ch)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DMCP_ResNet(nn.Module):
    """Pruned Redidual Networks class, based on
    `"DMCP: Differentiable Markov Channel Pruning for Neural Networks" <https://arxiv.org/abs/2005.03354>`_
    """
    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 normalize={'type': 'solo_bn'},
                 initializer={'method': 'msra'},
                 ceil_mode=False,
                 frozen_layers=[],
                 out_layers=[],
                 out_strides=[],
                 task='classification',
                 channel_config=None
                 ):
        r"""
        Arguments:
        - block (:obj:`torch.nn.Module`): block type
        - layers (:obj:`list` of 4 ints): how many layers in each stage
        - num_classes (:obj:`int`): number of classification classes
        - normalize (:obj:`dict`): configurations for normalize
        - initializer (:obj:`dict`): initializer method
        - ceil_mode (:obj:`bool`): if `True`, set the `ceil_mode=True` in maxpooling
        - frozen_layers (:obj:`list` of :obj:`int`): Index of frozen layers, [0,4]
        - out_layers (:obj:`list` of :obj:`int`): Index of output layers, [0,4]
        - out_strides (:obj:`list` of :obj:`int`): Stride of outputs
        - task (:obj:`str`): Type of task, 'classification' or object 'detection'
        """

        super(DMCP_ResNet, self).__init__()

        global NormLayer

        NormLayer = build_norm_layer(normalize)

        self.num_classes = num_classes
        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        self.task = task
        self.performance = None

        layer_out_planes = [channel_config['conv1']]
        layer_out_planes.extend([channel_config['layer{}'.format(i + 1)]['0']['conv1'][0] for i in range(1, 4)])
        layer_out_planes.append(channel_config['fc'])
        self.out_planes = [int(layer_out_planes[i]) for i in self.out_layers]

        conv1_out_ch = channel_config['conv1']
        self.conv1 = nn.Conv2d(3, conv1_out_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = NormLayer(conv1_out_ch)
        self.relu = nn.ReLU(inplace=True)
        if ceil_mode:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0], bottleneck_settings=channel_config['layer1'])
        self.layer2 = self._make_layer(block, layers[1], stride=2, bottleneck_settings=channel_config['layer2'])
        self.layer3 = self._make_layer(block, layers[2], stride=2, bottleneck_settings=channel_config['layer3'])
        self.layer4 = self._make_layer(block, layers[3], stride=2, bottleneck_settings=channel_config['layer4'])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel_config['fc'], num_classes)

        # classifier only for classification task
        if self.task == 'classification':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(channel_config['fc'], num_classes)

        if initializer is not None:
            initialize_from_cfg(self, initializer)

        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params to optimizer
        self.freeze_layer()

    def _make_layer(self, block, blocks, stride=1, avg_down=False, bottleneck_settings=None):
        layers = []
        layers.append(block(stride, bottleneck_settings=bottleneck_settings['0']))
        for i in range(1, blocks):
            layers.append(block(bottleneck_settings=bottleneck_settings[str(i)]))

        return nn.Sequential(*layers)

    def forward(self, input):
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


def dmcp_resnet18_45M(**kwargs):
    """
    equal to ResNet18-1/8
    """
    kwargs['channel_config'] = resnet18_45M
    return DMCP_ResNet(BasicBlock, [1, 1, 2, 2], **kwargs)


def dmcp_resnet18_47M(pretrained=False, **kwargs):
    """
    equal to ResNet18-1/8
    """
    kwargs['channel_config'] = resnet18_47M
    model = DMCP_ResNet(BasicBlock, [1, 1, 1, 2], **kwargs)
    model.performance = model_performances['dmcp_resnet18_47M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['dmcp_resnet18_47M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def dmcp_resnet18_51M(**kwargs):
    """
    equal to ResNet18-1/8
    """
    kwargs['channel_config'] = resnet18_51M
    return DMCP_ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def dmcp_resnet18_480M(**kwargs):
    """
    equal to ResNet18-1/2
    """
    kwargs['channel_config'] = resnet18_480M
    return DMCP_ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def dmcp_resnet18_1040M(**kwargs):
    """
    equal to ResNet18-3/4
    """
    kwargs['channel_config'] = resnet18_1040M
    return DMCP_ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def dmcp_resnet50_282M(**kwargs):
    """
    equal to ResNet50-1/4
    """
    kwargs['channel_config'] = resnet50_282M
    return DMCP_ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def dmcp_resnet50_1100M(**kwargs):
    """
    equal to ResNet50-1/2
    """
    kwargs['channel_config'] = resnet50_1100M
    return DMCP_ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def dmna_resnet18_1800M(pretrained=False, **kwargs):
    """
    equal to ResNet18
    """
    kwargs['channel_config'] = resnet18_1800M
    model = DMCP_ResNet(BasicBlock, [1, 2, 3, 3], **kwargs)
    model.performance = model_performances['dmna_resnet18_1800M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['dmna_resnet18_1800M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def dmna_resnet50_4100M(**kwargs):
    """
    equal to ResNet50
    """
    kwargs['channel_config'] = resnet50_4100M
    return DMCP_ResNet(Bottleneck, [2, 5, 7, 8], **kwargs)


if __name__ == '__main__':
    from prototype.spring.models import SPRING_MODELS_REGISTRY
    SPRING_MODELS_REGISTRY.register('dmna_resnet18_1800M', dmna_resnet18_1800M)

    cls_model = SPRING_MODELS_REGISTRY['dmna_resnet18_1800M'](pretrained=True)
    det_model = SPRING_MODELS_REGISTRY['dmna_resnet18_1800M'](
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
