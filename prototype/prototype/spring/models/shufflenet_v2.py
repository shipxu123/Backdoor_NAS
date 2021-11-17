import torch
import torch.nn as nn

from prototype.spring.models.utils.normalize import build_norm_layer
from prototype.spring.models.utils.initializer import initialize_from_cfg
from prototype.spring.models.utils.modify import modify_state_dict

NormLayer = None

__all__ = ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
           'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'shufflenet_v2_scale']

model_urls = {
    'shufflenet_v2_x0_5': 'http://spring.sensetime.com/drop/$/pPp5y.pth',
    'shufflenet_v2_x1_0': 'http://spring.sensetime.com/drop/$/Zn9uL.pth',
    'shufflenet_v2_x1_5': 'http://spring.sensetime.com/drop/$/lFZLo.pth',
    'shufflenet_v2_x2_0': 'http://spring.sensetime.com/drop/$/tZubU.pth'
}

model_performances = {
    'shufflenet_v2_x1_0': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 34.170,
            'input_size': (3, 224, 224), 'accuracy': 68.84},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 288.000,
            'input_size': (3, 224, 224), 'accuracy': 68.84},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 2177.889,
            'input_size': (3, 224, 224), 'accuracy': 68.84},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 1.404,
            'input_size': (3, 224, 224), 'accuracy': 66.536},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 2.990,
            'input_size': (3, 224, 224), 'accuracy': 66.536},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 12.962,
            'input_size': (3, 224, 224), 'accuracy': 66.536},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 5.223,
            'input_size': (3, 224, 224), 'accuracy': 69.488},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 56.803,
            'input_size': (3, 224, 224), 'accuracy': 69.488},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 504.057,
            'input_size': (3, 224, 224), 'accuracy': 69.488},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 1.969,
            'input_size': (3, 224, 224), 'accuracy': 67.906},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 14.970,
            'input_size': (3, 224, 224), 'accuracy': 67.906},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 3.424,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': None,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': None,
            'input_size': (3, 224, 224), 'accuracy': None},
    ],
    'shufflenet_v2_x0_5': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 30.321,
            'input_size': (3, 224, 224), 'accuracy': 58.756},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 157.950,
            'input_size': (3, 224, 224), 'accuracy': 58.756},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 1160.160,
            'input_size': (3, 224, 224), 'accuracy': 58.756},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 0.879,
            'input_size': (3, 224, 224), 'accuracy': 55.188},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 1.809,
            'input_size': (3, 224, 224), 'accuracy': 55.188},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 7.407,
            'input_size': (3, 224, 224), 'accuracy': 55.188},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 1.975,
            'input_size': (3, 224, 224), 'accuracy': 57.024},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 28.785,
            'input_size': (3, 224, 224), 'accuracy': 57.024},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 213.928,
            'input_size': (3, 224, 224), 'accuracy': 57.024},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 1.839,
            'input_size': (3, 224, 224), 'accuracy': 55.958},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 13.523,
            'input_size': (3, 224, 224), 'accuracy': 55.958},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 2.755,
            'input_size': (3, 224, 224), 'accuracy': 61.162},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 8.738,
            'input_size': (3, 224, 224), 'accuracy': 61.162},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 61.018,
            'input_size': (3, 224, 224), 'accuracy': 61.162}
    ],
    'shufflenet_v2_x1_5': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 58.315,
            'input_size': (3, 224, 224), 'accuracy': 69.376},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 409.817,
            'input_size': (3, 224, 224), 'accuracy': 69.376},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 2817.751,
            'input_size': (3, 224, 224), 'accuracy': 69.376},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 1.378,
            'input_size': (3, 224, 224), 'accuracy': 70.254},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 3.683,
            'input_size': (3, 224, 224), 'accuracy': 70.254},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 19.002,
            'input_size': (3, 224, 224), 'accuracy': 70.254},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 11.458,
            'input_size': (3, 224, 224), 'accuracy': 72.866},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 91.828,
            'input_size': (3, 224, 224), 'accuracy': 72.866},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 888.129,
            'input_size': (3, 224, 224), 'accuracy': 72.866},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 2.069,
            'input_size': (3, 224, 224), 'accuracy': 70.942},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 15.795,
            'input_size': (3, 224, 224), 'accuracy': 70.942},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 4.630,
            'input_size': (3, 224, 224), 'accuracy': 72.81},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 24.144,
            'input_size': (3, 224, 224), 'accuracy': 72.81},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 252.355,
            'input_size': (3, 224, 224), 'accuracy': 72.81}
    ],
    'shufflenet_v2_x2_0': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 74.199,
            'input_size': (3, 224, 224), 'accuracy': 73.506},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 528.193,
            'input_size': (3, 224, 224), 'accuracy': 73.506},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': None,
            'input_size': (3, 224, 224), 'accuracy': 73.506},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 1.773,
            'input_size': (3, 224, 224), 'accuracy': 72.32},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 4.780,
            'input_size': (3, 224, 224), 'accuracy': 72.32},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 25.881,
            'input_size': (3, 224, 224), 'accuracy': 72.32},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 31.696,
            'input_size': (3, 224, 224), 'accuracy': 74.522},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 204.436,
            'input_size': (3, 224, 224), 'accuracy': 74.522},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 2345.825,
            'input_size': (3, 224, 224), 'accuracy': 74.522},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 2.402,
            'input_size': (3, 224, 224), 'accuracy': 73.514},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 18.611,
            'input_size': (3, 224, 224), 'accuracy': 73.514},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 5.433,
            'input_size': (3, 224, 224), 'accuracy': 72.816},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': None,
            'input_size': (3, 224, 224), 'accuracy': 72.816},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': None,
            'input_size': (3, 224, 224), 'accuracy': 72.816}
    ]

}


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3,
                                    stride=self.stride, padding=1),
                NormLayer(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1,
                          stride=1, padding=0, bias=False),
                NormLayer(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            NormLayer(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features,
                                kernel_size=3, stride=self.stride, padding=1),
            NormLayer(branch_features),
            nn.Conv2d(branch_features, branch_features,
                      kernel_size=1, stride=1, padding=0, bias=False),
            NormLayer(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    """ShuffleNet model class, based on
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" <https://arxiv.org/abs/1807.11164>`_
    """

    def __init__(self,
                 stages_repeats,
                 stages_out_channels,
                 num_classes=1000,
                 normalize={'type': 'solo_bn'},
                 initializer={'method': 'msra'},
                 ceil_mode=False,
                 frozen_layers=[],
                 out_layers=[],
                 out_strides=[],
                 task='classification',):
        r"""
        Arguments:

        - stages_repeats (:obj:`list` of 3 ints): how many layers in each stage
        - stages_out_channels (:obj:`list` of 5 ints): output channels
        - num_classes (:obj:`int`): number of classification classes
        - normalize (:obj:`dict`): configurations for normalize
        - initializer (:obj:`dict`): initializer method
        - ceil_mode (:obj:`bool`): if `True`, set the `ceil_mode=True` in maxpooling
        - frozen_layers (:obj:`list` of :obj:`int`): Index of frozen layers, [0,4]
        - out_layers (:obj:`list` of :obj:`int`): Index of output layers, [0,4]
        - out_strides (:obj:`list` of :obj:`int`): Stride of outputs
        - task (:obj:`str`): Type of task, 'classification' or object 'detection'
        """

        super(ShuffleNetV2, self).__init__()

        global NormLayer

        NormLayer = build_norm_layer(normalize)

        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        self.task = task
        self.num_classes = num_classes
        self.performance = None

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        self.out_planes = [stages_out_channels[i] for i in out_layers]

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            NormLayer(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        if ceil_mode:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(
                    output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            NormLayer(output_channels),
            nn.ReLU(inplace=True),
        )

        # classifier only for classification task
        if self.task == 'classification':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(output_channels, num_classes)

        # initialization
        if initializer is not None:
            initialize_from_cfg(self, initializer)

        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params to optimizer
        self.freeze_layer()

    def forward(self, input):
        x = input['image'] if isinstance(input, dict) else input

        x = self.conv1(x)
        c1 = self.maxpool(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.conv5(c4)

        if self.task == 'classification':
            x = self.avgpool(c5)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

        outs = [c1, c2, c3, c4, c5]
        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}

    def freeze_layer(self):
        """
        Freezing layers during training.
        """
        layers = [
            nn.Sequential(self.conv1, self.maxpool),
            self.stage2, self.stage3, self.stage4, self.conv5
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


def shufflenet_v2_x0_5(pretrained=False, **kwargs):
    """
    Constructs a ShuffleNet-V2-0.5 model.
    """
    model = ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)
    model.performance = model_performances['shufflenet_v2_x0_5']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['shufflenet_v2_x0_5'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def shufflenet_v2_x1_0(pretrained=False, **kwargs):
    """
    Constructs a ShuffleNet-V2-1.0 model.
    """
    model = ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)
    model.performance = model_performances['shufflenet_v2_x1_0']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['shufflenet_v2_x1_0'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def shufflenet_v2_x1_5(pretrained=False, **kwargs):
    """
    Constructs a ShuffleNet-V2-1.5 model.
    """
    model = ShuffleNetV2([4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)
    model.performance = model_performances['shufflenet_v2_x1_5']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['shufflenet_v2_x1_5'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def shufflenet_v2_x2_0(pretrained=False, **kwargs):
    """
    Constructs a ShuffleNet-V2-2.0 model.
    """
    model = ShuffleNetV2([4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)
    model.performance = model_performances['shufflenet_v2_x2_0']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['shufflenet_v2_x2_0'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def shufflenet_v2_scale(**kwargs):
    """
    Constructs a custom ShuffleNet-V2 model.
    """
    model = ShuffleNetV2(**kwargs)
    return model
