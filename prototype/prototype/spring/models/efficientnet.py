import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from prototype.spring.models.utils.normalize import build_norm_layer
from prototype.spring.models.utils.initializer import initialize_from_cfg
from prototype.spring.models.utils.modify import modify_state_dict

import re
import math
import collections
from collections import OrderedDict


__all__ = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
           'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']

model_urls = {
    'efficientnet_b0': 'http://spring.sensetime.com/drop/$/w4HEB.pth',
    'efficientnet_b1': 'http://spring.sensetime.com/drop/$/NZhT7.pth',
    'efficientnet_b2': 'http://spring.sensetime.com/drop/$/AAD7d.pth',
    'efficientnet_b3': 'http://spring.sensetime.com/drop/$/nxuaW.pth',
    'efficientnet_b4': 'http://spring.sensetime.com/drop/$/CWiml.pth',
    'efficientnet_b5': 'http://spring.sensetime.com/drop/$/VBZBs.pth',
    'efficientnet_b6': 'http://spring.sensetime.com/drop/$/pH5Xn.pth',
    'efficientnet_b7': 'http://spring.sensetime.com/drop/$/9SHCY.pth'
}

model_performances = {
    'efficientnet_b0': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 3.083, 'accuracy': 50.96, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 9.932, 'accuracy': 50.96, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 54.426, 'accuracy': 50.96, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 106.047, 'accuracy': 76.172, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 874.424, 'accuracy': 76.172, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 7153.123, 'accuracy': 76.172, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 4.337, 'accuracy': 76.17, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 19.34, 'accuracy': 76.17, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 178.246, 'accuracy': 76.17, 'input_size': (3, 224, 224)},
    ],
    'efficientnet_b1': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 4.512, 'accuracy': 75.658, 'input_size': (3, 240, 240)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 15.418, 'accuracy': 75.658, 'input_size': (3, 240, 240)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 88.71, 'accuracy': 75.658, 'input_size': (3, 240, 240)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 169.272, 'accuracy': 78.626, 'input_size': (3, 240, 240)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 1403.502, 'accuracy': 78.626, 'input_size': (3, 240, 240)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 11345.106, 'accuracy': 78.626, 'input_size': (3, 240, 240)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 6.838, 'accuracy': 78.634, 'input_size': (3, 240, 240)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 33.494, 'accuracy': 78.634, 'input_size': (3, 240, 240)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 298.354, 'accuracy': 78.634, 'input_size': (3, 240, 240)},
    ],
    'efficientnet_b2': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 5.208, 'accuracy': 77.284, 'input_size': (3, 260, 260)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 19.339, 'accuracy': 77.284, 'input_size': (3, 260, 260)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 120.015, 'accuracy': 77.284, 'input_size': (3, 260, 260)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 230.831, 'accuracy': 79.678, 'input_size': (3, 260, 260)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 1863.609, 'accuracy': 79.678, 'input_size': (3, 260, 260)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 14908.349, 'accuracy': 79.678, 'input_size': (3, 260, 260)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 8.454, 'accuracy': 79.142, 'input_size': (3, 260, 260)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 45.16, 'accuracy': 79.142, 'input_size': (3, 260, 260)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 394.515, 'accuracy': 79.142, 'input_size': (3, 260, 260)},
    ],
    'efficientnet_b3': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 7.568, 'accuracy': 78.594, 'input_size': (3, 300, 300)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 30.244, 'accuracy': 78.594, 'input_size': (3, 300, 300)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 200.027, 'accuracy': 78.594, 'input_size': (3, 300, 300)},
    ],
    'efficientnet_b4': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 12.18, 'accuracy': 80.12, 'input_size': (3, 380, 380)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 59.686, 'accuracy': 80.12, 'input_size': (3, 380, 380)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 414.533, 'accuracy': 80.12, 'input_size': (3, 380, 380)},
    ],
    'efficientnet_b5': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 20.577, 'accuracy': 81.19, 'input_size': (3, 456, 456)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 116.029, 'accuracy': 81.19, 'input_size': (3, 456, 456)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 857.161, 'accuracy': 81.19, 'input_size': (3, 456, 456)},
    ],
    'efficientnet_b6': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 34.24, 'accuracy': 82.374, 'input_size': (3, 528, 528)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 197.373, 'accuracy': 82.374, 'input_size': (3, 528, 528)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 1487.339, 'accuracy': 82.374, 'input_size': (3, 528, 528)},
    ],
    'efficientnet_b7': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 53.459, 'accuracy': 82.342, 'input_size': (3, 600, 600)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 336.799, 'accuracy': 82.342, 'input_size': (3, 600, 600)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 2569.914, 'accuracy': 82.342, 'input_size': (3, 600, 600)},
    ],
}

GlobalParams = collections.namedtuple('GlobalParams', [
    'dropout_rate', 'data_format', 'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate',
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def efficientnet_params(model_name):
    """Get efficientnet params based on model name."""
    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet_b0': (1.0, 1.0, 224, 0.2),
        'efficientnet_b1': (1.0, 1.1, 240, 0.2),
        'efficientnet_b2': (1.1, 1.2, 260, 0.3),
        'efficientnet_b3': (1.2, 1.4, 300, 0.3),
        'efficientnet_b4': (1.4, 1.8, 380, 0.4),
        'efficientnet_b5': (1.6, 2.2, 456, 0.4),
        'efficientnet_b6': (1.8, 2.6, 528, 0.5),
        'efficientnet_b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]


def efficientnet(width_coefficient=None, depth_coefficient=None,
                 dropout_rate=0.2, drop_connect_rate=0.3, override_block=None):
    """Creates a efficientnet model."""
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    if override_block is not None:
        assert isinstance(override_block, dict)
        for k, v in override_block.items():
            blocks_args[int(k)] = v
    global_params = GlobalParams(dropout_rate=dropout_rate,
                                 drop_connect_rate=drop_connect_rate,
                                 data_format='channels_last',
                                 num_classes=1000,
                                 width_coefficient=width_coefficient,
                                 depth_coefficient=depth_coefficient,
                                 depth_divisor=8,
                                 min_depth=None)
    decoder = BlockDecoder()
    return decoder.decode(blocks_args), global_params


class BlockDecoder(object):
    """Block Decoder for readability."""

    def _decode_block_string(self, block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            strides=[int(options['s'][0]), int(options['s'][1])])

    def _encode_block_string(self, block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if block.se_ratio > 0 and block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    def decode(self, string_list):
        """Decodes a list of string notations to specify blocks inside the network.
        Args:
            string_list: a list of strings, each string is a notation of block.
        Returns:
            A list of namedtuples to represent blocks arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(self._decode_block_string(block_string))
        return blocks_args

    def encode(self, blocks_args):
        """Encodes a list of Blocks to a list of strings.
        Args:
            blocks_args: A list of namedtuples to represent blocks arguments.
        Returns:
            a list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(self._encode_block_string(block))
        return block_strings


def get_model_params(model_name, override_params=None, override_block=None):
    """Get the block args and global params for a given model."""
    if model_name.startswith('efficientnet'):
        width_coefficient, depth_coefficient, _, dropout_rate = (efficientnet_params(model_name))
        blocks_args, global_params = efficientnet(width_coefficient, depth_coefficient,
                                                  dropout_rate, override_block=override_block)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)

    if override_params is not None:
        # ValueError will be raised here if override_params has fields not included
        # in global_params.get_model_params
        global_params = global_params._replace(**override_params)

    return blocks_args, global_params


def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(x, training=False, drop_connect_rate=None):
    if drop_connect_rate is None:
        raise RuntimeError("drop_connect_rate not given")
    if not training:
        return x
    else:
        keep_prob = 1.0 - drop_connect_rate

        n = x.size(0)
        random_tensor = torch.rand([n, 1, 1, 1], dtype=x.dtype, device=x.device)
        random_tensor = random_tensor + keep_prob
        binary_mask = torch.floor(random_tensor)

        x = (x / keep_prob) * binary_mask

        return x


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


def activation(act_type='swish'):
    if act_type == 'swish':
        act = swish()
        return act
    else:
        act = nn.ReLU(inplace=True)
        return act


class MBConvBlock(nn.Module):
    def __init__(self, block_args):
        super(MBConvBlock, self).__init__()

        self._block_args = block_args

        self.has_se = (self._block_args.se_ratio is not None) and \
            (self._block_args.se_ratio > 0) and \
            (self._block_args.se_ratio <= 1)

        self._build(inp=self._block_args.input_filters, oup=self._block_args.output_filters,
                    expand_ratio=self._block_args.expand_ratio, kernel_size=self._block_args.kernel_size,
                    stride=self._block_args.strides)

    def block_args(self):
        return self._block_args

    def _build(self, inp, oup, expand_ratio, kernel_size, stride):
        module_lists = []

        self.use_res_connect = all([s == 1 for s in stride]) and inp == oup

        if expand_ratio != 1:
            module_lists.append(nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False))
            module_lists.append(NormLayer(inp * expand_ratio))
            module_lists.append(activation())

        module_lists.append(nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel_size,
                                      stride, kernel_size // 2, groups=inp * expand_ratio, bias=False))
        module_lists.append(NormLayer(inp * expand_ratio))
        module_lists.append(activation())

        self.in_conv = nn.Sequential(*module_lists)

        if self.has_se:
            se_size = max(1, int(inp * self._block_args.se_ratio))
            s = OrderedDict()
            s['conv1'] = nn.Conv2d(inp * expand_ratio, se_size, kernel_size=1, stride=1, padding=0)
            s['act1'] = activation()
            s['conv2'] = nn.Conv2d(se_size, inp * expand_ratio, kernel_size=1, stride=1, padding=0)
            s['act2'] = nn.Sigmoid()
            self.se_block = nn.Sequential(s)

        self.out_conv = nn.Sequential(
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            NormLayer(oup)
        )

    def forward(self, x, drop_connect_rate=None):
        out = self.in_conv(x)
        if self.has_se:
            weight = F.adaptive_avg_pool2d(out, output_size=1)
            weight = self.se_block(weight)
            out = out * weight

        out = self.out_conv(out)
        if self._block_args.id_skip:
            if self.use_res_connect:
                if drop_connect_rate is not None:
                    out = drop_connect(out, self.training, drop_connect_rate)
                out = out + x

        return out


class EfficientNet(nn.Module):
    """EfficientNet class, based on
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_
    """
    def __init__(self,
                 blocks_args=None,
                 global_params=None,
                 use_fc_bn=False,
                 fc_bn_init_scale=1.0,
                 num_classes=1000,
                 normalize={'type': 'solo_bn'},
                 initializer={'method': 'msra'},
                 frozen_layers=[],
                 out_layers=[],
                 out_strides=[],
                 task='classification',):
        r"""
        Arguments:
            - num_classes (:obj:`int`): Number of classes
            - blocks_args: specifying building block for efficientnet
            - global_params: specifying building block settings for efficientnet
            - normalize (:obj:`dict`): configurations for normalize
            - initializer (:obj:`dict`): initializer method
            - frozen_layers (:obj:`list` of :obj:`int`): Index of frozen layers, [0,4]
            - out_layers (:obj:`list` of :obj:`int`): Index of output layers, [0,4]
            - out_planes (:obj:`list` of :obj:`int`): Output planes for features
            - out_strides (:obj:`list` of :obj:`int`): Stride of outputs
            - task (:obj:`str`): Type of task, 'classification' or object 'detection'
        """
        super(EfficientNet, self).__init__()

        global NormLayer

        NormLayer = build_norm_layer(normalize)

        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        self.task = task
        self.num_classes = num_classes
        self.performance = None

        if not isinstance(blocks_args, list):
            raise ValueError('blocks_args should be a list.')

        self._global_params = global_params
        self._blocks_args = blocks_args
        self.use_fc_bn = use_fc_bn
        self.fc_bn_init_scale = fc_bn_init_scale

        self._build()

        # initialization
        if initializer is not None:
            initialize_from_cfg(self, initializer)

        if self.use_fc_bn:
            self.fc_bn = NormLayer(self._global_params.num_classes)
            init.constant_(self.fc_bn.weight, self.fc_bn_init_scale)
            init.constant_(self.fc_bn.bias, 0)

        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params to optimizer
        self.freeze_layer()

    def _build(self):

        out_planes = []
        self.stage_out_idx = [0]
        c_in = round_filters(32, self._global_params)
        self.stem = nn.Sequential(
            nn.Conv2d(3, c_in, kernel_size=3, stride=2, padding=1, bias=False),
            NormLayer(c_in),
            activation(),
        )
        out_planes.append(c_in)

        features = []
        _block_idx = 1
        for block_args in self._blocks_args:
            assert block_args.num_repeat > 0

            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            features.append(MBConvBlock(block_args))
            out_planes.append(block_args.output_filters)

            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1, 1])

            for _ in range(block_args.num_repeat - 1):
                features.append(MBConvBlock(block_args))
            _block_idx += block_args.num_repeat
            self.stage_out_idx.append(_block_idx - 1)
        self.features = nn.ModuleList(features)

        c_in = round_filters(320, self._global_params)
        c_final = round_filters(1280, self._global_params)
        self.head = nn.Sequential(
            nn.Conv2d(c_in, c_final, kernel_size=1, stride=1, padding=0, bias=False),
            NormLayer(c_final),
            activation(),
        )
        out_planes.append(c_final)
        self.stage_out_idx.append(_block_idx)
        self.out_planes = [out_planes[i] for i in self.out_layers]
        if self.task == 'classification':
            self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
            self.classifier = torch.nn.Linear(c_final, self._global_params.num_classes)

        if self._global_params.dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=self._global_params.dropout_rate, inplace=True)
        else:
            self.dropout = None

    def _forward_impl(self, input):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = input['image'] if isinstance(input, dict) else input
        x = self.stem(x)
        outs = [x]
        # blocks
        for idx, block in enumerate(self.features):
            drop_rate = self._global_params.drop_connect_rate
            if drop_rate:
                drop_rate *= float(idx) / len(self.features)
            x = block(x, drop_rate)
            if (idx + 1) in self.stage_out_idx:
                outs.append(x)
        x = self.head(x)
        outs.append(x)

        if self.task == 'classification':
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.classifier(x)

            if self.use_fc_bn and x.size(0) > 1:
                x = self.fc_bn(x.view(x.size(0), -1, 1, 1))
                x = x.view(x.size(0), -1)

            return x

        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}

    def forward(self, x):
        return self._forward_impl(x)

    def freeze_layer(self):
        """
        Freezing layers during training.
        """
        layers = [self.stem]

        start_idx = 0
        for stage_out_idx in self.stage_out_idx[1:-1]:
            end_idx = stage_out_idx
            stage = [self.features[i] for i in range(start_idx, end_idx)]
            layers.append(nn.Sequential(*stage))
            start_idx = end_idx
        layers.append(self.head)

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


def efficientnet_b0(pretrained=False, override_params=None, override_block=None, **kwargs):
    """
    Constructs a EfficientNet-B0 model.
    """
    model_name = 'efficientnet_b0'
    blocks_args, global_params = get_model_params(model_name, override_params, override_block)
    model = EfficientNet(blocks_args, global_params, **kwargs)
    model.performance = model_performances['efficientnet_b0']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['efficientnet_b0'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def efficientnet_b1(pretrained=False, override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B1 model.
    """
    model_name = 'efficientnet_b1'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)
    model.performance = model_performances['efficientnet_b1']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['efficientnet_b1'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def efficientnet_b2(pretrained=False, override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B2 model.
    """
    model_name = 'efficientnet_b2'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)
    model.performance = model_performances['efficientnet_b2']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['efficientnet_b2'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)

    return model


def efficientnet_b3(pretrained=False, override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B3 model.
    """
    model_name = 'efficientnet_b3'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)
    model.performance = model_performances['efficientnet_b3']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['efficientnet_b3'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def efficientnet_b4(pretrained=False, override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B4 model.
    """
    model_name = 'efficientnet_b4'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)
    model.performance = model_performances['efficientnet_b4']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['efficientnet_b4'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def efficientnet_b5(pretrained=False, override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B5 model.
    """
    model_name = 'efficientnet_b5'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)
    model.performance = model_performances['efficientnet_b5']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['efficientnet_b5'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def efficientnet_b6(pretrained=False, override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B6 model.
    """
    model_name = 'efficientnet_b6'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)
    model.performance = model_performances['efficientnet_b6']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['efficientnet_b6'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def efficientnet_b7(pretrained=False, override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B7 model.
    """
    model_name = 'efficientnet_b7'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)
    model.performance = model_performances['efficientnet_b7']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['efficientnet_b7'], map_location='cpu')
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
            out_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8],
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
        print('=' * 50)
