from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prototype.spring.models.utils.normalize import build_norm_layer
from prototype.spring.models.utils.initializer import initialize_from_cfg
from prototype.spring.models.utils.modify import modify_state_dict


__all__ = [
    'bignas_resnet50_2954M',
    'bignas_resnet50_3145M',
    'bignas_resnet50_3811M'
]


model_urls = {
    'bignas_resnet50_2954M': 'http://spring.sensetime.com/drop/$/fFb8k.pth',
    'bignas_resnet50_3145M': 'http://spring.sensetime.com/drop/$/q4ew1.pth',
    'bignas_resnet50_3811M': 'http://spring.sensetime.com/drop/$/BHjRB.pth'
}


model_performances = {
    'bignas_resnet50_2954M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 1.937, 'accuracy': 76.832, 'input_size': 224},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 6.270, 'accuracy': 76.832, 'input_size': 224},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 35.148, 'accuracy': 76.832, 'input_size': 224},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 73.754, 'accuracy': 76.986, 'input_size': 224},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 675.087, 'accuracy': 76.986, 'input_size': 224},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 5082.337, 'accuracy': 76.986, 'input_size': 224},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': 11.164, 'accuracy': 76.832, 'input_size': 224},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': 90.426, 'accuracy': 76.832, 'input_size': 224},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': 723.285, 'accuracy': 76.832, 'input_size': 224},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 2.269, 'accuracy': 76.992, 'input_size': 224},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 14.115, 'accuracy': 76.992, 'input_size': 224},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 119.705, 'accuracy': 76.992, 'input_size': 224},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 3.712, 'accuracy': 76.378, 'input_size': 224},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 28.758, 'accuracy': 76.378, 'input_size': 224},
    ],
    'bignas_resnet50_3145M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 2.136, 'accuracy': 77.158, 'input_size': 224},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 7.034, 'accuracy': 77.158, 'input_size': 224},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 37.407, 'accuracy': 77.158, 'input_size': 224},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 77.872, 'accuracy': 77.2, 'input_size': 224},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 685.734, 'accuracy': 77.2, 'input_size': 224},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 5458.041, 'accuracy': 77.2, 'input_size': 224},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': 12.217, 'accuracy': 76.89, 'input_size': 224},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': 95.765, 'accuracy': 76.89, 'input_size': 224},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': 767.351, 'accuracy': 76.89, 'input_size': 224},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 2.652, 'accuracy': 77.186, 'input_size': 224},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 14.457, 'accuracy': 77.186, 'input_size': 224},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 123.971, 'accuracy': 77.186, 'input_size': 224},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 3.905, 'accuracy': 76.64, 'input_size': 224},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 30.458, 'accuracy': 76.64, 'input_size': 224},
    ],
    'bignas_resnet50_3811M': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
            'latency': 2.37, 'accuracy': 77.776, 'input_size': 224},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
            'latency': 7.952, 'accuracy': 77.776, 'input_size': 224},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
            'latency': 43.466, 'accuracy': 77.776, 'input_size': 224},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
            'latency': 96.149, 'accuracy': 77.804, 'input_size': 224},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
            'latency': 815.468, 'accuracy': 77.804, 'input_size': 224},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
            'latency': 6616.761, 'accuracy': 77.804, 'input_size': 224},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
            'latency': 13.788, 'accuracy': 77.634, 'input_size': 224},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
            'latency': 110.119, 'accuracy': 77.634, 'input_size': 224},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
            'latency': 873.098, 'accuracy': 77.634, 'input_size': 224},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
            'latency': 3.360, 'accuracy': 77.788, 'input_size': 224},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
            'latency': 18.789, 'accuracy': 77.788, 'input_size': 224},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
            'latency': 162.799, 'accuracy': 77.788, 'input_size': 224},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
            'latency': 4.718, 'accuracy': 77.346, 'input_size': 224},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
            'latency': 36.526, 'accuracy': 77.346, 'input_size': 224},
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


class BottleneckBlock(nn.Module):

    def __init__(self, in_channel, out_channel,
                 kernel_size=3, expand_ratio=0.25, stride=1, act_func='relu'):
        super(BottleneckBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio

        self.stride = stride
        self.act_func = act_func

        # build modules
        middle_channel = make_divisible(self.out_channel * self.expand_ratio, 8)
        padding = get_same_padding(self.kernel_size)
        self.point_conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(self.in_channel, middle_channel, kernel_size=1, groups=1)),
            ('bn', nn.BatchNorm2d(middle_channel)),
            ('act', build_activation(self.act_func, inplace=True)),
        ]))

        self.normal_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(middle_channel, middle_channel, self.kernel_size, stride=self.stride, padding=padding)),
            ('bn', nn.BatchNorm2d(middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))

        self.point_conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(middle_channel, self.out_channel, kernel_size=1, groups=1)),
            ('bn', nn.BatchNorm2d(self.out_channel)),
        ]))
        self.act3 = build_activation(self.act_func, inplace=True)

        if self.in_channel == self.out_channel and self.stride == 1:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, groups=1, stride=stride)
            self.shortcutbn = nn.BatchNorm2d(self.out_channel)

    def forward(self, x):
        identity = x

        x = self.point_conv1(x)
        x = self.normal_conv(x)
        x = self.point_conv2(x)
        if self.shortcut is None:
            x += identity
        else:
            x += self.shortcutbn(self.shortcut(identity))
        return self.act3(x)


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


class BigNAS_ResNet_Bottleneck(nn.Module):

    def __init__(self,
                 num_classes=1000,
                 width=[64, 256, 512, 1024, 2048],
                 depth=[1, 3, 4, 6, 3],
                 stride_stages=[2, 2, 2, 2, 2],
                 kernel_size=[7, 3, 3, 3, 3],
                 expand_ratio=[0, 0.25, 0.25, 0.25, 0.25],
                 act_stages=['relu', 'relu', 'relu', 'relu', 'relu'],
                 dropout_rate=0.,
                 normalize={'type': 'solo_bn'},
                 initializer={'method': 'msra'},
                 frozen_layers=[],
                 out_layers=[],
                 out_strides=[],
                 task='classification'):
        r"""
        Arguments:

        - num_classes (:obj:`int`): number of classification classes
        - width (:obj:`list` of 5 (stages+1) ints): channel list
        - depth (:obj:`list` of 5 (stages+1) ints): depth list for stages
        - stride_stages (:obj:`list` of 5 (stages+1) ints): stride list for stages
        - kernel_size (:obj:`list` of 5 (stages+1) or 17 (blocks+1) ints): kernel size list for blocks
        - expand_ratio (:obj:`list` of 5 (stages+1) or 17 (blocks+1) ints): expand ratio list for blocks
        - act_stages(:obj:`list` of 5 (stages+1) ints): activation list for blocks
        - dropout_rate (:obj:`float`): dropout rate
        - normalize (:obj:`dict`): configurations for normalize
        - initializer (:obj:`dict`): initializer method
        - frozen_layers (:obj:`list` of :obj:`int`): Index of frozen layers, [0,4]
        - out_layers (:obj:`list` of :obj:`int`): Index of output layers, [0,4]
        - out_strides (:obj:`list` of :obj:`int`): Stride of outputs
        - task (:obj:`str`): Type of task, 'classification' or object 'detection'
        """

        super(BigNAS_ResNet_Bottleneck, self).__init__()

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

        # first conv layer
        self.first_conv = ConvBlock(
            in_channel=3, out_channel=self.width[0], kernel_size=self.kernel_size[0],
            stride=stride_stages[0], act_func=act_stages[0])

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
                basic_block = BottleneckBlock(
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


def bignas_resnet50_2954M(pretrained=False, **kwargs):
    """
    equal to ResNet50
    """
    kwargs['width'] = [48, 192, 448, 960, 2176]
    kwargs['depth'] = [1, 3, 3, 6, 2]
    kwargs['kernel_size'] = [5, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 0.25, 0.25, 0.25, 0.25]
    model = BigNAS_ResNet_Bottleneck(**kwargs)
    model.performance = model_performances['bignas_resnet50_2954M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['bignas_resnet50_2954M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def bignas_resnet50_3145M(pretrained=False, **kwargs):
    """
    equal to ResNet50
    """
    kwargs['width'] = [48, 192, 416, 960, 2112]
    kwargs['depth'] = [1, 3, 4, 7, 2]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 0.25, 0.25, 0.25, 0.25]
    model = BigNAS_ResNet_Bottleneck(**kwargs)
    model.performance = model_performances['bignas_resnet50_3145M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['bignas_resnet50_3145M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def bignas_resnet50_3811M(pretrained=False, **kwargs):
    """
    equal to ResNet50
    """
    kwargs['width'] = [80, 240, 416, 896, 2496]
    kwargs['depth'] = [1, 2, 5, 7, 3]
    kwargs['kernel_size'] = [5, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 0.25, 0.25, 0.25, 0.25]
    model = BigNAS_ResNet_Bottleneck(**kwargs)
    model.performance = model_performances['bignas_resnet50_3811M']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['bignas_resnet50_3811M'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


if __name__ == '__main__':
    from prototype.spring.models import SPRING_MODELS_REGISTRY
    SPRING_MODELS_REGISTRY.register('bignas_resnet50_2954M', bignas_resnet50_2954M)

    cls_model = SPRING_MODELS_REGISTRY['bignas_resnet50_2954M'](pretrained=True)
    det_model = SPRING_MODELS_REGISTRY['bignas_resnet50_2954M'](
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
    # output test
    print('detection output size: {}'.format(det_output['features'][0].size()))
    print('detection output size: {}'.format(det_output['features'][1].size()))
    print('detection output size: {}'.format(det_output['features'][2].size()))
    print('classification output size: {}'.format(cls_output.size()))
