import torch.nn as nn
import math
from prototype.utils.misc import get_logger, get_bn

BN = None

__all__ = ['resnet18', 'resnet26', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet_custom']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BN(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BN(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BN(planes * 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        bypass_bn_weight_list.append(self.bn3.weight)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
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
                 num_classes=1000,
                 deep_stem=False,
                 avg_down=False,
                 bypass_last_bn=False,
                 bn=None,
                 nnie_type=False):
        r"""
        Arguments:

        - layers (:obj:`list` of 4 ints): how many layers in each stage
        - num_classes (:obj:`int`): number of classification classes
        - deep_stem (:obj:`bool`): whether to use deep_stem as the first conv
        - avg_down (:obj:`bool`): whether to use avg_down when spatial downsample
        - bypass_last_bn (:obj:`bool`): whether use bypass_last_bn
        - bn (:obj:`dict`): definition of batchnorm
        """

        super(ResNet, self).__init__()

        logger = get_logger(__name__)

        global BN, bypass_bn_weight_list

        BN = get_bn(bn)
        bypass_bn_weight_list = []

        self.inplanes = inplanes
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.logger = get_logger(__name__)

        if self.deep_stem:
            self.conv1 = nn.Sequential(
                        nn.Conv2d(3, inplanes // 2, kernel_size=3, stride=2, padding=1, bias=False),
                        BN(inplanes // 2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(inplanes // 2, inplanes // 2, kernel_size=3, stride=1, padding=1, bias=False),
                        BN(inplanes // 2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(inplanes // 2, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                    )
        else:
            self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BN(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if nnie_type:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inplanes * 8 * block.expansion, num_classes)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        if bypass_last_bn:
            for param in bypass_bn_weight_list:
                param.data.zero_()
            logger.info('bypass {} bn.weight in BottleneckBlocks'.format(len(bypass_bn_weight_list)))

    def _make_layer(self, block, planes, blocks, stride=1, avg_down=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, bias=False),
                    BN(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    BN(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18(**kwargs):
    """
    Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet26(**kwargs):
    """
    Constructs a ResNet-26 model.
    """
    model = ResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """
    Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """
    Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """
    Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """
    Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
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