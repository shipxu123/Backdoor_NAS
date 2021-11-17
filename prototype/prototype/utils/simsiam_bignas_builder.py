import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import linklink as link
from linklink.nn import SyncBatchNorm2d
from prototype.utils.misc import get_bn

from prototype.spring.nas.bignas.ops.dynamic_ops import DynamicLinear

# change implmentation for aligning the old torch version
class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048, num_layers=3):
        super(projection_MLP, self).__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.num_layers = num_layers

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear1 = DynamicLinear(in_dim, hidden_dim)
        self.bn1 = BN(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = BN(hidden_dim)

        if self.num_layers == 3:
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.bn2 = BN(hidden_dim)
            self.relu2 = nn.ReLU(inplace=True)

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        b, _ = x.shape

        if self.num_layers == 3:
            # layer 1
            x = self.linear1(x)
            x = self.bn1(x)
            x = self.relu1(x)

            # layer 2
            x = self.linear2(x)
            x = self.bn2(x)
            x = self.relu2(x)

            # layer 3
            x = self.linear3(x)
            x = self.bn3(x)
        elif self.num_layers == 2:
            # layer 1
            x = self.linear1(x)
            x = self.bn1(x)
            x = self.relu1(x)

            # layer 3
            x = self.linear3(x)
            x = self.bn3(x)
        else:
            raise Exception
        return x

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super(prediction_MLP, self).__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = BN(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer2 = nn.Linear(hidden_dim, out_dim)
        # self.linear1 = DynamicLinear(in_dim, hidden_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        b, _ = x.shape

        # layer 1
        x = self.linear1(x)
        x.reshape(b, self.hidden_dim, 1)
        x = self.bn1(x)
        x = self.relu1(x)
        x.reshape(b, self.hidden_dim)

        # layer2
        x = self.layer2(x)
        return x

def D(p, z, version='original'):
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, p=2, dim=1) # l2-normalize 
        z = F.normalize(z, p=2, dim=1) # l2-normalize 
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

class SimSiam_BigNAS(nn.Module):

    def __init__(self,
                 backbone,
                 castrate=True,
                 bn=None,
                 proj_layers=None):

        global BN
        BN = get_bn(bn)

        super(SimSiam_BigNAS, self).__init__()

        if castrate:
            backbone.output_dim = max(backbone.classifier.in_features_list)

        self.encoder = backbone

        if proj_layers is not None:
            self.projector = projection_MLP(backbone.output_dim, num_layers=proj_layers)
        else:
            self.projector = projection_MLP(backbone.output_dim)

        # bruce subtitue
        self.encoder.module.classifier = self.projector

        self.predictor = prediction_MLP()
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, SyncBatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        f, h = self.encoder, self.predictor

        x1, x2 = torch.split(input, [3, 3], dim=1)
        x1 = x1.contiguous()
        x2 = x2.contiguous()

        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)

        return z1, z2, p1, p2

class Criterion(nn.Module):
    def __init__(self, symmetry=True):
        super(Criterion, self).__init__()
        self.symmetry = symmetry

    def forward(self, p1, z1, p2, z2):
        if self.symmetry:
            return -0.5 * (D(p1, z2)  + D(p2, z1))