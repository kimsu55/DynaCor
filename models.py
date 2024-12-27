'''
https://github.com/UCSC-REAL/cifar-10-100n/blob/main/models/resnet.py   # resnet from cirfar10n
https://github.com/AnanyaKumar/transfer_learning/blob/main/unlabeled_extrapolation/models/mlp.py

## pre-trained model for resnet 34
https://github.com/UCSC-REAL/SimiFeat/blob/main/resnet_image.py
--> weight dimension of conv1 and linear in pre-trained model dose not match with that of  model for cifar10/100, so therefore change the name of layers. 
--> set load_state_dict(strict=False)

'''


from collections import OrderedDict

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import itertools
import torch.nn.functional as F

from torch.hub import load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1_ = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_ = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.last = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1_(self.conv1_(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.last(out)
        return out

def PreResNet18(num_classes, pretrained=False):
    if pretrained:
        print('Pre-trained model for PreResNet18 is not availabel')
    return ResNet(PreActBlock, [2,2,2,2],num_classes=num_classes)


def ResNet18(num_classes, pretrained=False):
    model = ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)
    if pretrained:
        model = pretrain_model('resnet18', model)
    return model


def ResNet34(num_classes, pretrained=False):
    model = ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)
    if pretrained:
        model = pretrain_model('resnet34', model)
    return model

def ResNet34_mlp(num_classes):
    model = ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)
    model = pretrain_model('resnet34', model)
    for i, param in enumerate(model.parameters()):
        if i > 2:
            param.requires_grad = False
    model.last = MLP([512, 512], num_classes)
    
    return model

def ResNet34_linear(num_classes):
    model = ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)
    model = pretrain_model('resnet34', model)
    for i, param in enumerate(model.parameters()):
        if i > 2:
            param.requires_grad = False
    model.last = nn.Linear(512, num_classes)
    
    return model


def ResNet50(num_classes, pretrained=False):
    model =  ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)
    if pretrained:
        model = pretrain_model('resnet50', model)
    return model

def ResNet101(num_classes, pretrained=False):
    model =  ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)
    if pretrained:
        model = pretrain_model('resnet101', model)
    return model

def ResNet152(num_classes, pretrained=False):
    model = ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)
    if pretrained:
        model = pretrain_model('resnet152', model)
    return model

def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

def pretrain_model(arch, model):
    state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=True)
    model.load_state_dict(state_dict, strict=False)
    return model


class MLPDropout(nn.Module):
    '''
    A multilayer perception with ReLU activations and dropout layers.
    '''
    def __init__(self, dims, output_dim, dropout_probs):
        '''
        Constructor.
        Parameters
        ----------
        dims : list[int]
            Specifies the input and hidden layer dimensions.
        output_dim : int
            Specifies the output dimension.
        dropout_probs : list[float]
            Specifies the dropout probability at each layer. The length of this
            list must be equal to the length of dims. If the dropout
            probability of a layer is zero, then the dropout layer is omitted
            altogether.
        '''
        if len(dims) != len(dropout_probs):
            raise ValueError('len(dims) must equal len(dropout_probs)')
        if len(dims) < 1:
            raise ValueError('len(dims) must be at least 1')
        if any(prob < 0 or prob > 1 for prob in dropout_probs):
            raise ValueError('Dropout probabilities must be in [0, 1]')

        super(MLPDropout, self).__init__()
        layers = []
        if dropout_probs[0] > 0:  # Input dropout layer.
            layers.append(('Dropout1', nn.Dropout(p=dropout_probs[0])))

        for i in range(len(dims) - 1):
            layers.append((f'Linear{i + 1}', nn.Linear(dims[i], dims[i + 1])))
            layers.append((f'ReLU{i + 1}', nn.ReLU()))
            if dropout_probs[i + 1] > 0:
                dropout = nn.Dropout(p=dropout_probs[i + 1])
                layers.append((f'Dropout{i + 2}', dropout))

        layers.append((f'Linear{len(dims)}', nn.Linear(dims[-1], output_dim)))
        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.model(x)


class MLP(MLPDropout):
    '''
    A multilayer perceptron with ReLU activations.
    '''
    def __init__(self, dims, output_dim):
        '''
        Constructor.
        Parameters
        ----------
        dims : List[int]
            Specifies the input and hidden layer dimensions.
        output_dim : int
            Specifies the output dimension.
        '''
        super(MLP, self).__init__(dims, output_dim, [0] * len(dims))


class oneD_CNN_cluster(nn.Module):
    def __init__(self, num_epochs, logit_dim, out_dim=512, dropout=0.5, **kwargs):
        super(oneD_CNN_cluster, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(logit_dim, 16, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 32, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(in_features=32*num_epochs, out_features=out_dim)
        self.features = nn.BatchNorm1d(out_dim, eps=1e-05)
        self.v = 1
        self.centers = nn.Parameter(torch.Tensor(2, out_dim))

        self.__dict__.update(kwargs)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    def forward(self,x):
        x = F.normalize(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x
    
    def get_q(self, dist_sq):
        q = 1.0 / (1.0 + dist_sq / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q
    
    def target_distribution(self, q):
        weight = q**2 / q.sum(0)
        p = (weight.t() / weight.sum(1)).t()
        return p

    def cos_similarity(self, z):
        z = z.unsqueeze(dim=-1)
        mu = torch.transpose(self.centers, 1, 0).unsqueeze(dim=0)

        cos_sim = 1 - self.cos(z, mu) # N x 2 
        return cos_sim 
    

