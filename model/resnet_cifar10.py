'''
Refer to: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
Modified a bit by Huan Wang (wang.huan@northeastern.edu) for network pruning.

-------------
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .weight_normalization_layer import Conv2D_WN

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
    
class ZeroPaddingX2(nn.Module):
    def __init__(self):
        super(ZeroPaddingX2, self).__init__()
    def forward(self, x):
        num_channels = x.shape[1]
        y = F.pad(x[:, :, ::2, ::2], # stride = 2, per Kaiming's ResNet (CVPR'16) paper
            (0, 0, 0, 0, num_channels // 2, num_channels // 2), "constant", 0)
        return y

LambdaLayer2 = ZeroPaddingX2 # To keep back-compatibility (some pretrained ckpt needs this to be properly loaded)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option='A', conv_type='default'):
        super(BasicBlock, self).__init__()
        Conv2d = Conv2D_WN if conv_type == 'wn' else nn.Conv2d # @mst: weight normalization
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.downsample = ZeroPaddingX2()

            elif option == 'B':
                self.downsample = nn.Sequential(
                     Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # print(f'after conv2: {out.shape}', f'after downsample: {self.downsample(x).shape}', f'x.shape: {x.shape}')
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, conv_type='default', option='A', width_mul=1):
        super(ResNet, self).__init__()
        self.in_planes = 16 * width_mul

        Conv2d = Conv2D_WN if conv_type == 'wn' else nn.Conv2d # @mst: weight normalization
        self.conv1 = Conv2d(3, 16 * width_mul, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16 * width_mul)
        self.layer1 = self._make_layer(block, 16 * width_mul, num_blocks[0], stride=1, option=option)
        self.layer2 = self._make_layer(block, 32 * width_mul, num_blocks[1], stride=2, option=option)
        self.layer3 = self._make_layer(block, 64 * width_mul, num_blocks[2], stride=2, option=option)
        self.linear = nn.Linear(64 * width_mul, num_classes)

        # self.apply(_weights_init) # @mst: Commented because PyTorch has its own default init scheme (Kaiming uniform)

    def _make_layer(self, block, planes, num_blocks, stride, option):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option=option))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, conv_type=kwargs['conv_type'])


def resnet32(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, conv_type=kwargs['conv_type'])


def resnet44(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes, conv_type=kwargs['conv_type'])


def resnet56(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, conv_type=kwargs['conv_type'])


def resnet110(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes, conv_type=kwargs['conv_type'])

def resnet1202(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [200, 200, 200], num_classes=num_classes, conv_type=kwargs['conv_type'])

# ------------------------
# Below are the models with 1x1 conv as downsample layers
def resnet20_B(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, conv_type=kwargs['conv_type'], option='B')

def resnet32_B(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, conv_type=kwargs['conv_type'], option='B')

def resnet44_B(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes, conv_type=kwargs['conv_type'], option='B')

def resnet56_B(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, conv_type=kwargs['conv_type'], option='B')

def resnet110_B(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes, conv_type=kwargs['conv_type'], option='B')

def resnet1202_B(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [200, 200, 200], num_classes=num_classes, conv_type=kwargs['conv_type'], option='B')

# ------------------------
# Below are models with increased width
def resnet56x4(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, conv_type=kwargs['conv_type'], option='A', width_mul=4)
