'''
    This file includes mobilenets' architecture.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class dw_conv(nn.Module):
    def __init__(self, in_dim, out_dim, stride):
        super().__init__()
        self.dw_conv_k3 = nn.Conv2d(in_dim, out_dim, kernel_size=3,
        stride=stride, groups=in_dim, padding=(1,1), bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.dw_conv_k3(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class point_conv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.p_conv_k1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.p_conv_k1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MobileNetsV1(nn.Module):
    '''
        IMPORTANT: This is not the original version of MobileNet v1 that is designed for ImageNet.
        This is a CIFAR10 classifier, and the main changes are the number of strides and the size of 
        Average pooling layer.
    '''
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=(1,1)), # output size: 32*32*32
                nn.ReLU(inplace=True),
                dw_conv(32, 32, 1),
                point_conv(32, 64),
                dw_conv(64, 64, 2), # output size: 64*16*16
                point_conv(64, 128),
                dw_conv(128, 128, 1),
                point_conv(128, 128),
                dw_conv(128, 128, 1),
                point_conv(128, 256),
                dw_conv(256, 256, 1),
                point_conv(256, 256),
                dw_conv(256, 256, 1),
                point_conv(256, 512),
                dw_conv(512, 512, 2), # output size: 512*8*8
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 1024),
                dw_conv(1024, 1024, 2), # output size: 1024*4*4
                point_conv(1024, 1024),
                nn.AvgPool2d(4), # 7 >> 4
        )
        self.fc = nn.Linear(1024, self.num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def mobilenetV1(num_classes, **kwargs):
    r"""PyTorch implementation of the MobileNets architecture
    <https://arxiv.org/abs/1704.04861>`_.
    Model has been modified to work on CIFAR-10
    Args:
        num_classes (int): 10 for CIFAR-10
    """
    model = MobileNetsV1(num_classes, **kwargs)
    return model

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class SeModule(nn.Module):
    def __init__(self, in_dim, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, in_dim // reduction, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(in_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // reduction, in_dim, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(in_dim),
            hsigmoid()
        )
    def forward(self, x):
        return x * self.se(x)

class Block(nn.Module):
    '''
        expand + depthwise + pointwise
    '''
    def __init__(self, kernel_size, in_dim, expand_size, out_dim, active_func, se_module, stride):
        super().__init__()
        self.stride = stride
        self.se = se_module

        self.ex_conv = nn.Conv2d(in_dim, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.no_linear_1 = active_func
        self.dw_conv = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                                padding= kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.no_linear_2 = active_func
        self.pw_conv = nn.Conv2d(expand_size, out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_dim)

        self.short_cut = nn.Sequential()
        if stride == 1 and in_dim != out_dim:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1,padding=0,bias=False),
                nn.BatchNorm2d(out_dim),
            )
    
    def forward(self, x):
        out = self.no_linear_1(self.bn1(self.ex_conv(x)))
        out = self.no_linear_2(self.bn2(self.dw_conv(out)))
        out = self.bn3(self.pw_conv(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.short_cut(x) if self.stride==1 else out
        return out

class MobileNetV3_Small(nn.Module):
    '''
        IMPORTANT: This is not the original version of MobileNet v3 small that is designed for ImageNet.
        This is a CIFAR10 classifier, and the main changes are the number of strides and the size of 
        Average pooling layer.
    '''
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False) # stride: 2 >> 1 output size : 32
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bottle_neck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2), # output size : 16
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 1), # stride: 2 >> 1 output size : 16
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2), # output size : 8
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),# output size : 4
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )
        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.hs1(self.bn1(self.conv1(x)))
        x = self.bottle_neck(x)
        x = self.hs2(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, 4) # 7 >> 4
        x = x.view(x.size(0), -1)
        x = self.hs3(self.bn3(self.linear3(x)))
        x = self.linear4(x)
        return x

def mobilenetV3_small(num_classes, **kwargs):
    r"""PyTorch implementation of the MobileNets v3 architecture
    <https://arxiv.org/abs/1704.04861>`_.
    Model has been modified to work on CIFAR-10
    Args:
        num_classes (int): 10 for CIFAR-10
    """
    model = MobileNetV3_Small(num_classes, **kwargs)
    return model

if __name__ == "__main__":
    net = mobilenetV3_small(num_classes=10)
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(f'Test output: {y}, size: {y.size()}')