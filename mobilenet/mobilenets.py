'''
    This file includes mobilenets' architecture.
'''
import torch
import torch.nn as nn

class dw_conv(nn.Module):
    def __init__(self, in_dim, out_dim, stride):
        super(dw_conv, self).__init__()
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
        super(point_conv, self).__init__()
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
        super(MobileNetsV1, self).__init__()
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

