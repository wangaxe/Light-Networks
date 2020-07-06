import torch
import torch.nn as nn

class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups
    
    def forward():
        batchsize, channels, height, width = x.data.size()
        channels_per_group = int(channels / self.groups)

        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x

def DepthwiseConv2d(nn.Module):
    
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwis = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(output_channels)
        )
    
    def forward(self, x):
        return self.depthwise(x)

def PointwiseConv2d(nn.Module):

    def __init__(self, input_channels, out_channels, **kwargs):
        super().__init__()
        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, 1, **kwargs),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.pointwise(x)

class ShuffleNetUnit(nn.Module):

    def __init__(self, input_channels, output_channels, stage, kernel_size, stride, groups):
        super().__init__()

        self.gp1 = nn.Sequential(
            PointwiseConv2d(
                input_channels, 
                int(output_channels / 4), 
                groups=groups
            ),
            nn.ReLU(inplace=True)
        )

        self.channel_shuffle = ChannelShuffle(groups)

        self.depthwise = DepthwiseConv2d(
            int(output_channels / 4),
            int(output_channels / 4),
            kernel_size,
            groups = int(output_channels / 4),
            stride = stride,
            padding = 1
        )