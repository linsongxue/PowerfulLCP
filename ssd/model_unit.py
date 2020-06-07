import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter, init
import math


class ChannelAttention(nn.Module):

    def __init__(self, input_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1_Conv2d = nn.Conv2d(input_channels, input_channels // ratio, 1, bias=False)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.fc2_Conv2d = nn.Conv2d(input_channels // ratio, input_channels, 1, bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2_Conv2d(self.activation(self.fc1_Conv2d(self.avg_pool(x))))
        max_out = self.fc2_Conv2d(self.activation(self.fc1_Conv2d(self.max_pool(x))))
        out = self.sigmod(avg_out + max_out).expand_as(x)
        return out


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7!"

        padding = 3 if kernel_size == 7 else 1

        self.fc_Conv2d = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmod(self.fc_Conv2d(x))

        return out


class CBAMModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 ratio=16,
                 spatial_kernel=7,
                 deepwidth=True,
                 bn=True):
        super(CBAMModule, self).__init__()
        groups = in_channels if deepwidth else 1
        self.bn = bn
        self.conv_layer = nn.Sequential()
        self.conv_layer.add_module("Conv2d", nn.Conv2d(in_channels,
                                                       out_channels,
                                                       kernel_size,
                                                       stride,
                                                       padding,
                                                       groups=groups,
                                                       bias=not bn))
        if bn:
            self.conv_layer.add_module('BatchNorm2d', nn.BatchNorm2d(out_channels))
        self.channel_coefficient = ChannelAttention(out_channels, ratio)
        self.spatial_coefficient = SpatialAttention(spatial_kernel)

    def forward(self, x):
        out = self.conv_layer(x)
        out = out * self.channel_coefficient(out)
        out = out * self.spatial_coefficient(out)
        return out


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(MaskConv2d, self).__init__(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         bias=bias)
        self.register_buffer("selected_channels_mask", torch.ones(in_channels))  # init the weight with original weight

    def forward(self, input):
        mask = self.selected_channels_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(self.weight)
        return F.conv2d(input,
                        self.weight.mul(mask),
                        self.bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups)
