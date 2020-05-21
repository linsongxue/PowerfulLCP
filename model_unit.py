import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter, init
import math


class PartBatchNorm2d(nn.Module):
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine', 'all_features']

    def __init__(self, num_features, all_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(LimitBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.channel_part = int(all_features / 3)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        input[:, 2:4] = F.batch_norm(
            input[:, 2:4], self.running_mean[0:2], self.running_var[0:2], self.weight[0:2], self.bias[0:2],
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

        input[:, self.channel_part + 2: self.channel_part + 4] = F.batch_norm(
            input[:, self.channel_part + 2:self.channel_part + 4], self.running_mean[2:4], self.running_var[2:4],
            self.weight[2:4], self.bias[2:4], self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

        input[:, self.channel_part * 2 + 2: self.channel_part * 2 + 4] = F.batch_norm(
            input[:, self.channel_part * 2 + 2:self.channel_part * 2 + 4], self.running_mean[4:6],
            self.running_var[4:6], self.weight[4:6], self.bias[4:6], self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

        return input

    def extra_repr(self):
        return '{num_features}, eps={eps}, all_features={all_features}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(LimitBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class LimitBatchNorm2d(nn.Module):
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine', 'all_features']

    def __init__(self, num_features, all_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(LimitBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.channel_part = int(all_features / 3)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        gama = float(self.weight.detach().mean())
        beta = float(self.bias.detach().mean())
        min = -(beta / gama) - 1.2
        max = -(beta / gama) + 1.2
        self.weight.data = F.hardtanh(self.weight, min, max)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        input[:, 2:4] = F.batch_norm(
            input[:, 2:4], self.running_mean[0:2], self.running_var[0:2], self.weight[0:2], self.bias[0:2],
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

        input[:, self.channel_part + 2: self.channel_part + 4] = F.batch_norm(
            input[:, self.channel_part + 2:self.channel_part + 4], self.running_mean[2:4], self.running_var[2:4],
            self.weight[2:4], self.bias[2:4], self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

        input[:, self.channel_part * 2 + 2: self.channel_part * 2 + 4] = F.batch_norm(
            input[:, self.channel_part * 2 + 2:self.channel_part * 2 + 4], self.running_mean[4:6],
            self.running_var[4:6], self.weight[4:6], self.bias[4:6], self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

        return input

    def extra_repr(self):
        return '{num_features}, eps={eps}, all_features={all_features}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(LimitBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


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


class GhostModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, ratio=2):
        super(GhostModule, self).__init__()
        intrinsic = math.ceil(out_channels / ratio)
        remain = intrinsic * (ratio - 1)
        self.out_channels = out_channels
        self.primary_conv = nn.Conv2d(in_channels, intrinsic, kernel_size, stride, padding, bias=False)
        self.cheap_linear = nn.Conv2d(intrinsic, remain, kernel_size, stride, padding, groups=intrinsic, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        primary_out = self.primary_conv(x)
        remain_out = self.cheap_linear(primary_out)
        out = torch.cat([primary_out, remain_out], dim=1)
        out = self.relu(self.bn(out[:, :self.out_channels, :, :]))
        return out


class GhostCBAM(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=0,
                 ghost_ratio=2,
                 channel_ratio=16,
                 spatial_kernel=7,
                 bn=True):
        super(GhostCBAM, self).__init__()
        self.bn = bn
        self.conv_layer = nn.Sequential()
        self.conv_layer.add_module("Ghost",
                                   GhostModule(in_channels, out_channels, kernel_size, stride, padding, ghost_ratio))
        if bn:
            self.conv_layer.add_module('BatchNorm2d', nn.BatchNorm2d(out_channels))
        self.channel_coefficient = ChannelAttention(out_channels, channel_ratio)
        self.spatial_coefficient = SpatialAttention(spatial_kernel)

    def forward(self, x):
        out = self.conv_layer(x)
        out = out * self.channel_coefficient(out)
        out = out * self.spatial_coefficient(out)
        return out