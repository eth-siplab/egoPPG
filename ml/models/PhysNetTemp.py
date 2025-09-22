""" PhysNet
We repulicate the net pipeline of the orginal paper, but set the input as diffnormalized data.
orginal source:
Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks
British Machine Vision Conference (BMVC)} 2019,
By Zitong Yu, 2019/05/05
Only for research purpose, and commercial use is not allowed.
MIT License
Copyright (c) 2019
"""

import math
import pdb

import torch
import torch.nn as nn

from collections import OrderedDict


class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC',
                 negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm3d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


class TemporalAttentionGate(nn.Module):
    def __init__(self, nc, nt, reduction_ratio=16, mode='CRC'):
        super(TemporalAttentionGate, self).__init__()

        self.nc = nc
        self.nt = nt
        self.reduction_ratio = reduction_ratio
        self.mode = mode

        self.relu = nn.ReLU(inplace=True)
        self.temporal_attention_gate = sequential(# ResBlock(self.nc, self.nc, bias=True, mode=self.mode),
                                                  torch.nn.AdaptiveAvgPool3d((self.nt, 1, 1)),  # [64, 96, 1, 1]
                                                  nn.Conv3d(self.nc, 1, (1, 1, 1), stride=1, padding=0),
                                                  torch.nn.Flatten(),
                                                  torch.nn.Linear(self.nt, self.nt // self.reduction_ratio),
                                                  nn.ReLU(),
                                                  torch.nn.Linear(self.nt // self.reduction_ratio, self.nt),
                                                  nn.Sigmoid()
                                                  )

    """def forward(self, x):
        residual = x
        temp_att_scale = self.temporal_attention_gate(x).unsqueeze(1).unsqueeze(3).unsqueeze(4).expand_as(x)
        temp_att = temp_att_scale * x
        out = residual + temp_att
        out = self.relu(out)
        return out"""

    def forward(self, x):
        scale = self.temporal_attention_gate(x).unsqueeze(1).unsqueeze(3).unsqueeze(4).expand_as(x)
        return scale * x


class ChannelAttentionGate(nn.Module):
    def __init__(self, nc, nt, reduction_ratio=16, mode='CRC'):
        super(ChannelAttentionGate, self).__init__()

        self.nc = nc
        self.nt = nt
        self.reduction_ratio = reduction_ratio
        self.mode = mode

        self.relu = nn.ReLU(inplace=True)
        self.channel_attention_gate = sequential(
            torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(self.nc, self.nc // self.reduction_ratio),
            nn.ReLU(),
            torch.nn.Linear(self.nc // self.reduction_ratio, self.nc),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.channel_attention_gate(x).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return scale * x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialAttentionGate(nn.Module):
    def __init__(self, nwh, reduction_ratio=16, mode='CRC'):
        super(SpatialAttentionGate, self).__init__()

        self.nwh = nwh
        self.reduction_ratio = reduction_ratio
        self.mode = mode

        self.relu = nn.ReLU(inplace=True)
        self.compress = ChannelPool()
        self.conv1 = sequential(nn.Conv3d(2, 1, (1, 1, 1), bias=True, stride=1, padding=0),
                                nn.BatchNorm3d(1, momentum=0.9, eps=1e-04, affine=True),
                                nn.ReLU(inplace=True))

        self.pool = torch.nn.AdaptiveAvgPool3d((1, nwh, nwh))

        # self.conv1 = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        scale = self.pool(x)
        scale = self.compress(scale)
        scale = self.conv1(scale)

        return scale * x


class PhysNetTemp(nn.Module):
    def __init__(self, frames=128):
        super(PhysNetTemp, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(1, 16, (1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(4, 1, 1), stride=(2, 1, 1),
                               padding=(1, 0, 0)),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(4, 1, 1), stride=(2, 1, 1),
                               padding=(1, 0, 0)),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(64, 1, (1, 1, 1), stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # self.poolspa = nn.AdaptiveMaxPool3d((frames, 1, 1))    # pool only spatial space
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))   # better for EDA!

        # For CL 128
        self.TemporalAttentionGate1 = TemporalAttentionGate(64, frames//4, reduction_ratio=16, mode='CRC')
        self.TemporalAttentionGate2 = TemporalAttentionGate(64, frames//2, reduction_ratio=16, mode='CRC')

        # For CL 384
        # self.TemporalAttentionGate1 = TemporalAttentionGate(64, 96, reduction_ratio=16, mode='CRC')
        # self.TemporalAttentionGate2 = TemporalAttentionGate(64, 192, reduction_ratio=16, mode='CRC')

        # self.ChannelAttentionGate1 = ChannelAttentionGate(64, 96, reduction_ratio=16, mode='CRC')
        # self.ChannelAttentionGate2 = ChannelAttentionGate(64, 192, reduction_ratio=16, mode='CRC')

        # self.SpatialAttentionGate1 = SpatialAttentionGate(4, reduction_ratio=16, mode='CRC')
        # self.SpatialAttentionGate2 = SpatialAttentionGate(4, reduction_ratio=16, mode='CRC')

    def forward(self, x):  # Batch_size* [3, T, 64, 64]
        x_visual = x
        [batch, channel, length, width, height] = x.shape

        x1 = self.ConvBlock1(x)  # x [16, T, 64, 64]
        xmax1 = self.MaxpoolSpa(x1)  # x [16, T, 32, 32]  Spatial halve

        x2 = self.ConvBlock2(xmax1)  # x [32, T, 32, 32]
        x3 = self.ConvBlock3(x2)  # x [64, T, 32, 32]
        xmax2 = self.MaxpoolSpaTem(x3)  # [64, T/2, 16, 16]  Temporal + spatial halve

        x4 = self.ConvBlock4(xmax2)  # x [64, T/2, 16, 16]
        x5 = self.ConvBlock5(x4)  # x [64, T/2, 16, 16]
        xmax3 = self.MaxpoolSpaTem(x5)  # x [64, T/4, 8,8]  Temporal + spatial halve

        x6 = self.ConvBlock6(xmax3)  # x [64, T/4, 8,8]
        x7 = self.ConvBlock7(x6)  # x [64, T/4, 8,8]
        xmax4 = self.MaxpoolSpa(x7)  # x [64, T/4, 4,4]  Spatial halve

        x8 = self.ConvBlock8(xmax4)  # x [64, T/4, 4, 4]
        x9 = self.ConvBlock9(x8)  # x [64, T/4, 4, 4]

        x_temp_att_1 = self.TemporalAttentionGate1(x9)
        # x_channel_att_1 = self.ChannelAttentionGate1(x9)
        # x_spatial_att_1 = self.SpatialAttentionGate1(x_channel_att_1)

        xup1 = self.upsample(x_temp_att_1)  # x [64, T/2, 4, 4]  Temporal up sample

        x_temp_att_2 = self.TemporalAttentionGate2(xup1)
        # x_channel_att_2 = self.ChannelAttentionGate2(xup1)
        # x_spatial_att_2 = self.SpatialAttentionGate2(x_channel_att_2)

        xup2 = self.upsample2(x_temp_att_2)  # x [64, T, 4, 4]  Temporal up sample

        xpool = self.poolspa(xup2)  # x [64, T, 1, 1]  Spatial downsample to (1, 1)
        x10 = self.ConvBlock10(xpool)  # x [1, T, 1, 1]

        rPPG = x10.view(-1, length)  # Flatten

        return rPPG
