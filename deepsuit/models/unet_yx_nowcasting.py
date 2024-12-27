# -*-coding:utf-8 -*-

import torch
from torch import nn as nn
from torch.nn import functional as F

Norm = nn.BatchNorm2d

class Activation(nn.Module):
    def __init__(self, param, **kwargs):
        super(Activation, self).__init__()
        self.param = param
        self.kwargs = kwargs

        if self.param == 'relu':
            act = nn.ReLU(inplace=True)
        elif self.param == 'leakyrelu':
            act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif self.param == 'elu':
            act = nn.ELU(inplace=True)
        elif self.param == 'tanh':
            act = nn.Tanh()
        elif self.param == 'sigmoid':
            act = nn.Sigmoid()
        else:
            act = lambda x : x

        self.act = act

    def __call__(self, x):
        act = self.act(x)
        return act

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            Norm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            Norm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Model(nn.Module):
    def __init__(self, params, bilinear=False):
        num_in_ch = params['in_channels']
        num_out_ch = params['out_channels']
        self.num_out_ch = num_out_ch
        super(Model, self).__init__()
        self.n_channels = num_in_ch
        self.bilinear = bilinear
        self.inc = DoubleConv(num_in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.conv_last = nn.Conv2d(64, num_out_ch, 3, 1, 1)
        self.act_bias = Activation('leakyrelu')
        self.act_out = Activation('leakyrelu')

    def forward(self, input):
        B, S, C, W, H = tuple(input.shape)
        base_line = input[:, -1:, :, ...]
        base_line = base_line.repeat(1, self.num_out_ch, 1, 1, 1)  # 初始场作为训练时候的baseline
        x = input.reshape(B, -1, W, H)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.conv_last(self.conv_before_upsample(x))
        out = self.act_bias(out)
        # out = fuxi + out
        # out = self.act_out(out)
        out = torch.unsqueeze(out, dim=-3)
        return base_line, out


if __name__ == '__main__':
    print("********************************")
    in_varibales = 24
    in_times = 72
    out_varibales = 1
    out_times = 72
    input_size = in_times * in_varibales
    output_size = out_times * out_varibales
    params = {'in_channels': input_size,
              'out_channels': output_size}
    model = Model(params=params)
    sample = torch.randn(2, in_times, in_varibales, 57, 81)
    print(sample.shape)
    print(model(sample)[1].shape)