import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from . import ConcatConv2d, ODEBlock, UpsampleBlock, init_weights


class ODEFuncMSR(nn.Module):
    def __init__(self, hidden_size):
        super(ODEFuncMSR, self).__init__()
        self.conv1 = ConcatConv2d(hidden_size, hidden_size, kernel_size=3,
                                  padding=1)
        self.conv2 = ConcatConv2d(hidden_size, hidden_size, kernel_size=3,
                                  padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.nfe = 0
        init_weights([self.conv1._layer, self.conv2._layer], scale=0.1)

    def forward(self, t, x):
        self.nfe += 1
        out = self.lrelu(self.conv1(t, x))
        out = self.conv2(t, out)

        return x + out


class ODEMSR(nn.Module):
    def __init__(self, scale_factor, hidden_size, rtol, atol):
        self.scale_factor = scale_factor
        super(ODEMSR, self).__init__()
        self.conv_first = nn.Conv2d(3, hidden_size, kernel_size=3, padding=1,
                                    bias=True)
        self.ode = ODEBlock(ODEFuncMSR(hidden_size), rtol=rtol, atol=atol)
        self.upconv1 = nn.Conv2d(hidden_size, hidden_size * 4, kernel_size=3,
                                 padding=1, bias=True)
        self.upconv2 = nn.Conv2d(hidden_size, hidden_size * 4, kernel_size=3,
                                 padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(hidden_size, hidden_size, kernel_size=3,
                                padding=1, bias=True)
        self.conv_last = nn.Conv2d(hidden_size, 3, kernel_size=3, padding=1,
                                   bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        init_weights([self.conv_first, self.upconv1, self.upconv2,
                      self.HRconv, self.conv_last], scale=0.1)

    def forward(self, x):
        out = self.lrelu(self.conv_first(x))
        out = self.ode(out)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.scale_factor,
                             mode='bilinear', align_corners=False)

        return base + out


class ODEFuncSR(nn.Module):
    def __init__(self, hidden_size):
        super(ODEFuncSR, self).__init__()
        self.conv1 = ConcatConv2d(hidden_size, hidden_size, kernel_size=3,
                                  padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.prelu = nn.PReLU()
        self.conv2 = ConcatConv2d(hidden_size, hidden_size, kernel_size=3,
                                  padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.prelu(self.bn1(self.conv1(t, x)))
        out = self.bn2(self.conv2(t, out))

        return x + out


class ODESR(nn.Module):
    def __init__(self, scale_factor, hidden_size, rtol, atol):
        super(ODESR, self).__init__()
        upsample_block_num = int(math.log(scale_factor, 2))
        self.conv_first = nn.Conv2d(3, hidden_size, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()
        self.ode = ODEBlock(ODEFuncSR(hidden_size), rtol=rtol, atol=atol)

        self.conv_mid = nn.Conv2d(hidden_size, hidden_size, kernel_size=3,
                                  padding=1)
        self.bn_mid = nn.BatchNorm2d(hidden_size)

        upsample = [
            UpsampleBlock(hidden_size, 2) for _ in range(upsample_block_num)
        ]
        self.upsample = nn.Sequential(*upsample)
        self.conv_last = nn.Conv2d(hidden_size, 3, kernel_size=9, padding=4)

    def forward(self, x):
        pre = self.prelu(self.conv_first(x))
        ode = self.ode(pre)
        mid = self.bn_mid(self.conv_mid(ode))
        out = self.upsample(mid + pre)
        out = self.conv_last(out)
        out = torch.tanh(out)

        return out
