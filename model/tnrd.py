# tnrd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import show, log, parameter, irfft, ifft, fft, rfft, cm, conj, psf2otf


class ModelStage(nn.Module):
    def __init__(self, stage=1, lam=1.0):
        super(ModelStage, self).__init__()
        filter_size = 5
        filter_num = 24
        penalty_num = 63
        self.filter_size = filter_size
        self.lam = torch.tensor(0 if stage == 1 else log(0.1), dtype=torch.float).exp()
        self.mean = torch.linspace(-310, 310, penalty_num).view(1, 1, penalty_num, 1, 1)
        self.weight = torch.randn(1, filter_num, penalty_num, 1, 1) * 0.1
        self.filter = torch.randn(filter_num, 1, filter_size, filter_size) * 0.1
        self.pad = nn.ReplicationPad2d(filter_size // 2)
        self.crop = nn.ReplicationPad2d(-(filter_size // 2))
        self.weight = nn.Parameter(self.weight)
        self.filter = nn.Parameter(self.filter)

    # Bx1xHxW
    def forward(self, inputs):
        # y=x^0
        x, y, lam = inputs
        xx = x
        self.mean = self.mean.to(x.device)
        self.lam = self.lam.to(x.device)
        x = self.pad(x)
        x = F.conv2d(x, self.filter)
        x = x.unsqueeze(2)
        x = ((x - self.mean).pow(2) / -200).exp() * self.weight
        x = x.sum(2)
        x = F.conv_transpose2d(x, self.filter)
        x = self.crop(x)
        # log("filter", self.filter)
        return xx - (x + self.lam * (xx - y))


class ModelStack(nn.Module):
    def __init__(self, stage=1, weight=None):
        super(ModelStack, self).__init__()
        self.m = nn.ModuleList(ModelStage(i + 1) for i in range(stage))
        pad_width = self.m[0].filter_size + 1
        self.pad = nn.ReplicationPad2d(pad_width)
        self.crop = nn.ReplicationPad2d(-pad_width)
        self.stage = stage

    def forward(self, d):
        # tnrd pad and crop
        # x^t, y=x^0, s
        d[1] = self.pad(d[1])
        for i in self.m:
            d[0] = self.pad(d[0])
            d[0] = i(d)
            d[0] = self.crop(d[0])
        return d[0]