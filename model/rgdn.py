# tnrd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.checkpoint import checkpoint
from util import show, log, parameter

# from scipy.io import loadmat
from config import o


class D(nn.Module):
    def __init__(self, ci=1, co=1):
        super(D, self).__init__()
        c = lambda i, o: nn.Conv2d(i, o, 5, padding=2, bias=True)
        cr = lambda i, o: (c(i, o), nn.ReLU())
        cbr = lambda i, o: (c(i, o), nn.BatchNorm2d(o), nn.ReLU())
        cbr4 = (j for i in range(4) for j in cbr(64, 64))
        self.m = nn.Sequential(*cr(ci, 64), *cbr4, c(64, co))

    def forward(self, x):
        return self.m(x)


class ModelStage(nn.Module):
    def __init__(self, stage=1):
        super(ModelStage, self).__init__()
        penalty_num = o.penalty_num
        self.depth = o.depth
        self.channel = channel = filter_num = o.channel
        self.filter_size = filter_size = o.filter_size
        self.filter = torch.randn(channel, 1, filter_size, filter_size)
        self.bias = torch.zeros(channel)
        self.filter = parameter(self.filter, o.filter_scale)
        self.bias = parameter(self.bias)
        self.h = D()
        self.d = D()
        self.r = D(64, 64)

    # Bx1xHxW
    def forward(self, *inputs):
        x, y, lam = inputs
        t = F.conv2d(x, self.filter, self.bias, padding=2)
        t = self.r(t)
        t = F.conv_transpose2d(t, self.filter, padding=2)
        x = x - t
        return x


class ModelStack(nn.Module):
    def __init__(self, stage=1, weight=None):
        super(ModelStack, self).__init__()
        if type(stage) == int:
            stage = range(1, stage + 1)
        # self.m = nn.ModuleList(ModelStage(i) for i in stage)
        pad_width = o.filter_size + 1
        self.pad = nn.ReplicationPad2d(pad_width)
        self.crop = nn.ReplicationPad2d(-pad_width)
        self.stage = stage
        self.m = ModelStage()

    def forward(self, d):
        # tnrd pad and crop
        # x^t, y=x^0, s
        d[1] = self.pad(d[1])
        for i in range(len(self.stage)):
            d[0] = self.pad(d[0])
            if o.checkpoint:
                d[2].requires_grad = True
                d[0] = checkpoint(self.m, *d)
            else:
                d[0] = self.m(*d)
            d[0] = self.crop(d[0])
        return d[0]

