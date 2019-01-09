# symmetry model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import show, log, parameter


class ModelStage(nn.Module):
    def __init__(self, stage=1, lam=1.0):
        super(ModelStage, self).__init__()
        self.lam = 0.1
        self.filter_size = filter_size = 5
        self.depth = depth = 6
        in_channel = 1
        med_channel = 48
        out_channel = 48
        parameter_scale = .01
        self.w = [
            torch.randn(med_channel, in_channel, filter_size, filter_size),
            *(
                torch.randn(med_channel, med_channel, filter_size, filter_size)
                for i in range(depth - 2)
            ),
            torch.randn(out_channel, med_channel, filter_size, filter_size),
        ]
        self.b = [
            torch.randn(med_channel),
            *(torch.randn(med_channel) for i in range(depth - 2)),
            torch.randn(out_channel),
        ]
        self.w = parameter(self.w, parameter_scale)
        self.b = parameter(self.b, parameter_scale)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, inputs):
        # Bx1xHxW, Bx1xHxW, BxHxW, Bx1
        x, y, s = inputs
        c = x
        t = []
        data_norm_grad = self.lam * (x - y)
        p = [self.filter_size // 2] * 2
        w = [i / i.norm(2) for i in self.w]
        # w = self.w
        for i in range(self.depth):
            c = F.conv2d(c, w[i], padding=p)
            # if i < self.depth - 1:
            c = c.sigmoid()
            t.append(c)
        # c = self.bn(c)
        # log("cnnx", c)
        for i in reversed(range(self.depth)):
            # if i < self.depth - 1:
            if i < self.depth - 1:
                c = c * (t[i] * (1 - t[i]))
            c = F.conv_transpose2d(c, w[i], bias=None, padding=p)
        cnnx_square_grad = 2 * c
        # log("data_norm_grad", data_norm_grad)
        # log("cnnx_square_grad", cnnx_square_grad)
        # log("lam", self.lam)
        return x - (data_norm_grad + cnnx_square_grad)


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
