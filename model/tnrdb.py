# tnrd bn
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.checkpoint import checkpoint

from util import show, log, parameter, gen_dct2, kaiming_normal
from scipy.io import loadmat
from config import o


class BN(nn.BatchNorm2d):
    def __init__(self, *argv, **kwargs):
        super(BN, self).__init__(*argv, **kwargs)

        def hook(self, x, y):
            x = x[0]
            self.g = grad(y.sum(), x, retain_graph=True, create_graph=True)[0]

        # self.register_forward_hook(hook)


class ModelStage(nn.Module):
    def __init__(self, stage=1):
        super(ModelStage, self).__init__()
        penalty_num = o.penalty_num
        self.depth = o.depth if type(o.depth) is int else o.depth[stage - 1]
        self.channel = channel = filter_num = o.channel
        self.filter_size = filter_size = o.filter_size
        self.lam = torch.tensor(0 if stage == 1 else np.log(0.1), dtype=torch.float)
        space = o.penalty_space
        self.register_buffer(
            "mean",
            torch.linspace(-space, space, penalty_num).view(1, 1, penalty_num, 1, 1),
        )
        self.register_buffer(
            "ngammas",
            -torch.tensor(o.penalty_gamma or (2 * space / (penalty_num - 1))).pow(2).float(),
        )
        self.actw = [
            torch.randn(1, filter_num, penalty_num, 1, 1) for i in range(self.depth)
        ]
        self.filter = [
            torch.randn(channel, 1, filter_size, filter_size),
            *(
                torch.randn(channel, channel, filter_size, filter_size)
                for i in range(self.depth - 1)
            ),
        ]
        kaiming_normal(self.filter)

        self.bias = [torch.randn(channel) for i in range(self.depth)]
        self.pad = nn.ReplicationPad2d(filter_size // 2)
        self.crop = nn.ReplicationPad2d(-(filter_size // 2))
        self.lam = parameter(self.lam)
        self.bias = parameter(self.bias, o.bias_scale)
        self.filter = parameter(self.filter, o.filter_scale)
        self.actw = parameter(self.actw, o.actw_scale)
        # self.bn = [BN(o.channel, momentum=None) for i in range(self.depth*2)]
        self.bn = [
            BN(o.channel, track_running_stats=False) for i in range(self.depth * 2)
        ]
        self.bn = nn.ModuleList(self.bn)
        # self.inf = nn.InstanceNorm2d(channel)

    # checkpoint a function
    def act(self, x, w, gradient=False):
        if (
            x.shape[1] == 1
            or (o.mem_capacity == 1 and x.shape[-1] < o.patch_size * 2)
            or (o.mem_capacity == 2)
        ):
            x = x.unsqueeze(2)
            if not gradient:
                x = (((x - self.mean).pow(2) / self.ngammas / 2).exp() * w).sum(2)
            else:
                t = x - self.mean
                x = ((t.pow(2) / self.ngammas / 2).exp() * t / self.ngammas * w).sum(2)
        else:
            # do on each channel
            x, y = torch.empty_like(x), x
            for j in range(x.shape[1]):
                x[:, j, ...] = self.act(
                    y[:, j, ...].unsqueeze(1), w[:, j, ...].unsqueeze(1), gradient
                ).squeeze(1)
        return x

    # Bx1xHxW
    def forward(self, *inputs):
        x, y, lam = inputs
        x = x * o.ioscale
        y = y * o.ioscale
        xx = x
        f = self.filter
        t = []
        for i in range(self.depth):
            x = F.conv2d(self.pad(x), f[i], self.bias[i])
            # x = self.bn[i](x)
            t.append(x)
            x = self.act(x, self.actw[i])
        for i in reversed(range(self.depth)):
            if i != self.depth - 1:
                x *= self.act(t[i], self.actw[i], True)
            # x *= self.bn[i].g
            # x = self.bn[i+2](x)
            x = self.crop(F.conv_transpose2d(x, f[i]))
        return (xx - (x + self.lam.exp() * (xx - y))) / o.ioscale


class ModelStack(nn.Module):
    def __init__(self, stage=1, weight=None):
        super(ModelStack, self).__init__()
        if type(stage) == int:
            stage = range(1, stage + 1)
        self.m = nn.ModuleList(ModelStage(i) for i in stage)
        pad_width = self.m[0].filter_size + 1
        self.pad = nn.ReplicationPad2d(pad_width)
        self.crop = nn.ReplicationPad2d(-pad_width)
        self.stage = stage

    def forward(self, d):
        # x^t, y=x^0, s
        # with torch.enable_grad():
        d[1] = self.pad(d[1])
        t = []
        for i in self.m:
            d[0] = self.pad(d[0])
            if o.checkpoint:
                d[2].requires_grad = True
                d[0] = checkpoint(i, *d)
            else:
                d[0] = i(*d)
            d[0] = self.crop(d[0])
            t.append(d[0])
        return t
