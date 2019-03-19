# tnrd bn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.checkpoint import checkpoint

from config import o
from util import kaiming_normal, parameter


class BN(nn.BatchNorm2d):
    def __init__(self, *argv, **kwargs):
        super(BN, self).__init__(*argv, **kwargs)

        def hook(self, x, y):
            x = x[0]
            self.g = grad(y.sum(), x, retain_graph=True, create_graph=True)[0]

        self.register_forward_hook(hook)


class ModelStage(nn.Module):
    def __init__(self, stage=1):
        super(ModelStage, self).__init__()
        self.depth = o.depth if type(o.depth) is int else o.depth[stage - 1]
        self.lam = torch.tensor(1.0 if stage == 1 else 0.1).log()
        self.register_buffer(
            "mean",
            torch.linspace(-o.penalty_space, o.penalty_space, o.penalty_num).view(
                1, 1, o.penalty_num, 1, 1
            ),
        )
        self.register_buffer(
            "ngammas",
            -torch.tensor(
                o.penalty_gamma or (2 * o.penalty_space / (o.penalty_num - 1))
            )
            .float()
            .pow(2),
        )
        self.actw = [
            torch.randn(1, o.channel, o.penalty_num, 1, 1) for i in range(self.depth)
        ]
        self.filter = [
            torch.randn(o.channel, 1, o.filter_size, o.filter_size),
            *(
                torch.randn(o.channel, o.channel, o.filter_size, o.filter_size)
                for i in range(self.depth - 1)
            ),
        ]
        kaiming_normal(self.filter)
        self.bias = [torch.randn(o.channel) for i in range(self.depth)]
        self.bn = [
            BN(o.channel, track_running_stats=False) for i in range(self.depth * 2)
        ]

        self.pad = nn.ReplicationPad2d(o.filter_size // 2)
        self.crop = nn.ReplicationPad2d(-(o.filter_size // 2))
        self.lam = parameter(self.lam)
        self.bias = parameter(self.bias, o.bias_scale)
        self.filter = parameter(self.filter, o.filter_scale)
        self.actw = parameter(self.actw, o.actw_scale)
        self.bn = nn.ModuleList(self.bn)

    # checkpoint a function
    def act(self, x, w, grad=False):
        if (
            x.shape[1] == 1
            or (o.mem_capacity == 1 and x.shape[-1] < o.patch_size * 2)
            or (o.mem_capacity == 2)
        ):
            x = x.unsqueeze(2)
            if not grad:
                x = (((x - self.mean).pow(2) / self.ngammas / 2).exp() * w).sum(2)
            else:
                t = x - self.mean
                x = ((t.pow(2) / self.ngammas / 2).exp() * t / self.ngammas * w).sum(2)
        else:
            # do on each channel
            x, y = torch.empty_like(x), x
            for j in range(x.shape[1]):
                x[:, j, ...] = self.act(
                    y[:, j, ...].unsqueeze(1), w[:, j, ...].unsqueeze(1), grad
                ).squeeze(1)
        return x

    # Bx1xHxW
    def forward(self, *inputs):
        x, y, lam = inputs
        x, y = x * o.ioscale, y * o.ioscale
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
    def __init__(self, stage=1):
        super(ModelStack, self).__init__()
        self.stage = stage = range(1, stage + 1) if type(stage) == int else stage
        pad_width = o.filter_size + 1
        self.pad = nn.ReplicationPad2d(pad_width)
        self.crop = nn.ReplicationPad2d(-pad_width)
        self.m = nn.ModuleList(ModelStage(i) for i in stage)

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
