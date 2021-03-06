import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.checkpoint import checkpoint

from config import o
from util import kaiming_normal, parameter
o.filter_size = o.filter_size[0]
o.penalty_gamma = o.penalty_gamma[0]
o.penalty_space = o.penalty_space[0]
o.penalty_num = o.penalty_num[0]

class BN(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(BN, self).__init__(*args, **kwargs)

        def hook(self, x, y):
            x = x[0]
            self.g = grad(y.sum(), x, retain_graph=True, create_graph=True)[0]

        self.register_forward_hook(hook)


class Rbf(nn.Module):
    def __init__(
        self, ps=o.penalty_space, pn=o.penalty_num, pg=o.penalty_gamma, grad=False
    ):
        super(Rbf, self).__init__()
        self.register_buffer("mean", torch.linspace(-ps, ps, pn).view(1, 1, pn, 1, 1))
        self.register_buffer(
            "ngammas", -torch.tensor(pg or (2 * ps / (pn - 1))).float().pow(2)
        )
        self.grad = grad

    def act(self, x, w):
        if (
            x.shape[1] == 1
            or (o.mem_capacity == 1 and x.shape[-1] < 120)
            or (o.mem_capacity == 2)
        ):
            # use tensor boardcast
            x = x.unsqueeze(2)
            if not self.grad:
                x = (((x - self.mean).pow(2) / self.ngammas / 2).exp() * w).sum(2)
            else:
                x = x - self.mean
                x = ((x.pow(2) / self.ngammas / 2).exp() * x / self.ngammas * w).sum(2)
        else:
            # do on each channel
            x, y = torch.empty_like(x), x
            for j in range(x.shape[1]):
                x[:, j, ...] = self.act(
                    y[:, j, ...].unsqueeze(1), w[:, j, ...].unsqueeze(1)
                ).squeeze(1)
        return x

    def forward(self, *args):
        return self.act(*args)


class Stage(nn.Module):
    def __init__(self, stage=1):

        super(Stage, self).__init__()
        self.depth = o.depth if type(o.depth) is int else o.depth[stage - 1]
        self.lam = torch.tensor(1.0 if stage == 1 else 0.1).log()
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
        # kaiming_normal(self.filter)
        self.bias = [torch.randn(o.channel) for i in range(self.depth)]

        self.pad = nn.ReplicationPad2d(o.filter_size // 2)
        self.crop = nn.ReplicationPad2d(-(o.filter_size // 2))
        self.rbf = Rbf()
        self.rbfg = Rbf(grad=True)

        self.lam = parameter(self.lam)
        self.bias = parameter(self.bias, o.bias_scale)
        self.filter = parameter(self.filter, o.filter_scale)
        self.actw = parameter(self.actw, o.actw_scale)

    # Bx1xHxW
    def forward(self, *inputs):
        x, y, lam = inputs
        x, y = x * o.ioscale, y * o.ioscale
        xx = x
        f = self.filter
        t = []
        for i in range(self.depth):
            x = F.conv2d(self.pad(x), f[i], self.bias[i])
            t.append(x)
            x = self.rbf(x, self.actw[i])
        for i in reversed(range(self.depth)):
            if i != self.depth - 1:
                x *= self.rbfg(t[i], self.actw[i])
            x = self.crop(F.conv_transpose2d(x, f[i]))
        return (xx - (x + self.lam.exp() * (xx - y))) / o.ioscale


class Model(nn.Module):
    def __init__(self, stage=1):
        super(Model, self).__init__()
        self.stage = stage = range(1, stage + 1) if type(stage) == int else stage
        pad_width = o.filter_size + 1
        self.pad = nn.ReplicationPad2d(pad_width)
        self.crop = nn.ReplicationPad2d(-pad_width)
        self.m = nn.ModuleList(Stage(i) for i in stage)

    def forward(self, d):
        # x^t, y=x^0, s
        # with torch.enable_grad():
        d[1] = self.pad(d[1])
        t = []
        for i in self.m:
            d[0] = self.pad(d[0])
            if o.model_checkpoint:
                d[2].requires_grad = True
                d[0] = checkpoint(i, *d)
            else:
                d[0] = i(*d)
            d[0] = self.crop(d[0])
        # for mem
        t.append(d[0])
        return t
