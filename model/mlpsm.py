# replace elu with sigmoid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.checkpoint import checkpoint
from torch.autograd import grad
from util import show, log, parameter, gen_dct2
from scipy.io import loadmat
from config import o

# mlp
class C(nn.Module):
    def __init__(self, ci=o.channel, co=o.channel, cm=o.channel):
        super(C, self).__init__()
        self.m = nn.Sequential(
            nn.Linear(ci, cm),
            nn.Sigmoid(),
            # nn.Tanh(),
            # nn.ELU(),
            nn.Linear(cm, cm),
            nn.Sigmoid(),
            # nn.Tanh(),
            # nn.ELU(),
            nn.Linear(cm, cm),
            nn.Sigmoid(),
            # nn.Tanh(),
            # nn.ELU(),
            nn.Linear(cm, co),
            nn.Softplus(),
        )

    # BxCxHxW6
    def forward(self, x):
        if x.dim() == 1:
            return self.m(x)
        x = x.permute(0, 2, 3, 1)
        x = self.m(x)
        x = x.permute(0, 3, 1, 2)
        return x


# cnn module
class D(nn.Module):
    def __init__(self, ci=o.channel, co=o.channel):
        super(D, self).__init__()
        c = lambda ii, oo: nn.Conv2d(ii, oo, o.filter_size, padding=o.filter_size // 2)
        cr = lambda i, o: (c(i, o), nn.ReLU())
        cbr = lambda i, o: (c(i, o), nn.BatchNorm2d(o, track_running_stats=False), nn.ReLU())
        cbr4 = (j for i in range(4) for j in cbr(o.channel, o.channel))
        self.m = nn.Sequential(*cr(ci, o.channel), *cbr4, c(o.channel, co))

    def forward(self, x):
        return self.m(x)


class ModelStage(nn.Module):
    def __init__(self, stage=1):
        super(ModelStage, self).__init__()
        penalty_num = o.penalty_num
        self.depth = o.depth
        self.channel = channel = filter_num = o.channel
        self.filter_size = filter_size = o.filter_size
        self.lam = torch.tensor(0.0).view(1)
        # self.lam = torch.tensor(0, dtype=torch.float)
        # self.mean = torch.linspace(-310, 310, penalty_num).view(1, 1, penalty_num, 1, 1)
        # self.actw = torch.randn(1, filter_num, penalty_num, 1, 1)
        # self.actw *= 10 if stage == 1 else 5 if stage == 2 else 1
        # self.actw *= 10
        # self.actw = [torch.randn(1, filter_num, penalty_num, 1, 1) for i in range(self.depth)]
        # self.filter = [
        #     torch.randn(channel, 1, filter_size, filter_size),
        #     *(
        #         torch.randn(channel, channel, filter_size, filter_size)
        #         for i in range(self.depth - 1)
        #     ),
        # ]
        # self.bias = [torch.randn(channel) for i in range(self.depth)]
        # self.pad = nn.ReplicationPad2d(filter_size // 2)
        # self.crop = nn.ReplicationPad2d(-(filter_size // 2))
        self.lam = parameter(self.lam)
        # self.bias = parameter(self.bias, o.bias_scale)
        # self.filter = parameter(self.filter, o.filter_scale)
        # self.bias = parameter(self.bias, o.bias_scale)
        # self.filter = parameter(self.filter, o.filter_scale)
        # self.actw = parameter(self.actw, o.actw_scale)
        # self.k1 = D(1, 64)
        # self.k2 = D()
        self.k1 = nn.Conv2d(1, o.channel, o.filter_size, padding=o.filter_size // 2)
        self.k2 = nn.Conv2d(o.channel, o.channel, o.filter_size, padding=o.filter_size // 2)
        nn.init.normal_(self.k1.weight)
        nn.init.normal_(self.k2.weight)
        self.k1.bias.data *= o.bias_scale
        self.k2.bias.data *= o.bias_scale
        self.k1.weight.data *= o.filter_scale
        self.k2.weight.data *= o.filter_scale

        self.p1 = C()
        self.p2 = C()
        # self.n1 = nn.InstanceNorm2d(o.channel,track_running_stats=True)
        # self.n2 = nn.InstanceNorm2d(o.channel,track_running_stats=True)
        # self.p3 = C(1, 1, 16)
        # self.lam = D()s
        # self.inf = nn.InstanceNorm2d(channel)
        # self.k1 = D(1)
        # self.k2 = D()
   # Bx1xHxW
    def forward(self, *inputs):
        x, y, lam = inputs
        xx = x

        # f = self.filter
        x = self.k1(x)
        # x = self.n1(x)
        # x = F.conv2d(x, f[0], self.bias[0], padding=o.filter_size // 2)
        x = self.p1(x)
        # x = F.conv2d(x, f[1], self.bias[1], padding=o.filter_size // 2)
        x = self.k2(x)
        # x = self.n2(x)
        x = self.p2(x)

        # x = self.k1(x)
        # x = self.p1(x)
        # x = self.k2(x)
        # x = self.p2(x)
        x = grad(x.sum(), xx, create_graph=True)[0]
        return xx - x - self.lam * (xx - y)
        # x = x * 255
        # y = y * 255
        # xx = x
        # self.mean = self.mean.to(x.device)
        # f = self.filter
        # t = []
        # for i in range(self.depth):
        #     x = F.conv2d(self.pad(x), f[i], self.bias[i])
        #     t.append(x)
        #     x = self.act(x, self.actw[i])
        # x = self.crop(F.conv_transpose2d(x, f[self.depth - 1]))
        # for i in reversed(range(self.depth - 1)):
        #     c1 = t[i]
        #     x = x * self.act(c1, self.actw[i], True)
        #     x = self.crop(F.conv_transpose2d(x, f[i]))
        # return (xx - (x + self.lam.exp() * (xx - y))) / 255


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
        with torch.enable_grad():
            # tnrd pad and crop
            # x^t, y=x^0, s
            if not d[0].requires_grad:
                d[0].requires_grad = True
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
            # return d[0]
            return t
