# show prior
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.checkpoint import checkpoint

from util import show, log, parameter, gen_dct2, normalize ,sleep
from scipy.io import loadmat
from config import o, w

dontsave = False
# write x to tb
def save(x, name="?"):
    print(name)
    if name not in ["minus"]:
    # if name not in  ["d1","in","out","p2"]:
        return
    if dontsave:
        return
    assert x.shape[0] == 1
    for i in range(min(x.shape[1], 4)):
        w.add_image(name + "_" + str(i), normalize(x[0][i].unsqueeze(0)), 0)


class ModelStage(nn.Module):
    def __init__(self, stage=1):
        assert o.depth == 2
        super(ModelStage, self).__init__()
        penalty_num = o.penalty_num
        self.depth = o.depth
        self.channel = channel = filter_num = o.channel
        self.filter_size = filter_size = o.filter_size
        self.lam = torch.tensor(0 if stage == 1 else np.log(0.1), dtype=torch.float)
        # self.lam = torch.tensor(0, dtype=torch.float)
        self.mean = torch.linspace(-310, 310, penalty_num).view(1, 1, penalty_num, 1, 1)
        self.actw = torch.randn(1, filter_num, penalty_num, 1, 1)
        self.actw *= 10 if stage == 1 else 5 if stage == 2 else 1
        # self.actw *= 10
        self.actw = [
            torch.randn(1, filter_num, penalty_num, 1, 1),
            torch.randn(1, 1, penalty_num, 1, 1),
        ]
        self.filter = [
            torch.randn(channel, 1, filter_size, filter_size),
            torch.randn(1, channel, filter_size, filter_size),
        ]
        self.bias = [torch.randn(channel), torch.randn(1)]
        self.pad = nn.ReplicationPad2d(filter_size // 2)
        self.crop = nn.ReplicationPad2d(-(filter_size // 2))
        self.lam = parameter(self.lam)
        self.bias = parameter(self.bias, o.bias_scale)
        self.filter = parameter(self.filter, o.filter_scale)
        self.actw = parameter(self.actw, o.actw_scale)
        # p2_x = np.arange(-400, 400, 0.01)
        self.p2_y = np.loadtxt("tmp_prior_cum.txt")
        # hardcode
        # self.p2_x = torch.tensor(p2_x).to("cuda")
        # self.p2_y = torch.tensor(p2_y).to("cuda")
        # self.inf = nn.InstanceNorm2d(channel)

    # only run once, so
    def p2(self, x):
        assert x.dim() == 4 and x.shape[0] == 1 and x.shape[1] == 1
        x = x[0, 0, ...].clone().detach()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                a = round(x[i, j].item() / 0.01) + 40000
                if a < 0 or a > 80000:
                    a = 0
                x[i, j] = self.p2_y[a]
        return x.unsqueeze(0).unsqueeze(0)

    # checkpoint a function
    def act(self, x, w, gradient=False):
        if x.shape[-1] < o.patch_size * 2 or x.shape[1] == 1 or o.mem_infinity:
            x = x.unsqueeze(2)
            if not gradient:
                x = (((x - self.mean).pow(2) / -200).exp() * w).sum(2)
            else:
                x = (((x - self.mean).pow(2) / -200).exp() * (x - self.mean) / -100 * w).sum(2)
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
        x = x * 255
        y = y * 255
        xx = x
        self.mean = self.mean.to(x.device)
        f = self.filter
        t = []
        save(x, "in")
        for i in range(self.depth):
            x = F.conv2d(self.pad(x), f[i], self.bias[i])
            save(x, "c" + str(i + 1))
            t.append(x)
            # if i == 1:
            #     save(self.p2(x), "p2")
            x = self.act(x, self.actw[i])
            save(x, "a" + str(i + 1))
        x = self.crop(F.conv_transpose2d(x, f[self.depth - 1]))
        save(x, "d2")
        for i in reversed(range(self.depth - 1)):
            c1 = t[i]
            x = x * self.act(c1, self.actw[i], True)
            save(x, "b1")
            x = self.crop(F.conv_transpose2d(x, f[i]))
            save(x, "d1")
        save((x + self.lam.exp() * (xx - y)),"d1+data")
        xx = (xx - (x + self.lam.exp() * (xx - y))) / 255
        save(xx, "out")
        sleep(1)


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
        # tnrd pad and crop
        # x^t, y=x^0, s
        d[1] = self.pad(d[1])
        # d[0].require                                                                   _grad=True
        # d[1].requires_grad=True
        for i in self.m:
            d[0] = self.pad(d[0])
            if o.checkpoint:
                d[2].requires_grad = True
                d[0] = checkpoint(i, *d)
            else:
                d[0] = i(*d)
            d[0] = self.crop(d[0])
        return d[0]
        # d[0].requires_grad=True
        # d[1].requires_grad=True
        # d[2].requires_grad=True
        # return checkpoint_sequential(self.m,4,*d)
