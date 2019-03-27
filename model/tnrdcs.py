# force checkpoint for patch_size>100
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from config import o
from util import kaiming_normal, parameter


class BN(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(BN, self).__init__(*args, **kwargs)

        def hook(self, x, y):
            x = x[0]
            self.g = grad(y.sum(), x, retain_graph=True, create_graph=True)[0]

        self.register_forward_hook(hook)


def act(x, mean, ngammas, grad, w, mm=None):
    if (
        x.shape[1] == 1
        or (o.mem_capacity == 1 and x.shape[-1] < o.patch_size + o.filter_size * 3)
        or (o.mem_capacity == 2)
    ):
        x = x.unsqueeze(2)
        if not grad:
            x = (((x - mean).pow(2) / ngammas / 2).exp() * w).sum(2)
        else:
            x = x - mean
            x = ((x.pow(2) / ngammas / 2).exp() * x / ngammas * w).sum(2)
    else:
        # do on each channel
        exit(0)
        x, y = torch.empty_like(x), x
        for j in range(x.shape[1]):
            x[:, j, ...] = act(
                y[:, j, ...].unsqueeze(1), w[:, j, ...].unsqueeze(1)
            ).squeeze(1)
    if mm is not None:
        return x * mm
    return x


class Rbf(nn.Module):
    def __init__(self, once=False):
        super(Rbf, self).__init__()
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
        self.once = once
        self.grad = False
        # require assign after init
        self.w = None

    def forward(self, x):
        if not self.grad:
            self.cache = x
            x = act(x, self.mean, self.ngammas, self.grad, self.w)
        else:
            x = act(self.cache, self.mean, self.ngammas, self.grad, self.w, x)
        if not self.once:
            self.grad = not self.grad
        return x


class Stage(nn.Module):
    def __init__(self, stage=1):
        super(Stage, self).__init__()
        # make parameter
        depth = o.depth if type(o.depth) is int else o.depth[stage - 1]
        lam = torch.tensor(1.0 if stage == 1 else 0.1).log()
        actw = [torch.randn(1, o.channel, o.penalty_num, 1, 1) for i in range(depth)]
        filter = [
            torch.randn(o.channel, 1, o.filter_size, o.filter_size),
            *(
                torch.randn(o.channel, o.channel, o.filter_size, o.filter_size)
                for i in range(depth - 1)
            ),
        ]
        kaiming_normal(filter)
        bias = [torch.randn(o.channel) for i in range(depth)]
        lam = parameter(lam)
        bias = parameter(bias, o.bias_scale)
        filter = parameter(filter, o.filter_scale)
        actw = parameter(actw, o.actw_scale)

        # make modules
        pad = nn.ReplicationPad2d(o.filter_size // 2)
        crop = nn.ReplicationPad2d(-(o.filter_size // 2))
        conv = [
            nn.Conv2d(1, o.channel, o.filter_size),
            *(nn.Conv2d(o.channel, o.channel, o.filter_size) for i in range(depth - 1)),
        ]
        convt = [
            nn.ConvTranspose2d(o.channel, 1, o.filter_size, bias=False),
            *(
                nn.ConvTranspose2d(o.channel, o.channel, o.filter_size, bias=False)
                for i in range(depth - 1)
            ),
        ]
        rbf = [Rbf() for i in range(depth)]
        rbf[-1].once = True
        # rbfg = [Rbf(True) for i in range(depth)]
        # assign parameter
        for i in range(depth):
            # assert conv[i].weight.shape == filter[i].shape
            # assert conv[i].bias.shape == bias[i].shape
            # assert convt[i].weight.shape == filter[i].transpose(0,1).shape
            conv[i].weight = filter[i]
            conv[i].bias = bias[i]
            convt[i].weight = filter[i]
            rbf[i].w = actw[i]
            # rbfg[i].w = actw[i]
        # del rbfg[depth - 1]
        # make sequential
        pcf = lambda i: (pad, conv[i], rbf[i])
        cc = lambda i: (convt[i], crop)
        rcc = lambda i: (rbf[i], convt[i], crop)
        m = nn.Sequential(
            # pad,
            # conv[0],
            # rbf[0],
            # pad,
            # conv[1],
            # rbf[1],
            # pad,
            # conv[2],
            # rbf[2],
            # convt[2],
            # crop,
            # rbf[1],
            # convt[1],
            # crop,
            # rbf[0],
            # convt[0],
            # crop,
            *(j for i in range(depth) for j in pcf(i)),
            *cc(depth - 1),
            *(j for i in reversed(range(depth - 1)) for j in rcc(i))
        )
        self.m = m
        self.lam = lam

    # Bx1xHxW
    def forward(self, *inputs):
        x, y, lam = inputs
        x, y = x * o.ioscale, y * o.ioscale
        xx = x

        x.requires_grad = True
        x = checkpoint_sequential(self.m, o.stage_checkpoint, x)
        # x = self.m(x)
        # x.sum().backward()
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
