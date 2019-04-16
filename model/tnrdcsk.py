# checkpoint_sequential debug, multi way
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.checkpoint import checkpoint

from config import o
from util import kaiming_normal, parameter


stage_cp = (lambda f, *i: f(*i)) if not o.stage_checkpoint else checkpoint
model_cp = (lambda f, *i: f(*i)) if not o.model_checkpoint else checkpoint


def run_function(start, end, functions):
    def forward(*inputs):
        for j in range(start, end):
            if functions[j] is None:
                continue
            if isinstance(inputs, tuple):
                inputs = functions[j](*inputs)
            else:
                inputs = functions[j](inputs)
        return inputs

    return forward


class BN(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(BN, self).__init__(*args, **kwargs)

        def hook(self, x, y):
            x = x[0]
            self.g = grad(y.sum(), x, retain_graph=True, create_graph=True)[0]

        self.register_forward_hook(hook)


class Rbf(nn.Module):
    def __init__(self, ps=o.penalty_space, pn=o.penalty_num, pg=o.penalty_gamma):
        super(Rbf, self).__init__()
        self.register_buffer("mean", torch.linspace(-ps, ps, pn).view(1, 1, pn, 1, 1))
        self.register_buffer(
            "ngammas", -torch.tensor(pg or (2 * ps / (pn - 1))).float().pow(2)
        )
        self.grad = False
        # require assign after init
        self.w = None
        self.ps = ps
        self.b = ps * 4 / 3

    def act(self, x, w=None):
        if w is None:
            w = self.w
        if (
            x.shape[1] == 1
            or (o.mem_capacity == 1 and x.shape[-1] < o.patch_size + o.filter_size * 3)
            or (o.mem_capacity == 2)
        ):
            x = x.unsqueeze(2)
            if not self.grad:
                x = (((x - self.mean).pow(2) / self.ngammas / 2).exp() * w).sum(2)
            else:
                x = x - self.mean
                x = ((x.pow(2) / self.ngammas / 2).exp() * x / self.ngammas * w).sum(2)
        else:
            # do on each channel
            # exit(0)
            x, y = torch.empty_like(x), x
            for j in range(x.shape[1]):
                x[:, j, ...] = self.act(
                    y[:, j, ...].unsqueeze(1), w[:, j, ...].unsqueeze(1)
                ).squeeze(1)
        return x

    def forward(self, x, y=None):
        self.grad = False if y is None else True
        x = x.clamp(-self.b, self.b)
        x = self.act(x)
        return x if y is None else x * y


class Stage(nn.Module):
    def __init__(self, stage=1, depth=o.depth):
        super(Stage, self).__init__()
        # make parameter
        depth = self.depth = depth[stage - 1]
        lam = torch.tensor(1.0 if stage == 1 else 0.1).log()
        actw = [torch.randn(1, o.channel, o.penalty_num[i], 1, 1) for i in range(depth)]
        filter = [
            torch.randn(o.channel, 1, o.filter_size, o.filter_size),
            *(
                torch.randn(o.channel, o.channel, o.filter_size, o.filter_size)
                for i in range(depth - 1)
            ),
        ]
        kaiming_normal(filter)
        bias = [torch.randn(o.channel) for i in range(depth)]
        self.lam = lam = parameter(lam)
        self.bias = bias = parameter(bias, o.bias_scale)
        self.filter = filter = parameter(filter, o.filter_scale)
        self.actw = actw = parameter(actw, o.actw_scale)

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
        rbf = [
            Rbf(o.penalty_space[i], o.penalty_num[i], o.penalty_gamma[i])
            for i in range(depth)
        ]
        # assign parameter
        for i in range(depth):
            conv[i].weight = filter[i]
            conv[i].bias = bias[i]
            convt[i].weight = filter[i]
            rbf[i].w = actw[i]

        # make sequential
        pcf = lambda i: (pad, conv[i], rbf[i])
        cc = lambda i: (convt[i], crop)
        rcc = lambda i: (rbf[i], convt[i], crop)
        m = [
            None,
            *(j for i in range(depth) for j in pcf(i)),
            *cc(depth - 1),
            *(j for i in reversed(range(depth - 1)) for j in rcc(i)),
        ]
        self.a = nn.ModuleList(m)

    # Bx1xHxW
    def forward(self, *inputs):
        x, y, lam = inputs
        x, y = x * o.ioscale, y * o.ioscale
        xx = x
        if x.requires_grad == False:
            x.requires_grad = True
        t = []
        step = 3
        index = 0
        for i in range(self.depth):
            x = stage_cp(run_function(index, index + step, self.a), x)
            t.append(x)
            index += step
        x = stage_cp(run_function(index, index + step, self.a), x)
        index += step
        for i in reversed(range(1, self.depth - 1)):
            x = stage_cp(run_function(index, index + step, self.a), t[i], x)
            index += step
        x = run_function(index, len(self.a), self.a)(t[0], x)
        return (xx - (x + self.lam.exp() * (xx - y))) / o.ioscale


class Model(nn.Module):
    def __init__(self, stage=1):
        super(Model, self).__init__()
        self.stage = stage = range(1, stage + 1) if type(stage) == int else stage
        pad_width = o.filter_size + 1
        self.pad = nn.ReplicationPad2d(pad_width)
        self.crop = nn.ReplicationPad2d(-pad_width)
        self.m = nn.ModuleList(Stage(i) for i in stage)
        self.m2 = nn.ModuleList([Stage(stage[0] - 1)])
        self.k = nn.Conv2d(2, 1, 1)

    def forward(self, d):
        # x^t, y=x^0, s
        # with torch.enable_grad():
        x0 = d[0]
        d[1] = self.pad(d[1])
        # t = []
        for i in self.m:
            d[0] = self.pad(d[0])
            d[2].requires_grad = True
            d[0] = model_cp(i, *d)
            d[0] = self.crop(d[0])
        r1 = d[0]
        d[0] = x0
        for i in self.m2:
            d[0] = self.pad(d[0])
            d[2].requires_grad = True
            d[0] = model_cp(i, *d)
            d[0] = self.crop(d[0])
        r2 = d[0]
        r = self.k(torch.cat((r1, r2), 1))
        # for mem
        # t.append(d[0])
        return [r]
