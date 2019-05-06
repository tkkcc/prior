# replace all rbf by 3 convs
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


class Rbf(nn.Module):
    def __init__(
        self, ps=o.penalty_space, pn=o.penalty_num, pg=o.penalty_gamma, operator="*"
    ):
        super(Rbf, self).__init__()
        c = lambda ic=o.channel, oc=o.channel: nn.Conv2d(ic, oc, 1)
        r = nn.ReLU
        elu = nn.ELU
        sp = nn.Softplus
        sm = nn.Sigmoid
        ss = nn.Softsign
        lsm = nn.LogSigmoid
        th = nn.Tanh
        bn = lambda ic: nn.BatchNorm2d(ic)
        cc = o.cc
        csm = lambda: (c(cc, cc), sm())
        cr = lambda: (c(cc, cc), r())
        csp = lambda: (c(cc, cc), sp())
        self.c1 = nn.Sequential(c(1, cc), sm(), c(cc, cc), sm(), c(cc, 1))
        self.c2 = nn.Sequential(c(1, cc), sm(), c(cc, cc), sm(), c(cc, 1))
        self.grad = False
        # require assign after init
        self.w = None
        self.ps = ps
        self.operator = operator
        self.b = ps * 4 / 3

    def forward(self, x, y=None):
        self.grad = False if y is None else True
        if self.grad == False:
            x = self.c1(x)
        else:
            x = self.c2(x)
        return x if y is None else x * y if self.operator == "*" else x + y


class Stage(nn.Module):
    def __init__(self, stage=1):
        super(Stage, self).__init__()
        # make parameter
        depth = self.depth = o.depth[stage - 1]
        lam = torch.tensor(1.0 if stage == 1 else 0.1).log()
        actw = [torch.randn(1, o.channel, o.penalty_num[i], 1, 1) for i in range(depth)]
        filter = [
            torch.randn(o.channel, 1, o.filter_size[0], o.filter_size[0]),
            *(
                torch.randn(o.channel, o.channel, o.filter_size[i], o.filter_size[i])
                for i in range(1, depth)
            ),
        ]
        kaiming_normal(filter)
        bias = [torch.randn(o.channel) for i in range(depth)]
        self.lam = lam = parameter(lam)
        self.bias = bias = parameter(bias, o.bias_scale)
        self.filter = filter = parameter(filter, o.filter_scale)
        self.actw = actw = parameter(actw, o.actw_scale)

        # make modules
        pad = [nn.ReplicationPad2d(o.filter_size[i] // 2) for i in range(depth)]
        crop = [nn.ReplicationPad2d(-(o.filter_size[i] // 2)) for i in range(depth)]
        conv = [
            nn.Conv2d(1, o.channel, o.filter_size[0]),
            *(
                nn.Conv2d(o.channel, o.channel, o.filter_size[i])
                for i in range(1, depth)
            ),
        ]
        convt = [
            nn.ConvTranspose2d(o.channel, 1, o.filter_size[0], bias=False),
            *(
                nn.ConvTranspose2d(o.channel, o.channel, o.filter_size[i], bias=False)
                for i in range(1, depth)
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
        pcf = lambda i: (pad[i], conv[i], rbf[i])
        cc = lambda i: (convt[i], crop[i])
        rcc = lambda i: (rbf[i], convt[i], crop[i])
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
        for i in range(self.depth + 1):
            x = stage_cp(run_function(index, index + step, self.a), x)
            if i < self.depth - 1:
                t.append(x)
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
        pad_width = o.filter_size[0] + 1
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
            d[2].requires_grad = True
            d[0] = model_cp(i, *d)
            d[0] = self.crop(d[0])
        # for mem
        t.append(d[0])
        return t
