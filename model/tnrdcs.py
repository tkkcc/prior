# checkpoint_sequential debug
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from config import o
from util import kaiming_normal, parameter


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


# segments is length of each segment
def checkpoint_sequential(functions, segments, inputs):
    start = 0
    end = -1
    index = 0
    while end < len(functions) - segments[-1]:
        end = start + (segments[index] if index < len(segments) else segments[-1])
        index += 1
        inputs = checkpoint(run_function(start, end, functions), inputs)
        start = end
    return run_function(start, len(functions), functions)(inputs)


class BN(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(BN, self).__init__(*args, **kwargs)

        def hook(self, x, y):
            x = x[0]
            self.g = grad(y.sum(), x, retain_graph=True, create_graph=True)[0]

        self.register_forward_hook(hook)


class Rbf(nn.Module):
    def __init__(self):
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
        self.grad = False
        # require assign after init
        self.w = None

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
        x = self.act(x)
        return x if y is None else x * y


class Stage(nn.Module):
    def __init__(self, stage=1):
        super(Stage, self).__init__()
        # make parameter
        self.depth = depth = o.depth if type(o.depth) is int else o.depth[stage - 1]
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
        rbf = [Rbf() for i in range(depth)]

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
        x.requires_grad = True
        t = []
        step = 3
        index = 0
        for i in range(self.depth):
            x = checkpoint(run_function(index, index + step, self.a), x)
            t.append(x)
            index += step
        x = checkpoint(run_function(index, index + step, self.a), x)
        index += step
        for i in reversed(range(1, self.depth - 1)):
            x = checkpoint(run_function(index, index + step, self.a), t[i], x)
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
