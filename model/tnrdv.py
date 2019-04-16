# for visualization
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.checkpoint import checkpoint

from config import o, w
from util import kaiming_normal, parameter, normalize


class BN(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(BN, self).__init__(*args, **kwargs)

        def hook(self, x, y):
            x = x[0]
            self.g = grad(y.sum(), x, retain_graph=True, create_graph=True)[0]

        self.register_forward_hook(hook)


class RBF(nn.Module):
    def __init__(self, ps=o.penalty_space, grad=False):
        super(RBF, self).__init__()
        self.register_buffer(
            "mean",
            torch.linspace(-ps, ps, o.penalty_num).view(
                1, 1, o.penalty_num, 1, 1
            ),
        )
        self.register_buffer(
            "ngammas",
            -torch.tensor(o.penalty_gamma or (2 * ps / (o.penalty_num - 1)))
            .float()
            .pow(2),
        )
        self.grad = grad

    def act(self, x, w):
        if (
            x.shape[1] == 1
            or (o.mem_capacity == 1 and x.shape[-1] < o.patch_size * 1.2)
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
        kaiming_normal(self.filter)
        self.bias = [torch.randn(o.channel) for i in range(self.depth)]

        self.pad = nn.ReplicationPad2d(o.filter_size // 2)
        self.crop = nn.ReplicationPad2d(-(o.filter_size // 2))
        l = o.penalty_space
        l = l if type(l) == list else [l]
        
        self.rbf = [RBF(l[i] if i < len(l) else l[-1]) for i in range(self.depth)]
        self.rbfg = [
            RBF(l[i] if i < len(l) else l[-1], grad=True) for i in range(self.depth)
        ]

        self.lam = parameter(self.lam)
        self.bias = parameter(self.bias, o.bias_scale)
        self.filter = parameter(self.filter, o.filter_scale)
        self.actw = parameter(self.actw, o.actw_scale)
        self.stage = stage
        self.rbf=nn.ModuleList(self.rbf)
        self.rbfg=nn.ModuleList(self.rbfg)

    # Bx1xHxW
    def forward(self, *inputs):

        x, y, lam = inputs
        x, y = x * o.ioscale, y * o.ioscale
        xx = x
        f = self.filter
        t = []
        # w.add_histogram('x', x[0], 0)

        for i in range(self.depth):
            x = F.conv2d(self.pad(x), f[i], self.bias[i])

            def fg():
                p = "k" + str(i + 1)
                w.add_histogram(p, x[0], 0)

            fg()
            t.append(x)
            x = self.rbf[i](x, self.actw[i])

            def ff():
                # v
                p = ("p" + str(i + 1)) if i != self.depth - 1 else "p6`"
                # w.add_histogram(p, x[0], 0)
                l=o.penalty_space
                ps = l[i] if i < len(l) else l[-1]
                ps=310
                mean = torch.linspace(-ps, ps, 63).view(1, 63, 1, 1).to("cuda")
                for j in range(6):
                    w.add_image(
                        p + "-" + str(j + 1), normalize(x[0, j].unsqueeze(0)), 0
                    )
                    c = self.actw[i][:, j, ...]
                    for k in range(-ps, ps):
                        tt = (((k - mean).pow(2) / -200).exp() * c).sum()
                        w.add_scalar(p + "-" + str(j + 1), tt, k)

            # if self.stage==1:
            # ff()

        def ff():
            import numpy as np
            from scipy import integrate

            # make p6
            mean = torch.linspace(-310, 310, 63).view(1, 63, 1, 1).to("cuda")
            b = self.actw[-1]
            a = []
            for i in range(6):
                print(i)
                c = b[:, i, ...]
                x = torch.arange(-400, 400, 0.01).to("cuda")
                y = []
                for j in x:
                    tt = (((j - mean).pow(2) / -200).exp() * c).sum().item()
                    y.append(tt)
                x = np.arange(-400, 400, 0.01)
                z = integrate.cumtrapz(y, x, initial=0)
                # make p6 output
                x = t[-1][:, i, ...].clone().detach().cpu().numpy()
                for j in np.nditer(x, op_flags=["readwrite"]):
                    if -400 <= j <= 399:
                        j[...] = z[int(round((j + 400) / 0.01))]
                    else:
                        j[...] = 0
                x = torch.tensor(x)
                a.append(x)
                if i >= 6:
                    continue
                # w.add_image("p6" + "-" + str(i + 1), normalize(x), 0)
                # draw p6
                for j in range(-400, 400):
                    tt = z[int(round((j + 400) / 0.01))]
                    w.add_scalar("p6" + "-" + str(i + 1), tt, j)
            # w.add_histogram("p6", torch.cat(a, 0), 0)

        # ff()

        for i in reversed(range(self.depth)):
            if i != self.depth - 1:
                x *= self.rbfg[i](t[i], self.actw[i])

                def ff():
                    # v
                    p = "p" + str(i + 1) + "`"
                    # w.add_histogram(p, x, 0)
                    mean = torch.linspace(-310, 310, 63).view(1, 63, 1, 1).to("cuda")
                    for j in range(6):
                        # w.add_image(
                        #     p + "-" + str(j + 1), normalize(x[0, j].unsqueeze(0)), 0
                        # )
                        c = self.actw[i][:, j, ...]
                        for k in range(-400, 400):
                            tt = k - mean
                            tt = ((tt.pow(2) / -100 / 2).exp() * tt / -100 * c).sum()
                            w.add_scalar(p + "-" + str(j + 1), tt, k)

                # ff()
            x = self.crop(F.conv_transpose2d(x, f[i]))

            def fg():
                p = "k_" + str(i + 1)
                w.add_histogram(p, x[0], 0)

            # fg()
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
