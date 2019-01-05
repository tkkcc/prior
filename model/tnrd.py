# tnrd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import show, log, parameter, gen_dct2
from scipy.io import loadmat


def loadTNRD():
    from scipy.io import loadmat
    import numpy as np
    from util import gen_dct2

    mat = loadmat("/home/bilabila/code/code/GreedyTraining_5x5_400_180x180_sigma=25.mat")

    filter_size = 5
    m = filter_size ** 2 - 1
    filter_num = 24
    basis = gen_dct2(filter_size)
    filtN = 24

    s = 0
    vcof = mat["cof"][:, s]
    mfs = mat["MFS"]
    part1 = vcof[: filtN * m]
    cof_beta = np.reshape(part1, (filtN, m)).T
    part3 = vcof[filtN * m]
    p = np.exp(part3)
    part4 = vcof[filtN * m + 1 :]
    weights = np.reshape(part4, (filtN, 63)).T

    K = np.empty((filtN, filter_size, filter_size))
    for i in range(filtN):
        x_cof = cof_beta[:, i]
        filters = basis @ x_cof
        filters = filters / np.sum(filters ** 2) ** (1 / 2)
        K[i] = np.reshape(filters, (filter_size, filter_size)).T
    # 24x5x5 63x24 1
    return K, weights, np.log(p)


class ModelStage(nn.Module):
    def __init__(self, stage=1, lam=1.0):
        super(ModelStage, self).__init__()
        penalty_num = 63
        self.channel = channel = filter_num = 24
        self.filter_size = filter_size = 5
        self.basis = torch.tensor(gen_dct2(filter_size), dtype=torch.float)
        self.lam = torch.tensor(0 if stage == 1 else log(0.1), dtype=torch.float)
        # self.mean = torch.linspace(-310, 310, penalty_num).view(1, 1, penalty_num, 1, 1)
        # self.weight = torch.randn(1, filter_num, penalty_num, 1, 1) * 0.1
        mat = loadmat("data/w0_63_means.mat")
        self.mean = torch.tensor(mat["means"], dtype=torch.float).view(1, 1, penalty_num, 1, 1)

        # activation weight 1x24x63x1x1
        self.actw = torch.tensor(mat["w"], dtype=torch.float).view(1, 1, penalty_num, 1, 1)
        self.actw *= 10 if stage == 1 else 5 if stage == 2 else 1
        self.actw = self.actw.repeat(1, filter_num, 1, 1, 1)
        self.actw = [self.actw.clone().detach(), self.actw]
        # conv filter and bias
        self.filter = [
            torch.randn(channel, 1, filter_size, filter_size),
            torch.randn(channel, channel, filter_size, filter_size),
        ]
        # self.filter = [
        #     torch.eye(filter_size ** 2 - 1, channel * 1),
        #     torch.eye(filter_size ** 2 - 1, channel * channel),
        # ]
        self.bias = [torch.randn(channel), torch.randn(channel)]

        self.pad = nn.ReplicationPad2d(filter_size // 2)
        self.crop = nn.ReplicationPad2d(-(filter_size // 2))
        # load test
        filters, weights, p = loadTNRD()
        self.lam = torch.tensor(p, dtype=torch.float)
        self.filter = [
            torch.tensor(filters, dtype=torch.float).view(channel, 1, filter_size, filter_size)
        ]
        self.actw = [
            torch.tensor(1, dtype=torch.float),
            torch.tensor(weights, dtype=torch.float)
            .permute(1, 0)
            .view(1, filter_num, penalty_num, 1, 1),
        ]
        self.lam = parameter(self.lam)
        self.bias = parameter(self.bias)
        self.filter = parameter(self.filter)
        self.actw = parameter(self.actw)

    # Bx1xHxW
    def forward(self, inputs):
        # y=x^0
        x, y, lam = inputs
        # x *= 255
        # y *= 255
        xx = x
        self.mean = self.mean.to(x.device)
        self.basis = self.basis.to(x.device)
        # f = []
        # for i in self.filter:
        #     t = (
        #         (self.basis @ i)
        #         .permute(1, 0)
        #         .view(self.channel, -1, self.filter_size, self.filter_size)
        #     )
        #     f.append(t / t.norm(2))
        f = [
            i / i.view(self.channel, -1).norm(dim=1).view(self.channel, 1, 1, 1)
            for i in self.filter
        ]
        x = c1 = F.conv2d(self.pad(x), f[0])

        # x = (((x.unsqueeze(2) - self.mean).pow(2) / -200).exp() * self.actw[0]).sum(2)
        # x = F.conv2d(self.pad(x), f[1])

        x = (
            ((x.unsqueeze(2) - self.mean / 255).pow(2) / -(200 / 255 ** 2)).exp() * self.actw[1]
        ).sum(2)

        # x = self.crop(F.conv_transpose2d(x, f[1]))
        # c1=c1.unsqueeze(2)
        # x = x * (
        #     ((c1 - self.mean).pow(2) / -200).exp() * (c1 - self.mean) / -100 * self.actw[0]
        # ).sum(2)

        x = self.pad(x)
        x = F.conv2d(x, f[0].permute(1, 0, 2, 3).flip(2, 3))
        # x = self.crop(F.conv_transpose2d(x, f[0]))
        return xx - (x + self.lam.exp() * (xx - y))


class ModelStack(nn.Module):
    def __init__(self, stage=1, weight=None):
        super(ModelStack, self).__init__()
        self.m = nn.ModuleList(ModelStage(i + 1) for i in range(stage))
        pad_width = self.m[0].filter_size + 1
        self.pad = nn.ReplicationPad2d(pad_width)
        self.crop = nn.ReplicationPad2d(-pad_width)
        self.stage = stage

    def forward(self, d):
        # tnrd pad and crop
        # x^t, y=x^0, s
        d[1] = self.pad(d[1])
        for i in self.m:
            d[0] = self.pad(d[0])
            d[0] = i(d)
            d[0] = self.crop(d[0])
        return d[0]
