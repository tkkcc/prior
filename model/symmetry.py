# symmetry model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import show, log, parameter, irfft, ifft, fft, rfft, cm, conj, psf2otf


class ModelStage(nn.Module):
    def __init__(self, stage=1, lam=1.0):
        super(ModelStage, self).__init__()
        self.lam = 0.1
        self.filter_size = filter_size = (5, 5)
        self.depth = depth = 6
        in_channel = 1
        med_channel = 48
        out_channel = 1
        parameter_scale = 0.1
        self.w = [
            torch.randn(med_channel, in_channel, *filter_size),
            *(torch.randn(med_channel, med_channel, *filter_size) for i in range(depth - 2)),
            torch.randn(out_channel, med_channel, *filter_size),
        ]
        self.b = [
            torch.randn(med_channel),
            *(torch.randn(med_channel) for i in range(depth - 2)),
            torch.randn(out_channel),
        ]
        self.w = parameter(self.w, parameter_scale)
        self.b = parameter(self.b, parameter_scale)

    def forward(self, inputs):
        # Bx1xHxW, Bx1xHxW, BxHxW, Bx1
        x, y, k, s = inputs
        p = [i // 2 for i in k.shape[-2:]]
        c = x
        t = []
        # data
        y_otf = rfft(y.squeeze(1))
        x_otf = rfft(x.squeeze(1))
        k_otf = psf2otf(k, x.shape[-2:])
        k_otf_conj = conj(k_otf)
        # data_norm_grad = self.lam * irfft(cm(k_otf_conj, (cm(k_otf, x_otf) - y_otf))).unsqueeze(1)
        # data_norm_grad = self.lam * irfft(
        #     cm(cm(k_otf_conj, k_otf), x_otf) - cm(k_otf_conj, y_otf)
        # ).unsqueeze(1)
        data_norm_grad = self.lam * irfft(
            cm(k_otf_conj, rfft(irfft(cm(k_otf, x_otf)) - y.squeeze(1)))
        ).unsqueeze(1)
        # k = k.unsqueeze(1)
        # data_norm_grad = self.lam * F.conv_transpose2d(F.conv2d(x, k, padding=p) - y, k, padding=p)
        log("x", x)
        p = [(i - 1) // 2 for i in self.filter_size]
        for i in range(self.depth):
            c = F.conv2d(c, self.w[i], self.b[i], padding=p)
            if i < self.depth - 1:
                c = c.sigmoid()
                t.append(c)
        log("cnnx", c)
        for i in reversed(range(self.depth)):
            if i < self.depth - 1:
                c = c * (t[i] * (1 - t[i]))
            c = F.conv_transpose2d(c, self.w[i], bias=None, padding=p)
        cnnx_square_grad = 2 * c
        log("data_norm_grad", data_norm_grad)
        log("cnnx_square_grad", cnnx_square_grad)
        log("lam", self.lam)
        return x - (data_norm_grad + cnnx_square_grad)


class ModelStack(nn.Module):
    def __init__(self, stage=1, weight=None):
        super(ModelStack, self).__init__()
        self.m = nn.ModuleList(ModelStage(i + 1) for i in range(stage))
        self.stage = stage

    def forward(self, d):
        for i in self.m:
            d[0] = i(d)
        return d[0]
