# sgd self model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from util import show, log, parameter, irfft, ifft, fft, rfft, cm, conj, psf2otf


class ModelStage(nn.Module):
    def __init__(self, stage=1, lam=1, eta=0.1):
        super(ModelStage, self).__init__()
        k = (3, 3)
        p = [(i - 1) // 2 for i in k]
        conv_pad = (p[0], p[1])
        conv = lambda i, o: nn.Conv2d(i, o, k)
        convAct = lambda i, o: (conv(i, o), nn.BatchNorm2d(o), nn.ReLU())
        convActs = lambda i, o, n: (j for k in range(n) for j in convAct(i, o))
        self.cnn = nn.Sequential(
            *convAct(1, 16),
            nn.MaxPool2d(2),
            *convAct(16, 16),
            nn.MaxPool2d(2),
            *convAct(16, 16),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            conv(16, 16),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            conv(16, 1),
        )
        self.lam = lam
        self.eta = eta

    def forward(self, inputs):
        # Bx1xHxW, Bx1xHxW, BxHxW, Bx1
        x, y, k, s = inputs
        p = [i // 2 for i in k.shape[-2:]]
        y_otf = rfft(y.squeeze(1))
        x_otf = rfft(x.squeeze(1))
        k_otf = psf2otf(k, x.shape[-2:])
        k_otf_conj = conj(k_otf)
        data_norm_grad = self.lam * irfft(
            cm(k_otf_conj, rfft(irfft(cm(k_otf, x_otf)) - y.squeeze(1)))
        ).unsqueeze(1)
        cnnx = self.cnn(x)
        log("cnnx^2", cnnx)
        cnnx_square = cnnx.pow(2).sum()
        cnnx_square_grad = grad(cnnx_square, x, create_graph=True)[0]
        log("data_norm_grad", data_norm_grad)
        log("cnnx_square_grad", cnnx_square_grad)
        return x - self.eta * (data_norm_grad + cnnx_square_grad)


class ModelStack(nn.Module):
    def __init__(self, stage=1, weight=None):
        super(ModelStack, self).__init__()
        self.m = nn.ModuleList(ModelStage(i + 1) for i in range(stage))
        self.stage = stage

    def forward(self, d):
        for i in self.m:
            d[0] = i(d)
        return d[0]
