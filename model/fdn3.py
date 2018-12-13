import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from scipy.fftpack import idct

from util import log


class ModelStage(nn.Module):
    def __init__(self, stage=1, channel=1, eta=1, lam=1):
        if stage < 1:
            raise ValueError("stage<1")
        super(ModelStage, self).__init__()
        d = 6
        c = 16
        k = (3, 3)
        p = [(i - 1) // 2 for i in k]
        # top&bottom left&right
        conv_pad = (p[0], p[1])
        cbl = lambda i, o: (
            nn.Conv2d(i, o, k, padding=conv_pad),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
        )
        c4 = (j for i in range(d - 2) for j in cbl(c, c))
        self.cnn = nn.Sequential(
            nn.ReplicationPad2d(20),
            *cbl(channel, c),
            *c4,
            *cbl(c, channel),
            nn.ReplicationPad2d(-20)
        )
        self.eta = eta
        self.lam = lam

    def forward(self, inputs):
        if self.eta > 1:
            self.eta *= 0.98
        log("eta", self.eta)
        # Bx1xHxW, Bx1xHxW, BxHxW, Bx1
        x, y, k, s = inputs
        k = k.permute(1, 2, 0).unsqueeze(-1)
        k_otf = psf2otf(k, x.shape[-2:])[:, 0, ...]
        k_otf_conj = conj(k_otf)
        t1 = cm(cm(k_otf_conj, k_otf), rfft(x[:, 0, :, :])) - cm(k_otf_conj, rfft(y[:, 0, :, :]))
        t1 = self.lam * irfft(t1)
        t1 = t1.unsqueeze(1)
        t2 = self.cnn(x)
        log("t2", t2)
        out = x - self.eta * (t1 + t2)
        return out


def psf2otf(psf, img_shape):
    psf_shape = psf.shape
    psf_type = psf.dtype
    psf_device = psf.device
    midH = psf_shape[0] // 2
    midW = psf_shape[1] // 2
    top_left = psf[:midH, :midW, :, :]
    top_right = psf[:midH, midW:, :, :]
    bottom_left = psf[midH:, :midW, :, :]
    bottom_right = psf[midH:, midW:, :, :]
    zeros_bottom = torch.zeros(
        psf_shape[0] - midH,
        img_shape[1] - psf_shape[1],
        psf_shape[2],
        psf_shape[3],
        dtype=psf_type,
        device=psf_device,
    )
    zeros_top = torch.zeros(
        midH,
        img_shape[1] - psf_shape[1],
        psf_shape[2],
        psf_shape[3],
        dtype=psf_type,
        device=psf_device,
    )
    top = torch.cat((bottom_right, zeros_bottom, bottom_left), 1)
    bottom = torch.cat((top_right, zeros_top, top_left), 1)
    zeros_mid = torch.zeros(
        img_shape[0] - psf_shape[0],
        img_shape[1],
        psf_shape[2],
        psf_shape[3],
        dtype=psf_type,
        device=psf_device,
    )
    pre_otf = torch.cat((top, zeros_mid, bottom), 0)
    otf = rfft(pre_otf.permute(2, 3, 0, 1))
    return otf


def cm(t1, t2):
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


def conj(t, inplace=False):
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c


def r2c(t):
    return torch.stack((t, torch.zeros_like(t)), -1)


def irfft(t):
    return torch.irfft(t, 2, onesided=False)


def ifft(t):
    return torch.ifft(t, 2, onesided=False)


def fft(t):
    return torch.fft(t, 2, onesided=False)


def rfft(t):
    return torch.rfft(t, 2, onesided=False)


class ModelStack(nn.Module):
    def __init__(self, stage=1, weight=None):
        super(ModelStack, self).__init__()
        self.m = nn.ModuleList(ModelStage(i + 1) for i in range(stage))
        # self.m = nn.ModuleList(ModelStage(1) for i in range(stage))

    def forward(self, d):
        for i in self.m:
            d[0] = i(d)
        return d[0]
