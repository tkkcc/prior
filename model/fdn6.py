# gradient model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import idct
from util import show, log, parameter
from torch.autograd import grad


def dct_filters(filter_size=(3, 3)):
    N = filter_size[0] * filter_size[1]
    filters = np.zeros((N, N - 1), np.float32)
    for i in range(1, N):
        d = np.zeros(filter_size, np.float32)
        d.flat[i] = 1
        filters[:, i - 1] = idct(idct(d, norm="ortho").T, norm="ortho").real.flatten()
    return filters


# divide psf to 4 parts, and assign to corners of a zero tensor,
# size as shape, then do fft
# psf: BxCxhxw, shape: [H,W]
# out: BxCxHxW
def psf2otf(psf, shape):
    h, w = psf.shape[-2:]
    mh, mw = h // 2, w // 2
    otf = torch.zeros(psf.shape[:-2] + shape, dtype=psf.dtype, device=psf.device)
    otf[..., : h - mh, : w - mw] = psf[..., mh:, mw:]
    otf[..., -mh:, -mw:] = psf[..., :mh, :mw]
    otf[..., : h - mh, -mw:] = psf[..., mh:, :mw]
    otf[..., -mh:, : w - mw] = psf[..., :mh, mw:]
    otf = rfft(otf)
    return otf


# complex multiplication, axbxcx2,axbxcx2=>axbxcx2
def cm(t1, t2):
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


# complex's conjugation
def conj(t, inplace=False):
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c


# real to complex, axbxc=>axbxcx2
def r2c(t):
    return torch.stack((t, torch.zeros_like(t)), -1)


# 2d fft wrapper
# irfft: real to complex inverse fft, axbxc=>axbxcx2
# ifft: real to complex fft, axbxcx2=>axbxc
# fft: complex to complex fft, axbxcx2=>axbxcx2
# ifft: complex to complex inverse fft, axbxcx2=>axbxcx2
def irfft(t):
    return torch.irfft(t, 2, onesided=False)


def ifft(t):
    return torch.ifft(t, 2, onesided=False)


def fft(t):
    return torch.fft(t, 2, onesided=False)


def rfft(t):
    return torch.rfft(t, 2, onesided=False)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class Pow(nn.Module):
    def __init__(self, alpha):
        super(Pow, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return x.pow(self.alpha)


class ModelStage(nn.Module):
    def __init__(self, stage=1, lam=1, eta=0.1):
        assert stage > 0
        super(ModelStage, self).__init__()
        # cnn, BxCxHxW=>BxCxHxW
        # pad -> 6x(conv+elu) -> crop
        # conv: 32in32out3x3
        k = (3, 3)
        p = [(i - 1) // 2 for i in k]
        # top&bottom left&right
        conv_pad = (p[0], p[1])
        conv = lambda i, o: nn.Conv2d(i, o, k)
        convAct = lambda i, o: (conv(i, o), nn.BatchNorm2d(o), nn.ReLU())
        convActs = lambda i, o, n: (j for k in range(n) for j in convAct(i, o))
        self.cnn = nn.Sequential(
            # *convAct(1, 16),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # *convAct(16, 32),
            # *convAct(32, 32),
            # *convAct(32, 32),
            # *convActs(32, 32, 3),
            # conv(32, 32)
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
        data_norm_grad = self.lam * irfft(cm(k_otf_conj, (cm(k_otf, x_otf) - y_otf))).unsqueeze(1)
        data_norm_grad = self.lam * irfft(
            cm(cm(k_otf_conj, k_otf), x_otf) - cm(k_otf_conj, y_otf)
        ).unsqueeze(1)
        # cnn
        cnnx = self.cnn(x)
        log("cnnx^2", cnnx)
        cnnx_square = cnnx.pow(2).sum()
        cnnx_square_grad = grad(cnnx_square, x, create_graph=True)[0]
        log("data_norm_grad", data_norm_grad)
        log("cnnx_square_grad", cnnx_square_grad)
        return x - self.eta * (data_norm_grad + cnnx_square_grad)


# stack multi ModelStage
class ModelStack(nn.Module):
    def __init__(self, stage=1, weight=None):
        super(ModelStack, self).__init__()
        self.m = nn.ModuleList(ModelStage(i + 1) for i in range(stage))
        self.stage = stage

    def forward(self, d):
        for i in self.m:
            d[0] = i(d)
        return d[0]
