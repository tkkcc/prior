# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import idct
from util import show, log, parameter
from torch.autograd import grad


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)


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


class ModelStage(nn.Module):
    def __init__(self, stage=1, lam=1, eta=0.1):
        super(ModelStage, self).__init__()
        self.cnn = UNet(1, 1)
        self.lam = 1
        self.eta = 0.01

    def forward(self, inputs):
        # Bx1xHxW, Bx1xHxW, BxHxW, Bx1
        x, y, k, s = inputs
        p = [i // 2 for i in k.shape[-2:]]
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
        # cnn
        cnnx = self.cnn(x)
        cnnx_square = cnnx.pow(2).sum()
        cnnx_square_grad = grad(cnnx_square, x, create_graph=True)[0]
        log("cnnx^2", cnnx)
        log("data_grad", data_norm_grad)
        log("cnn_grad", cnnx_square_grad)
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
