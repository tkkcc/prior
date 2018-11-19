import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from scipy.fftpack import idct
from util import log_mean

class ModelStage(nn.Module):
    def __init__(self, stage=1, channel=1):
        super(ModelStage, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ELU(1),
            nn.Linear(16, 16),
            nn.ELU(1),
            nn.Linear(16, 16),
            nn.ELU(1),
            nn.Linear(16, 1),
            nn.Softplus(),
        )
        d = 6
        c = 32
        k = (5, 5)
        p = [(i - 1) // 2 for i in k]
        # top&bottom left&right
        conv_pad = (p[0], p[1])
        convElu = lambda i, o: (
            nn.Conv2d(i, o, k, padding=conv_pad),
            nn.BatchNorm2d(o),
            # nn.ELU(1),
            nn.ReLU(),
        )
        c4 = (j for i in range(d-2) for j in convElu(c, c))
        self.cnn = nn.Sequential(
            # nn.ReplicationPad2d(20),
            *convElu(channel, c),
            *c4,
            *convElu(c, channel),
            # nn.ReplicationPad2d(-20)
        )
        self.fdn = FDN(stage)
        self.stage=stage
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.constant_(m.weight, 0)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.weight, 0)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        print('stage',self.stage)
        x_in, y, k, s = inputs
        lam = self.mlp(s.pow(-2))
        x_adjustment = self.cnn(x_in)
        x_in = x_in.permute(0, 2, 3, 1)
        x_adjustment = x_adjustment.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        log_mean(' x_adjust',x_adjustment)
        log_mean(' lam',lam)
        x_out = self.fdn([x_in, x_adjustment, y, k, lam])
        return x_out.permute(0, 3, 1, 2)


def psf2otf_(psf, img_shape):
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
        dtype=psf_type,device=psf_device
    )
    zeros_top = torch.zeros(
        midH, img_shape[1] - psf_shape[1], psf_shape[2], psf_shape[3], dtype=psf_type,device=psf_device
    )
    top = torch.cat((bottom_right, zeros_bottom, bottom_left), 1)
    bottom = torch.cat((top_right, zeros_top, top_left), 1)
    zeros_mid = torch.zeros(
        img_shape[0] - psf_shape[0],
        img_shape[1],
        psf_shape[2],
        psf_shape[3],
        dtype=psf_type,device=psf_device
    )
    pre_otf = torch.cat((top, zeros_mid, bottom), 0)
    otf = rfft(pre_otf.permute(2, 3, 0, 1))
    return otf


def cm(t1, t2):
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack(
        [real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1
    )


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


class FDN(nn.Module):
    def __init__(self, stage=1, **kwargs):
        super(FDN, self).__init__()
        self.stage = stage

    def forward(self, inputs, mask=None):
        padded_inputs, adjustments, observations, blur_kernels, lambdas = inputs
        dtype = padded_inputs.dtype
        device = padded_inputs.device
        imagesize = padded_inputs.shape[1:3]
        kernelsize = blur_kernels.shape[1:3]
        padding = [i // 2 for i in kernelsize]
        mask_in = torch.ones(
            imagesize[0] - 2 * padding[0],
            imagesize[1] - 2 * padding[1],
            dtype=dtype,device = device 
        )
        mask_in = F.pad(mask_in, (padding[1], padding[1], padding[0], padding[0]))
        mask_in = mask_in.unsqueeze(0)
        k = blur_kernels.permute(1, 2, 0).unsqueeze(-1)
        k_otf = psf2otf_(k, imagesize)[:, 0, ...]
        if self.stage > 1:
            Kx_fft = cm(rfft(padded_inputs[:, :, :, 0]), k_otf)
            Kx = irfft(Kx_fft)
            Kx_outer = (1.0 - mask_in) * Kx
            y_inner = mask_in * observations[:, :, :, 0]
            y_adjusted = y_inner + Kx_outer
            dataterm_fft = cm(rfft(y_adjusted), conj(k_otf))
        else:
            observations_fft = rfft(observations[:, :, :, 0])
            dataterm_fft = cm(observations_fft, conj(k_otf))
        lambdas = lambdas.unsqueeze(-1)
        adjustment_fft = rfft(adjustments[:, :, :, 0])
        numerator_fft = cm(r2c(lambdas), dataterm_fft)
        KtK = k_otf.pow(2).sum(-1)
        # todo norm or abs
        # otf_term = adjustment_fft.pow(2).sum(-1)
        otf_term = adjustment_fft.norm(2,-1)
        log_mean(" otf_term",otf_term)
        denominator = lambdas * KtK + otf_term
        denominator = torch.stack((denominator,) * 2, -1)
        frac_fft = numerator_fft / denominator
        return irfft(frac_fft).unsqueeze(-1)


class ModelStack(nn.Module):
    def __init__(self, stage=1, weight=None):
        super(ModelStack, self).__init__()
        self.m = nn.ModuleList(ModelStage(i + 1) for i in range(stage))

    def forward(self, d):
        for i in self.m:
            d[0] = i(d)
        return d[0]
