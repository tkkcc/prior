import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from scipy.fftpack import idct
from util import show


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
    otf[:, :, :mh, :mw] = psf[:, :, :mh, :mw]
    otf[:, :, :mh, -w + mw :] = psf[:, :, :mh, mw:]
    otf[:, :, -h + mh :, :mw] = psf[:, :, mh:, :mw]
    otf[:, :, -h + mh :, -w + mw :] = psf[:, :, mh:, mw:]
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


# Fourier Deconvolution Model
# parameter: filter_weights
class FDN(nn.Module):
    def __init__(self, filter_size=(5, 5), stage=1, **kwargs):
        super(FDN, self).__init__()
        self.filter_size = filter_size
        self.stage = stage
        B = dct_filters(self.filter_size)
        self.B = torch.tensor(B, dtype=torch.float)
        self.nb_filters = B.shape[1]
        self.filter_weights = nn.Parameter(torch.tensor(np.eye(self.nb_filters), dtype=torch.float))

    def forward(self, inputs, mask=None):
        # C(Channel)=1
        # Bx1xHxW,     Bx1xHxW,    Bx1xHxW,       Bxhxw       ,Bx1
        # x^t,         CNN(x^t),   y=x^0,         k,          ,mlp(lambda)=omega(lambda)
        padded_inputs, adjustments, observations, blur_kernels, lambdas = inputs
        observations = observations.squeeze(1)
        padded_inputs = padded_inputs.squeeze(1)
        adjustments = adjustments.squeeze(1)
        blur_kernels = blur_kernels.unsqueeze(1)
        imagesize = padded_inputs.shape[-2:]
        kernelsize = blur_kernels.shape[-2:]
        padding = [i // 2 for i in kernelsize]
        # internal mask for boundary adjust
        mask_int = torch.ones(
            imagesize[0] - 2 * padding[0], imagesize[1] - 2 * padding[1], dtype=torch.float32
        )
        mask_int = F.pad(mask_int, (padding[1], padding[1], padding[0], padding[0]))
        mask_int = mask_int.unsqueeze(0)
        if self.filter_weights.device != self.B.device:
            self.B = self.B.to(self.filter_weights.device)
        filters = self.B.mm(self.filter_weights)
        filters = filters.permute(1, 0).view(1, self.nb_filters, *self.filter_size)
        # filters_otfs: 1x24xHxWx2
        filter_otfs = psf2otf(filters, imagesize)
        # otf_term: sum(|F(f)|^2) 1xHxW
        otf_term = filter_otfs.pow(2).sum(-1).sum(1)
        # k_otf: BxHxWx2
        k_otf = psf2otf(blur_kernels, imagesize).squeeze(1)

        # boundary adjust
        # dataterm_fft: F(phi) x conj(F(k))
        if self.stage > 1:
            # use y's inner and (k conv x)'s outer
            Kx_fft = cm(rfft(padded_inputs), k_otf)
            Kx = irfft(Kx_fft)
            Kx_outer = (1.0 - mask_int) * Kx
            y_inner = mask_int * observations
            y_adjusted = y_inner + Kx_outer
            dataterm_fft = cm(rfft(y_adjusted), conj(k_otf))
        else:
            # use edgetaping
            observations_fft = rfft(observations)
            dataterm_fft = cm(observations_fft, conj(k_otf))
        lambdas = lambdas.unsqueeze(-1)

        # numerator_fft: omega x dataterm + CNN(x^t)
        adjustment_fft = rfft(adjustments)
        numerator_fft = cm(r2c(lambdas), dataterm_fft) + adjustment_fft
        # KtK: |F(k)|^2 BxHxW
        KtK = k_otf.pow(2).sum(-1)
        # denominator_fft: omega x |F(k)|^2 + sum(|F(f)|^2)
        denominator_fft = lambdas * KtK + otf_term
        # numerator/denominator, complex/real, BxHxWx2/BxHxWx2=>BxHxWx2
        t = torch.stack((denominator_fft,) * 2, -1)
        frac_fft = numerator_fft / t
        return irfft(frac_fft).unsqueeze(1)


# one stage
# parameter: mlp, cnn, fdn
class ModelStage(nn.Module):
    def __init__(self, stage=1, channel=1):
        assert stage > 0
        super(ModelStage, self).__init__()
        # mlp, Bx1=>Bx1
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ELU(1),
            nn.Linear(16, 16),
            nn.ELU(1),
            nn.Linear(16, 16),
            nn.ELU(1),
            nn.Linear(16, 1),
            nn.Softplus(beta=1, threshold=1),
        )

        # cnn, BxCxHxW=>BxCxHxW
        # pad -> 6x(conv+elu) -> crop
        # conv: 32in32out3x3
        k = (3, 3)
        p = [(i - 1) // 2 for i in k]
        # top&bottom left&right
        conv_pad = (p[0], p[1])
        convElu = lambda i, o: (nn.Conv2d(i, o, k, padding=conv_pad), nn.ELU(1))
        c4 = (j for i in range(4) for j in convElu(32, 32))
        self.cnn = nn.Sequential(
            nn.ReplicationPad2d(20),
            *convElu(channel, 32),
            *c4,
            *convElu(32, channel),
            nn.ReplicationPad2d(-20)
        )

        # fdn, filter 5x5
        self.fdn = FDN((5, 5), stage)
        # use (0,1)uniform random init

    def forward(self, inputs):
        # Bx1xHxW, BxHxWx1, BxHxW, Bx1
        x_in, y, k, s = inputs
        lam = self.mlp(s.pow(-2))
        x_adjustment = self.cnn(x_in)
        x_out = self.fdn([x_in, x_adjustment, y, k, lam])
        return x_out


# stack multi ModelStage
class ModelStack(nn.Module):
    def __init__(self, stage=1, weight=None):
        super(ModelStack, self).__init__()
        self.m = nn.ModuleList(ModelStage(i + 1) for i in range(stage))
        self.stage=stage

    def forward(self, d):
        for i in self.m:
            d[0] = i(d)
        return d[0]
