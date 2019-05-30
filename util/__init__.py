import numpy as np
from scipy.signal import fftconvolve

import torch
from torch import nn
import torch.nn.functional as F
import random
from scipy.signal import convolve2d
from .gen_kernel import blurkernel_synthesis as gen_kernel
from .gen_dct2 import gen_dct2
from .ssim import nssim
from numpy import mean
from math import isnan
from collections import OrderedDict
from tensorboardX import SummaryWriter
from time import sleep
from pathlib import Path
from torch.utils.checkpoint import checkpoint


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# divide psf to 4 parts, and assign to corners of a zero tensor,
# size as shape, then do fft
# psf: *xhxw, shape: [H,W]
# out: *xHxW
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
    return torch.stack(
        [real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1
    )


# complex's conjugation
def conj(t, inplace=False):
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c


# real to complex, axbxc=>axbxcx2
def r2c(t):
    return torch.stack((t, torch.zeros_like(t)), -1)


# 2d fft wrapper
def irfft(t):
    return torch.irfft(t, 2, onesided=False)


def ifft(t):
    return torch.ifft(t, 2, onesided=False)


def fft(t):
    return torch.fft(t, 2, onesided=False)


def rfft(t):
    return torch.rfft(t, 2, onesided=False)


def show(x, save=None):
    import matplotlib.pyplot as plt
    dim = x.dim() if type(x) is torch.Tensor else x.ndim
    for i in range(dim - 2):
        x = x[0]
    if type(x) is torch.Tensor:
        x = x.detach().cpu().numpy()
    plt.clf()
    plt.imshow(x, interpolation="nearest", cmap="gray")
    plt.show()
    if save is not None:
        plt.savefig(save)


def pad(img, kernel, mode="reflect"):
    p = [(i - 1) // 2 for i in kernel.shape[-2:]]
    return F.pad(img, (p[-1], p[-1], p[-2], p[-2]), mode)


def crop(img, kernel):
    p = [(i - 1) // 2 for i in kernel.shape[-2:]]
    p = [-1 * i for i in p]
    return F.pad(img, (p[-1], p[-1], p[-2], p[-2]))
    # p = [(d - 1) // 2 for d in kernel.shape]
    # r = [slice(p[0], -p[0]), slice(p[1], -p[1])] + (img.ndim - 2) * [slice(None)]
    # return img[r]


def edgetaper_alpha(kernel, img_shape):
    v = []
    for i in range(2):
        z = np.fft.fft(np.sum(kernel, 1 - i), img_shape[i] - 1)
        z = np.real(np.fft.ifft(np.square(np.abs(z)))).astype(np.float32)
        z = np.concatenate([z, z[0:1]], 0)
        v.append(1 - z / np.max(z))
    return np.outer(*v)


def edgetaper(img, kernel, n_tapers=3):
    alpha = edgetaper_alpha(kernel, img.shape[0:2])
    _kernel = kernel
    if 3 == img.ndim:
        kernel = kernel[..., np.newaxis]
        alpha = alpha[..., np.newaxis]
    for i in range(n_tapers):
        blurred = fftconvolve(
            pad_for_kernel(img, _kernel, "wrap"), kernel, mode="valid"
        )
        img = alpha * img + (1 - alpha) * blurred
    return img


def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis, ..., np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img, 2, 0)[..., np.newaxis]


def from_tensor(img):
    return np.squeeze(np.moveaxis(img[..., 0], 0, -1))


def pad_for_kernel(img, kernel, mode):
    p = [(d - 1) // 2 for d in kernel.shape]
    padding = [(p[0], p[0]), (p[1], p[1])] + (img.ndim - 2) * [(0, 0)]
    return np.pad(img, padding, mode)


# def crop_for_kernel(img, kernel):
#     p = [(d - 1) // 2 for d in kernel.shape]
#     r = [slice(p[0], -p[0]), slice(p[1], -p[1])] + (img.ndim - 2) * [slice(None)]
#     return img[r]


# [...,h,w] torch/numpy
def center_crop(img, h, w=None):
    w = h if w is None else w
    hh, ww = img.shape[-2:]
    assert hh >= h and ww >= w
    if hh == h and ww == w:
        return img
    top, left = (hh - h) // 2, (ww - w) // 2
    return img[..., top : top + h, left : left + w]


# [...,h,w] torch/numpy
def rand_crop(img, h, w=None):
    w = h if w is None else w
    hh, ww = img.shape[-2:]
    assert hh >= h and ww >= w
    top, left = random.randint(0, hh - h), random.randint(0, ww - w)
    return img[..., top : top + h, left : left + w]


# [...,h,w] for torch only
def center_pad(img, h, w=None):
    w = h if w is None else w
    hh, ww = img.shape[-2:]
    assert hh <= h and ww <= w
    top, left = (hh - h) // 2, (ww - w) // 2
    bottom, right = h - top - hh, w - left - ww
    return F.pad(img, (left, right, top, bottom))


# abs mean
def log(a, name="", **arg):
    p = lambda *x: print(*x, **arg)
    if type(a) is str:
        name, a = a, name
    if type(a) is torch.Tensor:
        a = a.detach()
        if a.numel() == 1:
            p(name, f"{a.item():.4f}")
        else:
            p(
                name,
                f"{a.mean().item():.4f} {a.var().item():.4f} {a.max().item():.4f} {a.min().item():.4f}",
            )
    else:
        p(name, f"{a:.3f}")


# mean pow 2
def m2e(a, b, reduction="mean"):
    assert a.shape == b.shape and a.dim() == 4
    a = (a - b).pow(2).mean(2).mean(2).mean(2)
    return a.sum() if reduction == "sum" else a.mean()


# negative psnr, [B]xCxHxW
def npsnr(a, b, reduction="mean"):
    assert a.shape == b.shape
    l = a[0].numel()
    d = list(range(1, a.dim()))
    s = ((a - b).pow(2).sum(d) / l).log10()
    s = 10 * (s.mean() if reduction == "mean" else s.sum())
    return s


# align to get max psnr, [1]xCxHxW
def npsnr_align_max(a, b):
    assert a.shape == b.shape
    assert a.dim() == 3 or a.dim() == 4 and a.shape[0] == 1
    hh, ww = b.shape[-2:]
    h, w = a.shape[-2:]
    assert ww >= w and hh > h and (ww - w) % 2 == 0 and (hh - h) % 2 == 0
    psnr = 0
    for t in range(hh - h):
        for l in range(ww - w):
            r = b[..., t : t + h, l : l + w]
            npt = npsnr(a, r)
            p = -npt.detach().item()
            if p > psnr:
                left = l
                top = t
                psnr = p
                result = r
                npsnrt = npt
    left -= (ww - w) // 2
    top -= (hh - h) // 2
    return npsnrt, result, left, top


# load model state, DataParallel compatible, partial load
# d:path or OrdesredDict
def load(m, d):
    if type(d) is not OrderedDict:
        if not Path(d).exists():
            print("load path not exist: " + str(d))
            return
        d = torch.load(d, map_location='cpu')
    a = OrderedDict()
    s = m.state_dict()
    # cant use or operator
    for k in s:
        a[k] = d.get(k)
        if a[k] is None:
            a[k] = d.get(k[7:]) if k.startswith("module.") else d.get("module." + k)
        if a[k] is None or a[k].shape != s[k].shape:
            a[k] = s[k]

    # model specific code, load 5x5 into 7x7 center
    # f = lambda k: "filter" in k
    # for k in a:
    #     if not f(k):
    #         continue
    #     s1 = a[k].shape[-1]
    #     s2 = d[k[7:]].shape[-1]
    #     assert s1 % 2 == 1 and s2 % 2 == 1
    #     if s1 > s2:
    #         g = (s1 - s2) // 2
    #         a[k].copy_(torch.zeros_like(a[k]))
    #         a[k][..., g : g + s2, g : g + s2] = d[k[7:]]
    (m if hasattr(m, "load_state_dict") else m.module).load_state_dict(a)


def change_key(a, f):
    b = OrderedDict()
    for i in a:
        k = f(i)
        if k == None:
            continue
        b[k] = a[i]
    return b


def parameter(x, scale=1):
    if type(x) is not list:
        return nn.Parameter(x * scale)
    return nn.ParameterList([nn.Parameter(i * scale) for i in x])


# *xHxW
def augment(x):
    # counterclockwise rotate 0,90,180,270, flip up2down,left2right
    if type(x) is torch.Tensor:
        r = random.randint(0, 3)
        if r == 1:
            x = x.transpose(-2, -1).flip(-2)
        elif r == 2:
            x = x.flip(-2).flip(-1)
        elif r == 3:
            x = x.transpose(-2, -1).flip(-1)
        r = random.randint(0, 2)
        if r == 1:
            x = x.flip(-2)
        elif r == 2:
            x = x.flip(-1)
        return x
    x = np.rot90(x, random.randint(0, 3), (-2, -1))
    x = np.flip(x, random.randint(-2, -1)) if random.random() > 2 / 3 else x
    return x.copy()


def l2(g, x):
    return (g - x).pow(2).sum()


#  horizontal and vertical grad diff norm sum
def grad_diff(g, x, norm=1):
    a = g[..., 1:, :] - g[..., :-1, :]
    b = x[..., 1:, :] - x[..., :-1, :]
    c = g[..., :, 1:] - g[..., :, :-1]
    d = x[..., :, 1:] - x[..., :, :-1]
    return (a - b).norm(norm) + (c - d).norm(norm)


# min-max
def normalize(x):
    min_v = torch.min(x)
    range_v = torch.max(x) - min_v
    if range_v > 0:
        normalised = (x - min_v) / range_v
    else:
        normalised = torch.zeros(x.size())
    return normalised


def kaiming_normal(x):
    if type(x) is not list:
        return nn.init.kaiming_normal_(x)
    for i in x:
        nn.init.kaiming_normal_(i)


# repeat last element of a to length l
def repeat_last(a, l):
    a = [a] if type(a) is not list else a
    a.extend([a[-1]] * (l - len(a)))
    return a


# def checkpointor(func, flag):
#     def f(*args):
#         if flag:
#             args[0].requires_grad_(True)
#             return checkpoint(func, *args)
#         return func(*args)

#     return f
#     return lambda *args: checkpoint(func, *args) if flag else func(*args)

