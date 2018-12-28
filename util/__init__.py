import torch.nn.functional as F
from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import random
from scipy.signal import convolve2d
from .gen_kernel import blurkernel_synthesis as gen_kernel

from numpy import mean
from math import isnan
from collections import OrderedDict


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def show(x, save=None):
    for i in range(x.dim() - 2):
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
        blurred = fftconvolve(pad_for_kernel(img, _kernel, "wrap"), kernel, mode="valid")
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
def log(a, name=""):
    if type(a) is str:
        name, a = a, name
    if type(a) is torch.Tensor:
        a = a.detach()
        print(
            name,
            f"{a.mean().item():.4f} {a.var().item():.4f} {a.max().item():.4f} {a.min().item():.4f}",
        )
    else:
        print(name, f"{a:.3f}")


# negative psnr, [B]xCxHxW
def npsnr(a, b):
    assert a.shape == b.shape
    t = torch.tensor(10, dtype=a.dtype, device=a.device)
    if a.dim() == 3:
        return F.mse_loss(a, b).log10() * t
    # mean(dim=list) polyfill for 0.4
    l = a[0].numel()
    d = list(range(1, a.dim()))
    s = ((a - b).pow(2).sum(d) / l).log10().mean() * t
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
        d = torch.load(d)
    a = OrderedDict()
    s = m.state_dict()
    for k in s:
        a[k] = d[k] if k in d else d[k[7:]] if k.startswith("module.") else d["module." + k]
    (m if hasattr(m, "load_state_dict") else m.module).load_state_dict(a)


def change_key(a, f):
    b = OrderedDict()
    for i in a:
        b[f(i)] = a[i]
    return b


def parameter(x, scale=1):
    return nn.ParameterList([nn.Parameter(i * scale) for i in x])
