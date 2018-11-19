import torch.nn.functional as F
from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
import torch

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def show(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation="nearest", cmap="gray")
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


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
    padding = [p, p] + (img.ndim - 2) * [(0, 0)]
    return np.pad(img, padding, mode)


def crop_for_kernel(img, kernel):
    p = [(d - 1) // 2 for d in kernel.shape]
    r = [slice(p[0], -p[0]), slice(p[1], -p[1])] + (img.ndim - 2) * [slice(None)]
    return img[r]


# [...,h,w]
def center_crop(img, h, w=None):
    if w == None:
        w = h
    hh, ww = img.shape[-2:]
    if hh < h or ww < w:
        raise ValueError("size")
    top, left = (hh - h) // 2, (ww - w) // 2
    return img[..., top : top + h, left : left + w]
def log_mean(name, var):
    if type(var) is torch.Tensor:
        print(name,var.detach().mean().item())
    else:
        print(name,var)

# [...,h,w] todo
# def center_pad(img, h, w=None):
#     w = h if w is None else w
#     hh, ww = img.shape[-2:]
#     if hh > h or ww > w:
#         raise ValueError("size")
#     top, left = (hh - h) // 2, (ww - w) // 2
#     return img[..., top : top + h, left : left + w]
