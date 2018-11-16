import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import fftconvolve

from skimage.io import imread
from torch.utils.data import Dataset

# from util import pad
def pad(img, kernel, mode="reflect"):
    p = [(i - 1) // 2 for i in kernel.shape[-2:]]
    return F.pad(img, (p[-1], p[-1], p[-2], p[-2]), mode)


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
    padding = [p, p] + (img.ndim - 2) * [(0, 0)]
    return np.pad(img, padding, mode)


def crop_for_kernel(img, kernel):
    p = [(d - 1) // 2 for d in kernel.shape]
    r = [slice(p[0], -p[0]), slice(p[1], -p[1])] + (img.ndim - 2) * [slice(None)]
    return img[r]


class BSD(Dataset):
    # type=['train','test','val']
    def __init__(self, type="train"):
        d = Path(f"data/BSR/BSDS500/data/images/{type}").glob("**/*")
        self.d = [i for i in d if i.is_file()]
        k = np.loadtxt("data/example1.dlm").astype(np.float32)
        k = np.clip(k, 0, 1)
        k /= np.sum(k)
        # k = torch.from_numpy(k)
        self.k = k
        self.s = 255 * 0.01

    def __getitem__(self, i):
        g = imread(self.d[i]).astype(np.float32) / 255
        k = self.k
        y = to_tensor(edgetaper(pad_for_kernel(g, k, "edge"), k)).astype(np.float32)
        g = torch.from_numpy(g).permute(-1, 0, 1)[0:1, 50:300, 50:300]
        y= torch.from_numpy(y)
        # k = torch.stack((self.k,) * g.shape[0], 0)
        k = torch.from_numpy(k)
        s = torch.tensor((self.s,), dtype=torch.float)
        y = pad(g.view(1, *g.shape), k.view(1, 1, *k.shape))
        y = F.conv2d(y, k.view(1, 1, *k.shape))
        y = y.squeeze(0)
        y += torch.randn_like(y) * s
        # [1,250,250] [1,250,250] [1,19,19] [1]
        return g, y, k, s

    def __len__(self):
        return len(self.d)


if __name__ == "__main__":
    d = BSD()
    print(d[0])
