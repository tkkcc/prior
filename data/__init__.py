import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from skimage.color import rgb2gray
from skimage.io import imread
from torch.utils.data import Dataset

from util import (
    center_crop,
    center_pad,
    edgetaper,
    gen_kernel,
    pad,
    pad_for_kernel,
    rand_crop,
    show,
    to_tensor,
)

# return shape [1,284,284] [1,320,320] [37,37] [1]


class Sun(Dataset):
    def __init__(self):
        self.d = list(Path(f"data/input80imgs8kernels").glob("*_blurred.png"))

    def __getitem__(self, i):
        i = self.d[i]
        print(i)
        p = str(i.name)
        g, k = p.split("_")[:2]
        g = imread(i.parent / f"img{g}_groundtruth_img.png")
        k = imread(i.parent / f"kernel{k}_groundtruth_kernel.png")
        y = imread(i)
        [g, k, y] = [i.astype(np.float32) / 255 for i in [g, k, y]]
        # k = k[::-1, ::-1]
        k = np.clip(k, 0, 1)
        k /= np.sum(k)
        y = to_tensor(edgetaper(pad_for_kernel(y, k, "edge"), k)).astype(np.float32)
        g = torch.from_numpy(g).unsqueeze(0)
        y = torch.from_numpy(y).squeeze(-1)
        k = torch.from_numpy(k)
        s = torch.tensor((2.55,), dtype=torch.float)
        return g, y, k, s

    __len__ = lambda self: len(self.d)


class Levin(Dataset):
    def __init__(self):
        self.d = list(Path(f"data/Levin09blurdata").iterdir())

    def __getitem__(self, i):
        print(self.d[i])
        mat = loadmat(self.d[i])
        g = mat["x"].astype(np.float32)
        y = mat["y"].astype(np.float32)
        k = mat["f"].astype(np.float32)
        # flip kernel
        k = k[::-1, ::-1]
        k = np.clip(k, 0, 1)
        k /= np.sum(k)
        y = to_tensor(edgetaper(pad_for_kernel(y, k, "edge"), k)).astype(np.float32)
        g = torch.from_numpy(g).unsqueeze(0)
        y = torch.from_numpy(y).squeeze(-1)
        k = torch.from_numpy(k)
        s = torch.tensor((1.5,), dtype=torch.float)
        return g, y, k, s

    __len__ = lambda self: len(self.d)


# class BSD3000(Dataset):
#     def __init__(self, total=3000):
#         d = Path(f"data/BSR/BSDS500/data/images/").glob("t*/*")
#         self.d = [i for i in d if i.is_file()]
#         self.gs = 180
#         self.s = 25
#         self.total = total
#         random.seed(0)

#     def __getitem__(self, i):
#         g = imread(random.choice(self.d)) / 255
#         g = rgb2gray(g).astype(np.float32)
#         g = rand_crop(g, self.gs)
#         g = torch.from_numpy(g).view(1, *g.shape)
#         s = torch.tensor((self.s,), dtype=torch.float)
#         y = torch.tensor(g)
#         y += torch.randn_like(y) * s / 255
#         return g, y, s, s

#     def __len__(self):
#         return self.total


# #deblur
class BSD3000(Dataset):
    def __init__(self, total=100,noise=True,edgetaper=True):
        d = Path(f"data/BSR/BSDS500/data/images/").glob("t*/*")
        self.d = [i for i in d if i.is_file()]
        self.gs = 180
        self.ks = 31
        self.total = total
        self.noise=noise
        self.edgetaper=edgetaper
        random.seed(0)

    def __getitem__(self, i):
        g = imread(random.choice(self.d)) / 255
        g = rgb2gray(g).astype(np.float32)
        g = rand_crop(g, self.gs + self.ks - 1)
        g = torch.from_numpy(g).view(1, *g.shape)

        k = gen_kernel(self.ks).astype(np.float32)
        k = torch.from_numpy(k)
        s = random.uniform(1, 3)
        s = torch.tensor((s,), dtype=torch.float)

        y = F.conv2d(g.view(1, *g.shape), k.view(1, 1, *k.shape)).squeeze(0)
        assert y.shape[-1] == self.gs
        # noise
        if self.noise:
            y += torch.randn_like(y) * s / 255
        # show(y[0])
        # edgetaping, todo convert to torch, and move to model stage
        if self.edgetaper:
            y = y.permute(1, 2, 0)
            y = to_tensor(edgetaper(pad_for_kernel(y.numpy(), k.numpy(), "edge"), k.numpy())).astype(
                np.float32
            )
            y = torch.from_numpy(y).squeeze(-1)
        g = center_crop(g, self.gs)
        # [1,284,284] [1,320,320] [37,37] [1]
        return g, y, k, s

    def __len__(self):
        return self.total
