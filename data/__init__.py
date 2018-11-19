import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import rgb2gray

from skimage.io import imread
from torch.utils.data import Dataset

# from torchvision.transforms import Grayscale
from util import pad, pad_for_kernel, to_tensor, edgetaper, center_crop, show

# def pad(img, kernel, mode="reflect"):
#     p = [(i - 1) // 2 for i in kernel.shape[-2:]]
#     return F.pad(img, (p[-1], p[-1], p[-2], p[-2]), mode)


class BSD(Dataset):
    # type=['train','test','val']
    def __init__(self, type="train"):
        d = Path(f"data/BSR/BSDS500/data/images/{type}").glob("**/*")
        self.d = [i for i in d if i.is_file()]
        self.k = []
        for i in range(1, 9):
            k = (
                imread(f"data/kernel/kernel{i}_groundtruth_kernel.png").astype(
                    np.float32
                )
                / 255
            )
            k = np.clip(k, 0, 1)
            k /= np.sum(k)
            self.k.append(torch.from_numpy(center_crop(k, 13)))
        self.s = 0.01 * 255

    def __getitem__(self, i):
        g = imread(self.d[i // 8])/ 255
        # g=np.clip(g,0,1)
        g = rgb2gray(g).astype(np.float32)
        # g = np.clip(g, 0, 1)
        k = self.k[i % 8]
        g = torch.from_numpy(g)
        s = torch.tensor((self.s,), dtype=torch.float)
        g = g.view(1, *g.shape)
        y = F.conv2d(g.view(1, *g.shape), k.view(1, 1, *k.shape))
        g = center_crop(g, 250)
        y = center_crop(y, 250)
        y = y.squeeze(0).permute(1, 2, 0)
        # y += torch.randn_like(y) * s / 255
        # todo convert to torch, and move to model stage
        y = to_tensor(
            edgetaper(pad_for_kernel(y.numpy(), k.numpy(), "edge"), k.numpy())
        ).astype(np.float32)
        y = torch.from_numpy(y).squeeze(-1)
        # y=y.clamp(0,1)
        # [1,250,250] [1,250,250] [1,19,19] [1]
        # g = np.clip(g, 0, 1)
        return g, y, k, s

    def __len__(self):
        return len(self.d) * 8

