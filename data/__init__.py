import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from skimage.io import imread
from torch.utils.data import Dataset

def pad(img, kernel, mode="reflect"):
    p = [(i - 1) // 2 for i in kernel.shape[-2:]]
    return F.pad(img, (p[-1], p[-1], p[-2], p[-2]), mode)

class BSD(Dataset):
    # type=['train','test','val']
    def __init__(self, type="train"):
        d = Path(f"data/BSR/BSDS500/data/images/{type}").glob("**/*")
        self.d = [i for i in d if i.is_file()]

    def __getitem__(self, i):
        g = imread(self.d[i]).astype(np.float32) / 255
        g = torch.from_numpy(g).view(1, -1, *g.shape[:2])
        k = np.loadtxt("data/example1.dlm").astype(np.float32)
        k = torch.from_numpy(k).view(1, -1, *k.shape[:2])
        k= torch.cat((k,)*g.shape[1],1)
        s = 255 * 0.01
        y = pad(g, k)
        y = F.conv2d(y, k)
        y += torch.randn_like(y) * s
        return g, k, s, y

    def __len__(self):
        return len(self.d)


if __name__ == "__main__":
    d = BSD()
    print(d[0])