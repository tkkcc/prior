import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from skimage.color import rgb2gray
from skimage.io import imread
from torch.utils.data import Dataset

from util import rand_crop, show, augment
from config import o


def _denoise(path, test=False):
    class C(Dataset):
        def __init__(self):
            d = path
            d = Path(d) if type(d) is str else d
            self.d = [i for i in d if i.is_file()]
            self.gs = o.patch_size
            self.s = o.sigma

        def __getitem__(self, i):
            # g = imread(random.choice(self.d)) / 255
            g = imread(self.d[i]) / 255
            g = rgb2gray(g).astype(np.float32)
            g = augment(rand_crop(g, self.gs)) if not test else g
            g = torch.from_numpy(g).view(1, *g.shape)
            s = torch.tensor((self.s,), dtype=torch.float)
            y = g.clone().detach()
            y += torch.randn_like(y) * s / 255
            return g, y, s
            # [1,180,180] [1,180,180] [1]

        def __len__(self):
            return len(self.d)

    return C


_f = _denoise

# train
BSD400 = _f(Path(f"data/BSR/BSDS500/data/images/").glob("t*/*"))
TNRD400 = _f(Path(f"data/FoETrainingSets180/").glob("*"))
WED4744 = _f(Path(f"data/pristine_images/").glob("*"))
# test
TNRD68 = _f(sorted(Path(f"data/68imgs/").glob("*")), test=True)
TNRD68_03 = _f(Path(f"data/68imgs/").glob("test003*"), test=True)

