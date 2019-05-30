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
from config import o, w

random.seed(o.random_seed)
np.random.seed(o.random_seed)
torch.manual_seed(o.random_seed)
w.add_text("extra", "torch cudnn reproducibility off")
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def _denoise(path, test=False, sigma=None):
    # glob return generator, list it to reuse
    path = list(path)

    class C(Dataset):
        def __init__(self):
            d = [i for i in path if i.is_file()]
            self.d = d

        def __getitem__(self, i):
            g = imread(self.d[i]) / 255
            g = rgb2gray(g).astype(np.float32)
            g = augment(rand_crop(g, o.patch_size)) if not test else g
            g = torch.from_numpy(g).view(1, *g.shape)
            s = (
                sigma or o.sigma
                if test
                else o.sigma * (random.random() if o.sigma_range else 1)
            )
            s = torch.tensor((s,), dtype=torch.float)
            y = g.clone().detach()
            y += torch.randn_like(y) * s / 255
            return g, y, s
            # [C,H,W] [C,H,W] [1]

        def __len__(self):
            return len(self.d)

    return C


_f = _denoise

# train
BSD400 = _f(Path(f"data/BSD500/").glob("t*/*"))
TNRD400 = _f(Path(f"data/FoETrainingSets180/").glob("*"))
WED4744 = _f(Path(f"data/WED4744/").glob("*"))
ILSVRC12 = _f(Path(f"data/ILSVRC12/").glob("*"))
# test
Set12 = _f(sorted(Path(f"data/Set12/").glob("*")), test=True, sigma=o.sigma_test)
BSD68 = _f(sorted(Path(f"data/BSD68/").glob("*")), test=True, sigma=o.sigma_test)
Urban100 = _f(sorted(Path(f"data/Urban100/").glob("*")), test=True, sigma=o.sigma_test)
Urban100_06 = _f(sorted(Path(f"data/Urban100/").glob("img_006.png")), test=True, sigma=o.sigma_test)
Urban100_63 = _f(sorted(Path(f"data/Urban100/").glob("img_063.png")), test=True, sigma=o.sigma_test)

BSD68_03 = _f(Path(f"data/BSD68/").glob("test003*"), test=True, sigma=o.sigma_test)
BSD68_03_ = _f(Path(f"data/BSD68/").glob("test003*"))
