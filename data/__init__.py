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

random.seed(o.random_seed)
np.random.seed(o.random_seed)
torch.manual_seed(o.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _denoise(path, test=False, sigma=None):
    class C(Dataset):
        def __init__(self):
            d = path
            d = Path(d) if type(d) is str else d
            d= [i for i in d if i.is_file()]
            
            # d = [imread(i) / 255 for i in d if i.is_file()]
            # d = [rgb2gray(i).astype(np.float32) for i in d]
            self.d = d

        def __getitem__(self, i):
            # g = imread(random.choice(self.d)) / 255
            g = imread(self.d[i]) / 255
            g = rgb2gray(g).astype(np.float32)
            # g = random.choice(self.d)
            g = augment(rand_crop(g, o.patch_size)) if not test else g
            g = torch.from_numpy(g).view(1, *g.shape)
            s = sigma or o.sigma if test else o.sigma * (random.random() if o.sigma_range else 1)
            s = torch.tensor((s,), dtype=torch.float)
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
ILSVRC12 = _f(Path(f"data/ILSVRC12/").glob("*"))
# test
TNRD68 = _f(sorted(Path(f"data/68imgs/").glob("*")), test=True, sigma=25)
TNRD68_03 = _f(Path(f"data/68imgs/").glob("test003*"), test=True, sigma=25)
Set12 = _f(sorted(Path(f"data/Set12/").glob("*")), test=True, sigma=25)
BSD68 = _f(sorted(Path(f"data/BSD68/").glob("*")), test=True, sigma=25)
