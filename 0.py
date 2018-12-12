from util import gen_kernel, npsnr
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.io import imsave
from data import BSD3000
from util import crop,show
import numpy as np
# plt.imshow(gen_kernel(), interpolation="nearest", cmap="gray")
# plt.show()

# a = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.3]])
# b = torch.tensor([[0.2, 0.2, 0.3], [0.4, 0.5, 0.3]])
# c = npsnr(a, b)
# print(c)

d = BSD3000()
d = d[0]
g, y, k, s = d
g = g[0]
y = crop(y[0], k)
assert y.shape==g.shape
imsave('g.png',g)
imsave('y.png',y)
# imsave('k.tiff', k)
# show(k)
np.savetxt('k.dlm', k)
print(s.item())
