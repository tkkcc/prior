# import scipy
# import scipy.io
# read Levin09blurdata
# import numpy as np
# import pathlib
# import matplotlib.pyplot as plt
# # import scipy.misc.imshow
# from os import environ
# environ['SCIPY_PIL_IMAGE_VIEWER'] = 'feh'
# # for i in pathlib.Path('Levin09blurdata').iterdir():
# mat = scipy.io.loadmat('data/Levin09blurdata/im05_flit03.mat')
# plt.imshow(mat['y'],cmap='gray')
# plt.show()
# print(mat)

from torch.utils.data import DataLoader,Subset
from util import show
from data import BSD
d = BSD()
d = DataLoader(d, 2, num_workers=0)
for i in d:
  g, y, k, s = i
  # show(g[0,0,...])
  # show(y[0,0,...])
  print()
exit(0)
g,y,k,s = d[0]
print(g.shape, y.shape, k.shape, s.shape)
print