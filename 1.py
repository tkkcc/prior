import numpy as np
from tensorboardX import SummaryWriter

import torch

from scipy.io import loadmat
# show actw
a = torch.load("save/g_initwithpre.tar", map_location="cpu")
w = SummaryWriter(comment='actw')
for i in range(2):
    b = a['0.actw.0']
    for j in range(-400, 400):
        x = (((x.unsqueeze(2) - self.mean).pow(2) / -200).exp() * self.actw[i]).sum(2)
        w.add_scalar('actw[0]')
exit()
# a = torch.load("save/g_initliketnrd.tar", map_location="cpu")
a = torch.load("save/g_initwithpre.tar", map_location="cpu")
# a = torch.load("save/g.tar", map_location="cpu")
exit()
mat = loadmat("/home/bilabila/code/GreedyTraining_5x5_400_180x180_sigma=25.mat")
filter_size = 5
m = filter_size ** 2 - 1
filter_num = 24
filtN = 24
for i in range(5):
    vcof = mat["cof"][:, i]
    # part4 = vcof[filtN * m + 1 :]
    # weights = np.reshape(part4, (filtN, 63))
    part3 = vcof[filtN*m]
    p=part3
    # p = np.exp(part3)
    print(p)

