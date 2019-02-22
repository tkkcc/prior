#!/usr/bin/env python
import numpy as np
from tensorboardX import SummaryWriter

import torch

from scipy.io import loadmat

w = SummaryWriter(comment="sigmoid")
# for i in range(10):

b = torch.linspace(-100, 100, 30, dtype=torch.float)
c = torch.arange(-150, 150, 1, dtype=torch.float)
for i in range(100):
    a = torch.rand(30) - 0.5
    d = (torch.sigmoid(c.view(-1, 1) - b.view(1, -1)) * a.view(1, -1)).sum(1)
    for j in range(c.numel()):
        w.add_scalar("y" + str(i), d[j], j - 150)
exit()
# show actw
a = torch.load("save/g_initwithpre.tar", map_location="cpu")
w = SummaryWriter(comment="actw")
mean = torch.linspace(-310, 310, 63).view(1, 63, 1, 1).to("cuda")
for i in range(2):
    b = a["m.0.actw." + str(i)].to("cuda")
    for j in range(b.shape[1]):
        c = b[:, j, ...]
        print(j)
        for k in range(-400, 400):
            x = (((k - mean).pow(2) / -200).exp() * c).sum()
            w.add_scalar("actw" + str(i) + "_filter" + str(j), x, k)
exit()
# show all parameters
# a = torch.load("save/g_initliketnrd.tar", map_location="cpu")
a = torch.load("save/g_initwithpre.tar", map_location="cpu")
# a = torch.load("save/g.tar", map_location="cpu")
exit()
# show tnrd's lambda
mat = loadmat("/home/bilabila/code/GreedyTraining_5x5_400_180x180_sigma=25.mat")
filter_size = 5
m = filter_size ** 2 - 1
filter_num = 24
filtN = 24
for i in range(5):
    vcof = mat["cof"][:, i]
    # part4 = vcof[filtN * m + 1 :]
    # weights = np.reshape(part4, (filtN, 63))
    part3 = vcof[filtN * m]
    p = part3
    # p = np.exp(part3)
    print(p)

