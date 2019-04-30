# fit rbf by 3 convs sequential
import json
import time

import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm, trange

from config import o, w
from data import BSD68_03

o.model = "tnrdcs"
from model import Model
from util import change_key, isnan, load, mean, normalize, npsnr, nssim, show, sleep

# replace all rbf by 3 convs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.checkpoint import checkpoint

from config import o
from util import kaiming_normal, parameter


# def v():
#     from importlib import reload

#     o.model = "tnrdcsc"
#     import model

#     reload(model)
#     m2 = DataParallel(model.Model([1])).to(o.device)
#     a = torch.load("save/g1_csc0.tar")
#     load(m2, a)
#     b = m2.module
#     c = b.m[0].a
#     assert not b["m.0.a.33.c2.0.weight"].equal(b["m.0.a.3.c2.0.weight"])

#     i = 0
#     j = len(c) - 1
#     while i < j:
#         if type(c[i]) != model.tnrdcsc.Rbf:
#             i += 1
#         if type(c[j]) != model.tnrdcsc.Rbf:
#             j -= 1
#         if type(c[i]) == model.tnrdcsc.Rbf and type(c[i]) == model.tnrdcsc.Rbf:
#             print(i, j)
#             c[j] = c[i]
#             i += 1
#             j -= 1
#     b = m2.module.state_dict()
#     assert b["m.0.a.33.c2.0.weight"].equal(b["m.0.a.3.c2.0.weight"])

#     torch.save(b, "save/g1_csc1.tar")
def vi():
    m1 = DataParallel(Model([1])).to(o.device)
    a = torch.load("save/g1_tnrd6p256e30.tar")
    load(m1, a)
    from importlib import reload

    o.model = "tnrdcsc"
    import model

    reload(model)
    m2 = DataParallel(model.Model([1])).to(o.device)
    a = torch.load("save/g1_csc1.tar")
    load(m2, a)
    # input
    d = DataLoader(BSD68_03(), 1)
    for i in d:
        g, y, s = [x.to(o.device) for x in i]
    torch.set_grad_enabled(False)
    # show(g)
    # show(m1([y,y,s])[-1])
    # show(m2([y,y,s])[-1])
    n = 0
    for ii in range(len(m1.module.m[0].a)):
        rbf = m1.module.m[0].a[ii]
        act = m2.module.m[0].a[ii]
        if type(act) is not model.tnrdcsc.Rbf:
            continue
        ps = 400 if n == 0 else 40
        x = torch.empty(1, 64, 1, 1).to("cuda")
        for i in range(-ps, ps):
            x.fill_(i)
            xx = x[0, :, 0, 0]
            if n % 2 == 0:
                y1 = rbf(x)[0, :, 0, 0]
                y2 = act(x)[0, :, 0, 0]
            for j in range(3):
                s = "_" + str(n) + "_" + str(j + 1)
                w.add_scalar("xx" + s, xx[j], i)
                w.add_scalar("y1" + s, y1[j], i)
                w.add_scalar("y2" + s, y2[j], i)
        n += 1
        print("finish")
        sleep(10)
        return


def m():
    m1 = DataParallel(Model([1])).to(o.device)
    a = torch.load("save/g1_tnrd6p256e30.tar")
    load(m1, a)
    from importlib import reload

    o.model = "tnrdcsc"
    import model

    reload(model)
    m2 = DataParallel(model.Model([1])).to(o.device)
    a = torch.load("save/g1_csc2.tar")

    load(m2, a)
    n = 0
    for ii in range(len(m1.module.m[0].a)):
        rbf = m1.module.m[0].a[ii]
        if rbf is None or not hasattr(rbf, "w"):
            continue
        act = m2.module.m[0].a[ii]
        ps = 400 if n == 0 else 40
        # plot fit
        import numpy as np
        import matplotlib.pyplot as plt

        x = torch.empty(1, 64, 1, 1).to("cuda")
        y = []
        for i in range(-ps, ps):
            x.fill_(i)
            y.append(rbf(x)[0, :, 0, 0].detach().cpu().numpy())
        y = np.transpose(y)
        x = np.arange(-ps, ps)
        sigma = 150 if n == 0 else 15
        w = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-x ** 2 / (2 * sigma ** 2))
        deg = len(act.c1)
        for i in range(64):
            z = np.polyfit(x, y[i], deg, w=w)
            # show in plt
            p = np.poly1d(z)
            _ = plt.plot(x, y[i], "--", x, p(x), "-")
            plt.show()
            # convert to conv parameter
            for j in range(deg + 1):
                if j == 0:
                    act.c1[j].weight[i, 0, 0, 0] = z[~j - 1]
                    act.c1[j].bias[i] = z[~j]
                else:
                    act.c1[j].weight[i, 0, 0, 0] = z[~j - 1]
                    act.c1[j].bias[i] = 1
                z /= z[~j - 1]

        # show in tb
        x = torch.empty(1, 64, 1, 1).to("cuda")
        for i in range(-ps, ps):
            x.fill_(i)
            # xx = x[0, :, 0, 0]
            # y1 = rbf(x)[0, :, 0, 0]
            y2 = act(x)[0, :, 0, 0]
            for j in range(1):
                s = "_" + str(n) + "_" + str(j + 1)
                # w.add_scalar("xx" + s, xx[j], i)
                # w.add_scalar("y1" + s, y1[j], i)
                w.add_scalar("y2" + s, y2[j], i)
        sleep(10)
        return
        if n == 5:
            break
        # grad=True

        n += 1

    # save
    # torch.save(m2.module.state_dict(), "save/g1_csc0.tar")


# v()
# vi()
m()
