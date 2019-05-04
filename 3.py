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
        # grad=False
        o.lr = 1e-1
        o.epoch = 10000
        o.milestones = [6000, 9000]
        optimizer = Adam(m2.parameters(), lr=o.lr)
        scheduler = MultiStepLR(optimizer, milestones=o.milestones, gamma=0.1)
        num = 0
        # o.epoch=0
        for i in trange(o.epoch, desc="epoch", mininterval=1):
            num += 1
            # x = torch.randn(4, 64, 60, 60).to("cuda")
            # x = x * (150 if n == 0 else 15)
            x = torch.rand(4, 64, 60, 60).to("cuda")
            x = (x - 0.5) * (800 if n == 0 else 80)
            with torch.no_grad():
                o1 = rbf(x)
            o2 = act(x)
            loss = (o1 - o2).abs().mean()
            loss.backward()
            w.add_scalar("loss_" + str(n), loss.item(), num)
            w.add_scalar("lr_" + str(n), optimizer.param_groups[0]["lr"], num)

            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
        torch.save(m2.module.state_dict(), "save/g1_csc2.tar")
        ps = 400 if n == 0 else 40
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
        optimizer = Adam(m2.parameters(), lr=o.lr)
        scheduler = MultiStepLR(optimizer, milestones=o.milestones, gamma=0.1)
        num = 0
        for i in trange(o.epoch, desc="epoch", mininterval=1):
            num += 1
            x = torch.rand(4, 64, 60, 60).to("cuda")
            x = (x - 0.5) * 800
            with torch.no_grad():
                o1 = rbf(x, 1)
            o2 = act(x, 1)
            loss = (o1 - o2).pow(2).mean()
            loss.backward()
            w.add_scalar("loss_" + str(n) + "_", loss.item(), num)
            w.add_scalar("lr_" + str(n) + "_", optimizer.param_groups[0]["lr"], num)
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
        n += 1

    # save
    # torch.save(m2.module.state_dict(), "save/g1_csc0.tar")


# v()
# vi()
m()
