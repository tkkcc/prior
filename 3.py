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

from config import o, w,w2
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

from scipy.signal import savgol_filter


def taker(table, bias=0):
    table = [torch.tensor(i).cuda().float() for i in table]

    def f(x, *args):
        return torch.cat(
            [
                table[i].take((x[:, i, :, :].unsqueeze(1) + bias).long())
                for i in range(x.shape[1])
            ],
            1,
        )

    return f


def m():
    m1 = DataParallel(Model([1])).to(o.device)
    a = torch.load("save/g1_tnrd6p256+e60.tar")
    load(m1, a)
    from importlib import reload

    o.model = "tnrdcsc"
    import model

    reload(model)
    m2 = DataParallel(model.Model([1])).to(o.device)
    # a = torch.load("save/g1_csc1.tar")

    load(m2, a)
    n = 0
    for ii in range(len(m1.module.m[0].a)):
        rbf = m1.module.m[0].a[ii]
        if type(rbf) is not model.tnrdcs.Rbf:
            continue
        act = m2.module.m[0].a[ii]
        ps = 400 if n == 0 else 100
        o.lr = 1e-3
        o.epoch = 27001
        o.milestones = [o.epoch]
        optimizer = Adam(m2.parameters(), lr=o.lr)
        scheduler = MultiStepLR(optimizer, milestones=o.milestones, gamma=0.1)

        def smooth(extra=None):
            # smooth rbf
            ys = []
            for j in range(o.channel):
                x = torch.empty(1, 1, 1, 1).cuda()
                y = []
                y_ = []
                for i in range(-ps, ps):
                    x.fill_(i)
                    y.append(rbf(x, extra)[:, j, :, :].item())
                y = savgol_filter(y, 33, 3)
                ys.append(y)
            return taker(ys, ps)


        def fit(extra=None):
            num = 0
            for i in trange(o.epoch, desc="epoch", mininterval=1):
                num += 1
                # x = torch.randn(4, channel, 60, 60).to("cuda")
                # x = x * (150 if n == 0 else 15)
                x = torch.rand(4, o.channel, 60, 60).to("cuda")
                x = (x - 0.5) * 2 * ps
                x.trunc_()
                with torch.no_grad():
                    o1 = rbf(x, extra)
                o2 = act(x, extra)
                loss = ((o1 - o2).pow(2)).mean()
                loss.backward()
                w.add_scalar("loss_" + str(n), loss.item(), num)
                w.add_scalar("lr_" + str(n), optimizer.param_groups[0]["lr"], num)

                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
            # if i % 3000 == 0:
            with torch.no_grad():
                x = torch.empty(1, o.channel, 1, 1).to("cuda")
                for j in range(-ps, ps):
                    x.fill_(j)
                    y1 = rbf(x, extra)[0, :, 0, 0]
                    y2 = act(x, extra)[0, :, 0, 0]
                    for k in range(o.channel):
                        s = str(i) + "_" + str(n) + "_" + str(k + 1)
                        w.add_scalar("y_" + s, y1[k], j)
                        w2.add_scalar("y_" + s, y2[k], j)
        rbf = smooth()
        fit()
        return
        if n == 5:
            break
        rbf = smooth(1)        
        fit(1)
        n += 1
    torch.save(m2.module.state_dict(), "save/g1_csc0.tar")


m()
