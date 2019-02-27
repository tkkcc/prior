#!/usr/bin/env python
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from tqdm import tqdm
from tqdm import trange
from config import o, w
from data import *
from model import Model
from util import change_key, isnan, load, mean, npsnr, show, nssim, normalize, sleep
import json

o.device = "cuda" if torch.cuda.is_available() else "cpu"
w.add_text("config", json.dumps(o))
# m:model to train, p:pre models
def train(m, p=None):
    d = DataLoader(
        WED4744(),
        o.batch_size,
        num_workers=o.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    optimizer = Adam(m.parameters(), lr=o.lr)
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.3, cooldown=0, patience=10)
    scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    num = 0
    for i in trange(o.epoch, desc="epoch", mininterval=1):
        scheduler.step()
        for j in tqdm(d, desc="batch", mininterval=1):
            optimizer.zero_grad()
            g, y, s = [x.to(o.device) for x in j]
            x = y.clone().detach()
            if p:
                with torch.no_grad():
                    x = p([x, y, s])
            out = m([x, y, s])
            if type(out) == list:
                loss = 0
                for k in out:
                    loss += npsnr(g, k, reduction="sum")
            else:
                loss = npsnr(g, out, reduction="sum")
            # loss = nssim(g, out, reduction="sum")
            # loss = l2(g, out) + grad_diff(g, out)
            loss.backward()
            optimizer.step()
            loss = loss.detach().item()
            assert not isnan(loss)
            num += 1
            w.add_scalar("loss", loss, num)
            # w.add_scalar("lr", optimizer.param_groups[0]["lr"], num)
        psnr = _test(m, p)
        w.add_scalar("psnr", psnr, i)
        for name, param in m.named_parameters():
            w.add_histogram(name, param.clone().detach().cpu().numpy(), i)


# greedy train the i stage
def greedy(stage=1):
    p = None
    m = DataParallel(Model([stage])).to(o.device)
    if stage > 1:
        p = DataParallel(Model(stage - 1)).to(o.device)
        load(p, o.load)
        p.eval()
        p.stage = stage - 1
        if o.init_from_last:
            a = change_key(p.module.m[-1].state_dict(), lambda x: f"m.0.{x}")
            load(m, a)
    train(m, p)
    # concat and save
    a = change_key(m.module.m[0].state_dict(), lambda x: f"m.{stage-1}." + x)
    if p:
        a.update(p.module.state_dict())
    torch.save(a, o.save)
    return m


def joint(stage=1):
    m = DataParallel(Model(stage)).to(o.device)
    load(m, o.load)
    train(m)
    torch.save(m.module.state_dict(), o.save)
    return m


def test(stage=1):
    m = DataParallel(Model(stage)).to(o.device)
    load(m, o.load)
    m = _test(m, save=True)
    w.add_text("average", str(m), 0)
    print(m)


def _test(m, p=None, save=False):
    with torch.no_grad():
        d = globals()[o.test_set]
        d = DataLoader(d(), 1)
        losss = []
        for index, i in enumerate(tqdm(d, desc="test", mininterval=1)):
            g, y, s = [x.to(o.device) for x in i]
            # assert g.dim() == 4
            # n = y - g
            # w.add_image("gt", normalize(g[0]), 0)
            # w.add_image("noise", normalize(n[0]), 0)
            # sleep(1)
            # return
            x = y.clone().detach()
            x = p([x, y, s]) if p else x
            out = m([x, y, s])
            if type(out) == list:
                loss = npsnr(g, out[-1])
            else:
                # w.add_image("noise-d1", normalize((g-out)[0]), 0)
                loss = npsnr(g, out)
            losss.append(-loss.detach().item())
            assert not isnan(losss[-1])
            if save:
                w.add_scalar("result", losss[-1], index)
                # w.add_image("test", torch.cat((y[0], g[0], out[0]), -1), index)
            del loss
            del out
        return mean(losss)


if __name__ == "__main__":
    print(o)
    locals()[o.run](o.stage)
    # test()
    # joint(o.stage)
    # finetuned(1)
    # m = greedy(o.stage)
    # print("========test==========")
    # p = Model(1).to(o.device)
    # load(p ,"save/g_initliketnrd.tar")
    # m = Model([2]).to(o.device)
    # d = torch.load('save/g_initliketnrd.tar')
    # from collections import OrderedDict
    # a = OrderedDict()
    # s = m.state_dict()
    # for k in s:
    #     a[k] = d['m.1'+k[3:]]
    # (m if hasattr(m, "load_state_dict") else m.module).load_state_dict(a)
    # p= None
    # m = Model(2).to(o.device)
    # load(m ,"save/g_initliketnrd.tar")
    # print(test(m,p))
