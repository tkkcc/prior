#!/usr/bin/env python
import json
import time

import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm, trange

from config import o, w
from data import *
from model import Model
from util import change_key, isnan, load, mean, normalize, npsnr, nssim, show, sleep
torch.set_num_threads(o.num_thread)
o.device = "cuda" if torch.cuda.is_available() else "cpu"
o.device_count = torch.cuda.device_count()
w.add_text("config", json.dumps(o))
w.add_text("extra", "training dataset add ILSVRC12, kaiming_normal", 0)
w.add_text("extra", "clip in rbf input", 1)
# assert o.device == "cuda"
assert o.batch_size_ >= o.device_count
# m:model to train, p:pre models
def train(m, p=None):
    d = DataLoader(
        ConcatDataset([globals()[i]() for i in o.train_set]),
        o.batch_size_,
        num_workers=o.num_workers,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )
    optimizer = Adam(m.parameters(), lr=o.lr)
    scheduler = MultiStepLR(optimizer, milestones=o.milestones, gamma=0.1)
    num = 0
    for i in trange(o.epoch, desc="epoch", mininterval=1):
        scheduler.step()
        if i <= o.pass_epoch:
            continue
        for j in tqdm(d, desc="batch", mininterval=1):
            g, y, s = [x.to(o.device) for x in j]
            # continue
            x = y.clone().detach()
            if p:
                with torch.no_grad():
                    x = p([x, y, s])[-1]
            out = m([x, y, s])
            if o.join_loss:
                loss = 0
                for k in out:
                    loss += npsnr(g, k, reduction="sum")
            else:
                loss = npsnr(g, out[-1], reduction="sum")
            num += 1
            loss.backward()
            loss = loss.detach().item()
            assert not isnan(loss)
            if num % (o.batch_size // o.batch_size_) == 0:
                # w.add_scalar("loss", loss, num)
                w.add_scalar("loss/-batch", -loss / o.batch_size_, num)
                w.add_scalar("lr", optimizer.param_groups[0]["lr"], num)
                optimizer.step()
                optimizer.zero_grad()
        # continue
        m.eval()
        psnr = _test(m, p)
        m.train()
        w.add_scalar("psnr", psnr, i)
        for name, param in m.named_parameters():
            if "lam" in name:
                w.add_histogram(name, param.clone().detach().cpu().numpy(), i)
        if i % 10 == 0 and i != 0:
            if p:
                a = change_key(
                    m.module.m[0].state_dict(), lambda x: f"m.{o.stage-1}." + x
                )
                a.update(p.module.state_dict())
            else:
                a = m.module.state_dict()
            torch.save(a, o.save[:-4] + f"e{i}.tar")


# greedy train stage i
def greedy(stage=1):
    p = None
    m = DataParallel(Model([stage])).to(o.device)
    if stage > 1:
        p = DataParallel(Model(stage - 1)).to(o.device)
        load(p, o.load)
        p.eval()
        if o.init_from == "last":
            a = change_key(p.module.m[-1].state_dict(), lambda x: f"m.0.{x}")
            load(m, a)
    if o.init_from == "load":
        # m.9 => m.0
        def last2first(x):
            a = x.split(".")
            return "m.0." + ".".join(a[2:]) if a[1] == str(stage - 1) else None

        a = change_key(torch.load(o.load), last2first)
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
        # load group to no group
    # if o.g2ng:
    if o.g2ng:
        import model
        d=torch.load(o.load)
        for ii in range(len(m.module.m[0].a)):
            g = m.module.m[0].a[ii]
            if type(g) is not model.tnrdcscss.Rbf:
                continue
            # 4 conv in c1
            for j in range(4):
                c1w = d['m.0.a.'+str(ii)+'.c1.'+str(2*j)+'.weight']
                c2w = d['m.0.a.'+str(ii)+'.c2.'+str(2*j)+'.weight']
                c1b = d['m.0.a.'+str(ii)+'.c1.'+str(2*j)+'.bias']
                c2b = d['m.0.a.'+str(ii)+'.c2.'+str(2*j)+'.bias']
                g.c1[j*2].weight.data.fill_(0)
                g.c2[j*2].weight.data.fill_(0)
                ci = c1w.shape[1]
                co = c1w.shape[0]
                gci =  g.c1[j*2].weight.data.shape[1]
                gco =  g.c1[j*2].weight.data.shape[0]
                def f(index,step):
                    if step == 0:
                        return slice(0,1)
                    return slice(index*step,(index+1)*step)
                groups = o.channel
                s1 = 0
                s2 = 0
                for i in range(groups):
                    assert g.c1[j*2].weight.data[ f(i,gco//groups),  f(i,gci//groups), :, :].shape==c1w[  f(i,co//groups),0:ci, :, :].shape
                    g.c1[j*2].weight.data[ f(i,gco//groups),  f(i,gci//groups), :, :].copy_(c1w[  f(i,co//groups), 0:ci, :, :])
                    g.c2[j*2].weight.data[ f(i,gco//groups),  f(i,gci//groups), :, :].copy_(c2w[  f(i,co//groups), 0:ci, :, :])
                    # g.c2[j * 2].weight.data[i * ci:(i + 1) * ci, i * ci:(i + 1) * ci,:,:].copy_(c2w[i * ci:(i + 1) * ci,:,:,:])
                    s1+=c1w[  f(i,co//groups), 0:ci, :, :].sum()
                    s2 += g.c1[j * 2].weight.data[f(i, gco // groups), f(i, gci // groups),:,:].sum()
                print(s1-s2)
                print(g.c1[j*2].weight.data.sum()-c1w.sum())
                g.c1[j*2].bias.data.copy_(c1b)
                g.c2[j*2].bias.data.copy_(c2b)
    m.eval()
    m = _test(m, benchmark=True)
    print(m)

from imageio import imwrite
def _test(m, p=None, benchmark=False):
    with torch.no_grad():
        d = globals()[o.test_set]
        d = DataLoader(d(), 1)
        losss = []
        times = []
        for index, i in enumerate(tqdm(d, desc="test", mininterval=1)):
            g, y, s = [x.to(o.device) for x in i]
            # n = '028'
            # imwrite(f'./box/noise/{index+1:03d}.png', np.clip(y[0, 0].detach().cpu().numpy(), 0, 1))
            # imwrite(f'./box/gt/{index+1:03d}.png',np.clip(g[0, 0].detach().cpu().numpy(), 0, 1))
            # continue
            # g = imread(f'./box/gt/{n}.png').astype(np.float32)/255
            # g = torch.from_numpy(g).view(1,1, *g.shape).to(o.device)
            # y = imread(f'./box/noise/{n}.png').astype(np.float32)/255
            # y = torch.from_numpy(y).view(1, 1, *y.shape).to(o.device)
            # print(npsnr(g, y))
            # return
            x = y.clone().detach()
            # if benchmark:
            #     torch.cuda.synchronize()
            #     start = time.time()
            x = p([x, y, s])[-1] if p else x
            out = m([x, y, s])[-1]
            # if benchmark:
            #     torch.cuda.synchronize()
            #     times.append(time.time() - start)
            loss = npsnr(g, out)
            losss.append(-loss.detach().item())
            assert not isnan(losss[-1])
            if benchmark and o.save_image:
                # from skimage.io import imsave
                # imsave(f'./box/noise/{i+1:03d}.png', np.clip(result, 0, 1))
                # imsave(f'./box/gt/{i+1:03d}.png', np.clip(result, 0, 1))
                imwrite(f'./box/our/{n}.png', np.clip(out[0,0].detach().cpu().numpy(), 0, 1))
                print(losss[-1])
                w.add_scalar("result", losss[-1], index)
                # w.add_image("test", torch.cat((y[0], g[0], out[0]), -1), index)
        if benchmark:
            return mean(losss), mean(times)
        return mean(losss)


if __name__ == "__main__":
    print(o)
    locals()[o.run](o.stage)
