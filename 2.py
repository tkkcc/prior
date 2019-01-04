import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import o
from data import *
from model import ModelStack, ModelStage
from util import center_crop, change_key, crop, isnan, load, log, mean, npsnr, npsnr_align_max, show

o.device = "cuda" if torch.cuda.is_available() else "cpu"

# m:model to train, p:pre models
def train(m, p=None):
    d = DataLoader(TNRD400(), o.batch_size, num_workers=o.num_workers)
    optimizer = torch.optim.Adam(m.parameters(), lr=o.lr)
    iter_num = len(d)
    num = 0
    losss = []
    stage = 1 if not p else p.stage + 1
    for epoch in range(o.epoch):
        for i in tqdm(d, mininterval=1):
            g, y, s = [x.to(o.device) for x in i]
            x = torch.tensor(y, requires_grad=False)
            if p:
                with torch.no_grad():
                    x = p([x, y, s])
            optimizer.zero_grad()
            out = m([x, y, s])
            # loss = F.mse_loss(g, out,reduction='elementwise_mean')
            loss = (g - out).pow(2).sum()
            loss.backward()
            optimizer.step()
            losss.append(loss.detach().item())
            assert not isnan(losss[-1])
            print("stage", stage, "epoch", epoch + 1)
            log("loss", mean(losss[-5:]))
            log("psnr", npsnr(g, out.detach()))
            num += 1
            # if num > (o.epoch * iter_num - 4):
            if num % 200 == 0:
                show(
                    torch.cat((y[0, 0], g[0, 0], out[0, 0]), 1),
                    # save=f"save/{stage:02}{epoch:02}.png",
                )
    plt.clf()
    plt.plot(range(len(losss)), losss)
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.title(f"{iter_num} iter x {o.epoch} epoch")
    plt.savefig(f"save/{stage:02}loss.png")


# greedy train the i stage
def greedy(stage=1):
    p = None
    m = DataParallel(ModelStack(1)).to(o.device)
    # load(m, "save/01-10g.tar")
    if stage > 1:
        p = DataParallel(ModelStack(stage - 1)).to(o.device)
        load(p, "save/01-10g.tar")
        p.eval()
        p.stage = stage - 1
        # init stage using stage-1
        a = change_key(p.module.m[-1].state_dict(), lambda x: f"m.0.{x}")
        load(m, a)
    train(m, p)
    # concat and save
    a = change_key(m.module.m[0].state_dict(), lambda x: f"m.{stage-1}." + x)
    if p:
        a.update(p.module.state_dict())
    torch.save(a, "save/01-10g.tar")


def test(m):
    m.eval()
    with torch.no_grad():
        d = DataLoader(TNRD68_03(), 1)
        losss = []
        for i in tqdm(d):
            g, y, s = [x.to(o.device) for x in i]
            x = torch.tensor(y)
            out = m([x, y, s])
            loss = npsnr(g, out)
            losss.append(-loss.detach().item())
            assert not isnan(losss[-1])
            log("input psnr", npsnr(g, y))
            log("psnr", losss[-1])
            show(torch.cat((y[0, 0], g[0, 0], out[0, 0]), 1))
        log("psnr avg", sum(losss) / len(losss))


if __name__ == "__main__":
    print(o)
    greedy(1)
    print("========test==========")
    m = ModelStack(1).to(o.device)
    load(m, "save/01-10g.tar")
    test(m)
