import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from config import o
from data import BSD, BSD3000, Levin
from model import ModelStack, ModelStage
from util import crop, log_mean, npsnr, show

o.device = "cuda" if torch.cuda.is_available() else "cpu"
print("use " + o.device)

# m:model to train, p:pre models
def train(m, p=None):
    d = DataLoader(BSD3000(), o.batch_size, num_workers=o.num_workers, shuffle=True)
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=o.lr, weight_decay=o.wd)
    iter_num = len(d)
    num = 0
    losss = []
    loss_sum = 0
    for epoch in range(o.epoch):
        for i in tqdm(d):
            # Bx1x284x284, Bx1x2,267, Bx13x13, Bx1
            # y is lager than g because of edgetaping
            g, y, k, s = [x.to(o.device) for x in i]
            if p:
                with torch.no_grad():
                    y = p([y, y, k, s])
            optimizer.zero_grad()
            # x^0 = y
            out = m([y, y, k, s])
            log_mean("out", out)
            out = crop(out, k)
            loss = npsnr(out, g)
            loss.backward()
            optimizer.step()
            losss.append(loss.detach().item())
            # loss_sum += loss.detach().item()
            # losss.append(loss_sum / (1 + len(losss) - epoch * iter_num))
            log_mean("loss", losss[-1])
            num += 1
            # if num > (o.epoch * iter_num - 5):
            # if num % 50 == 1:
            #     show(torch.cat((g[0, 0], crop(y, k)[0, 0], out[0, 0]), 1))
        # loss_sum = 0
    plt.plot([i + 1 for i in range(len(losss))], losss)
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.title(f"{iter_num} iter x {o.epoch} epoch")
    # plt.show()
    plt.savefig(f"save/{m.stage:02}loss.png")


# greedy train the i stage
def greedy(stage=1):
    p = None
    m = ModelStack(1).to(o.device)
    if stage > 1:
        p = ModelStack(stage - 1).to(o.device)
        for i in range(1, stage):
            p.m[i - 1].load_state_dict(torch.load(f"save/{i:02}g.tar"))
    train(m, p)
    torch.save(m.m[0].state_dict(), f"save/{stage:02}g.tar")


def test(m):
    m.eval()
    # todo
    torch.no_grad()
    d = Levin()
    losss = []
    for i in tqdm(d):
        g, y, k, s = [x.to(o.device) for x in i]
        out = m([y, y, k, s])
        out = crop(out, k)
        loss = npsnr(out, g)
        losss.append(loss.detach().item())
    log_mean("psnr avg", sum(losss) / len(losss))
    log_mean("psnr max", max(losss))


if __name__ == "__main__":
    for i in range(1, 11):
        print("greedy train stage " + str(i))
        greedy(i)
