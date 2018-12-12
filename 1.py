import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from config import o
from data import BSD
from model import ModelStack, ModelStage
from util import crop, log_mean, show

o.device = "cuda" if torch.cuda.is_available() else "cpu"
print("use " + o.device)

m = ModelStack(1).to(o.device)

# m.m[0].to('cuda:1')
def train():
    # load 'train' and 'test' dataset, 400x8 total
    d = DataLoader(
        ConcatDataset((BSD(), BSD("test"))), o.batch_size, num_workers=o.num_workers, shuffle=True
    )
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=o.lr, weight_decay=o.wd, amsgrad=True)
    iter_num = len(d)
    num = 0
    losss = []
    loss_sum = 0
    for epoch in range(o.epoch):
        for i in tqdm(d):
            # Bx1x250x250, Bx1x267,267, Bx1x13x13, Bx1
            # y is lager than g because of edgetaping
            g, y, k, s = [x.to(o.device) for x in i]
            optimizer.zero_grad()
            # x^0 = y
            out = m([y, y, k, s])
            log_mean("out", out)
            out = crop(out, k)
            loss = mse(out, g)
            # log_mean("loss", loss)
            loss.backward()
            optimizer.step()
            # avg loss for each epoch
            loss_sum += loss.detach().item()
            losss.append(loss_sum / (1 + len(losss) - epoch * iter_num))
            log_mean("loss", losss[-1])

            # show last 5 output
            num += 1
            # if num > (o.epoch * iter_num - 5):
            if num % 50 == 1:
                show(
                    torch.cat(
                        (
                            g.detach().cpu()[0, 0],
                            crop(y.detach(), k).cpu()[0, 0],
                            # y.detach().cpu()[0, 0],
                            out.detach().cpu()[0, 0],
                        ),
                        0,
                    )
                )
        loss_sum = 0
    plt.plot([i + 1 for i in range(len(losss))], losss)
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.title(f"{iter_num} iter x {o.epoch} epoch")
    plt.show()


def test():
    # use 'val', 100x8 total
    d = DataLoader(BSD("val"), o.batch_size, num_workers=0)
    mse = torch.nn.MSELoss()
    loss_sum = 0
    for i in tqdm(d):
        g, y, k, s = [x.to(o.device) for x in i]
        out = m([y, y, k, s])
        out = crop(out, k)
        loss = mse(out, g)
        loss_sum += loss.detach().item()
        # print(loss)
        # show(torch.cat((out.detach()[0, 0, ...], g.detach()[0, 0, ...]), 0))
    print("test", f"{loss_sum/len(d):.3f}")


if __name__ == "__main__":
    train()
    # test()
    pass
