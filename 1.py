from data import BSD
from model import ModelStack, ModelStage, FDN
import os
import torch
from config import o
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F
from util import show, crop, log_mean

# m = ModelStage().to(o.device)
m = ModelStack(2).to(o.device)


def train():
    d = DataLoader(BSD(), o.batch_size, num_workers=o.num_workers, shuffle=1)
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=o.lr)
    num = 0
    for epoch in range(o.epoch):
        for i in tqdm(d):
            g, y, k, s = [x.to(o.device) for x in i]
            optimizer.zero_grad()
            out = m([y, y, k, s])
            log_mean("out", out)
            out = crop(out, k)
            loss = mse(out, g)

            num += 1
            if (num % 50) == 1:
                show(
                    torch.cat(
                        (
                            g.detach().cpu()[0, 0],
                            crop(y.detach(), k).cpu()[0, 0],
                            out.detach().cpu()[0, 0],
                        ),
                        0,
                    )
                )
            log_mean("loss", loss)
            # log_mean("epoch", epoch)
            loss.backward()
            optimizer.step()


def test():
    d = DataLoader(Subset(BSD(), (0, 1)), o.batch_size, num_workers=0)
    mse = torch.nn.MSELoss()
    for i in d:
        g, y, k, s = [x.to(o.device) for x in i]
        out = m([y, y, k, s])
        out = crop(out, k)
        loss = mse(out, g)
        print(loss)
        show(torch.cat((out.detach()[0, 0, ...], g.detach()[0, 0, ...]), 0))


if __name__ == "__main__":
    train()
    # test()
    pass
