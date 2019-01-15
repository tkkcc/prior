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
from util import change_key, isnan, load, log, mean, npsnr, show, l2, grad_diff
import json

o.device = "cuda" if torch.cuda.is_available() else "cpu"
w.add_text("config", json.dumps(o))
w.add_text("extra", "WED4744 npsnr fixlr allrandninit bias")
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
    scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.333)
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
            loss = npsnr(g, out, reduction="sum")
            # loss = l2(g, out) + grad_diff(g, out)
            loss.backward()
            optimizer.step()
            loss = loss.detach().item()
            assert not isnan(loss)
            num += 1
            w.add_scalar("loss", loss, num)
            w.add_scalar("lr", optimizer.param_groups[0]["lr"], num)
        psnr = test(m, p)
        w.add_scalar("psnr", psnr, i)
        for name, param in m.named_parameters():
            w.add_histogram(name, param.clone().detach().cpu().numpy(), i)


# greedy train the i stage
def greedy(stage=1):
    p = None
    m = DataParallel(Model(1)).to(o.device)
    # load(m, "save/e.tar")
    if stage > 1:
        p = DataParallel(Model(stage - 1)).to(o.device)
        load(p, "save/e.tar")
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
    torch.save(a, "save/e.tar")
    return m


def test(m, p=None, write=False):
    # m.eval()
    with torch.no_grad():
        d = DataLoader(TNRD68(), 1)
        losss = []
        for index, i in enumerate(tqdm(d, desc="test", mininterval=1)):
            g, y, s = [x.to(o.device) for x in i]
            x = y.clone().detach()
            x = p([x, y, s]) if p else x
            out = m([x, y, s])
            loss = npsnr(g, out)
            losss.append(-loss.detach().item())
            assert not isnan(losss[-1])
            if write:
                w.add_image("test", torch.cat((y[0], g[0], out[0]), -1), index)
        return mean(losss)


if __name__ == "__main__":
    print(o)
    m = greedy(o.stage)
    print("========test==========")
    # m = Model(1).to(o.device)
    # load(m, "save/e.tar")
    # load(m, "save/01-10g_tnrd159+200_0.5e-3.tar")
    # print(test(m))
