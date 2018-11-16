from data import BSD
from model import ModelStack, ModelStage, FDN
import os
import torch
from config import o
from torch.utils.data import DataLoader,Subset
from tqdm import tqdm
import torch.nn.functional as F
from util import show


m = ModelStage()
def train():
    d = DataLoader(Subset(BSD(),(0,1)), o.batch_size, num_workers=0)
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=o.lr)
    for epoch in range(1):
        for i in tqdm(d):
            g,y, k, s = [x.to(o.device) for x in i]
            optimizer.zero_grad()
            out = m([g, y, k, s])
            loss = mse(out, g)
            # show(torch.cat((out.detach()[0,0,...],g.detach()[0,0,...]),0).numpy())
            print(epoch,loss)
            loss.backward()
            optimizer.step()
    
def test():
    d = DataLoader(Subset(BSD(),(0,1)), o.batch_size, num_workers=0)
    mse = torch.nn.MSELoss()
    for i in d:
        g,y, k, s = [x.to(o.device) for x in i]
        out = m([g, y, k, s])
        loss = mse(out, g)
        print(loss)
        show(torch.cat((out.detach()[0,0,...],g.detach()[0,0,...]),0).numpy())

if __name__ == "__main__":
    o.device = "cpu"
    m.to(o.device)
    train()
    test()
    pass
