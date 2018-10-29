from data import BSD
from model import FDN
# import model
import os
import torch as t
from config import opt as o
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(**kwargs):
    o.parse(kwargs)
    m=FDN().to(o.device)
    d = DataLoader(BSD(), o.batch_size, num_workers=o.num_workers)
    mse = t.nn.MSELoss()
    lr = o.lr
    optimizer = t.optim.Adam(m.parameters(), lr=lr, weight_decay=o.weight_decay)
    for epoch in range(o.max_epoch):
        for d in tqdm(d):
            print(d)
            # train model
            # input = Variable(data)
            # target = Variable(label)
            # if o.use_gpu:
            #     input = input.cuda()
            #     target = target.cuda()
            
            optimizer.zero_grad()
            out = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.data[0])
            confusion_matrix.add(score.data, target.data)

            if ii % o.print_freq == o.print_freq - 1:
                vis.plot("loss", loss_meter.value()[0])

                # 进入debug模式
                if os.path.exists(o.debug_file):
                    import ipdb

                    ipdb.set_trace()

        # model.save()

if __name__ == "__main__":
    train()
    pass
