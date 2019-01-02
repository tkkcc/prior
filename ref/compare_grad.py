# manual grad vs auto grad 
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad
import time
from profilehooks import timecall


def eq(a, b):
    print(F.mse_loss(a, b))


def parameter(x):
    return nn.ParameterList([nn.Parameter(i) for i in x])


batch_size = 3
in_channel = 3
med_channel = 32
out_channel = 3
filter_size = (5, 5)
depth = 5
iteration = 5


class M1(nn.Module):
    def __init__(self):
        super(M1, self).__init__()
        self.w = [
            torch.randn(med_channel, in_channel, *filter_size),
            *(torch.randn(med_channel, med_channel, *filter_size) for i in range(depth - 2)),
            torch.randn(out_channel, med_channel, *filter_size),
        ]

        self.b = [
            torch.randn(med_channel),
            *(torch.randn(med_channel) for i in range(depth - 2)),
            torch.randn(out_channel),
        ]

        self.w = parameter(self.w)
        self.b = parameter(self.b)

    def forward(self, x):
        p = [(i - 1) // 2 for i in filter_size]
        c = x
        for i in range(depth):
            c = F.conv2d(c, self.w[i], self.b[i], padding=p)
            c = c.sigmoid()
        c = c.pow(2).sum()
        c = grad(c, x, create_graph=True)[0]
        return c


class M2(nn.Module):
    def __init__(self):
        super(M2, self).__init__()
        self.w = [
            torch.randn(med_channel, in_channel, *filter_size),
            *(torch.randn(med_channel, med_channel, *filter_size) for i in range(depth - 2)),
            torch.randn(out_channel, med_channel, *filter_size),
        ]

        self.b = [
            torch.randn(med_channel),
            *(torch.randn(med_channel) for i in range(depth - 2)),
            torch.randn(out_channel),
        ]

        self.w = parameter(self.w)
        self.b = parameter(self.b)

    def forward(self, x):
        p = [(i - 1) // 2 for i in filter_size]
        c = x
        t = []
        for i in range(depth):
            c = F.conv2d(c, self.w[i], self.b[i], padding=p)
            c = c.sigmoid()
            t.append(c)
        for i in reversed(range(depth)):
            c = c * (t[i] * (1 - t[i]))
            c = F.conv_transpose2d(c, self.w[i], bias=None, padding=p)
        return 2 * c


torch.manual_seed(0)
a = torch.randn(batch_size, in_channel, 100,120)
a1 = torch.tensor(a, requires_grad=True)
a2 = torch.tensor(a, requires_grad=False)

torch.manual_seed(0)
m1 = M1()
torch.manual_seed(0)
m2 = M2()
assert m1.w[0].equal(m2.w[0])
assert m1.b[0].equal(m2.b[0])

if torch.cuda.is_available():
    a1=a1.to('cuda')
    a2=a2.to('cuda')
    m1=m1.to('cuda')
    m2=m2.to('cuda')


@timecall
def t1():
    for i in range(iteration):
        b1 = m1(a1)
        b1.sum().backward()
    return b1

@timecall
def t2():
    for i in range(iteration):
        b2 = m2(a2)
        b2.sum().backward()
    return b2

b2=t2()
b1=t1()

eq(b1, b2)
assert m1.w[0].grad is not None
eq(m1.w[0].grad, m2.w[0].grad)
eq(m1.b[0].grad, m2.b[0].grad)
'''
  t1 (0.py:106):
    16.081 seconds
  t2 (0.py:114):
    0.580 seconds
tensor(4.0627e-16, grad_fn=<SumBackward0>)
tensor(3.5087e-13)
tensor(3.8132e-13)

  t2 (0.py:113):
    6.549 seconds


  t1 (0.py:106):
    0.379 seconds

tensor(4.4095e-07, device='cuda:0', grad_fn=<SumBackward0>)
tensor(0.0002, device='cuda:0')
tensor(0.0002, device='cuda:0')

  t2 (0.py:113):
    6.680 seconds


  t1 (0.py:106):
    0.204 seconds

tensor(4.4095e-07, device='cuda:0', grad_fn=<SumBackward0>)
tensor(0.0000, device='cuda:0')
tensor(0.0001, device='cuda:0')
'''