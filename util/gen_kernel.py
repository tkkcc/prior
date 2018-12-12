from math import cos, sin
import numpy as np
from numpy import zeros, ones, prod, array, pi, log, min, mod, arange, sum, mgrid, exp, pad, round
from numpy.random import randn, rand
from scipy.signal import convolve2d

# https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
def fspecial_gauss(size, sigma):
    x, y = mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
    g = exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def blurkernel_synthesis(h=37, w=None):
    w = h if w is None else w
    kdims = [h, w]
    x = randomTrajectory(200)
    k = kernelFromTrajectory(x)
    # center pad to kdims
    pad_width = ((kdims[0] - k.shape[0]) // 2, (kdims[1] - k.shape[1]) // 2)
    pad_width = [(pad_width[0],), (pad_width[1],)]
    k = pad(k, pad_width, "constant")
    # import matplotlib.pyplot as plt
    # plt.imshow(k, interpolation="nearest", cmap="gray")
    # plt.show()
    return k


def kernelFromTrajectory(x):
    h = 5 - log(rand()) / 0.15
    h = round(min([h, 27])).astype(int)
    h = h + 1 - mod(h, 2)
    w = h
    k = zeros((h, w))

    xmin = min(x[0, :])
    xmax = max(x[0, :])
    ymin = min(x[1, :])
    ymax = max(x[1, :])
    xthr = arange(xmin, xmax, (xmax - xmin) / w)
    ythr = arange(ymin, ymax, (ymax - ymin) / h)

    for i in range(1, xthr.size):
        for j in range(1, ythr.size):
            idx = (
                (x[0, :] >= xthr[i - 1])
                & (x[0, :] < xthr[i])
                & (x[1, :] >= ythr[j - 1])
                & (x[1, :] < ythr[j])
            )
            k[i - 1, j - 1] = sum(idx)
    k = k / sum(k)
    k = convolve2d(k, fspecial_gauss(3, 1), "same")
    k = k / sum(k)
    return k


def randomTrajectory(T):
    x = zeros((3, T))
    v = randn(3, T)
    r = zeros((3, T))
    trv = 1 / 1
    trr = 2 * pi / T
    for t in range(1, T):
        F_rot = randn(3) / (t + 1) + r[:, t - 1]
        F_trans = randn(3) / (t + 1)
        r[:, t] = r[:, t - 1] + trr * F_rot
        v[:, t] = v[:, t - 1] + trv * F_trans
        st = v[:, t]
        st = rot3D(st, r[:, t])
        x[:, t] = x[:, t - 1] + st
    return x


def rot3D(x, r):
    Rx = array([[1, 0, 0], [0, cos(r[0]), -sin(r[0])], [0, sin(r[0]), cos(r[0])]])
    Ry = array([[cos(r[1]), 0, sin(r[1])], [0, 1, 0], [-sin(r[1]), 0, cos(r[1])]])
    Rz = array([[cos(r[2]), -sin(r[2]), 0], [sin(r[2]), cos(r[2]), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    x = R @ x
    return x

