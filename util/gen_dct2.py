import numpy as np
from scipy.fftpack import idct


def gen_dct2(n=3):
    C = np.empty((n ** 2, n ** 2))
    for i in range(n):
        for j in range(n):
            A = np.zeros((n, n))
            A[i, j] = 1
            B = idct(A, type=2, axis=0, norm="ortho")
            B = idct(B, type=2, axis=1, norm="ortho")
            C[:, i * n + j] = B.flatten()
    return C[:, 1:]
