import scipy
import scipy.io
import numpy as np
import pathlib
import matplotlib.pyplot as plt
# import scipy.misc.imshow
from os import environ
environ['SCIPY_PIL_IMAGE_VIEWER'] = 'feh'
# for i in pathlib.Path('Levin09blurdata').iterdir():
mat = scipy.io.loadmat('Levin09blurdata/im05_flit03.mat')
plt.imshow(mat['y'],cmap='gray')
plt.show()
print(mat)