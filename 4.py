# draw box
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from skimage.io import imread, imsave

from pathlib import Path
def border(img, box, color, width=5):
    x1, y1, x2, y2 = box
    if type(color) == str:
        color = [int(color[i : i + 2], 16) / 255 for i in range(0, len(color), 2)]
    # r, g, b = color
    for i in range(3):
        img[x1:x2, y1 : y1 + width, i] = color[i]
        img[x1:x2, y2 - width : y2, i] = color[i]
        img[x1 : x1 + width, y1:y2, i] = color[i]
        img[x2 - width : x2, y1:y2, i] = color[i]


def m(dir='gt'):
    file = f"./box/{dir}/028.png"

    img = imread(file)/255
    if not img.shape[1] % 2 == 0:
        img = img[:, :-1]
    img = np.tile(np.expand_dims(img, -1), [1, 1, 3])

    size = 80
    b1 = [500, 250]
    b1.extend([i + size for i in b1])
    b2 = [250, 550]
    b2.extend([i + size for i in b2])
    croped1 = img[b1[0]:b1[2], b1[1]:b1[3], ...]
    croped2 = img[b2[0] : b2[2], b2[1] : b2[3], ...]
    scale_width = img.shape[1] // 2
    scale_height = croped1.shape[0] * scale_width // croped1.shape[1]
    croped1 = resize(croped1, (scale_height, scale_width))
    croped2 = resize(croped2, (scale_height, scale_width))
    border(croped1, [0, 0, *croped1.shape[:2]], "FF4081")
    border(croped2, [0, 0, *croped2.shape[:2]], "40C4FF")
    border(img, b1, "FF4081")
    border(img, b2, "40C4FF")
    croped = np.concatenate((croped1, croped2), 1)
    img = np.concatenate((img, croped), 0)
    file = Path(file).with_name('box.png')
    imsave(file, img)
    p = imread(file)
    print(p.shape)
    # plt.imshow(file)
    plt.imshow(img)
    # plt.imshow(img_croped, cmap='gray')
    plt.axis("off")
    plt.show()

for i in ['gt','noise','tnrd','dncnn','our']:
    m(i)
