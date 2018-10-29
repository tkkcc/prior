import argparse
import datetime
import json
import os
import sys
from pprint import pprint

import keras
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Conv2D, Dense, Input, Lambda
from keras.models import Model
from scipy.fftpack import idct
from scipy.signal import fftconvolve
from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.measure import compare_psnr
from skimage.transform import resize


# https://stackoverflow.com/a/43357954
def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis, ..., np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img, 2, 0)[..., np.newaxis]

def is_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    data = parser.add_argument_group("input")
    data.add_argument(
        "--image",
        metavar=None,
        type=str,
        default="data/example2.png",
        help="blurred image",
    )
    data.add_argument(
        "--kernel",
        metavar=None,
        type=str,
        default="data/example2.dlm",
        help="blur kernel",
    )
    data.add_argument(
        "--sigma",
        metavar=None,
        type=float,
        default=1.5,
        help="standard deviation of Gaussian noise",
    )
    data.add_argument(
        "--flip-kernel",
        metavar=None,
        type=str2bool,
        default=False,
        const=True,
        nargs="?",
        help="rotate blur kernel by 180 degrees",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--model-dir",
        metavar=None,
        type=str,
        default="models/sigma_0.1-12.75",
        help="path to model",
    )
    model.add_argument(
        "--n-stages",
        metavar=None,
        type=int,
        default=1,
        help="number of model stages to use",
    )
    model.add_argument(
        "--finetuned",
        metavar=None,
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="use finetuned model weights",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output",
        metavar=None,
        type=str,
        default=None,
        help="deconvolved result image",
    )
    output.add_argument(
        "--save-all-stages",
        metavar=None,
        type=str2bool,
        default=False,
        const=True,
        nargs="?",
        help="save all intermediate results (if finetuned is false)",
    )
    parser.add_argument(
        "--quiet",
        metavar=None,
        type=str2bool,
        default=False,
        const=True,
        nargs="?",
        help="don't print status messages",
    )
    return parser.parse_args()

def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis, ..., np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img, 2, 0)[..., np.newaxis]

def from_tensor(img):
    return np.squeeze(np.moveaxis(img[..., 0], 0, -1))


def pad_for_kernel(img, kernel, mode):
    p = [(d - 1) // 2 for d in kernel.shape]
    padding = [p, p] + (img.ndim - 2) * [(0, 0)]
    return np.pad(img, padding, mode)


def crop_for_kernel(img, kernel):
    p = [(d - 1) // 2 for d in kernel.shape]
    r = [slice(p[0], -p[0]), slice(p[1], -p[1])] + (img.ndim - 2) * [slice(None)]
    return img[r]


def edgetaper_alpha(kernel, img_shape):
    v = []
    for i in range(2):
        z = np.fft.fft(np.sum(kernel, 1 - i), img_shape[i] - 1)
        z = np.real(np.fft.ifft(np.square(np.abs(z)))).astype(np.float32)
        z = np.concatenate([z, z[0:1]], 0)
        v.append(1 - z / np.max(z))
    return np.outer(*v)


def edgetaper(img, kernel, n_tapers=3):
    alpha = edgetaper_alpha(kernel, img.shape[0:2])
    _kernel = kernel
    if 3 == img.ndim:
        kernel = kernel[..., np.newaxis]
        alpha = alpha[..., np.newaxis]
    for i in range(n_tapers):
        blurred = fftconvolve(
            pad_for_kernel(img, _kernel, "wrap"), kernel, mode="valid"
        )
        img = alpha * img + (1 - alpha) * blurred
    return img


def load_json(path, fname="config.json"):
    with open(os.path.join(path, fname), "r") as f:
        return json.load(f)

def save_result(result, path):
    path = path if path.find(".") != -1 else path + ".png"
    ext = os.path.splitext(path)[-1]
    if ext in (".txt", ".dlm"):
        np.savetxt(path, result)
    else:
        imsave(path, np.clip(result, 0, 1))


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation="nearest", cmap="gray")
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def _get_inputs(img_shape=(None, None, 1), kernel_shape=(None, None)):
    x_in = Input(shape=img_shape, name="x_in")
    y = Input(shape=img_shape, name="y")
    k = Input(shape=kernel_shape, name="k")
    s = Input(shape=(1,), name="s")
    return x_in, y, k, s


def model_stage(stage):
    init = "zeros"
    assert 1 <= stage
    x_in, y, k, s = _get_inputs()

    # MLP for noise-adaptive regularization weight
    layer = Lambda(lambda u: 1 / (u * u), name="1_over_s_squared")(s)
    for i in range(3):
        layer = Dense(
            16, activation="elu", name="dense%d" % (i + 1), kernel_initializer=init
        )(layer)
    lamb = Dense(1, activation="softplus", name="lambda", kernel_initializer=init)(
        layer
    )

    layer = Pad(20, "REPLICATE", name="x_in_padded")(x_in)
    nconvs = 5
    for i in range(nconvs):
        layer = Conv2D(
            32,
            (3, 3),
            activation="elu",
            padding="same",
            name="conv%d" % (i + 1),
            kernel_initializer=init,
        )(layer)
    layer = Conv2D(
        1,
        (3, 3),
        activation="linear",
        padding="same",
        name="conv%d" % (nconvs + 1),
        kernel_initializer=init,
    )(layer)

    x_adjustment = Crop(20, name="x_adjustment")(layer)
    x_out = FourierDeconvolution((5, 5), stage, name="x_out")(
        [x_in, x_adjustment, y, k, lamb]
    )
    return Model([x_in, y, k, s], x_out)

class ModelStage(nn.Module):
    def __init__(self, stage=1, channel=1):
        super(ModelStage, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ELU(1),
            nn.Linear(16, 16),
            nn.ELU(1),
            nn.Linear(16, 16),
            nn.ELU(1),
            nn.Linear(16, 1),
            nn.Softplus(beta=1, threshold=1),
        )
        k = (3, 3)
        p = [(i - 1) // 2 for i in k]
        # top&bottom left&right
        conv_pad = (p[0], p[1])
        convElu = lambda i, o: (nn.Conv2d(i, o, k, padding=conv_pad), nn.ELU(1))
        c4 = (j  for i in range(4) for j in convElu(32, 32))
        self.cnn = nn.Sequential(
            nn.ReplicationPad2d(20),
            *convElu(channel, 32),
            *c4,
            *convElu(32, channel),
            nn.ReplicationPad2d(-20)
        )
        self.fdn = FDN((5, 5), stage)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        x_in, y, k, s = inputs
        lam = self.mlp(s.pow(-2))
        x_in = x_in.permute(0, 3, 1, 2)
        x_adjustment = self.cnn(x_in)
        x_in = x_in.permute(0, 2, 3, 1)
        x_adjustment = x_adjustment.permute(0, 2, 3, 1)
        x_out = self.fdn([x_in, x_adjustment, y, k, lam])
        return x_out

def dct_filters(filter_size=(3, 3)):
    N = filter_size[0] * filter_size[1]
    filters = np.zeros((N, N - 1), np.float32)
    for i in range(1, N):
        d = np.zeros(filter_size, np.float32)
        d.flat[i] = 1
        filters[:, i - 1] = idct(idct(d, norm="ortho").T, norm="ortho").real.flatten()
    return filters

def psf2otf(psf, img_shape):
    psf_shape = tf.shape(psf)
    psf_type = psf.dtype
    midH = tf.floor_div(psf_shape[0], 2)
    midW = tf.floor_div(psf_shape[1], 2)
    top_left = psf[:midH, :midW, :, :]
    top_right = psf[:midH, midW:, :, :]
    bottom_left = psf[midH:, :midW, :, :]
    bottom_right = psf[midH:, midW:, :, :]
    zeros_bottom = tf.zeros(
        [psf_shape[0] - midH, img_shape[1] - psf_shape[1], psf_shape[2], psf_shape[3]],
        dtype=psf_type,
    )
    zeros_top = tf.zeros(
        [midH, img_shape[1] - psf_shape[1], psf_shape[2], psf_shape[3]], dtype=psf_type
    )
    top = tf.concat([bottom_right, zeros_bottom, bottom_left], 1)
    bottom = tf.concat([top_right, zeros_top, top_left], 1)
    zeros_mid = tf.zeros(
        [img_shape[0] - psf_shape[0], img_shape[1], psf_shape[2], psf_shape[3]],
        dtype=psf_type,
    )
    pre_otf = tf.concat([top, zeros_mid, bottom], 0)
    otf = tf.fft2d(tf.cast(tf.transpose(pre_otf, perm=[2, 3, 0, 1]), tf.complex64))
    return otf

class Pad(Layer):
    def __init__(self, border=0, mode="REPLICATE", **kwargs):
        assert border >= 0
        assert mode in ["REPLICATE", "ZEROS"]
        self.border = border
        self.mode = mode
        super(Pad, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[1] + 2 * self.border if input_shape[1] is not None else None,
            input_shape[2] + 2 * self.border if input_shape[2] is not None else None,
            input_shape[3],
        )

    def call(self, x, mask=None):
        if self.mode == "REPLICATE":
            # hack: iterate 1-pixel symmetric padding to get replicate padding
            for i in range(self.border):
                x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
        elif self.mode == "ZEROS":
            x = tf.pad(
                x,
                [
                    [0, 0],
                    [self.border, self.border],
                    [self.border, self.border],
                    [0, 0],
                ],
                "CONSTANT",
            )
        return x

class Crop(Layer):
    def __init__(self, border=0, **kwargs):
        assert border >= 0
        self.border = border
        super(Crop, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[1] - 2 * self.border if input_shape[1] is not None else None,
            input_shape[2] - 2 * self.border if input_shape[2] is not None else None,
            input_shape[3],
        )

    def call(self, x, mask=None):
        return (
            x[:, self.border : -self.border, self.border : -self.border, :]
            if self.border > 0
            else x
        )

class FourierDeconvolution(Layer):
    def __init__(self, filter_size, stage, **kwargs):
        self.filter_size = filter_size
        self.stage = stage
        super(FourierDeconvolution, self).__init__(**kwargs)

    def build(self, input_shapes):
        # construct filter basis B and define filter weights variable
        B = dct_filters(self.filter_size)
        self.B = K.variable(B, name="B", dtype="float32")
        self.nb_filters = B.shape[1]
        self.filter_weights = K.variable(
            np.eye(self.nb_filters), name="filter_weights", dtype="float32"
        )
        self.trainable_weights = [self.filter_weights]
        super(FourierDeconvolution, self).build(input_shapes)

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]

    def call(self, inputs, mask=None):

        padded_inputs, adjustments, observations, blur_kernels, lambdas = inputs
        imagesize = tf.shape(padded_inputs)[1:3]
        kernelsize = tf.shape(blur_kernels)[1:3]
        padding = tf.floor_div(kernelsize, 2)

        mask_int = tf.ones(
            (imagesize[0] - 2 * padding[0], imagesize[1] - 2 * padding[1]),
            dtype=tf.float32,
        )
        mask_int = tf.pad(
            mask_int,
            [[padding[0], padding[0]], [padding[1], padding[1]]],
            mode="CONSTANT",
        )
        mask_int = tf.expand_dims(mask_int, 0)
        filters = tf.matmul(self.B, self.filter_weights)

        filters = tf.reshape(
            filters, [self.filter_size[0], self.filter_size[1], 1, self.nb_filters]
        )
        filter_otfs = psf2otf(filters, imagesize)
        otf_term = tf.reduce_sum(tf.square(tf.abs(filter_otfs)), axis=1)
        k = tf.expand_dims(tf.transpose(blur_kernels, [1, 2, 0]), -1)
        k_otf = psf2otf(k, imagesize)[:, 0, :, :]

        if self.stage > 1:
            Kx_fft = tf.fft2d(tf.cast(padded_inputs[:, :, :, 0], tf.complex64)) * k_otf
            Kx = tf.to_float(tf.ifft2d(Kx_fft))
            Kx_outer = (1.0 - mask_int) * Kx
            y_inner = mask_int * observations[:, :, :, 0]
            y_adjusted = y_inner + Kx_outer
            dataterm_fft = tf.fft2d(tf.cast(y_adjusted, tf.complex64)) * tf.conj(k_otf)
        else:
            observations_fft = tf.fft2d(tf.cast(observations[:, :, :, 0], tf.complex64))
            dataterm_fft = observations_fft * tf.conj(k_otf)
        lambdas = tf.expand_dims(lambdas, -1)
        adjustment_fft = tf.fft2d(tf.cast(adjustments[:, :, :, 0], tf.complex64))
        numerator_fft = tf.cast(lambdas, tf.complex64) * dataterm_fft + adjustment_fft
        KtK = tf.square(tf.abs(k_otf))
        denominator_fft = lambdas * KtK + otf_term
        denominator_fft = tf.cast(denominator_fft, tf.complex64)
        frac_fft = numerator_fft / denominator_fft
        return tf.expand_dims(tf.to_float(tf.ifft2d(frac_fft)), -1)

def psf2otf_(psf, img_shape):
    psf_shape = psf.shape
    psf_type = psf.dtype
    midH = psf_shape[0] // 2
    midW = psf_shape[1] // 2
    top_left = psf[:midH, :midW, :, :]
    top_right = psf[:midH, midW:, :, :]
    bottom_left = psf[midH:, :midW, :, :]
    bottom_right = psf[midH:, midW:, :, :]
    zeros_bottom = torch.zeros(
        psf_shape[0] - midH,
        img_shape[1] - psf_shape[1],
        psf_shape[2],
        psf_shape[3],
        dtype=psf_type,
    )
    zeros_top = torch.zeros(
        midH, img_shape[1] - psf_shape[1], psf_shape[2], psf_shape[3], dtype=psf_type
    )
    top = torch.cat((bottom_right, zeros_bottom, bottom_left), 1)
    bottom = torch.cat((top_right, zeros_top, top_left), 1)
    zeros_mid = torch.zeros(
        img_shape[0] - psf_shape[0],
        img_shape[1],
        psf_shape[2],
        psf_shape[3],
        dtype=psf_type,
    )
    pre_otf = torch.cat((top, zeros_mid, bottom), 0)
    otf = rfft(pre_otf.permute(2, 3, 0, 1))
    return otf

def cm(t1, t2):
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack(
        [real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1
    )

def conj(t, inplace=False):
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c


def r2c(t):
    return torch.stack((t, torch.zeros_like(t)), -1)


def irfft(t):
    return torch.irfft(t, 2, onesided=False)


def ifft(t):
    return torch.ifft(t, 2, onesided=False)


def fft(t):
    return torch.fft(t, 2, onesided=False)


def rfft(t):
    return torch.rfft(t, 2, onesided=False)

class FDN(nn.Module):
    def __init__(self, filter_size=(5, 5), stage=1, **kwargs):
        super(FDN, self).__init__()
        self.filter_size = filter_size
        self.stage = stage
        B = dct_filters(self.filter_size)
        self.B = torch.tensor(B, dtype=torch.float)
        self.nb_filters = B.shape[1]
        # self.filter_weights = torch.tensor(np.eye(self.nb_filters), dtype=torch.float)
        self.filter_weights = nn.Parameter(torch.tensor(np.eye(self.nb_filters), dtype=torch.float))

    def forward(self, inputs, mask=None):
        padded_inputs, adjustments, observations, blur_kernels, lambdas = inputs
        imagesize = padded_inputs.shape[1:3]
        kernelsize = blur_kernels.shape[1:3]
        padding = [i // 2 for i in kernelsize]
        mask_int = torch.ones(
            imagesize[0] - 2 * padding[0],
            imagesize[1] - 2 * padding[1],
            dtype=torch.float32,
        )
        mask_int = F.pad(mask_int, (padding[1], padding[1], padding[0], padding[0]))
        mask_int = mask_int.unsqueeze(0)
        filters = self.B.mm(self.filter_weights)
        filters = filters.reshape(
            self.filter_size[0], self.filter_size[1], 1, self.nb_filters
        )
        filter_otfs = psf2otf_(filters, imagesize)
        otf_term = filter_otfs.pow(2).sum(-1).sum(1)
        k = blur_kernels.permute(1, 2, 0).unsqueeze(-1)
        k_otf = psf2otf_(k, imagesize)[:, 0, ...]
        if self.stage > 1:
            Kx_fft = cm(rfft(padded_inputs[:, :, :, 0]), k_otf)
            Kx = irfft(Kx_fft)
            Kx_outer = (1.0 - mask_int) * Kx
            y_inner = mask_int * observations[:, :, :, 0]
            y_adjusted = y_inner + Kx_outer
            dataterm_fft = cm(rfft(y_adjusted), conj(k_otf))
        else:
            observations_fft = rfft(observations[:, :, :, 0])
            dataterm_fft = cm(observations_fft, conj(k_otf))
        lambdas = lambdas.unsqueeze(-1)
        adjustment_fft = rfft(adjustments[:, :, :, 0])
        numerator_fft = cm(r2c(lambdas), dataterm_fft) + adjustment_fft
        KtK = k_otf.pow(2).sum(-1)
        denominator_fft = lambdas * KtK + otf_term
        t = torch.stack((denominator_fft,) * 2, -1)
        frac_fft = numerator_fft / t
        return irfft(frac_fft).unsqueeze(-1)


def model_stacked(n_stages, weights=None):
    assert weights is None or len(weights) == n_stages
    x = [*_get_inputs()]
    x0=[*x]
    for t in range(n_stages):
        stage = model_stage(t + 1)
        if weights is not None:
            stage.load_weights(weights[t])
        x[0] = stage(x)
    return Model(x0, x[0])

class ModelStack(nn.Module):
    def __init__(self, stage=1, weight=None):
        super(ModelStack, self).__init__()
        self.m = nn.ModuleList(ModelStage(i+1) for i in range(stage))

    def forward(self, d):
        for i in self.m:
            d[0] = i(d)
        return d[0]
