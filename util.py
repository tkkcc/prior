class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation="nearest", cmap="gray")
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()
