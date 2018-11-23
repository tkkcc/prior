from util import dotdict

o = dict(model="FDN", device="cuda", batch_size=16, num_workers=4, epoch=3, lr=1e-1, wd=1e-6)

o = dotdict(o)
