from util import dotdict

o = dict(model="FDN", device="cuda", batch_size=16, num_workers=4, epoch=3, lr=0.03, wd=0)

o = dotdict(o)
