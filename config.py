from util import dotdict

o = dict(model="FDN", device="cuda", batch_size=8, num_workers=0, epoch=100, lr=1e-3)
# lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
o = dotdict(o)
