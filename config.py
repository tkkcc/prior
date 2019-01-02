from util import dotdict

o = dict(model="FDN", device="cuda", batch_size=16, num_workers=0, epoch=1, lr=0.002)
# lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
o = dotdict(o)
