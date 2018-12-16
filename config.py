from util import dotdict

o = dict(model="FDN", device="cuda", batch_size=128, num_workers=4, epoch=5, lr=0.03)
# lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
o = dotdict(o)
