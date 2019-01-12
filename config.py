from util import dotdict
from tensorboardX import SummaryWriter

o = dict(
    model="tnrd",
    stage=1,
    batch_size=4,
    num_workers=4,
    epoch=1,
    lr=1e-3,
    penalty_num=23,
    depth=2,
    channel=24,
    bias_scale=0,
    filter_scale=0.01,
    actw_scale=0.01,
    patch_size=80,
    sigma=75,
    sigma_range=True,
)
o = dotdict(o)
w = SummaryWriter()
# writer = SummaryWriter(comment="".join("_" + str(i) for i in o.values()))
