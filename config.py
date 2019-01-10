from util import dotdict
from tensorboardX import SummaryWriter

o = dict(
    stage=1,
    batch_size=8,
    num_workers=0,
    epoch=20,
    lr=1e-3,
    penalty_num=63,
    depth=2,
    channel=24,
    bias_scale=0,
    filter_scale=0.01,
    actw_scale=0.01,
    patch_size=80,
    sigma=25,
    sigma_range=False,
)
o = dotdict(o)
writer = SummaryWriter()
# writer = SummaryWriter(comment="".join("_" + str(i) for i in o.values()))
