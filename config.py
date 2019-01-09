from util import dotdict
from tensorboardX import SummaryWriter
o = dict(
    batch_size=8,
    num_workers=0,
    epoch=20,
    lr=1e-3,
    penalty_num=63,
    depth=3,
    channel=24,
    bias_scale=0,
    filter_scale=0.01,
    actw_scale=0.01,
    patch_size=80,
    sigma=25,
)
o = dotdict(o)
writer = SummaryWriter()
# writer = SummaryWriter(comment="".join("_" + str(i) for i in o.values()))
