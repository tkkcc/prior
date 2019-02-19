from util import dotdict
from tensorboardX import SummaryWriter

o = dict(
    model="tnrd",
    train="greedy",
    stage=2,
    batch_size=8,
    num_workers=4,
    epoch=30,
    lr=1e-3,
    penalty_num=63,
    depth=2,
    channel=64,
    bias_scale=0,
    filter_scale=0.01,
    actw_scale=0.01,
    patch_size=80,
    sigma=25,
    sigma_range=False,
    mem_infinity=True,
    # greedy train, init current(n) stage using n-1 stage, else init like TNRD
    init_from_last=True,
)
o = dotdict(o)
w = SummaryWriter()
# writer = SummaryWriter(comment="".join("_" + str(i) for i in o.values()))
