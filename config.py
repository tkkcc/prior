from util import dotdict
from tensorboardX import SummaryWriter
from pathlib import Path

o = dict(
    model="tnrd",
    run="joint",
    # test_set="BSD68",
    # test_set="Set12",
    test_set="TNRD68",
    stage=5,
    batch_size=4,
    num_workers=1,
    epoch=30,
    lr=1e-4,
    penalty_num=63,
    depth=2,
    channel=64,
    bias_scale=0,
    filter_scale=0.01,
    actw_scale=0.01,
    patch_size=80,
    sigma=25,
    sigma_range=False,
    mem_infinity=False,
    # greedy train, init current(n) stage using n-1 stage, else init like TNRD
    init_from_last=True,
    checkpoint=True,
    load="save/g5_initwithper_j4.tar",
    # load="save/j4.tar",
    # save="save/j4_+30.tar",
    save="save/j5.tar",
)
o = dotdict(o)
c = o.test_set if o.run == "test" else Path(o.save).stem
w = SummaryWriter(comment=c)
# writer = SummaryWriter(comment="".join("_" + str(i) for i in o.values()))
