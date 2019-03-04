from util import dotdict
from tensorboardX import SummaryWriter
from pathlib import Path

o = dict(
    model="tnrd",
    run="greedy",
    # test_set="BSD68",
    # test_set="Set12",
    # test_set="TNRD68_03",
    test_set="TNRD68",
    stage=1,
    batch_size=4,
    num_workers=0,
    epoch=150,
    lr=1e-3,
    # lr*=0.1 when epoch=3, start from 0
    milestones=[60, 120],
    join_loss=False,
    filter_size=5,
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
    # init_from = "last",
    # greedy train, init current(n) stage from load, for continue train
    init_from="none",
    checkpoint=False,
    load="save/g1_tnrd5.tar",
    # load="save/g1_tnrdprior.tar",
    # load="save/j4.tar",
    # save="save/g1_tddnrdi.tar",
    save="save/g1_tnrd5.tar",
)
o = dotdict(o)
c = o.test_set if o.run == "test" else Path(o.save).stem
w = SummaryWriter(comment=c)
# writer = SummaryWriter(comment="".join("_" + str(i) for i in o.values()))
