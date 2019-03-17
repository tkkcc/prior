from util import dotdict
from tensorboardX import SummaryWriter
from pathlib import Path

o = dict(
    model="tnrdb",
    run="greedy",
    test_set="BSD68",
    stage=1,
    batch_size=4,
    # internal batch size
    batch_size_=4,
    num_workers=0,
    epoch=130,
    lr=1e-3,
    random_seed=0,
    # lr*=0.1 when epoch=3, start from 0
    milestones=[120],
    join_loss=False,
    filter_size=5,
    ioscale=255,
    penalty_space=310,
    penalty_num=32,
    penalty_gamma=10,
    depth=2,
    channel=64,
    bias_scale=1,
    filter_scale=1,
    actw_scale=1,
    patch_size=60,
    sigma=25,
    sigma_range=False,
    save_image=False,
    # 0: for loop, 1: tensor in train for loop in test, 2:tensor
    mem_capacity=1,
    # greedy train, init current(n) stage using n-1 stage, else init like TNRD
    # init_from = "last",
    # greedy train, init current(n) stage from load, for continue train
    init_from="none",
    checkpoint=False,
    load="save/g1_tnrd5.tar",
    save="save/g1_tnrdb_6p32g10.tar",
)
o = dotdict(o)
c = o.test_set if o.run == "test" else Path(o.save).stem
w = SummaryWriter(comment=c)
