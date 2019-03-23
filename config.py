from util import dotdict
from tensorboardX import SummaryWriter
from pathlib import Path

o = dict(
    model="tnrdcs",
    run="test",
    test_set="BSD68",
    batch_size=4,
    # internal batch size
    batch_size_=4,
    num_workers=4,
    epoch=120,
    lr=1e-3,
    # lr*=0.1 when epoch=3, start from 0
    milestones=[90,110],
    sigma=25,
    sigma_range=False,
    patch_size=60,
    join_loss=False,
    ## model
    stage=1,
    depth=6,
    channel=64,
    filter_size=5,
    ioscale=255,
    penalty_space=310,
    penalty_num=63,
    penalty_gamma=None,
    bias_scale=1,
    filter_scale=1,
    actw_scale=1,
    ## extra
    save_image=True,
    random_seed=0,
    # 0: for loop, 1: tensor boardcast in train for loop in test, 2:tensor
    mem_capacity=2,
    # greedy train, init current(n) stage using n-1 stage, else init like TNRD
    # init_from = "last",
    # greedy train, init current(n) stage from load, for continue train
    init_from="none",
    rbf_checkpoint=False,
    stage_checkpoint=False,
    load="save/g1_tnrd6p100.tar",
    save="save/g1_tnrd6p200.tar",
)
o = dotdict(o)
c = o.test_set if o.run == "test" else Path(o.save).stem
w = SummaryWriter(comment=c)
