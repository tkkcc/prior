from util import dotdict
from tensorboardX import SummaryWriter
from pathlib import Path
o = dict(
    model="tnrdcs",
    run="greedy",
    test_set="BSD68",
    batch_size=4,
    # internal batch size
    batch_size_=4,
    num_workers=1,
    epoch=60,
    lr=1e-3,
    # lr*=0.1 when epoch in milestones, start from 0
    milestones=[0,30],
    # sigma only for train
    sigma=25,
    sigma_range=False,
    patch_size=100,
    join_loss=False,
    ## model
    stage=2,
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
    save_image=False,
    random_seed=0,
    # 0: for loop, 1: tensor boardcast in train for loop in test, 2:tensor
    mem_capacity=1,
    # "last": greedy train, init current(n) stage using n-1 stage
    # "load": greedy train, init current(n) stage from load, for continue train
    # other: default random init 
    init_from="load",
    model_checkpoint=False,
    stage_checkpoint=False,
    load="save/g2_tnrd6p100e10.tar",
    save="save/g2_tnrd6p100+.tar",
)

o = dotdict(o)
c = o.test_set if o.run == "test" else Path(o.save).stem
w = SummaryWriter(comment=c)