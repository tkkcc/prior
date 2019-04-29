from util import dotdict, repeat_last
from tensorboardX import SummaryWriter
from pathlib import Path

o = dict(
    model="tnrdcs4",
    run="greedy",
    test_set="BSD68",
    batch_size=4,
    # internal batch size
    batch_size_=1,
    num_workers=1,
    epoch=120,
    lr=1e-3,
    # lr*=0.1 when epoch in milestones, start from 0
    milestones=[90,110],
    # sigma for train
    sigma=25,
    sigma_range=False,
    sigma_test=25,
    patch_size=100,
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
    save_image=False,
    random_seed=0,
    # 0: for loop, 1: tensor boardcast in train for loop in test, 2:tensor
    mem_capacity=1,
    # "last": greedy train, init current(n) stage using n-1 stage
    # "load": greedy train, init current(n) stage from load, for continue train
    # other: default random init
    init_from="none",
    model_checkpoint=False,
    stage_checkpoint=False,
    load="save/g1_tnrd5.tar",
    save="save/g1_cs4_5.tar",
    cs4=5,
)

o = dotdict(o)
c = o.test_set + "_" + Path(o.load).stem if o.run == "test" else Path(o.save).stem
w = SummaryWriter(comment=c)
# entend 
o.depth = repeat_last(o.depth, o.stage)
o.filter_size = repeat_last(o.filter_size, max(o.depth))
o.penalty_gamma = repeat_last(o.penalty_gamma, max(o.depth))
o.penalty_space = repeat_last(o.penalty_space, max(o.depth))
o.penalty_num = repeat_last(o.penalty_num, max(o.depth))

