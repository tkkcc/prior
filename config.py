from util import dotdict, repeat_last
from tensorboardX import SummaryWriter
from pathlib import Path

o = dict(
    model="tnrdcs",
    run="greedy",
    test_set="BSD68",
    batch_size=4,
    # internal batch size
    batch_size_=1,
    num_workers=0,
    epoch=30,
    lr=1e-3,
    # lr*=0.1 when epoch in milestones, start from 0
    milestones=[15,25],
    # sigma for train
    sigma=75,
    sigma_range=False,
    sigma_test=75,
    patch_size=256,
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
    # "load": greedy train, init current(n) stage from load, for continue training
    # other: default random init
    init_from="load",
    model_checkpoint=False,
    stage_checkpoint=True,
    load="save/g1_tnrd6p256++e70.tar",
    save="save/s75.tar",
    # epoch <= -1 will pass
    pass_epoch=-1,
    g2ng=False,
    color=False,
    train_set=["BSD400","ILSVRC12","WED4744"],
    num_thread=1,
)

o = dotdict(o)
c = o.test_set + "_" + Path(o.load).stem if o.run == "test" else Path(o.save).stem
# import datetime

# date = str(datetime.datetime.now())
# w = SummaryWriter(comment=c, log_dir=o.logdir + "/" + date)
# w2 = SummaryWriter(comment=c, log_dir=o.logdir + "/" + date + "_")
w = SummaryWriter(comment=c)
# w2 = SummaryWriter(comment=c + "_")
# entend
o.depth = repeat_last(o.depth, o.stage)
o.filter_size = repeat_last(o.filter_size, max(o.depth))
o.penalty_gamma = repeat_last(o.penalty_gamma, max(o.depth))
o.penalty_space = repeat_last(o.penalty_space, max(o.depth))
o.penalty_num = repeat_last(o.penalty_num, max(o.depth))
