from util import dotdict
from tensorboardX import SummaryWriter
from pathlib import Path

o = dict(
    model="mlp",
    run="greedy",
    # test_set="BSD68",
    # test_set="Set12",
    # test_set="TNRD68_03",
    test_set="TNRD68",
    stage=1,
    batch_size=4,
    num_workers=0,
    epoch=40,
    lr=1e-3,
    filter_size=5,
    penalty_num=63,
    depth=2,
    channel=64,
    bias_scale=0,
    filter_scale=1,
    actw_scale=0.01,
    patch_size=80,
    sigma=25,
    sigma_range=False,
    mem_infinity=True,
    # greedy train, init current(n) stage using n-1 stage, else init like TNRD
    init_from_last=True,
    checkpoint=False,
    load="save/g1_mlpa.tar",
    # load="save/g1_tnrdprior.tar",
    # load="save/j4.tar",
    # save="save/g1_tddnrdi.tar",
    save="save/g1_mlp_kpkp.tar",
)
o = dotdict(o)
c = o.test_set if o.run == "test" else Path(o.save).stem
w = SummaryWriter(comment=c)
# writer = SummaryWriter(comment="".join("_" + str(i) for i in o.values()))
