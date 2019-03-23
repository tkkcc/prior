#!/usr/bin/env bash
# ./0.sh teston Set12
c(){
    sed -ri 's|^(\s*'$1'=).*,|\1'$2',|' config.py
}
r(){
    python 2.py
}
i(){
    c model \"tnrd\"
    c run \"greedy\"
    c test_set \"BSD68\"
    c batch_size 4
    c batch_size_ 4
    c num_workers 4
    c epoch 120
    c lr 1e-3
    c milestones [90,110]
    c sigma 25
    c sigma_range False
    c patch_size 60
    c join_loss False
    c stage 1
    c depth 6
    c channel 64
    c filter_size 5
    c ioscale 255
    c penalty_space 310
    c penalty_num 63
    c penalty_gamma None
    c bias_scale 1
    c filter_scale 1
    c actw_scale 1
    c save_image False
    c random_seed 0
    c mem_capacity 1
    c init_from \"none\"
    c rbf_checkpoint False
    c stage_checkpoint False
    c load \"save/g1_tnrd5.tar\"
    c save \"save/g1_tnrd6p200.tar\"
}
teston(){
    i
    c run \"test\"
    c test_set \"$1\"
    c model \"tnrdcs\"
    # c depth 6
    # c penalty_num 100
    c mem_capacity 2
    c save_image True
    # c stage 1
    # c sigma 25
    c load \"save/g1_tnrd6p100.tar\"
    r
}
tnrd(){
    a(){
        i
        c model \"tnrdcs\"
        # c run \"test\"
        # c test_set \"BSD68\"
        # c batch_size_ 1
        # c num_workers 4
        #c rbf_checkpoint True
        # c depth 6
        #c mem_capacity 1
        c channel 96
        c patch_size 100
        #c sigma 30
        c load \"save/g1_tnrd6p60.tar\"
        c save \"save/g1_tnrd6p100c96.tar\"
    }
    a
    r
}
"$@"