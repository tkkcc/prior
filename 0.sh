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
    c sigma_test 25
    c patch_size 100
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
    c model_checkpoint False
    c stage_checkpoint False
    c load \"save/g1_tnrd5.tar\"
    c save \"save/g1_tnrd6p200.tar\"
}
teston(){
    i
    c model \"tnrdv\"
    c run \"test\"
    c test_set \""${1:-BSD68_03}"\"
    # c stage 1
    # c depth 2
    # c sigma_test 15
    # c stage_checkpoint 3
    # c model \"tnrdcs\"
    # c depth 6
    # c penalty_num 100
    # c mem_capacity 2
    # c save_image True
    # c stage 1
    # c sigma 25
    # c load \"save/g1_tnrd6p100+++e80.tar\"
    # c load \"save/j2_tnrd6p100e50.tar\"
    # c load \"save/j2_tnrd6p100e10.tar\"
    c load \"save/g1_tnrd6p100+e250.tar\"
    # c load \"save/g1_tnrd5.tar\"
    r
}
tnrd(){
    a(){
        i
        c model \"tnrdcs\"
        #c run \"test\"
        #c test_set \"BSD68_03\"
        c batch_size_ 2
        c num_workers 1
        #c rbf_checkpoint True
        #c mem_capacity 1
        # c channel 96
        #c filter_size 6
        #c patch_size 100
        #c penalty_num 96
        #c depth 6
        #c random_seed 1
        # c stage_checkpoint True
        c penalty_space [310,50]
        # c epoch 90
        # c milestones [30,60]
        #c sigma 30
        # c init_from \"load\"
        #c load_filter False
        #c load \"save/g1_tnrd6p100+e250.tar\"
        #c load \"save/g1_tnrd6p100c96e100.tar\"
        # c load \"save/g1_tnrd6p100c96+1.tar\"
        # c save \"save/g1_tnrd6p100c96+2.tar\"
        # c save \"save/g1_tnrd6p100pn96.tar\"
        c save \"save/g1_tnrd6p100pn50.tar\"
    }
    a
    r
}
tnrd100(){
    a(){
        i
        c model \"tnrd\"
        c num_workers 1
        #c batch_size 128
        c batch_size_ 1
        c patch_size 100
        c random_seed 4
        c epoch 60
        c milestones [0,0,0,50]
        c init_from \"load\"
        c load \"save/g1_tnrd6p100+e250.tar\"
        c save \"save/g1_tnrd6p100+3.tar\"
    }
    a
    r
}
j2(){
    a(){
        i
        c model \"tnrdcs\"
        c run \"joint\"
        # c run \"test\"
        # c test_set \"BSD68_03\"
        #c batch_size_ 4
        c num_workers 1
        #c rbf_checkpoint True
        #c mem_capacity 2
        #c channel 96
        #c filter_size 7
        #c patch_size 100
        #c depth 6
        #c random_seed 1
        #c model_checkpoint True
        c stage 2
        c epoch 60
        c milestones [0,0,30]
        #c sigma 30
        #c init_from \"load\"
        c load \"save/g2_tnrd6p100e50.tar\"
        c save \"save/g2_tnrd6p100+1.tar\"
    }
    a
    r
}
tnrdt(){
    a(){
        i
        c model \"tnrd\"
        # c run \"test\"
        # c test_set \"BSD68_03\"
        #c batch_size_ 4
        # c num_workers 4
        #c rbf_checkpoint True
        #c mem_capacity 2
        #c channel 96
        # c filter_size 5
        #c patch_size 60
        c depth 4
        #c random_seed 1
        #c stage_checkpoint 2
        c epoch 250
        c milestones [0,200]
        #c sigma 30
        c init_from \"load\"
        c load \"save/g1_tnrd4re.tar\"
        c save \"save/tmpg1_tnrd.tar\"
    }
    a
    r
}
# tmp(){
#     i
#     c num_workers 0
#     c load \"save/g1_tnrd5.tar\"
#     c save \"save/tmp.tar\"
# }
"$@"