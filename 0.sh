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
    c model_checkpoint False
    c stage_checkpoint False
    c load \"save/g1_tnrd5.tar\"
    c save \"save/g1_tnrd6p200.tar\"
}
teston(){
    i
    c model \"tnrdcsc\"
    c run \"test\"
    c test_set \""${1:-BSD68_03}"\"
    # c save_image True
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
    #c penalty_space [310,50]
    # c load \"save/g1_tnrd6p100+++e80.tar\"
    # c load \"save/j2_tnrd6p100e50.tar\"
    # c load \"save/j2_tnrd6p100e10.tar\"
    # c load \"save/g1_tnrd6p100+e250.tar\"
    # c load \"save/g1_tnrd6p120ps50e30.tar\"
    # c load \"save/g1_tnrd6p256e30.tar\"
    #c load \"save/g1_cs2.tar\"
    c load \"save/g1_csc0.tar\"
    r
}
# one branch p1,5
cs2(){
    a(){
        i
        c model \"tnrdcs2\"
        c batch_size_ 2
        c num_workers 1
        #c stage_checkpoint True
        c save \"save/g1_cs2.tar\"
    }
    a
    r
}
# one branch kpk
cs3(){
    a(){
        i
        c model \"tnrdcs3\"
        c batch_size_ 2
        c num_workers 1
        #c mem_capacity 0
        c patch_size 200
        c milestones [100,100,100,100]
        c stage_checkpoint True
        c init_from \"load\"
        c load \"save/g1_cs3.tar\"
        #c stage_checkpoint True
        c save \"save/g1_cs3+.tar\"
    }
    a
    r
}
# c(p6',p3)
cs4(){
    n=5
    a(){
        i
        c model \"tnrdcs4\"
        c batch_size_ 1
        c num_workers 1
        c cs4 $n
        #c stage_checkpoint True
        c save \"save/g1_cs4_$n.tar\"
    }
    a
    r
}
# replace rbf with conv
csc(){
    a(){
        i
        c model \"tnrdcsc\"
        c batch_size_ 1
        c num_workers 0

        c init_from \"load\"
        c init \"save/g1_csc0.tar\"
        c save \"save/tmp.tar\"
    }
    a
    r
}
rbf(){
    a(){
        i
        c model \"tnrdcs\"
        c batch_size_ 1
        c num_workers 1
        # c stage_checkpoint True
        c penalty_num [80,30]
        c penalty_space [400,300]
        c save \"save/g1_rbf.tar\"
    }
    a
    r
}
tnrd(){
    a(){
        i
        c model \"tnrdcs\"
        #c batch_size 4
        c batch_size_ 1
        c num_workers 1
        #c patch_size 100
        #c stage_checkpoint True
        #c penalty_space [310,40]
        #c milestones [60,110]
        #c init_from \"load\"
        #c load \"save/g1_tnrd6p100pn50.tar\"
        c save \"save/g1_tnrd6p100_clip_re.tar\"
    }
    a
    r
}
tnrd96(){
    a(){
        i
        c model \"tnrdcs\"
        c batch_size_ 2
        c num_workers 1
        c stage 2
        c channel 96
        c stage_checkpoint True
        c epoch 90
        c milestones [30,60]
        c load \"save/g1_tnrd6p100c96+1.tar\"
        c save \"save/g1_tnrd6p100c96+2.tar\"
    }
    a
    r
}
tnrd100(){
    a(){
        i
        c model \"tnrdcs\"
        c num_workers 1
        #c batch_size 4
        c batch_size_ 2
        c patch_size 256
        c random_seed 2
        c epoch 210
        c stage_checkpoint True
        c milestones [30,90,150]
        c init_from \"load\"
        c load \"save/g1_tnrd6p256e30.tar\"
        c save \"save/g1_tnrd6p256+.tar\"
    }
    a
    r
}
tnrd75(){
    a(){
        i
        c model \"tnrdcs\"
        #c num_workers 0
        #c batch_size 4
        c batch_size_ 1
        c filter_size [7,5]
        #c patch_size 256
        #c random_seed 2
        #c epoch 90
        #c stage_checkpoint True
        #c milestones [0,65,80]
        #c init_from \"load\"
        #c load \"save/g1_tnrd6p200.tar\"
        c save \"save/g1_75.tar\"
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
g2(){
    a(){
        i
        c model \"tnrdcs\"
        c batch_size_ 1
        c num_workers 1
        #c mem_capacity 2
        #c channel 96
        #c filter_size 7
        #c depth 6
        #c random_seed 1
        #c model_checkpoint True
        #c stage_checkpoint True
        c stage 2
        c lr 1e-0
        c epoch 10
        c load \"save/g1_tnrd6p100+e250.tar\"
        c save \"save/g2_tnrd6p100lr0.tar\"
    }
    a
    r
    a
    c lr 1e-1
    c save \"save/g2_tnrd6p100lr1.tar\"
    r
    a
    c lr 1e-2
    c save \"save/g2_tnrd6p100lr2.tar\"
    r
    a
    c lr 1e-3
    c save \"save/g2_tnrd6p100lr3.tar\"
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