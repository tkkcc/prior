#!/usr/bin/env bash
# ./0.sh teston Set12
c(){
    sed -ri 's|^(\s*'$1'=).*,|\1'$2',|' config.py
}
r(){
    python 2.py
}

c3(){
    r(){
        python 3.py
    }
    cc=4
    c run \"train\"
    c save \"save/csc0_${cc}ch4elu.tar\"
    # c logdir \"tmpc${ch}\"
    c cc ${cc}
    r
    #for i in $(seq 0 63);do
    #c logdir \"tmp5\"
    #c ccc $i
    #r
    #done
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
    c model \"tnrdcss\"
    c run \"test\"
    c test_set \""${1:-BSD68_03}"\"
    c load \"save/g1_tnrd6p256+e60.tar\"
    # c load \"save/csc0_4ch4elu.tar\"
    r
}
# test for paper
tp(){
    a(){
        i
        c model \"tnrdcs\"
        c run \"test\"
        c test_set \""${1:-BSD68_03}"\"
    }
    # 256
    a
    c test_set \"BSD68\"
    c load \"save/g1_tnrd6p256++.tar\"
    r # 29.230
    a
    c test_set \"Set12\"
    c load \"save/g1_tnrd6p256++.tar\"
    r # 30.548
    a
    c test_set \"Urban100\"
    c load \"save/g1_tnrd6p256++.tar\"
    r # 30.120
    # csc
    a
    c model \"tnrdcscs\"
    c test_set \"BSD68\"
    c load \"save/cscs_p256.tar\"
    r # 29.273
    # sigma 15
    a
    c sigma_test 15
    c test_set \"BSD68\"
    c load \"save/s15e50.tar\"
    r #31.720
    a
    c sigma_test 15
    c test_set \"Set12\"
    c load \"save/s15e50.tar\"
    r #32.953
    a
    c sigma_test 15
    c test_set \"Urban100\"
    c load \"save/s15e50.tar\"
    r #32.769
    # sigma 50
    a
    c sigma_test 50
    c test_set \"BSD68\"
    c load \"save/s50e30.tar\"
    r #26.262
    a
    c sigma_test 50
    c test_set \"Set12\"
    c load \"save/s50e30.tar\"
    r #27.298
    a
    c sigma_test 50
    c test_set \"Urban100\"
    c load \"save/s50e30.tar\"
    r #26.502
}
# replace rbf with conv
csc(){
    a(){
        i
        c model \"tnrdcsc\"
        c batch_size_ 4
        c num_workers 1
        c epoch 320
        c milestones [20,120,220]
        c init_from \"load\"
        #c load \"save/csc0_4ch4elu.tar\"
        c load \"save/csc_plain_inite80.tar\"
        c save \"save/csc_plain_init+.tar\"
        c load \"save/csce80.tar\"
        c save \"save/csc+.tar\"
    }
a
r
}
# replace rbf with conv and split convt and conv
cscs(){
    a(){
        i
        c model \"tnrdcscs\"
        c batch_size_ 2
        c num_workers 0
        c epoch 400
        c patch_size 256
        c milestones [100,200,300]
        c init_from \"load\"
        c load \"save/csc_p256.tar\"
        c save \"save/cscs_p256.tar\"
    }
a
r
}
# split parameters
css(){
    a(){
        i
        c model \"tnrdcss\"
        c batch_size_ 1
        c num_workers 0
        c patch_size 256
        c epoch 60
        c stage_checkpoint True
        c milestones [10,40]
        c init_from \"load\"
        c load \"save/g1_tnrd6p256+e60.tar\"
        c save \"save/css_p256.tar\"
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
tnrd256(){
    a(){
        i
        c model \"tnrdcs\"
        c num_workers 1
        #c batch_size 4
        c batch_size_ 2
        c patch_size 256
        c random_seed 2
        c epoch 120
        c stage_checkpoint True
        c milestones [0,30,90]
        c init_from \"load\"
        c load \"save/g1_tnrd6p256+e90.tar\"
        c save \"save/g1_tnrd6p256++.tar\"
    }
a
r
}
s15(){
    s=50
    a(){
        i
        c model \"tnrdcs\"
        c num_workers 1
        c batch_size_ 2
        c sigma $s
        c sigma_test $s
        c epoch 20
        c milestones [0]
        c init_from \"load\"
        c load \"save/g1_tnrd6p256++e70.tar\"
        c save \"save/s$s.tar\"
        # s15 31.73 target
        # s50 26.23 target
    }
#. $HOME/miniconda3/etc/profile.d/conda.sh
#conda activate
#a
#r
#conda activate t
a
c patch_size 256
c epoch 60
c stage_checkpoint True
c milestones [10,40]
#c load \"save/s15.tar\"
#c save \"save/s15+.tar\"
r
}
"$@"