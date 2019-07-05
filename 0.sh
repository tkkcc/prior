#!/usr/bin/env bash
# ./0.sh teston Set12
c() {
    sed -ri 's|^(\s*'$1'=).*,|\1'$2',|' config.py
}
r() {
    python 2.py
}
# old 1 conv model for comparison with tnrd
x1() {
    a() {
        i
        c model \"tnrd\"
        c bias_scale 0
        c filter_scale 0.01
        # c actw_scale 0.01
        c patch_size 60
        c depth 5
        c mem_capacity 1
        c save \"save/g1_tnrd5p60.tar\"
    }
    a
    r
    #a
    #c depth 2
    #c save \"save/g1_tnrd2p60.tar\"
    #r
}
c3() {
    r() {
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
i() {
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
    c pass_epoch -1
    c g2ng False
    c color False
    c train_set [\"BSD400\",\"ILSVRC12\",\"WED4744\"]
    c num_thread 1
}
teston() {
    i
    c model \"tnrdcss\"
    c run \"test\"
    c test_set \""${1:-BSD68_03}"\"
    c load \"save/g1_tnrd6p256+e60.tar\"
    # c load \"save/csc0_4ch4elu.tar\"
    r
}

# train & test for paper, color
trpc() {
    # pytorch>=1.0.1 is needed for urban100
    a() {
        i
        c model \"tnrdcs\"
        c run \"test\"
        c color True
        #c num_workers 0
        c test_set \"${1:-BSD68_03}\"
    }
    c batch_size_ 4
    c run \"greedy\"
    c test_set \"CBSD68\"
    c epoch 100
    c patch_size 80
    c milestones []
    c init_from \"none\"
    c save \"save/color_p80.tar\"
    #r
    a
    c batch_size_ 4
    c stage_checkpoint True
    c run \"greedy\"
    c test_set \"CBSD68\"
    c epoch 20
    c patch_size 256
    c milestones [12,18]
    c init_from \"load\"
    c load \"save/color_p80.tar\"
    c save \"save/color_p256.tar\"
    r
    return
    # sigma=25
    a
    c test_set \"CBSD68\"
    c load \"save/g1_tnrd6p256++.tar\"
    r # 29.230
    a
    c test_set \"Kodak24\"
    c load \"save/g1_tnrd6p256++.tar\"
    r # 30.548
    a
    c test_set \"McMaster\"
    c load \"save/g1_tnrd6p256++.tar\"
    r # 30.120
}
# test for paper
tp() {
    # pytorch>=1.0.1 is needed for urban100
    a() {
        i
        c model \"tnrdcs\"
        c run \"test\"
        c test_set \""${1:-BSD68_03}"\"
    }
    # sigma 50 final
    a
    c sigma_test 50
    c test_set \"BSD68\"
    c load \"save/s50+.tar\"
    r #26.273
    a
    c sigma_test 50
    c test_set \"Set12\"
    c load \"save/s50+.tar\"
    r #27.301
    a
    c sigma_test 50
    c test_set \"Urban100\"
    c load \"save/s50+.tar\"
    r #26.517996
    return
    # sigma 15 final
    a
    c sigma_test 15
    c test_set \"BSD68\"
    c load \"save/s15.tar\"
    r #31.7205
    a
    c sigma_test 15
    c test_set \"Set12\"
    c load \"save/s15.tar\"
    r #32.953
    a
    c sigma_test 15
    c test_set \"Urban100\"
    c load \"save/s15.tar\"
    r #32.769
    return
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
    #a
    #c sigma_test 15
    #c test_set \"BSD68\"
    #c load \"save/s15e50.tar\"
    #r #31.720
    #a
    #c sigma_test 15
    #c test_set \"Set12\"
    #c load \"save/s15e50.tar\"
    #r #32.953
    #a
    #c sigma_test 15
    #c test_set \"Urban100\"
    #c load \"save/s15e50.tar\"
    #r #32.769
    # sigma 50
    #a
    #c sigma_test 50
    #c test_set \"BSD68\"
    #c load \"save/s50e30.tar\"
    #r #26.262
    #a
    #c sigma_test 50
    #c test_set \"Set12\"
    #c load \"save/s50e30.tar\"
    #r #27.298
    #a
    #c sigma_test 50
    #c test_set \"Urban100\"
    #c load \"save/s50e30.tar\"
    #r #26.502
}
# replace rbf with conv
csc() {
    a() {
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
cscs() {
    a() {
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
# replace rbf with conv and split convt and conv and no group
cscss() {
    a() {
        i
        c model \"tnrdcscss\"
        c batch_size_ 1
        c num_workers 0
        c epoch 120
        c patch_size 256
        c milestones [30,60,90]
        c init_from \"load\"
        c g2ng True
        c load \"save/cscs_p256e140.tar\"
        c save \"save/cscss_p256.tar\"
    }
    a
    c run \"test\"
    c test_set \"BSD68_03\"
    r
}
# split parameters
css() {
    a() {
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
tnrd() {
    a() {
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
tnrd256() {
    a() {
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
s15() {
    s=50
    a() {
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
    c batch_size_ 1
    c patch_size 256
    c epoch 60
    c stage_checkpoint True
    c milestones [10,40]
    c load \"save/s50e40.tar\"
    c pass_epoch 40
    c save \"save/s$s+.tar\"
    r
}
# gray s35 75
grays35() {
    s=${1:-35}
    a() {
        i
        c model \"tnrdcs\"
        c num_workers 0
        c batch_size_ 1
        c sigma $s
        c sigma_test $s
        c epoch 30
        c milestones [15,25]
        c init_from \"load\"
        c load \"save/g1_tnrd6p256++e70.tar\"
        c save \"save/s${s}.tar\"
        c stage_checkpoint True
        c patch_size 256
    }
    a
    r
}
# color s15 50 75
color() {
    s=$1
    a() {
        i
        c model \"tnrdcs\"
        c num_workers 0
        c batch_size_ 1
        c sigma $s
        c sigma_test $s
        c epoch 30
        c milestones [15,25]
        c init_from \"load\"
        c load \"save/color_p256.tar\"
        c save \"save/color_s${s}_1525.tar\"
        c stage_checkpoint True
        c patch_size 256
        c color True
        c test_set \"CBSD68\"
    }
    a
    r
}
# gray s25
gray() {
    a() {
        i
        c model \"tnrdcs\"
        c run \"greedy\"
        c num_workers 0
        c test_set \"BSD68\"
    }
    c batch_size_ 2
    c epoch 100
    c patch_size 80
    c milestones []
    c init_from \"none\"
    c save \"save/gray_p80.tar\"
    r
    a
    c batch_size_ 1
    c stage_checkpoint True
    c epoch 20
    c patch_size 256
    c milestones [12,18]
    c init_from \"load\"
    c load \"save/gray_p80.tar\"
    c save \"save/gray_p256.tar\"
    r
}
# gray s25 extra
graye() {
    a() {
        i
        c model \"tnrdcs\"
        c run \"greedy\"
        c num_workers 0
        c test_set \"BSD68\"
    }
    a
    c batch_size_ 1
    c stage_checkpoint True
    c epoch 30
    c patch_size 256
    c milestones [5,25]
    c init_from \"load\"
    c load \"save/gray_p80.tar\"
    c save \"save/gray_p256_0525.tar\"
    r
}
# test for paper, color
tpc() {
    a=(CBSD68 Kodak24 McMaster)
    b=(15 25 35 50 75)
    for i in ${a[@]}; do
        for j in ${b[@]}; do
            if [ $j == 25 ]; then
                suffix=p256
            else
                suffix=s${j}_1525
            fi
            file=save/color_$suffix.tar
            [[ ! -e $file ]] && exit 1
            echo $i $j $suffix
            i
            c model \"tnrdcs\"
            c run \"test\"
            c color True
            c sigma_test $j
            c test_set \"$i\"
            c load \"$file\"
            r
        done
    done
}
"$@"
