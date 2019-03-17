#!/usr/bin/env bash
# ./0.sh teston Set12
c(){
    sed -ri 's|^(\s*'$1'=).*,|\1'$2',|' config.py
}
r(){
    python 2.py
}
teston(){
    c run \"test\"
    c test_set \"$1\"
    c model \"tnrd\"
    c depth 6
    c penalty_num 32
    c stage 1
    c load \"save/g1_tnrd6p32.tar\"
    c sigma 25
    c save_image False
    c checkpoint False
    r
}
tnrd6p32(){
    i(){
        c model \"tnrd\"
        c lr 1e-3
        c penalty_num 32
        c run \"greedy\"
        c filter_scale 1
        c bias_scale 1
        c filter_size 5
        c patch_size 60
        c batch_size 256
        c batch_size_ 4
        c mem_capacity 1
        c channel 64
        c depth 6
        c stage 1
        c test_set \"Set12\"
        c epoch 130
        c milestones [60]
        c random_seed 1
        c init_from \"load\"
        c load \"save/g1_tnrd6p32.tar\"
        c checkpoint False
        c save \"save/g1_tnrd6p32+.tar\"
    }
    i
    # c run \"test\"
    r
}
tnrdb(){
    i(){
        c model \"tnrdb\"
        c lr 1e-3
        c ioscale 255
        c penalty_space 310
        c penalty_num 32
        c penalty_gamma 10
        c run \"greedy\"
        c filter_scale 1
        c bias_scale 1
        c random_seed 0
        c filter_size 5
        c batch_size 4
        c batch_size_ 4
        c patch_size 60
        c depth 2
        c stage 1
        c test_set \"BSD68\"
        c epoch 130
        c milestones [120]
        c init_from \"none\"
        c load \"save/g1_tnrd5.tar\"
        c checkpoint False
        c save \"save/g1_tnrdb_6p32g10.tar\"
    }
    i
    r
}
tnrdc(){
    i(){
        c model \"tnrdc\"
        c lr 1e-3
        c penalty_num 63
        c run \"greedy\"
        c filter_scale 1
        c bias_scale 1
        c filter_size 5
        c patch_size 60
        c depth 4
        c stage 1
        c test_set \"Set12\"
        c epoch 130
        c milestones [120]
        c init_from \"none\"
        c load \"save/g1_tnrd5.tar\"
        c checkpoint False
        c save \"save/g1_tnrdc2.tar\"
    }
    i
    r
}
tnrd1(){
    i(){
        c model \"tnrd\"
        c lr 1e-3
        c run \"greedy\"
        c filter_scale 1
        c bias_scale 1
        c filter_size 5
        c patch_size 70
        c depth 1
        c stage 1
        c test_set \"BSD68\"
        c epoch 140
        c milestones [100,130]
        c init_from \"none\"
        c load \"save/g1_tnrd5.tar\"
        c checkpoint False
        c save \"save/g1_tnrd1.tar\"
    }
    i
    r
}
tnrdii(){
    i(){
        c model \"tnrdii\"
        c lr 1e-3
        c run \"greedy\"
        c filter_scale 1
        c bias_scale 1
        c filter_size 5
        c patch_size 70
        c depth 2
        c stage 1
        c test_set \"BSD68\"
        c epoch 140
        c milestones [100,130]
        c init_from \"none\"
        c load \"save/g1_tnrd5.tar\"
        c checkpoint False
        c save \"save/g1_tnrdii.tar\"
    }
    i
    r
}
tnrdri(){
    i(){
        c model \"tnrd\"
        c lr 1e-3
        c patch_size 80
        c actw_scale 1
        c bias_scale 1
        c filter_scale 1
        c filter_size 5
        c run \"greedy\"
        c channel 64
        c depth 2
        c init_from \"none\"
        c mem_infinity False
        c checkpoint False
        c test_set \"BSD68\"
    }
    i
    #r
    c run \"greedy\"
    c epoch 50
    c milestones [0,20,40]
    c stage 3
    c load \"save/j2_tnrd5.tar\"
    c save \"save/g3_tnrdri.tar\"
    #r
    i
    c patch_size 80
    c run \"joint\"
    c stage 3
    c epoch 80
    c milestones [0,30,60]
    c load \"save/g3_tnrdri.tar\"
    c save \"save/j3_tnrdri.tar\"
    c checkpoint True
    r
}
tnrd4321(){
    i(){
        c model \"tnrd\"
        c lr 1e-3
        c patch_size 70
        c penalty_num 63
        c actw_scale 1
        c bias_scale 1
        c filter_scale 1
        c filter_size 5
        c run \"greedy\"
        c channel 64
        c depth [4,3,2,1]
        c epoch 120
        c milestones [90,110]
        c stage 1
        c init_from \"none\"
        c load \"save/g1_tnrd5.tar\"
        c save \"save/g1_tnrdx4.tar\"
        c mem_infinity False
        c checkpoint False
        c test_set \"BSD68\"
    }
    i
    c load \"save/g1_tnrdx4.tar\"
    c run \"test\"
    c test_set \"TNRD68_03\"
    #r
    i
    c checkpoint True
    c run \"greedy\"
    c stage 2
    #    c epoch 60
    #c milestones [5,55]
    c epoch 10
    c milestones [5]
    c init_from \"none\"
    c load \"save/g1_tnrdx4.tar\"
    c save \"save/g2_tnrd4321.tar\"
    #r
    i
    c checkpoint True
    c run \"joint\"
    c stage 2
    #    c epoch 60
    #c milestones [5,40]
    c epoch 20
    c milestones [0,5]
    c load \"save/g2_tnrd4321.tar\"
    c save \"save/j2_tnrd4321.tar\"
    r
    return
    i
    c checkpoint True
    c run \"greedy\"
    c stage 3
    c epoch 20
    c milestones [5,15]
    c load \"save/j2_tnrd4321.tar\"
    c save \"save/g3_tnrd4321.tar\"
    r
    i
    c checkpoint True
    c run \"joint\"
    c stage 3
    c epoch 30
    c milestones [0,15]
    c load \"save/g3_tnrd4321.tar\"
    c save \"save/j3_tnrd4321.tar\"
    r
}
tnrdifn(){
    i(){
        c model \"tnrdii\"
        c lr 1e-3
        c patch_size 70
        c depth 2
        c filter_size 5
        c run \"greedy\"
        c epoch 120
        c milestones [90,110]
        c stage 1
        c init_from \"none\"
        c load \"save/g1_tnrd5.tar\"
        c save \"save/g1_tnrdii_l255.tar\"
        c checkpoint False
        c test_set \"TNRD68\"
    }
    i
    r
    return
    i
    c run \"greedy\"
    c stage 1
    c epoch 40
    c milestones [30]
    c init_from \"load\"
    c load \"save/g1_tnrdii.tar\"
    c save \"save/g1_tnrdii+.tar\"
    r
}
tnrd5+(){
    i(){
        c model \"tnrd\"
        c lr 1e-3
        c filter_scale 1
        c filter_size 5
        c init_from \"load\"
        c load \"save/j5_tnrd5.tar\"
        c save \"save/j5_tnrd5+.tar\"
        c checkpoint True
        c mem_capacity 1
        c batch_size 256
        c batch_size_ 4
        c run \"joint\"
        c depth 2
        c random_seed 1
        c patch_size 60
        c stage 5
        c epoch 90
        c channel 64
        c penalty_num 63
        c milestones [0,30,60]
    }
    i
    r
}

tnrdiv(){
    c model \"tnrdiv\"
    c load \"save/g1_tnrdii.tar\"
    c test_set \"TNRD68_03\"
    c stage 1
    c run \"test\"
    c lr 1e-3
    c filter_scale .01
    c filter_size 5
    c checkpoint False
    r
}
tnrdi(){
    i(){
        c model \"tnrdi\"
        c lr 1e-3
        c filter_scale .01
        c filter_size 5
        c init_from \"last\"
        c checkpoint False
    }
    i
    c run \"greedy\"
    # c epoch 30
    # c milestones [0,20]
    c epoch 120
    c milestones [100]
    c stage 1
    c init_from \"none\"
    c load \"save/g1_tnrd5.tar\"
    c save \"save/g1_tnrdi_ifn.tar\"
    r
    return
    i
    c run \"greedy\"
    c stage 2
    c epoch 120
    c milestones [100]
    c load \"save/g1_tnrdi.tar\"
    c save \"save/g2_tnrdi.tar\"
    r
    i
    c run \"joint\"
    c stage 2
    c epoch 30
    c milestones [0,10]
    c load \"save/g2_tnrdi.tar\"
    c save \"save/j2_tnrdi.tar\"
    r
    i
    c run \"greedy\"
    c epoch 30
    c milestones [0,10]
    c stage 3
    c load \"save/j2_tnrdi.tar\"
    c save \"save/g3_tnrdi.tar\"
    r
    i
    c run \"joint\"
    c stage 3
    c epoch 30
    c milestones [0,0,10]
    c load \"save/g3_tnrdi.tar\"
    c save \"save/j3_tnrdi.tar\"
    r
    i
    c run \"greedy\"
    c epoch 40
    c milestones [0,30]
    c stage 4
    c load \"save/j3_tnrdi.tar\"
    c save \"save/g4_tnrdi.tar\"
    r
    i
    c run \"joint\"
    c stage 4
    c epoch 40
    c milestones [0,0,20]
    c load \"save/g4_tnrdi.tar\"
    c save \"save/j4_tnrdi.tar\"
    r
    i
    c run \"greedy\"
    c epoch 40
    c milestones [0,20]
    c stage 5
    c load \"save/j4_tnrdi.tar\"
    c save \"save/g5_tnrdi.tar\"
    r
    i
    c run \"joint\"
    c stage 5
    c epoch 40
    c milestones [0,0,20]
    c load \"save/g5_tnrdi.tar\"
    c save \"save/j5_tnrdi.tar\"
    c checkpoint True
    r
    c run \"greedy\"
    c epoch 40
    c milestones [5,20,35]
    c stage 6
    c load \"save/j5_tnrdi.tar\"
    c save \"save/g6_tnrdi.tar\"
    #r
    c run \"joint\"
    c stage 6
    c epoch 40
    c milestones [0,15,35]
    c load \"save/g6_tnrdi.tar\"
    c save \"save/j6_tnrdi.tar\"
    #r
}
mlpsm(){
    
    c model \"mlpsm\"
    c lr 1e-3
    c bias_scale 1
    c filter_scale 1
    c filter_size 5
    c run \"greedy\"
    c epoch 90
    c milestones [60]
    c stage 1
    c init_from \"none\"
    c load \"save/g1_mlpsm.tar\"
    c save \"save/g1_mlpth.tar\"
    r
}
mlpelup(){
    
    c model \"mlpelup\"
    c lr 1e-3
    c filter_scale 1
    c filter_size 5
    c batch_size 64
    c run \"greedy\"
    c epoch 2
    c milestones [60,70]
    c stage 1
    c init_from \"none\"
    c save \"save/g1_mlpelup1.tar\"
    r
    c init_from \"load\"
    c batch_size 4
    c epoch 80
    c load \"save/g1_mlpelup1.tar\"
    c save \"save/g1_mlpelup.tar\"
    r
}
mlp1x1(){
    
    c model \"mlp1x1\"
    c lr 1e-3
    c filter_scale 1
    c filter_size 5
    c run \"greedy\"
    c epoch 150
    c milestones [90,120]
    c stage 1
    c init_from \"none\"
    c load \"save/g1_mlp1x1.tar\"
    c save \"save/g1_mlp1x1_group1.tar\"
    r
}
mlpexplam(){
    
    c checkpoint False
    
    c model \"mlpexplam\"
    c lr 1e-3
    c filter_scale 1
    c run \"greedy\"
    c epoch 90
    c milestones [30,60]
    c stage 2
    c init_from \"last\"
    c load \"save/g1_mlp.tar\"
    c save \"save/g2_mlpexplam.tar\"
    #r
    c run \"joint\"
    c stage 2
    c epoch 40
    c milestones [0,20]
    c load \"save/g2_mlpexplam.tar\"
    c save \"save/j2_mlpexplam.tar\"
    #r
    c run \"greedy\"
    c epoch 40
    c milestones [10,30]
    c stage 3
    c load \"save/j2_mlpexplam.tar\"
    c save \"save/g3_mlpexplam.tar\"
    r
    c run \"joint\"
    c stage 3
    c epoch 40
    c milestones [0,15,30]
    c load \"save/g3_mlpexplam.tar\"
    c save \"save/j3_mlpexplam.tar\"
    r
}
mlpnolam(){
    
    c checkpoint False
    
    c model \"mlpnolam\"
    c lr 1e-3
    c filter_scale 1
    c run \"greedy\"
    c epoch 90
    c milestones [30,60]
    c stage 2
    c init_from \"last\"
    c load \"save/g1_mlp.tar\"
    c save \"save/g2_mlpnolam.tar\"
    r
    c run \"joint\"
    c stage 2
    c epoch 40
    c milestones [0,20]
    c load \"save/g2_mlpnolam.tar\"
    c save \"save/j2_mlpnolam.tar\"
    r
    c run \"greedy\"
    c epoch 40
    c milestones [10,30]
    c stage 3
    c load \"save/j2_mlpnolam.tar\"
    c save \"save/g3_mlpnolam.tar\"
    r
    c run \"joint\"
    c stage 3
    c epoch 40
    c milestones [0,15,30]
    c load \"save/g3_mlpnolam.tar\"
    c save \"save/j3_mlpnolam.tar\"
    r
}
tnrd5(){
    
    c model \"tnrd\"
    c lr 1e-3
    c filter_scale .01
    c filter_size 5
    c run \"greedy\"
    c epoch 120
    c milestones [100]
    c stage 1
    c init_from \"none\"
    c load \"save/g1_tnrd5.tar\"
    c save \"save/g1_tnrd5.tar\"
    #r
    c run \"greedy\"
    c stage 2
    c epoch 120
    c milestones [100]
    c init_from \"last\"
    c load \"save/g1_tnrd5.tar\"
    c save \"save/g2_tnrd5.tar\"
    #r
    c run \"joint\"
    c stage 2
    c epoch 20
    c milestones [0,5]
    c load \"save/g2_tnrd5.tar\"
    c save \"save/j2_tnrd5.tar\"
    #r
    c run \"greedy\"
    c epoch 20
    c milestones [0,10]
    c stage 3
    c load \"save/j2_tnrd5.tar\"
    c save \"save/g3_tnrd5.tar\"
    #r
    c run \"joint\"
    c stage 3
    c epoch 20
    c milestones [0,0,10]
    c load \"save/g3_tnrd5.tar\"
    c save \"save/j3_tnrd5.tar\"
    #r
    c init_from \"load\"
    c run \"joint\"
    c stage 3
    c epoch 20
    c milestones [0,0,10]
    c load \"save/j3_tnrd5.tar\"
    c save \"save/j3_tnrd5+.tar\"
    #r
    c init_from \"last\"
    c run \"greedy\"
    c epoch 40
    c milestones [0,30]
    c stage 4
    c load \"save/j3_tnrd5.tar\"
    c save \"save/g4_tnrd5.tar\"
    #r
    c run \"joint\"
    c stage 4
    c epoch 40
    c milestones [0,0,20]
    c load \"save/g4_tnrd5.tar\"
    c save \"save/j4_tnrd5.tar\"
    #r
    c run \"greedy\"
    c epoch 40
    c milestones [0,20]
    c stage 5
    c load \"save/j4_tnrd5.tar\"
    c save \"save/g5_tnrd5.tar\"
    #r
    c run \"joint\"
    c stage 5
    c epoch 40
    c milestones [0,0,20]
    c load \"save/g5_tnrd5.tar\"
    c save \"save/j5_tnrd5.tar\"
    c checkpoint True
    # r
    c run \"greedy\"
    c epoch 40
    c milestones [5,20,35]
    c stage 6
    c load \"save/j5_tnrd5.tar\"
    c save \"save/g6_tnrd5.tar\"
    #r
    c run \"joint\"
    c stage 6
    c epoch 40
    c milestones [0,15,35]
    c load \"save/g6_tnrd5.tar\"
    c save \"save/j6_tnrd5.tar\"
    #r
}

mlp(){
    
    c model \"mlp\"
    c lr 1e-3
    c filter_scale 1
    c run \"greedy\"
    c epoch 90
    c milestones [30,60]
    c stage 2
    c init_from \"last\"
    c load \"save/g1_mlp.tar\"
    c save \"save/g2_mlp_last.tar\"
    #r
    c run \"joint\"
    c stage 2
    c epoch 40
    c milestones [0,20]
    c load \"save/g2_mlp_last.tar\"
    c save \"save/j2_mlp_last.tar\"
    #r
    c run \"greedy\"
    c epoch 40
    c milestones [10,30]
    c stage 3
    c load \"save/j2_mlp_last.tar\"
    c save \"save/g3_mlp_last.tar\"
    #r
    c run \"joint\"
    c stage 3
    c epoch 40
    c milestones [0,15,30]
    c load \"save/g3_mlp_last.tar\"
    c save \"save/j3_mlp_last.tar\"
    #r
    c run \"greedy\"
    c epoch 40
    c milestones [10,30]
    c stage 4
    c load \"save/j3_mlp_last.tar\"
    c save \"save/g4_mlp_last.tar\"
    #r
    c run \"joint\"
    c stage 4
    c epoch 40
    c milestones [0,15,30]
    c load \"save/g4_mlp_last.tar\"
    c save \"save/j4_mlp_last.tar\"
    #r
    c run \"greedy\"
    c epoch 40
    c milestones [5,20,30]
    c stage 5
    c load \"save/j4_mlp_last.tar\"
    c save \"save/g5_mlp_last.tar\"
    #r
    c run \"joint\"
    c stage 5
    c epoch 40
    c milestones [0,15,30]
    c load \"save/g5_mlp_last.tar\"
    c save \"save/j5_mlp_last.tar\"
    #r
    c run \"greedy\"
    c epoch 40
    c milestones [5,20,35]
    c stage 6
    c load \"save/j5_mlp_last.tar\"
    c save \"save/g6_mlp_last.tar\"
    #r
    c run \"joint\"
    c stage 6
    c epoch 40
    c milestones [0,15,35]
    c load \"save/g6_mlp_last.tar\"
    c save \"save/j6_mlp_last.tar\"
    #r
}

tnrd7(){
    
    # bad at gj2
    # 7x7
    c stage 1
    c model \"tnrd\"
    c lr 1e-3
    c run \"greedy\"
    c milestones [40]
    c load \"save/g1_7.tar\"
    c save \"save/g1_7.tar\"
    r
    c run \"greedy\"
    c stage 2
    c milestones [30]
    c load \"save/g1_7.tar\"
    c save \"save/g2_7.tar\"
    r
    c run \"joint\"
    c stage 2
    c milestones [0]
    c load \"save/g2_7.tar\"
    c save \"save/j2_7.tar\"
    r
    c run \"greedy\"
    c stage 3
    c milestones [0]
    c load \"save/j2_7.tar\"
    c save \"save/g3_7.tar\"
    r
    c run \"joint\"
    c stage 3
    c milestones [0]
    c load \"save/g3_7.tar\"
    c save \"save/j3_7.tar\"
    r
    c run \"greedy\"
    c stage 4
    c milestones [0]
    c load \"save/j3_7.tar\"
    c save \"save/g4_7.tar\"
    r
    c run \"joint\"
    c stage 4
    c milestones [0]
    c load \"save/g4_7.tar\"
    c save \"save/j4_7.tar\"
    r
    c run \"greedy\"
    c stage 5
    c milestones [0]
    c load \"save/j4_7.tar\"
    c save \"save/g5_7.tar\"
    r
    c run \"joint\"
    c stage 5
    c milestones [0]
    c load \"save/g5_7.tar\"
    c save \"save/j5_7.tar\"
    r
    c run \"greedy\"
    c stage 6
    c milestones [0]
    c load \"save/j5_7.tar\"
    c save \"save/g6_7.tar\"
    r
    c run \"joint\"
    c stage 6
    c milestones [0]
    c load \"save/g6_7.tar\"
    c save \"save/j6_7.tar\"
    r
    c run \"greedy\"
    c stage 7
    c milestones [0]
    c load \"save/j6_7.tar\"
    c save \"save/g7_7.tar\"
    r
    c run \"joint\"
    c stage 7
    c milestones [0]
    c load \"save/g7_7.tar\"
    c save \"save/j7_7.tar\"
    r
    c run \"greedy\"
    c stage 8
    c milestones [0]
    c load \"save/j7_7.tar\"
    c save \"save/g8_7.tar\"
    r
    c run \"joint\"
    c stage 8
    c milestones [0]
    c load \"save/g8_7.tar\"
    c save \"save/j8_7.tar\"
    r
}
"$@"