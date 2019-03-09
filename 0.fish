#!/usr/bin/env fish
function c
    sed -ri 's|^(\s*'$argv[1]'=).*,|\1'$argv[2]',|' config.py
end
function r
    python 2.py
end
function test_set12
    c model \"tnrdi\"
    c lr 1e-3
    c filter_scale .01
    c filter_size 5
    c init_from \"last\"
    c checkpoint False
end
function tnrdi
    function i
        c model \"tnrdi\"
        c lr 1e-3
        c filter_scale .01
        c filter_size 5
        c init_from \"last\"
        c checkpoint False
    end
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
end
function mlpsm
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
end
function mlpelup
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
end
function mlp1x1
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
end
function mlpexplam
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
end
function mlpnolam
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
end
function tnrd5
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
    r
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
end

function mlp
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
end

function tnrd7
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
end
#tnrd5
tnrdi
#mlp1x1
#mlpsm
#mlpelup
#mlpexplam
#mlpnolam