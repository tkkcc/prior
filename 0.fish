#!/usr/bin/env fish
function c
    sed -ri 's|^(\s*'$argv[1]'=).*,|\1'$argv[2]',|' config.py
end
function r
    python 2.py
end

function tnrd5
    c model \"tnrd\"
    c lr 1e-3
    c filter_scale .01
    c filter_size 5
    c run \"greedy\"
    c epoch 150
    c milestones [60,120]
    c stage 1
    c init_from \"none\"
    c load \"save/g1_tnrd5.tar\"
    c save \"save/g1_tnrd5.tar\"
    r
    c run \"greedy\"
    c stage 2
    c epoch 150
    c milestones [60,120]
    c init_from \"last\"
    c load \"save/g1_tnrd5.tar\"
    c save \"save/g2_tnrd5.tar\"
    #r
    c run \"joint\"
    c stage 2
    c epoch 120
    c milestones [30,90]
    c load \"save/g2_tnrd5.tar\"
    c save \"save/j2_tnrd5.tar\"
    #r
    c run \"greedy\"
    c epoch 40
    c milestones [10,30]
    c stage 3
    c load \"save/j2_tnrd5.tar\"
    c save \"save/g3_tnrd5.tar\"
    #r
    c run \"joint\"
    c stage 3
    c epoch 40
    c milestones [0,15,30]
    c load \"save/g3_tnrd5.tar\"
    c save \"save/j3_tnrd5.tar\"
    #r
    c run \"greedy\"
    c epoch 40
    c milestones [10,30]
    c stage 4
    c load \"save/j3_tnrd5.tar\"
    c save \"save/g4_tnrd5.tar\"
    #r
    c run \"joint\"
    c stage 4
    c epoch 40
    c milestones [0,15,30]
    c load \"save/g4_tnrd5.tar\"
    c save \"save/j4_tnrd5.tar\"
    #r
    c run \"greedy\"
    c epoch 40
    c milestones [5,20,30]
    c stage 5
    c load \"save/j4_tnrd5.tar\"
    c save \"save/g5_tnrd5.tar\"
    #r
    c run \"joint\"
    c stage 5
    c epoch 40
    c milestones [0,15,30]
    c load \"save/g5_tnrd5.tar\"
    c save \"save/j5_tnrd5.tar\"
    #r
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
tnrd5
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

