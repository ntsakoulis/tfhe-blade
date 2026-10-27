#!/bin/bash
# P1 baselines at the same PBS budget as r4 (102,109 = 64.4% of dense)
cd "$(dirname "$0")/.." && source scripts/common.sh

echo "=== P1 BASELINES (E17, E18) ==="
SRC="checkpoints_B/B_VGG9_W2_A2_IN8_NR1.pth"
COMMON="--ckpt $SRC --target_pbs_ratio 0.644 --step 0.10 --ft_epochs 40 --max_acc_drop -1 --alpha 0"

# E18: hand-crafted, only conv1/conv2, filter L1-norm, same fraction in both
run "E18_only_conv12_l1norm" real_pruning.py $COMMON \
    --criterion l1norm --alloc uniform --only_layers features.0,features.3 \
    --save_dir checkpoints_struct_e18 --results_dir results_struct_e18

# E17: classic filter pruning (Li et al. 2017), same fraction in every layer
run "E17_all_l1norm_uniform" real_pruning.py $COMMON \
    --criterion l1norm --alloc uniform \
    --save_dir checkpoints_struct_e17 --results_dir results_struct_e17