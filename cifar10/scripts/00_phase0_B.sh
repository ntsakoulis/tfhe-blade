#!/bin/bash
# Move from scripts/ to the project root and source common.sh
cd "$(dirname "$0")/.." && source scripts/common.sh

echo "=== PHASE 0: PATH B (KNOWLEDGE DISTILLATION) ==="

TEACHER_DIR="checkpoints_teacher"
TEACHER_CKPT="${TEACHER_DIR}/ResNet18_Teacher_FP32.pth"
mkdir -p "$TEACHER_DIR" checkpoints_B results_B results_fhe_B

# 1. Train FP32 Teacher if it does not already exist
if [ ! -f "$TEACHER_CKPT" ]; then
    echo "Teacher checkpoint not found. Training FP32 Teacher first..."
    run "Train_FP32_Teacher" train_fp32_teacher.py \
        --epochs 200 \
        --batch 128 \
        --save_dir "$TEACHER_DIR"
else
    echo "Found existing FP32 Teacher checkpoint: ${TEACHER_CKPT}"
fi

# 2. Train and evaluate all quantized student configurations via Knowledge Distillation
for conf in "${CONFIGS[@]}"; do
    read W A IN NR TAG <<< "$conf"
    
    # 2a. Train Student with Knowledge Distillation
    run "Train_B_${TAG}" qat_cifar10_B.py \
        --w_bits $W \
        --a_bits $A \
        --in_bits $IN \
        --narrow_range $NR \
        --epochs 400 \
        --teacher_ckpt "$TEACHER_CKPT" \
        --save_dir checkpoints_B
    
    # 2b. Extract integer profile without FHE compile
    for R in 6 8 None; do
        run "Profile_B_${TAG}_Rnd${R}" fhe_eval.py \
            --ckpt checkpoints_B/B_VGG9_W${W}_A${A}_IN${IN}_NR${NR}.pth \
            --method B_kd \
            --tag Profile_B_${TAG}_Rnd${R} \
            --w_bits $W \
            --a_bits $A \
            --in_bit_width $IN \
            --narrow_range $NR \
            --rounding $R \
            --exec_n 0 \
            --out results_fhe_B
    done
done
