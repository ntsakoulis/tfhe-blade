#!/bin/bash
# Move from scripts/ to the project root and source common.sh
cd "$(dirname "$0")/.." && source scripts/common.sh

echo "=== PHASE 1: PBS-AWARE REAL STRUCTURED PRUNING ==="

# Deepest PBS target (the max_acc_drop guard may stop earlier); integer tag for file names
RATIO=0.3
RATIO_TAG=30
# Cost exponent: 1 = PBS-aware Taylor, 0 = plain Taylor (ablation)
ALPHA=1
# Minimum fraction of channels kept in every layer
MIN_FRAC=0.25

mkdir -p checkpoints_struct results_struct results_fhe_struct

for M in B A; do
    for conf in "${CONFIGS[@]}"; do
        read W A IN NR TAG <<< "$conf"
        SRC="checkpoints_${M}/${M}_VGG9_W${W}_A${A}_IN${IN}_NR${NR}.pth"
        if [ ! -f "$SRC" ]; then
            echo "[MISSING] ${SRC} (skipped)"
            continue
        fi
        NAME="${M}_VGG9_W${W}_A${A}_IN${IN}_NR${NR}_SP${RATIO_TAG}_a${ALPHA}"

        # 1. Structured pruning + fine-tuning (final model + one snapshot per accepted round)
        run "Prune_${M}_${TAG}_SP${RATIO_TAG}_a${ALPHA}" real_pruning.py \
            --ckpt "$SRC" \
            --target_pbs_ratio $RATIO \
            --alpha $ALPHA \
            --min_ch_frac $MIN_FRAC

        # 2. Accuracy-PBS curve: every round snapshot, rounding 8 only
        for SNAP in checkpoints_struct/${NAME}_r*_pbs*.pth; do
            [ -f "$SNAP" ] || continue
            SN=$(basename "$SNAP" .pth)
            run "Profile_${SN}_Rnd8" fhe_eval.py \
                --ckpt "$SNAP" \
                --method ${M}_struct_round \
                --tag "Profile_${SN}_Rnd8" \
                --w_bits $W --a_bits $A --in_bit_width $IN --narrow_range $NR \
                --rounding 8 \
                --exec_n 0 \
                --out results_fhe_struct
        done

        # 3. Final pruned model (after the final fine-tune): all rounding settings
        for R in 6 8 None; do
            run "Profile_${NAME}_Rnd${R}" fhe_eval.py \
                --ckpt "checkpoints_struct/${NAME}.pth" \
                --method ${M}_struct \
                --tag "Profile_${NAME}_Rnd${R}" \
                --w_bits $W --a_bits $A --in_bit_width $IN --narrow_range $NR \
                --rounding $R \
                --exec_n 0 \
                --out results_fhe_struct
        done
    done
done