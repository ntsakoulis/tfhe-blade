#!/bin/bash
# Move from scripts/ to the project root and source common.sh
cd "$(dirname "$0")/.." && source scripts/common.sh

echo "=== PHASE 2: ACCUMULATOR RESTRICTION (l1 / fanin / zama) ==="

# Input checkpoints: dense (phase 0) and/or pruned (phase 1). Comment out what you do not need.
INPUTS=(
    "checkpoints_struct_s10/B_VGG9_W2_A2_IN8_NR1_SP30_a1_r4_pbs102109.pth"
    # "checkpoints_B/B_VGG9_W2_A2_IN8_NR1.pth"
)
# Constraints compared at the SAME target bit-width (start with l1 only)
CONSTRAINTS=(l1)
# Target accumulator bit-widths
B_TARGETS=(10)
# Rounding settings for the FHE evaluation
ROUNDINGS=(None 8 6)

mkdir -p checkpoints_accum results_accum results_fhe_accum

for SRC in "${INPUTS[@]}"; do
    if [ ! -f "$SRC" ]; then
        echo "[MISSING] ${SRC} (skipped)"
        continue
    fi
    BASE=$(basename "$SRC" .pth)
    # Bit-widths from the file name: <M>_VGG9_W<w>_A<a>_IN<in>_NR<nr>...
    if [[ "$BASE" =~ ^([A-Za-z]+)_VGG9_W([0-9]+)_A([0-9]+)_IN([0-9]+)_NR([0-9]) ]]; then
        M=${BASH_REMATCH[1]}; W=${BASH_REMATCH[2]}; A=${BASH_REMATCH[3]}
        IN=${BASH_REMATCH[4]}; NR=${BASH_REMATCH[5]}
    else
        echo "[SKIP] cannot parse bit-widths from ${BASE}"
        continue
    fi

    for C in "${CONSTRAINTS[@]}"; do
        for B in "${B_TARGETS[@]}"; do
            NAME="${BASE}_${C}_B${B}"

            # 1. Accumulator restriction + fine-tuning
            run "Accum_${NAME}" accum_restrict.py \
                --ckpt "$SRC" \
                --constraint $C \
                --b_target $B

            # 2. FHE compile + simulation
            for R in "${ROUNDINGS[@]}"; do
                run "Profile_${NAME}_Rnd${R}" fhe_eval.py \
                    --ckpt "checkpoints_accum/${NAME}.pth" \
                    --method ${M}_accum_${C} \
                    --tag "Profile_${NAME}_Rnd${R}" \
                    --w_bits $W --a_bits $A --in_bit_width $IN --narrow_range $NR \
                    --B_target $B \
                    --rounding $R \
                    --sim_n 10000 \
                    --exec_n 0 \
                    --out results_fhe_accum
            done
        done
    done
done
