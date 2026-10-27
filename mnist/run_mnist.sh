#!/bin/bash
# =============================================================================
# BLADE on the second network: the 6-layer CNN of Stoian et al. (2023) on MNIST
#
# Three models with the SAME topology (4 conv 8-8-16-16, FC 120, W2/A2, 2-bit input):
#   Zama   : their recipe, L1 unstructured pruning (n_active = 190)   -> zama_mnist_cnn.py
#   Dense  : our QAT, no pruning                                       -> qat_mnist.py
#   BLADE  : Dense + our PBS-aware structured pruning, pruned round by round (-10% PBS each)
#            until the validation accuracy drops by more than 1 point    -> real_pruning.py
# Every model: compile without rounding, 8-bit and 6-bit rounding (simulation on 10k test images),
# overflow audit, adversarial calibration, and FHE latency on 3 images at 6-bit rounding.
#
# Launch (detached):
#   cd mnist && sudo -v && nohup bash run_mnist.sh > terminal_logs/run_mnist.out 2>&1 &
# Every finished step leaves terminal_logs/<step>.DONE and is skipped on a re-run.
# =============================================================================
set -e
set -o pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${BLADE_DATA:-./data}"
CORES="0,2,4,6,8,10,12,14"          # P-cores of the i7-12700K
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONWARNINGS="ignore" PYTHONUNBUFFERED=1

# Performance governor during the run, restored on exit
sudo -v
while true; do sudo -n true; sleep 60; done 2>/dev/null &
KEEPALIVE=$!
restore() {
    echo powersave | sudo -n tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null || true
    kill "$KEEPALIVE" 2>/dev/null || true
}
trap restore EXIT
echo performance | sudo -n tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null

mkdir -p terminal_logs
run() {
    local name="$1"; shift
    if [ -f "terminal_logs/${name}.DONE" ]; then echo "[SKIP] ${name}"; return 0; fi
    echo -e "\n=== [$(date '+%F %T')] ${name} ==="
    if taskset -c "$CORES" "$PYTHON_BIN" "$@" 2>&1 | tee "terminal_logs/${name}.log"; then
        touch "terminal_logs/${name}.DONE"
    else
        echo "[FAILED] ${name} -- see terminal_logs/${name}.log"; exit 1
    fi
}

DENSE="checkpoints_dense/Dense_CNN6_W2_A2_IN2_NR1.pth"
BLADE="checkpoints_struct/Dense_CNN6_W2_A2_IN2_NR1_SP30_a1.pth"
ZAMA="checkpoints_zama/Zama_MNIST_CNN6_W2_A2_NR0_K190_S0.pth"
EVAL="--w_bits 2 --a_bits 2 --in_bit_width 2 --narrow_range 1 --calib_n 1000 --sim_n 10000"

# --- 1. Zama baseline: train with their recipe, then compile / simulate --------------------------
run M01_Zama_train_eval zama_mnist_cnn.py --mode all --n_active 190 --bits 2 --narrow 0 \
    --data_root "$DATA_ROOT" --rounding None,8,6 --calib_n 1000 --sim_n 10000 --exec_n 0

# --- 2. Dense: our QAT ---------------------------------------------------------------------------
run M02_Dense_train qat_mnist.py --w_bits 2 --a_bits 2 --in_bits 2 --epochs 60

# --- 3. BLADE: PBS-aware pruning of Dense, as far as accuracy allows ------------------------------
# Every round removes 10% of the remaining PBS and fine-tunes. The loop stops at the first round whose
# validation accuracy is more than 1 point below Dense (that round is rolled back), when the minimum
# width per layer is reached, or at 30% of the dense PBS. Every accepted round is also saved as
# checkpoints_struct/..._r<k>_pbs<N>.pth.
run M03_BLADE_prune real_pruning.py --ckpt "$DENSE" --target_pbs_ratio 0.30 --step 0.10 \
    --alpha 1 --max_acc_drop 0.01 --max_rounds 20

# --- 4. Compile + simulate Dense and BLADE (default calibration) --------------------------------
for R in None 8 6; do
    run M04_Dense_Rnd$R fhe_eval.py --ckpt "$DENSE" --method dense --tag Dense_Rnd$R --rounding $R $EVAL
    run M04_BLADE_Rnd$R fhe_eval.py --ckpt "$BLADE" --method blade --tag BLADE_Rnd$R --rounding $R $EVAL
done

# --- 5. Overflow audit against the compiled bit-widths -------------------------------------------
run M05_Dense_overflow overflow_test.py --ckpt "$DENSE" --fhe_json results_fhe/Dense_RndNone.json
run M05_BLADE_overflow overflow_test.py --ckpt "$BLADE" --fhe_json results_fhe/BLADE_RndNone.json

# --- 6. Adversarial calibration (safe mode) ------------------------------------------------------
for R in None 8 6; do
    run M06_Dense_cadv_Rnd$R fhe_eval.py --ckpt "$DENSE" --method dense_cadv --tag Dense_cadv_Rnd$R \
        --rounding $R --calib_adv 64 $EVAL
    run M06_BLADE_cadv_Rnd$R fhe_eval.py --ckpt "$BLADE" --method blade_cadv --tag BLADE_cadv_Rnd$R \
        --rounding $R --calib_adv 64 $EVAL
done
run M06_Dense_cadv_overflow overflow_test.py --ckpt "$DENSE" --fhe_json results_fhe/Dense_cadv_RndNone.json
run M06_BLADE_cadv_overflow overflow_test.py --ckpt "$BLADE" --fhe_json results_fhe/BLADE_cadv_RndNone.json

# --- 7. FHE latency: 3 images at 6-bit rounding, all three models ---------------------------------
LAT="--w_bits 2 --a_bits 2 --in_bit_width 2 --narrow_range 1 --calib_n 1000 --sim_n 100 --exec_n 3 --no_warmup"
run M07_BLADE_latency fhe_eval.py --ckpt "$BLADE" --method blade --tag BLADE_lat_Rnd6 --rounding 6 $LAT
run M07_Dense_latency fhe_eval.py --ckpt "$DENSE" --method dense --tag Dense_lat_Rnd6 --rounding 6 $LAT
run M07_Zama_latency zama_mnist_cnn.py --mode eval --ckpt "$ZAMA" --tag Zama_MNIST_CNN6_lat --bits 2 --narrow 0 \
    --n_active 190 --data_root "$DATA_ROOT" --rounding 6 --calib_n 1000 --sim_n 100 --exec_n 3 --no_warmup

echo -e "\n=== [$(date '+%F %T')] MNIST PIPELINE DONE ==="