#!/bin/bash
# =============================================================================
# PHASE 5: Real FHE inference latency (final paper measurements)
#   - R6: 3 images per model (Pruned, Dense, Zama)  -> mean +/- std, FHE == Sim
#   - R8: 1 image for Pruned only                     -> validates the complexity
#                                                        proxy at a second rounding level
# Estimated total: ~11.5 h. Models run sequentially, with a 180 s cool-down
# after each FHE run (handled by run() in common.sh).
#
# Launch (detached, survives closing the terminal):
#   sudo -v && nohup bash scripts/05_fhe_infer.sh > terminal_logs/05_fhe_infer.out 2>&1 &
# Follow progress:
#   tail -f terminal_logs/05_fhe_infer.out
# =============================================================================
cd "$(dirname "$0")/.." && source scripts/common.sh

echo "=== PHASE 5: FHE INFERENCE LATENCY ==="

# ---------------------------------------------------------------------------
# Checkpoints used for the paper results (taken from the earlier latency JSONs)
# ---------------------------------------------------------------------------
PRUNED_CKPT="checkpoints_struct_s10/B_VGG9_W2_A2_IN8_NR1_SP30_a1_r4_pbs102109.pth"
DENSE_CKPT="checkpoints_B/B_VGG9_W2_A2_IN8_NR1.pth"
ZAMA_CKPT="${ZAMA_CKPT:-checkpoints_zama/best.tar}"
ZAMA_SCRIPT="zama_vgg_cifar10.py"

# Same calibration size as the accuracy runs, so that the compiled circuits
# (bit-widths, PBS count, complexity) are identical to those in the paper
CALIB_N=500
N_IMG=3          # 3 images, classes 0, 1, 2 (class-balanced selection)
SIM_N=100        # simulation accuracy is already measured on 10k; keep this short

OUT="results_fhe_latency"
mkdir -p "$OUT"

for f in "$PRUNED_CKPT" "$DENSE_CKPT" "$ZAMA_CKPT" "$ZAMA_SCRIPT"; do
    if [ ! -f "$f" ]; then
        echo "[ERROR] Missing file: $f  (edit the paths at the top of this script)"
        exit 1
    fi
done

# Common arguments for our networks (W2/A2, 8-bit input, narrow range)
OURS="--w_bits 2 --a_bits 2 --in_bit_width 8 --narrow_range 1 \
      --calib_n $CALIB_N --sim_n $SIM_N --no_warmup --out $OUT"

# ---------------------------------------------------------------------------
# 1) R6, 3 images per model (fastest model first)
# ---------------------------------------------------------------------------
run "Lat_Pruned_R6_n${N_IMG}" fhe_eval.py \
    --ckpt "$PRUNED_CKPT" --method B_struct --tag "Lat_Pruned_R6_n${N_IMG}" \
    --rounding 6 --exec_n $N_IMG $OURS

run "Lat_Dense_R6_n${N_IMG}" fhe_eval.py \
    --ckpt "$DENSE_CKPT" --method B_kd --tag "Lat_Dense_R6_n${N_IMG}" \
    --rounding 6 --exec_n $N_IMG $OURS

run "Lat_Zama_R6_n${N_IMG}" "$ZAMA_SCRIPT" \
    --mode eval --ckpt "$ZAMA_CKPT" --tag "Lat_Zama_R6_n${N_IMG}" \
    --rounding 6 --calib_n $CALIB_N --sim_n $SIM_N \
    --exec_n $N_IMG --no_warmup --out "$OUT"

# ---------------------------------------------------------------------------
# 2) R8, 1 image, Pruned only (proxy validation at 8-bit rounding, ~3.4 h)
# ---------------------------------------------------------------------------
run "Lat_Pruned_R8_n1" fhe_eval.py \
    --ckpt "$PRUNED_CKPT" --method B_struct --tag "Lat_Pruned_R8_n1" \
    --rounding 8 --exec_n 1 $OURS

# ---------------------------------------------------------------------------
# Summary: latency, std, keygen, FHE == Sim, complexity and seconds per unit
# ---------------------------------------------------------------------------
"$PYTHON_BIN" - "$OUT" <<'EOF'
import glob, json, os, sys
out = sys.argv[1]
print(f"\n{'run':28s} {'n':>2s} {'mean_s':>9s} {'std_s':>7s} {'keygen_s':>9s} "
      f"{'complexity':>11s} {'s/unit':>9s}  fhe==sim")
for p in sorted(glob.glob(os.path.join(out, "Lat_*.json"))):
    r = json.load(open(p))
    # Records may be flat or nested per rounding setting
    recs = [r] if "latency_mean_s" in r or "exec_times_s" in r else \
           [v for v in r.values() if isinstance(v, dict) and "exec_times_s" in v]
    for x in recs:
        t = x.get("exec_times_s") or []
        mean = x.get("latency_mean_s", sum(t) / len(t) if t else float("nan"))
        std = x.get("latency_std_s", float("nan"))
        cx = x.get("complexity")
        spu = mean / float(cx) if cx else float("nan")
        same = x.get("fhe_matches_sim", x.get("fhe_eq_sim", "see log"))
        print(f"{os.path.basename(p)[:-5]:28s} {len(t):2d} {mean:9.1f} {std:7.1f} "
              f"{str(x.get('keygen_s', '-')):>9s} {str(cx):>11s} {spu:9.2e}  {same}")
EOF

echo -e "\n=== [$(date '+%F %T')] PHASE 5 DONE ==="
