#!/bin/bash
set -e
set -o pipefail

# Explicit Python interpreter in Conda environment
PYTHON_BIN="${PYTHON_BIN:-python}"

# Ask for sudo password once and keep-alive
sudo -v
while true; do sudo -n true; sleep 60; done 2>/dev/null &
SUDO_KEEPALIVE_PID=$!

restore_gov() {
    echo powersave | sudo -n tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null || true
    kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
}
trap restore_gov EXIT
trap "echo -e '\n[INTERRUPTED]'; exit 1" INT TERM

echo "Setting all CPUs to PERFORMANCE mode..."
echo performance | sudo -n tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONWARNINGS="ignore"
export PYTHONUNBUFFERED=1

# P-Cores affinity for Intel 12th gen
CORES="0,2,4,6,8,10,12,14"

# Format: W A IN NR TAG
#"2 2 2 1 M3_W2A2_IN2"
#"4 4 8 1 M4_W4A4_IN8"
#"4 4 4 1 M5_W4A4_IN4"
#"2 2 4 1 M2_W2A2_IN4"

CONFIGS=(
    "2 2 8 1 M1_W2A2_IN8"
)

mkdir -p terminal_logs checkpoints_A results_A results_fhe_A

run() {
    local name="$1"; shift
    local log="terminal_logs/${name}.log"

    if [ -f "terminal_logs/${name}.DONE" ]; then
        echo "[SKIP] ${name} (already done)"
        return 0
    fi

    echo -e "\n=== [$(date '+%F %T')] ${name} ==="

        if taskset -c "$CORES" "$PYTHON_BIN" "$@" 2>&1 | tee "$log"; then
        touch "terminal_logs/${name}.DONE"
        # Cooldown after every run (training, pruning, compile, FHE execution)
        echo "-> Cooldown: Sleeping for 3 minutes to prevent thermal throttling..."
        sleep 180
    else
        echo "[FAILED] ${name} — check ${log}"
        exit 1
    fi
}