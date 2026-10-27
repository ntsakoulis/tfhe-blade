# BLADE: PBS-Cost-Aware Pruning and Accumulator Auditing for Encrypted CNN Inference with TFHE

Code for the paper *BLADE: PBS-Cost-Aware Pruning and Accumulator Auditing for Encrypted CNN Inference with TFHE* (IEEE Access, under review).

BLADE has two parts:

1. **PBS-cost-aware structured pruning.** A cost model attributes every programmable bootstrapping (PBS) operation of a Concrete ML circuit to a channel. Channels are then removed in rounds, ranked by Taylor importance per PBS operation.
2. **Accumulator audit.** Every compiled accumulator width is compared with a certified per-neuron bound. Uncertified layers are searched for overflowing inputs (witnesses), and adversarial calibration is evaluated as a remedy.

Both parts are evaluated against the two convolutional networks published by the Concrete ML authors: VGG-9 on CIFAR-10 and the 6-layer CNN on MNIST.

## Environment

Tested on Ubuntu with Python 3.10 and an Intel Core i7-12700K (64 GB RAM).

```bash
pip install -r requirements.txt
export BLADE_DATA=/path/to/datasets      # CIFAR-10 / MNIST are downloaded here (default: ./data)
export PYTHON_BIN=python                 # interpreter used by the shell scripts
```

The latency scripts set the CPU governor to `performance` and therefore ask for `sudo`.

## Reference model (CIFAR-10)

The VGG-9 reference checkpoint is not redistributed here. Download it from the Concrete ML repository (`use_case_examples/cifar/cifar_brevitas_training`, experiment `CNV_2W2A_2W2A_20221114_131345`), then set:

```bash
export ZAMA_CKPT=/path/to/CNV_2W2A_2W2A_20221114_131345/checkpoints/best.tar
```

## Trained models

The checkpoints used in the paper are attached to the GitHub release (and the Zenodo record):

| File | Model |
|---|---|
| `ResNet18_Teacher_FP32.pth` | full-precision teacher for distillation (CIFAR-10) |
| `B_VGG9_W2_A2_IN8_NR1.pth` | Dense VGG-9 (ours) |
| `B_VGG9_W2_A2_IN8_NR1_SP30_a1_r4_pbs102109.pth` | Pruned VGG-9 (ours, round 4) |
| `B_VGG9_W2_A2_IN8_NR1_SP30_a1_r4_pbs102109_l1_B10.pth` | Pruned VGG-9 with certified l1 restriction (Appendix A) |
| `Dense_CNN6_W2_A2_IN2_NR1.pth` | Dense 6-layer CNN (ours) |
| `Dense_CNN6_W2_A2_IN2_NR1_SP30_a1.pth` | Pruned 6-layer CNN (ours) |
| `Zama_MNIST_CNN6_W2_A2_NR0_K190_S0.pth` | reconstructed MNIST reference model |

## Reproducing the paper

All CIFAR-10 scripts are run from `cifar10/`:

```bash
cd cifar10
bash scripts/00_phase0_B.sh        # teacher + Dense QAT (with distillation) + compilation
bash scripts/01_phase1_struct.sh   # PBS-cost-aware pruning (alpha = 1) and plain Taylor (alpha = 0)
bash scripts/03_p1_baselines.sh    # pruning baselines: uniform l1-norm, l1-norm on conv1-2 only
bash scripts/02_phase2_accum.sh    # certified per-neuron l1 restriction (Appendix A)
bash scripts/05_fhe_infer.sh       # encrypted latency: 3 images per model at 6-bit rounding, Pruned at 8 bits
```

Compilation, simulation, and audit of a single checkpoint:

```bash
# rounding: None, 8 or 6; --calib_adv N adds N adversarial calibration inputs; --dump_graph writes the compiled graph
python fhe_eval.py --ckpt <ckpt> --method <name> --tag <tag> --w_bits 2 --a_bits 2 --in_bit_width 8 \
                   --narrow_range 1 --rounding 6 --calib_n 500 --dump_graph
python overflow_test.py --ckpt <ckpt> --fhe_json results_fhe/<tag>.json      # natural / synthetic / adversarial overflows
python sim_repeat.py --ckpt <ckpt> --tag <tag> --rounding 6                   # variability of simulation under approximate rounding
python zama_vgg_cifar10.py --mode eval --ckpt "$ZAMA_CKPT" --tag Zama --rounding 6
```

The whole MNIST pipeline (reference reconstruction, Dense, pruning, compilation, audit, latency) is one script:

```bash
cd mnist
bash run_mnist.sh
```

### Where each result comes from

| Paper | Script |
|---|---|
| Fig. 2 (weights, MACs, PBS per layer group), PBS cost per channel | `accum.py` via `fhe_eval.py` (PBS check: theory vs. compiler) |
| Tables 2-3 (accuracy, PBS, complexity per rounding) | `fhe_eval.py` on Reference / Dense / Pruned; MNIST: `run_mnist.sh` |
| Fig. 3 (accuracy vs. complexity) | same JSONs as Tables 2-3 |
| Fig. 4 (channels retained) | `01_phase1_struct.sh`, `run_mnist.sh` |
| Fig. 5 (pruning ablation) | `01_phase1_struct.sh` (alpha = 0, 1), `03_p1_baselines.sh` |
| Table 4 (latency) | `05_fhe_infer.sh`; MNIST: `run_mnist.sh` |
| Fig. 6 (certified vs. compiled widths) | `fhe_eval.py --dump_graph` |
| Tables 5-6 (audit, adversarial calibration) | `overflow_test.py`, `fhe_eval.py --calib_adv` |
| Appendix A | `02_phase2_accum.sh` |

The JSON files behind every table are in `results/`.

## Citation

```bibtex
@article{tsakoulis2026blade,
  author  = {Tsakoulis, Thanasis},
  title   = {{BLADE}: PBS-Cost-Aware Pruning and Accumulator Auditing for Encrypted {CNN} Inference with {TFHE}},
  journal = {IEEE Access},
  year    = {2026},
  note    = {under review}
}
```
