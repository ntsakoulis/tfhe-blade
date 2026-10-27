"""Check whether FHE simulation is stochastic under approximate rounding.

Compiles a model exactly as fhe_eval.py does (same calibration, same rounding),
then runs fhe="simulate" several times on the same images and reports:
  - the predictions on the three images used for the FHE latency runs,
  - the disagreement rate between independent simulation runs on n test images.
If independent simulations disagree with each other, FHE == Sim cannot be
expected to hold image by image, and the FHE check must be read statistically.
"""
import argparse
import json
import os
import numpy as np
import torch

import accum
import arc_cifar10 as arc
import utils_cifar10 as utils

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--rounding", type=int, default=6)
p.add_argument("--calib_n", type=int, default=500)
p.add_argument("--n", type=int, default=1000)
p.add_argument("--reps", type=int, default=5)
p.add_argument("--out", default="results_fhe_latency")
a = p.parse_args()

utils.set_seed(0)
_, val_l, test_l = utils.get_data_loaders(batch_size=128, augmentation=False)

# Pruned checkpoints store their architecture (cfg, cls_cfg); dense ones use the default
ck = torch.load(a.ckpt, map_location="cpu")
cfg = ck.get("cfg") if isinstance(ck, dict) else None
cls_cfg = ck.get("cls_cfg") if isinstance(ck, dict) else None
print(f"[{a.tag}] cfg={cfg} cls_cfg={cls_cfg}")
model = arc.nnet_cifar10(cfg=cfg, cls_cfg=cls_cfg, w_bit_width=2, act_bit_width=2, in_bit_width=8,
                         narrow_range=True, unsigned_requant=False)
accum.load_state(model, a.ckpt)
model.eval()

from concrete.fhe import Configuration, ParameterSelectionStrategy
from concrete.ml.torch.compile import compile_brevitas_qat_model

calib = utils.get_calibration_set(val_l, num_samples=a.calib_n)
xs, ys = [], []
for x, y in test_l:
    xs.append(x)
    ys.append(y)
    if sum(t.shape[0] for t in xs) >= a.n:
        break
X = np.ascontiguousarray(torch.cat(xs)[:a.n].numpy(), dtype=np.float32)
Y = torch.cat(ys)[:a.n].numpy()

cfg = Configuration(parameter_selection_strategy=ParameterSelectionStrategy.MULTI)
qm = compile_brevitas_qat_model(model, calib, configuration=cfg,
                                rounding_threshold_bits={"n_bits": a.rounding, "method": "APPROXIMATE"})

# Same three images as the latency runs: first test image of classes 0, 1, 2
exec_idx = [int(np.where(Y == c)[0][0]) for c in range(3)]

p_dis = np.concatenate([qm.forward(X[i:i + 100], fhe="disable").argmax(1) for i in range(0, a.n, 100)])
sims = []
for r in range(a.reps):
    sims.append(np.concatenate([qm.forward(X[i:i + 100], fhe="simulate").argmax(1)
                                for i in range(0, a.n, 100)]))
    print(f"  sim run {r + 1}/{a.reps}: acc={float((sims[-1] == Y).mean()):.4f} "
          f"exec images -> {sims[-1][exec_idx].tolist()}")
sims = np.stack(sims)

pair = [float((sims[i] != sims[j]).mean()) for i in range(a.reps) for j in range(i + 1, a.reps)]
unstable = float((sims != sims[0]).any(0).mean())
rec = dict(tag=a.tag, rounding=a.rounding, n=a.n, reps=a.reps,
           labels_exec=Y[exec_idx].tolist(), disable_exec=p_dis[exec_idx].tolist(),
           sim_exec=sims[:, exec_idx].tolist(),
           acc_sim_runs=[float((s == Y).mean()) for s in sims],
           disagree_sim_vs_sim_mean=float(np.mean(pair)),
           frac_images_unstable=unstable,
           disagree_sim_vs_disable_mean=float(np.mean([(s != p_dis).mean() for s in sims])))
print(f"[{a.tag}] labels={rec['labels_exec']} disable={rec['disable_exec']}")
print(f"[{a.tag}] sim-vs-sim disagreement: {rec['disagree_sim_vs_sim_mean']:.4f} | "
      f"images whose prediction changes across runs: {unstable:.4f} | "
      f"sim-vs-disable: {rec['disagree_sim_vs_disable_mean']:.4f}")
os.makedirs(a.out, exist_ok=True)
json.dump(rec, open(os.path.join(a.out, f"SimRepeat_{a.tag}.json"), "w"), indent=1)