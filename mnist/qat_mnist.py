"""Dense QAT of the MNIST CNN (step 1 of the pipeline, the counterpart of qat_cifar10_A.py).

Saves checkpoints_dense/Dense_CNN6_W{w}_A{a}_IN{in}_NR{nr}.pth (best validation accuracy) and a JSON
with the training curves.

Example:
  python qat_mnist.py --w_bits 2 --a_bits 2 --in_bits 2 --epochs 60
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

import arc_mnist as arc
import utils_mnist as utils

p = argparse.ArgumentParser(description="Dense QAT of the MNIST CNN")
p.add_argument("--w_bits", type=int, default=2)
p.add_argument("--a_bits", type=int, default=2)
p.add_argument("--in_bits", type=int, default=2)
p.add_argument("--narrow_range", type=int, default=1)
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--save_dir", type=str, default="checkpoints_dense")
p.add_argument("--results_dir", type=str, default="results_dense")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
utils.set_seed(args.seed)
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.results_dir, exist_ok=True)

base = f"Dense_CNN6_W{args.w_bits}_A{args.a_bits}_IN{args.in_bits}_NR{args.narrow_range}"
ckpt_path = os.path.join(args.save_dir, f"{base}.pth")

train_l, val_l, test_l = utils.get_data_loaders(batch_size=args.batch)
model = arc.nnet_mnist(w_bit_width=args.w_bits, act_bit_width=args.a_bits, in_bit_width=args.in_bits,
                       narrow_range=bool(args.narrow_range)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
stopper = utils.BestCheckpoint(path=ckpt_path)

print(f"=== DENSE QAT (MNIST) | {base} | device={device} ===", flush=True)
curves = dict(epoch=[], train_acc=[], val_acc=[], lr=[])
t_start = time.time()
for ep in range(args.epochs):
    _, t_acc, t_time = utils.training_phase(model, train_l, criterion, optimizer, device, clip_weights=True)
    _, v_acc = utils.evaluate(model, val_l, criterion, device)
    curves["epoch"].append(ep + 1)
    curves["train_acc"].append(t_acc)
    curves["val_acc"].append(v_acc)
    curves["lr"].append(float(optimizer.param_groups[0]["lr"]))
    print(f"Ep[{ep + 1:03d}/{args.epochs}] t_acc={t_acc:.4f} v_acc={v_acc:.4f} ({t_time:.1f}s)", flush=True)
    scheduler.step()
    stopper(v_acc, model)

model.load_state_dict(torch.load(ckpt_path, map_location=device))
_, test_acc = utils.evaluate(model, test_l, criterion, device)
print(f"\n[DONE] {base}: best val={stopper.best_acc:.4f} test={test_acc:.4f} "
      f"({(time.time() - t_start) / 60:.1f} min)")

with open(os.path.join(args.results_dir, f"{base}_metrics.json"), "w") as f:
    json.dump(dict(model_tag=base, args=vars(args), best_val_accuracy=stopper.best_acc, test_accuracy=test_acc,
                   train_time_sec=round(time.time() - t_start, 1), curves=curves), f, indent=2)
