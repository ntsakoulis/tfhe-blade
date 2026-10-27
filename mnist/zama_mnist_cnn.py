"""Zama-style 6-layer CNN on MNIST (reconstruction): train + integer profile + FHE compile.

Zama never released code for the MNIST conv networks of Stoian et al. 2023 (arXiv:2302.10906).
This script REBUILDS the "6-layer CNN" row of their Table 1 from what the paper states and from
the conventions of their public code. It is not their code.

From the paper (Table 1 + footnote 3):
  - four conv layers with 8, 8, 16, 16 filters, one FC layer with 120 neurons, one final FC layer
  - 2-bit weights / 2-bit activations, wide (non-narrow) range, 190 active neurons
  - reported: 8-bit circuit, 98.7% accuracy
From Zama's repository:
  - quantizers and layer style: use_case_examples/mnist/model.py and the CIFAR CNV model
    (constant scale 1.0, signed QuantIdentity activations after BatchNorm, no ReLU, no conv bias,
    AvgPool followed by a QuantIdentity, weights initialised with uniform(-1, 1))
  - pruning: torch l1_unstructured with amount = (fan_in - n_active) * fan_out, applied BEFORE
    training on every layer whose fan-in exceeds n_active (TinyQATCNN / MNISTQATModel logic).
    This is a layer-wide budget (mean fan-in), not a per-neuron or per-filter limit.
  - training: Adam lr=0.02, wd=1e-4, StepLR(step=1, gamma=0.33), 20 epochs, batch 32 (MNIST notebook)
  - no bias in any conv / linear layer (all QuantLinear(..., False, ...) in the release-0.5.x model)
NOT stated anywhere, chosen here (all exposed as arguments where it matters):
  - 3x3 kernels, no padding, 2x2 AvgPool after conv2 and conv4  ->  16*4*4 = 256 features
  - BatchNorm momentum 0.1 (the MNIST notebook uses 0.999, which makes evaluation very noisy)

Example:
  python zama_mnist_cnn.py --n_active 190 --bits 2 --narrow 0 --rounding None,8,6
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import FloatToIntImplType, RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear
from brevitas.quant.solver import ActQuantSolver, WeightQuantSolver
from dependencies import value
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

CONV_CFG = [(8, False), (8, True), (16, False), (16, True)]  # (filters, AvgPool after)
FC_HIDDEN = 120
KERNEL_SIZE = 3


# ----------------------------------------------------------------------------
# Quantizers (same as Zama's MNIST / CIFAR examples)
# ----------------------------------------------------------------------------
class CommonQuant(ExtendedInjector):
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_per_output_channel = False
    narrow_range = True
    signed = True

    @value
    def quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        if bit_width == 1:
            return QuantType.BINARY
        return QuantType.INT


class CommonWeightQuant(CommonQuant, WeightQuantSolver):
    scaling_const = 1.0


class CommonActQuant(CommonQuant, ActQuantSolver):
    min_val = -1.0
    max_val = 1.0


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class ZamaStyleMNISTCNN(nn.Module):
    def __init__(self, bits=2, narrow=False, n_active=190, bn_momentum=0.1, fc_bias=False, num_classes=10):
        super().__init__()
        self.bits, self.narrow, self.n_active = bits, bool(narrow), int(n_active)
        self.features = nn.ModuleList()
        self.classifier = nn.ModuleList()

        def act():
            return QuantIdentity(act_quant=CommonActQuant, bit_width=bits, narrow_range=self.narrow)

        # As in Zama's MNIST example, the input is quantized with the activation bit-width
        self.features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=bits,
                                           narrow_range=self.narrow, return_quant_tensor=True))
        in_ch, size = 1, 28
        for out_ch, pool in CONV_CFG:
            self.features.append(QuantConv2d(in_ch, out_ch, KERNEL_SIZE, bias=False,
                                             weight_quant=CommonWeightQuant, weight_bit_width=bits,
                                             weight_narrow_range=self.narrow))
            self.features.append(nn.BatchNorm2d(out_ch, momentum=bn_momentum))
            self.features.append(act())
            in_ch, size = out_ch, size - KERNEL_SIZE + 1
            if pool:
                self.features.append(nn.AvgPool2d(kernel_size=2))
                self.features.append(act())  # re-quantization after pooling, as in Zama's CNV
                size //= 2
        self.flat_dim = in_ch * size * size

        self.classifier.append(QuantLinear(self.flat_dim, FC_HIDDEN, bias=bool(fc_bias),
                                           weight_quant=CommonWeightQuant, weight_bit_width=bits,
                                           weight_narrow_range=self.narrow))
        self.classifier.append(nn.BatchNorm1d(FC_HIDDEN, momentum=bn_momentum))
        self.classifier.append(act())
        # No bias in any layer, as in Zama's MNIST model (release 0.5.x: QuantLinear(..., False, ...))
        self.classifier.append(QuantLinear(FC_HIDDEN, num_classes, bias=False,
                                           weight_quant=CommonWeightQuant, weight_bit_width=bits,
                                           weight_narrow_range=self.narrow))

        for m in self.modules():
            if isinstance(m, (QuantConv2d, QuantLinear)):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def weight_layers(self):
        return [m for m in list(self.features) + list(self.classifier)
                if isinstance(m, (QuantConv2d, QuantLinear))]

    def toggle_pruning(self, enable):
        """Zama's pruning rule: keep n_active * fan_out weights in every layer with fan_in > n_active."""
        pruned = []
        for layer in self.weight_layers():
            fan_out, fan_in = layer.weight.shape[0], int(np.prod(layer.weight.shape[1:]))
            if fan_in > self.n_active:
                if enable:
                    prune.l1_unstructured(layer, "weight", (fan_in - self.n_active) * fan_out)
                elif prune.is_pruned(layer):
                    prune.remove(layer, "weight")
                pruned.append((fan_in, fan_out))
        return pruned

    def forward(self, x):
        for mod in self.features:
            x = mod(x)
        x = torch.flatten(x, 1)
        for mod in self.classifier:
            x = mod(x)
        return x


# ----------------------------------------------------------------------------
# Data / training (Zama MNIST notebook recipe)
# ----------------------------------------------------------------------------
def get_sets(data_root):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    return (datasets.MNIST(data_root, train=True, download=True, transform=tf),
            datasets.MNIST(data_root, train=False, download=True, transform=tf))


def _tensor(out):
    return out.value if hasattr(out, "value") else out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (_tensor(model(x)).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def train(args, model, device, train_set, test_set):
    kw = dict(num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True, **kw)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False, **kw)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    curves = dict(epoch=[], train_loss=[], train_acc=[], test_acc=[], lr=[])

    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        t0, run_loss, correct, total = time.time(), 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = _tensor(model(x))
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        acc = evaluate(model, test_loader, device)
        curves["epoch"].append(epoch + 1)
        curves["train_loss"].append(run_loss / total)
        curves["train_acc"].append(correct / total)
        curves["test_acc"].append(acc)
        curves["lr"].append(optimizer.param_groups[0]["lr"])
        print(f"Ep[{epoch + 1:02d}/{args.epochs}] loss={run_loss / total:.4f} t_acc={correct / total:.4f} "
              f"test_acc={acc:.4f} lr={curves['lr'][-1]:.6f} ({time.time() - t0:.1f}s)", flush=True)
        scheduler.step()
    # As in Zama's notebook there is no checkpoint selection: the final model is used
    return dict(train_time_sec=round(time.time() - t_start, 1), epochs_trained=args.epochs, curves=curves)


# ----------------------------------------------------------------------------
# Integer profile (same conventions as accum.py: B_cert = certified worst case, B_emp = observed)
# ----------------------------------------------------------------------------
def _bits_signed(bound):
    """Two's complement bits needed for the range [-bound, bound]."""
    if bound <= 0:
        return 1
    return 1 + max(math.ceil(math.log2(bound + 1)), math.ceil(math.log2(bound)))


@torch.no_grad()
def integer_profile(model, loader, max_batches=10):
    """Per-layer fan-in / L1 / accumulator bits and the theoretical PBS count."""
    model.eval()
    rows, pbs_act = {}, {}
    mods = [(f"features.{i}", m) for i, m in enumerate(model.features)] + [("flatten", None)] + \
           [(f"classifier.{i}", m) for i, m in enumerate(model.classifier)]
    for bi, (x, _) in enumerate(loader):
        if bi >= max_batches:
            break
        cur, first_act = None, True
        for name, m in mods:
            if m is None:
                x = torch.flatten(_tensor(x), 1)
            elif isinstance(m, QuantIdentity):
                x = m(x)
                bits, scale = int(m.quant_act_bit_width()), float(m.quant_act_scale())
                # Signed activations: largest magnitude is 2^(b-1) (wide) or 2^(b-1)-1 (narrow)
                cur = (2 ** (bits - 1) - (1 if model.narrow else 0), scale)
                if not first_act:
                    pbs_act[name] = int(_tensor(x)[0].numel())  # one table lookup per element
                first_act = False
            elif isinstance(m, (QuantConv2d, QuantLinear)):
                out = _tensor(m(x))
                qw = m.quant_weight()
                ws = float(qw.scale.flatten()[0])
                q = torch.round(qw.value / ws).reshape(qw.value.shape[0], -1)
                lin = out
                if m.bias is not None:  # float bias is not part of the integer MAC
                    lin = out - m.bias.reshape([1, -1] + [1] * (out.dim() - 2))
                acc = lin / (cur[1] * ws)
                r = rows.setdefault(name, dict(obs=0.0, resid=0.0))
                r["obs"] = max(r["obs"], float(acc.abs().max()))
                r["resid"] = max(r["resid"], float((acc - acc.round()).abs().max()))
                if "fan_in" not in r:
                    k_mask = (m.weight.reshape(q.shape[0], -1) != 0).sum(1).float()
                    nnz, l1 = (q != 0).sum(1).float(), q.abs().sum(1)
                    fan_in = int(q.shape[1])
                    r.update(
                        name=name, kind="conv" if isinstance(m, QuantConv2d) else "fc",
                        fan_in=fan_in, out_ch=int(q.shape[0]), in_max=float(cur[0]),
                        pruned=fan_in > model.n_active, n_active_target=min(model.n_active, fan_in),
                        Kmask_mean=float(k_mask.mean()), Kmask_max=int(k_mask.max()), Kmask_min=int(k_mask.min()),
                        K_mean=float(nnz.mean()), K_max=int(nnz.max()), K_min=int(nnz.min()),
                        L1_mean=float(l1.mean()), L1_max=float(l1.max()),
                        B_cert_max=_bits_signed(cur[0] * float(l1.max())),
                        B_cert_mean=float(np.mean([_bits_signed(cur[0] * float(v)) for v in l1])),
                        int_sparsity=float(1 - nnz.sum() / q.numel()),
                        w_int_max=int(q.abs().max()), has_float_bias=m.bias is not None,
                    )
                x = out
            else:
                x = m(_tensor(x))
    layers = []
    for r in rows.values():
        obs = r.pop("obs")
        r["B_emp"] = 1 + math.ceil(math.log2(obs + 1)) if obs > 0 else 1
        r["int_residual"] = round(r.pop("resid"), 6)  # ~0 means the integer scales are right
        layers.append(r)
    return layers, pbs_act


# ----------------------------------------------------------------------------
# FHE compile / simulate / execute
# ----------------------------------------------------------------------------
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    return str(o)


def fhe_measure(model, calib, X, Y, rnd_val, args, rec):
    from concrete.fhe import Configuration, ParameterSelectionStrategy
    from concrete.ml.torch.compile import compile_brevitas_qat_model

    cfg = Configuration(parameter_selection_strategy=ParameterSelectionStrategy.MULTI)
    kwargs = dict(configuration=cfg,
                  rounding_threshold_bits={"n_bits": rnd_val, "method": "APPROXIMATE"} if rnd_val else None)
    if args.p_error is not None:
        kwargs["p_error"] = args.p_error
    try:
        t0 = time.time()
        qm = compile_brevitas_qat_model(model, calib, **kwargs)
        rec["compile_s"] = round(time.time() - t0, 1)
        c = qm.fhe_circuit
        st = c.statistics
        st = st() if callable(st) else st
        rec["B_meas"] = int(c.graph.maximum_integer_bit_width())
        # Dump the compiled graph to locate the widest integer nodes
        with open(os.path.join(args.out, f"{rec['tag']}_graph.txt"), "w") as f:
            f.write(c.graph.format(maximum_constant_length=8))
        rec["pbs"] = st.get("programmable_bootstrap_count")
        rec["complexity"] = getattr(c, "complexity", None)
        # Key / ciphertext sizes (bytes) and operation counts, available without key generation
        rec["circuit_stats"] = {k: v for k, v in st.items() if isinstance(v, (int, float))}
        rec["pbs_per_parameter"] = st.get("programmable_bootstrap_count_per_parameter")

        p_dis, p_sim = [], []
        for i in range(0, len(X), 100):
            xb = X[i:i + 100]
            p_dis.append(qm.forward(xb, fhe="disable").argmax(1))
            p_sim.append(qm.forward(xb, fhe="simulate").argmax(1))
        p_dis, p_sim = np.concatenate(p_dis), np.concatenate(p_sim)
        rec["sim"] = dict(n=int(len(X)), acc_disable=float((p_dis == Y).mean()),
                          acc_sim=float((p_sim == Y).mean()),
                          disagree_sim_vs_disable=float((p_sim != p_dis).mean()))
        print(f"  COMPILE OK ({rec['compile_s']}s) B_meas={rec['B_meas']} PBS={rec['pbs']} "
              f"complexity={rec['complexity']} acc_sim={rec['sim']['acc_sim']:.4f}", flush=True)
        if rec["B_meas"] > rec["B_cert_global"] + 1:
            print(f"  WARNING: B_meas ({rec['B_meas']}) is far above B_cert ({rec['B_cert_global']}). "
                  f"The compiler may not be using the Brevitas weight quantizer for some layer; "
                  f"inspect the circuit and try --fc_bias 1.")
    except Exception as e:
        rec["compile_error"] = str(e)[:1000]
        print(f"  COMPILE/SIM FAILED: {str(e)[:300]}")
        return

    if args.exec_n <= 0:
        return
    try:
        Xe, Ye = X[:args.exec_n], Y[:args.exec_n]
        t0 = time.time()
        c.keygen()
        rec["keygen_s"] = round(time.time() - t0, 2)
        try:
            rec["eval_key_size_mb"] = round(len(c.client.evaluation_keys.serialize()) / 2 ** 20, 2)
        except Exception:
            rec["eval_key_size_mb"] = None
        if not args.no_warmup:
            qm.forward(Xe[:1], fhe="execute")
        times, preds = [], []
        for i in range(len(Xe)):
            t0 = time.time()
            out = qm.forward(Xe[i:i + 1], fhe="execute")
            times.append(time.time() - t0)
            preds.append(int(out.argmax(1)[0]))
            print(f"    FHE inference {i + 1}/{len(Xe)}: {times[-1]:.2f}s", flush=True)
        sim_preds = qm.forward(Xe, fhe="simulate").argmax(1)
        rec.update(exec_n=len(Xe), exec_times_s=[round(t, 2) for t in times],
                   latency_mean_s=round(float(np.mean(times)), 2),
                   latency_std_s=round(float(np.std(times)), 2), warmup_used=not args.no_warmup,
                   fhe_matches_sim=bool((np.array(preds) == sim_preds).all()),
                   fhe_acc_exec=float((np.array(preds) == Ye).mean()))
    except Exception as e:
        rec["exec_error"] = str(e)[:1000]
        print(f"  FHE EXECUTION FAILED: {str(e)[:300]}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Zama-style 6-layer CNN on MNIST (reconstruction)")
    p.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    p.add_argument("--ckpt", type=str, default=None, help="state_dict saved by this script")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="checkpoints_zama")
    p.add_argument("--out", type=str, default="results_fhe_zama")
    p.add_argument("--tag", type=str, default="Zama_MNIST_CNN6")
    # Paper row: 2/2 bits, wide range, 190 active neurons
    p.add_argument("--bits", type=int, default=2, help="Weight and activation bit-width")
    p.add_argument("--narrow", type=int, default=0, help="1 = narrow range, 0 = wide range (paper)")
    p.add_argument("--n_active", type=int, default=190, help="Active neurons kept per layer on average")
    p.add_argument("--fc_bias", type=int, default=0, help="1 = bias in the hidden FC layer")
    p.add_argument("--bn_momentum", type=float, default=0.1)
    # Zama MNIST notebook recipe
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=0.33)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--debug_subset", type=int, default=0, help="Use only N training samples (smoke test)")
    # FHE evaluation
    p.add_argument("--rounding", type=str, default="None,8,6")
    p.add_argument("--calib_n", type=int, default=1000)
    p.add_argument("--sim_n", type=int, default=10000)
    p.add_argument("--exec_n", type=int, default=0, help="Real FHE executions (0 = skip)")
    p.add_argument("--p_error", type=float, default=None)
    p.add_argument("--no_warmup", action="store_true")
    p.add_argument("--no_compile", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = f"{args.tag}_W{args.bits}_A{args.bits}_NR{args.narrow}_K{args.n_active}_S{args.seed}"
    ckpt_path = args.ckpt or os.path.join(args.save_dir, f"{base}.pth")

    train_set, test_set = get_sets(args.data_root)
    full_train = train_set
    if args.debug_subset:
        train_set = torch.utils.data.Subset(train_set, range(args.debug_subset))

    model = ZamaStyleMNISTCNN(bits=args.bits, narrow=args.narrow, n_active=args.n_active,
                              bn_momentum=args.bn_momentum, fc_bias=args.fc_bias).to(device)
    train_info = None
    if args.mode in ("train", "all") and not args.ckpt:
        pruned = model.toggle_pruning(True)  # one-shot pruning on the random initial weights
        print(f"=== Training {base} on {device} | pruned layers (fan_in, fan_out): {pruned} ===")
        train_info = train(args, model, device, train_set, test_set)
        model.toggle_pruning(False)          # make the masks permanent
        torch.save(model.state_dict(), ckpt_path)
        with open(os.path.join(args.out, f"{base}_train.json"), "w") as f:
            json.dump(jsonable(train_info), f, indent=2)
    if args.mode == "train":
        return

    print(f"=== Evaluating {ckpt_path} ===")
    model = model.cpu()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)
    calib = np.ascontiguousarray(
        torch.stack([full_train[i][0] for i in range(args.calib_n)]).numpy(), dtype=np.float32)
    n_sim = min(args.sim_n, len(test_set))
    X = np.ascontiguousarray(torch.stack([test_set[i][0] for i in range(n_sim)]).numpy(), dtype=np.float32)
    Y = np.array([test_set[i][1] for i in range(n_sim)])

    acc_test = evaluate(model, test_loader, "cpu")
    layers, pbs_act = integer_profile(model, test_loader, max_batches=10)
    base_rec = dict(
        tag=base, method="zama_style_mnist_cnn6_reconstruction", ckpt=ckpt_path, seed=args.seed,
        setting=dict(w=args.bits, a=args.bits, inp=args.bits, narrow=bool(args.narrow), signed_act=True),
        n_active=args.n_active, fc_bias=bool(args.fc_bias), bn_momentum=args.bn_momentum,
        time=time.strftime("%Y-%m-%d %H:%M"), cpu_count=os.cpu_count(), torch_threads=torch.get_num_threads(),
        acc_torch_test=acc_test,
        B_cert_global=max(r["B_cert_max"] for r in layers),
        B_emp_global=max(r["B_emp"] for r in layers),
        pbs_theory=sum(pbs_act.values()), pbs_theory_per_act=pbs_act,
        train=None if train_info is None else {k: v for k, v in train_info.items() if k != "curves"},
        layers=layers,
    )
    print(f"[{base}] acc={acc_test:.4f} B_cert={base_rec['B_cert_global']} "
          f"B_emp={base_rec['B_emp_global']} PBS_theory={base_rec['pbs_theory']}")
    for r in layers:
        print(f"  {r['name']} ({r['kind']}, fan_in={r['fan_in']}, pruned={r['pruned']}): "
              f"mask K mean/max={r['Kmask_mean']:.1f}/{r['Kmask_max']} | "
              f"int K mean/max={r['K_mean']:.1f}/{r['K_max']} | L1_max={r['L1_max']:.0f} | "
              f"B_cert={r['B_cert_max']} B_emp={r['B_emp']} resid={r['int_residual']}")

    for r in [s.strip() for s in args.rounding.split(",") if s.strip()]:
        rnd_val = None if r.lower() == "none" else int(r)
        rec = dict(base_rec, rounding_bits=rnd_val, tag=f"{base}_Rnd{r}")
        if not args.no_compile:
            print(f"--- Rounding = {rnd_val} ---", flush=True)
            fhe_measure(model, calib, X, Y, rnd_val, args, rec)
        path = os.path.join(args.out, f"{rec['tag']}.json")
        with open(path, "w") as f:
            json.dump(jsonable(rec), f, indent=2)
        print(f"Saved {path}")
        if args.no_compile:
            break


if __name__ == "__main__":
    main()