"""Zama VGG-9 (CNV 2W2A) CIFAR-10 baseline: train + integer profile + FHE compile/simulate/execute.

Faithful port of concrete-ml/use_case_examples/cifar/cifar_brevitas_training (release 1.9.x),
which is itself a port of the Brevitas bnn_pynq CNV example:
  - model: 6 conv (64,64,128,128,256,256) + 2 FC (512,512) + classifier, no bias, no pruning
  - signed narrow-range activations (QuantIdentity, no ReLU), constant weight scale 1.0
  - training: Adam lr=0.02, wd=0, batch 100, lr halved every 40 epochs, CrossEntropy,
    RandomCrop+Flip, clip_weights(-1, 1) after every step, best checkpoint picked on the TEST set
The module layout is identical to Zama's, so their released checkpoint (best.tar) loads directly.

Examples:
  # Evaluate Zama's released checkpoint (no training):
  python zama_vgg_cifar10.py --mode eval --ckpt path/to/best.tar --rounding None,8,6
  # Train from scratch with Zama's recipe, then evaluate:
  python zama_vgg_cifar10.py --mode all --epochs 1000 --rounding None,8,6
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
import torch.nn.init as init
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import FloatToIntImplType, RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear
from brevitas.quant.solver import ActQuantSolver, WeightQuantSolver
from dependencies import value
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

# ----------------------------------------------------------------------------
# Model (identical to Zama's models/model.py, models/common.py, models/tensor_norm.py)
# ----------------------------------------------------------------------------
CNV_OUT_CH_POOL = [(64, False), (64, True), (128, False), (128, True), (256, False), (256, False)]
INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]
LAST_FC_IN_FEATURES = 512
KERNEL_SIZE = 3


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


class TensorNorm(nn.Module):
    def __init__(self, eps=1e-4, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.rand(1))
        self.bias = nn.Parameter(torch.rand(1))
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))
        self.reset_running_stats()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            mean = x.mean()
            unbias_var = x.var(unbiased=True)
            biased_var = x.var(unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.detach()
            inv_std = 1 / (biased_var + self.eps).pow(0.5)
            return (x - mean) * inv_std * self.weight + self.bias
        return ((x - self.running_mean) / (self.running_var + self.eps).pow(0.5)) * self.weight + self.bias


class CNV(nn.Module):
    def __init__(self, num_classes=10, weight_bit_width=2, act_bit_width=2, in_bit_width=8, in_ch=3):
        super().__init__()
        self.conv_features = nn.ModuleList()
        self.linear_features = nn.ModuleList()

        # Q1.7 input format
        self.conv_features.append(
            QuantIdentity(
                act_quant=CommonActQuant,
                return_quant_tensor=True,
                bit_width=in_bit_width,
                min_val=-1.0,
                max_val=1.0 - 2.0 ** (-7),
                narrow_range=False,
                restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            )
        )

        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(
                QuantConv2d(kernel_size=KERNEL_SIZE, in_channels=in_ch, out_channels=out_ch, bias=False,
                            weight_quant=CommonWeightQuant, weight_bit_width=weight_bit_width)
            )
            in_ch = out_ch
            self.conv_features.append(nn.BatchNorm2d(in_ch, eps=1e-4))
            self.conv_features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))
            if is_pool_enabled:
                self.conv_features.append(nn.AvgPool2d(kernel_size=2))
                self.conv_features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(
                QuantLinear(in_features=in_features, out_features=out_features, bias=False,
                            weight_quant=CommonWeightQuant, weight_bit_width=weight_bit_width)
            )
            self.linear_features.append(nn.BatchNorm1d(out_features, eps=1e-4))
            self.linear_features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))

        self.linear_features.append(
            QuantLinear(in_features=LAST_FC_IN_FEATURES, out_features=num_classes, bias=False,
                        weight_quant=CommonWeightQuant, weight_bit_width=weight_bit_width)
        )
        self.linear_features.append(TensorNorm())

        for m in self.modules():
            if isinstance(m, (QuantConv2d, QuantLinear)):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in list(self.conv_features) + list(self.linear_features):
            if isinstance(mod, (QuantConv2d, QuantLinear)):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        for mod in self.conv_features:
            x = mod(x)
        x = torch.flatten(x, 1)
        for mod in self.linear_features:
            x = mod(x)
        return x


# ----------------------------------------------------------------------------
# Data (same as Zama: inputs scaled to [-1, 1], RandomCrop + Flip for training)
# ----------------------------------------------------------------------------
NORM = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # equivalent to 2*x - 1
TRAIN_TF = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), NORM])
TEST_TF = transforms.Compose([transforms.ToTensor(), NORM])


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


def save_ckpt(path, model, optimizer, epoch, best_acc):
    # Same package format as Zama's trainer (best.tar)
    torch.save({"state_dict": model.state_dict(), "optim_dict": optimizer.state_dict(),
                "epoch": epoch + 1, "best_val_acc": best_acc}, path)


def load_weights(model, path):
    package = torch.load(path, map_location="cpu")
    sd = package["state_dict"] if isinstance(package, dict) and "state_dict" in package else package
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    crit = [k for k in missing if k.endswith(".weight") or "running_" in k]
    if crit or unexpected:
        print(f"WARNING: missing={crit} unexpected={list(unexpected)}")
    return package if isinstance(package, dict) else {}


def train(args, model, device, best_path, last_path):
    train_set = CIFAR10(root=args.data_root, train=True, download=True, transform=TRAIN_TF)
    test_set = CIFAR10(root=args.data_root, train=False, download=True, transform=TEST_TF)
    if args.debug_subset:
        train_set = torch.utils.data.Subset(train_set, range(args.debug_subset))
        test_set = torch.utils.data.Subset(test_set, range(args.debug_subset))
    kw = dict(batch_size=args.batch, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_set, shuffle=True, **kw)
    test_loader = DataLoader(test_set, shuffle=False, **kw)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    start_epoch, best_acc = 1, 0.0
    curves = dict(epoch=[], train_loss=[], train_acc=[], test_acc=[], lr=[])

    if args.resume and os.path.isfile(last_path):
        pkg = load_weights(model, last_path)
        optimizer.load_state_dict(pkg["optim_dict"])
        start_epoch, best_acc = pkg["epoch"], pkg["best_val_acc"]
        curves = pkg.get("curves", curves)
        print(f"Resumed from {last_path} at epoch {start_epoch} (best={best_acc:.4f})")

    t_start = time.time()
    # Zama's loop is range(1, epochs): "--epochs 1000" means 999 training epochs
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, run_loss, correct, total = time.time(), 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.clip_weights(-1, 1)
            run_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        lr_used = optimizer.param_groups[0]["lr"]
        if epoch % 40 == 0:  # Zama "FIXED" scheduler
            optimizer.param_groups[0]["lr"] *= 0.5

        acc = evaluate(model, test_loader, device)
        curves["epoch"].append(epoch)
        curves["train_loss"].append(run_loss / total)
        curves["train_acc"].append(correct / total)
        curves["test_acc"].append(acc)
        curves["lr"].append(lr_used)
        flag = ""
        if acc >= best_acc:
            best_acc = acc
            save_ckpt(best_path, model, optimizer, epoch, best_acc)
            flag = " [BEST]"
        pkg = {"state_dict": model.state_dict(), "optim_dict": optimizer.state_dict(),
               "epoch": epoch + 1, "best_val_acc": best_acc, "curves": curves}
        torch.save(pkg, last_path)
        print(f"Ep[{epoch:04d}/{args.epochs - 1}] loss={run_loss / total:.4f} t_acc={correct / total:.4f} "
              f"test_acc={acc:.4f} lr={lr_used:.5f} ({time.time() - t0:.1f}s){flag}", flush=True)

    return dict(best_test_accuracy=best_acc, train_time_sec=round(time.time() - t_start, 1),
                epochs_trained=args.epochs - 1, curves=curves)


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
    """Per-layer fan-in / L1 / accumulator bits and the theoretical PBS count.

    Activations are signed here (no ReLU), so the worst case is in_max * L1 on both sides.
    """
    model.eval()
    rows, pbs_act = {}, {}
    for bi, (x, _) in enumerate(loader):
        if bi >= max_batches:
            break
        cur = None  # (integer max of the current activation, its scale)
        mods = [("conv_features", i, m) for i, m in enumerate(model.conv_features)] + \
               [("flatten", 0, None)] + \
               [("linear_features", i, m) for i, m in enumerate(model.linear_features)]
        first_act = True
        for grp, i, m in mods:
            name = f"{grp}.{i}"
            if m is None:
                x = torch.flatten(_tensor(x), 1)
                continue
            if isinstance(m, QuantIdentity):
                x = m(x)
                bits = int(m.quant_act_bit_width())
                scale = float(m.quant_act_scale())
                narrow = not first_act  # only the input quantizer is non-narrow
                cur = (2 ** (bits - 1) - (1 if narrow else 0), scale)
                if not first_act:
                    pbs_act[name] = int(_tensor(x)[0].numel())  # one table lookup per element
                first_act = False
                continue
            if isinstance(m, (QuantConv2d, QuantLinear)):
                out = _tensor(m(x))
                qw = m.quant_weight()
                ws = float(qw.scale.flatten()[0])
                q = torch.round(qw.value / ws).reshape(qw.value.shape[0], -1)
                acc = out / (cur[1] * ws)
                r = rows.setdefault(name, dict(obs=0.0, resid=0.0))
                r["obs"] = max(r["obs"], float(acc.abs().max()))
                r["resid"] = max(r["resid"], float((acc - acc.round()).abs().max()))
                if "fan_in" not in r:
                    nnz, l1 = (q != 0).sum(1).float(), q.abs().sum(1)
                    r.update(
                        name=name, kind="conv" if isinstance(m, QuantConv2d) else "fc",
                        fan_in=int(q.shape[1]), out_ch=int(q.shape[0]), in_max=float(cur[0]),
                        K_mean=float(nnz.mean()), K_max=int(nnz.max()), K_min=int(nnz.min()),
                        L1_mean=float(l1.mean()), L1_max=float(l1.max()),
                        B_cert_max=_bits_signed(cur[0] * float(l1.max())),
                        B_cert_mean=float(np.mean([_bits_signed(cur[0] * float(v)) for v in l1])),
                        int_sparsity=float(1 - nnz.sum() / q.numel()),
                        w_int_max=int(q.abs().max()),
                    )
                x = out
                continue
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


def select_class_balanced(labels, n, n_classes=10):
    per_class = {c: list(np.where(labels == c)[0]) for c in range(n_classes)}
    idx, rnd = [], 0
    while len(idx) < n and rnd < max(len(v) for v in per_class.values()):
        for c in range(n_classes):
            if len(idx) < n and rnd < len(per_class[c]):
                idx.append(int(per_class[c][rnd]))
        rnd += 1
    return idx


def fhe_measure(model, calib, X, Y, rnd_val, args, rec):
    from concrete.fhe import Configuration, ParameterSelectionStrategy
    from concrete.ml.torch.compile import compile_brevitas_qat_model

    cfg = Configuration(parameter_selection_strategy=ParameterSelectionStrategy.MULTI)
    kwargs = dict(
        n_bits={"model_inputs": 8, "model_outputs": 8},  # as in Zama's evaluate_torch_cml.py
        configuration=cfg,
        rounding_threshold_bits={"n_bits": rnd_val, "method": "APPROXIMATE"} if rnd_val else None,
    )
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
        rec["circuit_layers"] = circuit_layer_stats(c)

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
    except Exception as e:  # keep going with the next rounding setting
        rec["compile_error"] = str(e)[:1000]
        print(f"  COMPILE/SIM FAILED: {str(e)[:300]}")
        return

    if args.exec_n <= 0:
        return qm
    
    try:
        idx = select_class_balanced(Y, args.exec_n)
        Xe, Ye = X[idx], Y[idx]
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

    return qm

def circuit_layer_stats(circuit):
    """Per-layer accumulator bit-width and range of the compiled circuit (same logic as fhe_eval.py)."""
    g = circuit.graph

    def layer_of(node, depth=0):
        if node.tag:
            last = node.tag.split("/")[-1]
            return node.tag.rsplit(".", 1)[0] if "." in last else node.tag
        if depth > 20:
            return None
        for p in g.ordered_preds_of(node):
            t = layer_of(p, depth + 1)
            if t:
                return t
        return None

    def as_int(v):
        return int(v) if isinstance(v, (int, np.integer)) else None

    out = {}
    for n in g.query_nodes():
        if type(n.output.dtype).__name__ != "Integer" or not n.tag:
            continue
        name = n.properties.get("name", "")
        if name not in ("conv2d", "matmul", "dot", "add", "subtract", "sum", "multiply"):
            continue
        bw = max(n.properties.get("original_bit_width", n.output.dtype.bit_width),
                 n.properties.get("bit_width_hint", -1))
        lo, hi = (None, None) if n.bounds is None else (as_int(n.bounds[0]), as_int(n.bounds[1]))
        d = out.setdefault(layer_of(n), dict(acc_bits=-1, acc_min=None, acc_max=None))
        d["acc_bits"] = max(d["acc_bits"], bw)
        if lo is not None:
            d["acc_min"] = lo if d["acc_min"] is None else min(d["acc_min"], lo)
            d["acc_max"] = hi if d["acc_max"] is None else max(d["acc_max"], hi)
    return out


def compiled_limits(circuit_layers):
    """{layer: (bits, lo, hi)} of every conv / linear accumulator of the compiled circuit."""
    lim = {}
    for tag, d in circuit_layers.items():
        parts = tag.strip("/").split("/")
        if len(parts) < 2 or d.get("acc_bits", -1) <= 0 or parts[-1].split(".")[0] not in ("Conv", "Gemm", "MatMul"):
            continue
        b = int(d["acc_bits"])
        signed = d.get("acc_min") is None or d["acc_min"] < 0
        lim[parts[-2]] = (b, *((-(2 ** (b - 1)), 2 ** (b - 1) - 1) if signed else (0, 2 ** b - 1)))
    return lim


class AccProbe:
    """Integer accumulator of every conv / linear layer (output / (input_scale * weight_scale))."""

    def __init__(self, model):
        self.store, self.hooks, self.names = {}, [], []
        mods = [(f"conv_features.{i}", m) for i, m in enumerate(model.conv_features)] + \
               [(f"linear_features.{i}", m) for i, m in enumerate(model.linear_features)]
        last_act = None
        for name, m in mods:
            if isinstance(m, QuantIdentity):
                last_act = m
            elif isinstance(m, (QuantConv2d, QuantLinear)):
                div = float(last_act.quant_act_scale()) * float(m.quant_weight().scale.flatten()[0])
                self.hooks.append(m.register_forward_hook(self._mk(name, div)))
                self.names.append(name)

    def _mk(self, name, div):
        def h(m, inp, out):
            self.store[name] = _tensor(out) / div
        return h

    def remove(self):
        for h in self.hooks:
            h.remove()


def adversarial(model, probe, target, x0, steps, alpha, lo=-1.0, hi=1.0):
    """Sign-gradient ascent on the largest |accumulator| of one layer, inside the pixel range."""
    x = x0.clone()
    for _ in range(steps):
        x.requires_grad_(True)
        model(x)
        obj = probe.store[target].abs().flatten(1).amax(1).sum()
        grad, = torch.autograd.grad(obj, x)
        with torch.no_grad():
            x = (x + alpha * (hi - lo) * grad.sign()).clamp(lo, hi)
    return x.detach()


def adversarial_set(model, x0, args):
    """One adversarial batch per layer (each layer gets its own start points)."""
    for q in model.parameters():
        q.requires_grad_(False)
    probe = AccProbe(model)
    n = len(x0) // len(probe.names)
    out = [adversarial(model, probe, t, x0[i * n:(i + 1) * n], args.adv_steps, args.adv_alpha)
           for i, t in enumerate(probe.names)]
    probe.remove()
    return torch.cat(out), probe.names


@torch.no_grad()
def count_overflow(model, lim, batches):
    probe = AccProbe(model)
    n, bad, per = 0, 0, {}
    for x in batches:
        model(x)
        flag = torch.zeros(x.shape[0], dtype=torch.bool)
        for name, a in probe.store.items():
            if name not in lim:
                continue
            _, lo, hi = lim[name]
            a = torch.round(a).flatten(1)
            img = ((a < lo) | (a > hi)).any(1)
            if img.any():
                per[name] = per.get(name, 0) + int(img.sum())
            flag |= img
        n += x.shape[0]
        bad += int(flag.sum())
    probe.remove()
    return dict(n=n, images_with_overflow=bad, frac_with_overflow=bad / max(n, 1), per_layer=per)


@torch.no_grad()
def overflow_flags(model, lim, x):
    """Per-image flag: True if any accumulator of the image leaves its compiled range."""
    probe = AccProbe(model)
    flags = []
    for xb in x.split(256):
        model(xb)
        f = torch.zeros(xb.shape[0], dtype=torch.bool)
        for name, a in probe.store.items():
            if name in lim:
                _, lo, hi = lim[name]
                a = torch.round(a).flatten(1)
                f |= ((a < lo) | (a > hi)).any(1)
        flags.append(f)
    probe.remove()
    return torch.cat(flags).numpy()


def overflow_report(model, rec, test_loader, args, qm):
    lim = compiled_limits(rec["circuit_layers"])
    bcert = {r["name"]: r["B_cert_max"] for r in rec["layers"]}
    print(f"  {'layer':<20}{'B_cert':>7}{'compiled':>9}")
    for k, (b, _, _) in lim.items():
        print(f"  {k:<20}{bcert.get(k, -1):>7}{b:>9}")
    natural = count_overflow(model, lim, (x for x, _ in test_loader))
    x0 = torch.cat([x for x, _ in test_loader])[:args.adv_images * 9]
    xa, _ = adversarial_set(model, x0, args)
    adv = count_overflow(model, lim, xa.split(256))
    print(f"  OVERFLOW natural: {natural['images_with_overflow']}/{natural['n']} {natural['per_layer']} | "
          f"adversarial: {adv['images_with_overflow']}/{adv['n']} "
          f"({100 * adv['frac_with_overflow']:.2f}%) {adv['per_layer']}", flush=True)

    # Does the overflow change the FHE output? Simulated FHE vs. the clear quantized model,
    # separately on adversarial images that overflow and on those that do not
    xa_np = np.ascontiguousarray(xa.numpy(), dtype=np.float32)
    p_dis = np.concatenate([qm.forward(xa_np[i:i + 100], fhe="disable").argmax(1)
                            for i in range(0, len(xa_np), 100)])
    p_sim = np.concatenate([qm.forward(xa_np[i:i + 100], fhe="simulate").argmax(1)
                            for i in range(0, len(xa_np), 100)])
    flags = overflow_flags(model, lim, xa)
    wrong = p_sim != p_dis
    sim_check = dict(
        n_overflow=int(flags.sum()), n_clean=int((~flags).sum()),
        disagree_overflow=float(wrong[flags].mean()) if flags.any() else None,
        disagree_clean=float(wrong[~flags].mean()) if (~flags).any() else None,
    )
    print(f"  FHE-SIM vs CLEAR on adversarial images: disagree {sim_check['disagree_overflow']} "
          f"on {sim_check['n_overflow']} overflowing vs {sim_check['disagree_clean']} "
          f"on {sim_check['n_clean']} clean", flush=True)
    return dict(compiled={k: v[0] for k, v in lim.items()}, B_cert=bcert,
                natural=natural, adversarial=adv, sim_check=sim_check)
# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Zama VGG-9 (CNV) CIFAR-10 baseline")
    p.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint to evaluate (e.g. Zama's best.tar)")
    p.add_argument("--data_root", type=str, default=os.environ.get("BLADE_DATA", "./data"))
    p.add_argument("--save_dir", type=str, default="checkpoints_zama")
    p.add_argument("--out", type=str, default="results_fhe_zama")
    p.add_argument("--tag", type=str, default="Zama_VGG9")
    # Quantization (Zama: 2/2/8)
    p.add_argument("--w_bits", type=int, default=2)
    p.add_argument("--a_bits", type=int, default=2)
    p.add_argument("--in_bits", type=int, default=8)
    # Training (Zama defaults)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    p.add_argument("--debug_subset", type=int, default=0, help="Use only N samples (smoke test)")
    # FHE evaluation
    p.add_argument("--rounding", type=str, default="None,8,6", help="Comma separated, e.g. None,8,6")
    p.add_argument("--calib_n", type=int, default=100)
    p.add_argument("--sim_n", type=int, default=1000)
    p.add_argument("--exec_n", type=int, default=0, help="Real FHE executions (0 = skip)")
    p.add_argument("--p_error", type=float, default=None)
    p.add_argument("--no_warmup", action="store_true")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--calib_adv", type=int, default=0,
                   help="Adversarial images per layer added to the calibration set (0 = off)")
    p.add_argument("--overflow", action="store_true", help="Run the overflow test after compilation")
    p.add_argument("--adv_images", type=int, default=128, help="Start images per layer for the overflow test")
    p.add_argument("--adv_steps", type=int, default=40)
    p.add_argument("--adv_alpha", type=float, default=0.01)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = f"{args.tag}_W{args.w_bits}_A{args.a_bits}_IN{args.in_bits}"
    best_path = os.path.join(args.save_dir, f"{base}_best.tar")
    last_path = os.path.join(args.save_dir, f"{base}_last.tar")
    model = CNV(weight_bit_width=args.w_bits, act_bit_width=args.a_bits, in_bit_width=args.in_bits).to(device)

    train_info = None
    if args.mode in ("train", "all") and not args.ckpt:
        print(f"=== Training {base} with Zama's recipe on {device} ===")
        train_info = train(args, model, device, best_path, last_path)
        with open(os.path.join(args.out, f"{base}_train.json"), "w") as f:
            json.dump(jsonable(train_info), f, indent=2)
    if args.mode == "train":
        return

    ckpt = args.ckpt or best_path
    print(f"=== Evaluating {ckpt} ===")
    model = model.cpu()
    load_weights(model, ckpt)
    model.eval()

    test_set = CIFAR10(root=args.data_root, train=False, download=True, transform=TEST_TF)
    # Calibration from the training set, without augmentation
    calib_set = CIFAR10(root=args.data_root, train=True, download=True, transform=TEST_TF)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=args.num_workers)
    calib = np.ascontiguousarray(
        torch.stack([calib_set[i][0] for i in range(args.calib_n)]).numpy(), dtype=np.float32)
    
    if args.calib_adv > 0:
        x0 = torch.stack([calib_set[i][0] for i in range(args.calib_n, args.calib_n + 9 * args.calib_adv)])
        xa, _ = adversarial_set(model, x0, args)
        calib = np.ascontiguousarray(np.concatenate([calib, xa.numpy()]), dtype=np.float32)
        print(f"  CALIBRATION -> {len(calib)} samples (incl. adversarial)", flush=True)
    
    acc_test = evaluate(model, test_loader, "cpu")
    layers, pbs_act = integer_profile(model, test_loader, max_batches=10)
    base_rec = dict(
        tag=base, method="zama_vgg9_dense", ckpt=ckpt,
        setting=dict(w=args.w_bits, a=args.a_bits, inp=args.in_bits, narrow=True, signed_act=True),
        checkpoint_selected_on="test",
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

    X = np.ascontiguousarray(torch.stack([test_set[i][0] for i in range(args.sim_n)]).numpy(), dtype=np.float32)
    Y = np.array([test_set[i][1] for i in range(args.sim_n)])

    for r in [s.strip() for s in args.rounding.split(",") if s.strip()]:
        rnd_val = None if r.lower() == "none" else int(r)
        rec = dict(base_rec, rounding_bits=rnd_val, tag=f"{base}_Rnd{r}")
        if not args.no_compile:
            print(f"--- Rounding = {rnd_val} ---", flush=True)
            qm = fhe_measure(model, calib, X, Y, rnd_val, args, rec)
            if args.overflow and qm is not None and "circuit_layers" in rec:
                rec["overflow"] = overflow_report(model, rec, test_loader, args, qm)
            
        path = os.path.join(args.out, f"{rec['tag']}.json")
        with open(path, "w") as f:
            json.dump(jsonable(rec), f, indent=2)
        print(f"Saved {path}")
        if args.no_compile:
            break


if __name__ == "__main__":
    main()
