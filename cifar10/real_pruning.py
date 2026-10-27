"""Real structured pruning with FHE cost (PBS) awareness for the VGG9 CIFAR-10 pipeline.

For every dense checkpoint of a folder (or a single --ckpt):
  1. Rank conv channels / FC neurons with first-order Taylor importance (Molchanov et al.),
     normalised per layer and divided by the PBS cost of the channel (cost-aware score).
  2. Physically remove the lowest-scoring channels (model surgery -> narrower network).
     Each round removes a fixed fraction of the REMAINING PBS (geometric schedule).
  3. Fine-tune after every round with learning-rate rewinding (Renda et al., 2020):
     a fresh optimizer and cosine schedule restarting from the base learning rate.
  4. Stop when the theoretical PBS count reaches the target ratio, run a final fine-tune,
     profile the integer accumulators (accum.py) and save the pruned checkpoint together
     with its architecture (cfg, cls_cfg) and a JSON with all metrics.

The PBS cost of a channel is the number of table lookups it produces: one per element of every
QuantReLU / QuantIdentity output that it feeds (conv TLU, re-quantization after pooling,
re-quantization after flatten). This is the same count as pbs_total in fhe_eval.py.

Examples:
  python real_pruning.py --ckpt_dir checkpoints_A --target_pbs_ratio 0.5
  python real_pruning.py --ckpt checkpoints_B/B_VGG9_W2_A2_IN4_NR1.pth --alpha 0   # plain Taylor (ablation)
"""
import argparse
import copy
import glob
import json
import math
import os
import re
import time

import brevitas.nn as qnn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import accum
import arc_cifar10 as arc
import utils_cifar10 as utils

FNAME_RE = re.compile(r"^(?P<prefix>[A-Za-z]+)_VGG9_W(?P<w>\d+)_A(?P<a>\d+)_IN(?P<inp>\d+)_NR(?P<nr>\d)")


# ----------------------------------------------------------------------------
# Helpers: settings, model construction, layer bookkeeping
# ----------------------------------------------------------------------------
def parse_setting(path):
    base = os.path.splitext(os.path.basename(path))[0]
    m = FNAME_RE.match(base)
    if m is None:
        raise ValueError(f"Cannot parse bit-widths from file name: {base}")
    return dict(base=base, prefix=m["prefix"], w=int(m["w"]), a=int(m["a"]), inp=int(m["inp"]),
                narrow=bool(int(m["nr"])))


def build_model(s, cfg=None, cls_cfg=None):
    return arc.nnet_cifar10(w_bit_width=s["w"], act_bit_width=s["a"], in_bit_width=s["inp"],
                            narrow_range=s["narrow"], cfg=cfg, cls_cfg=cls_cfg)


def prunable_layers(model):
    """Conv and hidden FC layers in forward order (the classifier output layer is excluded)."""
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, (qnn.QuantConv2d, qnn.QuantLinear))]
    return layers[:-1]


def owners(model, kinds):
    """Map every module of the given kinds to the most recent prunable layer before it."""
    prun = {n for n, _ in prunable_layers(model)}
    cur, out = None, {}
    for n, m in model.named_modules():
        if n in prun:
            cur = n
        elif isinstance(m, kinds) and n != "quant_input" and cur is not None:
            out[n] = cur
    return out


def n_out(m):
    return m.out_channels if isinstance(m, qnn.QuantConv2d) else m.out_features


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def arch_of(model):
    return dict(cfg=list(model.cfg), cls_cfg=list(model.cls_cfg))


# ----------------------------------------------------------------------------
# PBS cost model
# ----------------------------------------------------------------------------
@torch.no_grad()
def pbs_cost(model, sample):
    """Returns (per-channel PBS cost per prunable layer, total theoretical PBS, PBS per activation)."""
    model.eval()
    acts = owners(model, (qnn.QuantReLU, qnn.QuantIdentity))
    numel, hooks = {}, []

    def mk(name):
        def h(m, inp, out):
            o = out.value if hasattr(out, "value") else out
            numel[name] = int(o[0].numel())
        return h

    mods = dict(model.named_modules())
    for n in acts:
        hooks.append(mods[n].register_forward_hook(mk(n)))
    model(sample)
    for h in hooks:
        h.remove()

    per_ch = {n: 0.0 for n, _ in prunable_layers(model)}
    for act, own in acts.items():
        per_ch[own] += numel[act] / n_out(mods[own])
    return per_ch, int(sum(numel.values())), numel


# ----------------------------------------------------------------------------
# First-order Taylor importance (Molchanov et al.)
# ----------------------------------------------------------------------------
def taylor_importance(model, loader, device, n_batches):
    """|mean_spatial(activation * gradient)| summed over samples, measured at the BatchNorm output."""
    model.eval()  # BN running statistics are not updated
    bns = owners(model, (nn.BatchNorm1d, nn.BatchNorm2d))
    mods = dict(model.named_modules())
    imp = {own: torch.zeros(n_out(mods[own]), device=device) for own in bns.values()}
    hooks = []

    def mk(own):
        def fwd(m, inp, out):
            o = out.value if hasattr(out, "value") else out
            if not o.requires_grad:
                return

            def bwd(g):
                t = o.detach() * g
                t = t.mean(dim=(2, 3)) if t.dim() == 4 else t
                imp[own] += t.abs().sum(0)
            o.register_hook(bwd)
        return fwd

    for bn, own in bns.items():
        hooks.append(mods[bn].register_forward_hook(mk(own)))
    criterion = nn.CrossEntropyLoss()
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        loss = criterion(utils._unwrap_out(model(x)), y)
        loss.backward()
    for h in hooks:
        h.remove()
    model.zero_grad(set_to_none=True)
    # Layer-wise L2 normalisation, as in Molchanov et al. (2017)
    return {n: (v / (v.norm() + 1e-12)).cpu() for n, v in imp.items()}


def select_keep(model, scores, costs, pbs_now, pbs_target, min_keep):
    """Greedy global removal of the lowest-score channels until the PBS target is met."""
    mods = dict(model.named_modules())
    cand = sorted((float(s), n, i) for n, v in scores.items() for i, s in enumerate(v))
    removed = {n: set() for n in scores}
    for _, n, i in cand:
        if pbs_now <= pbs_target:
            break
        if n_out(mods[n]) - len(removed[n]) - 1 < min_keep[n]:
            continue
        removed[n].add(i)
        pbs_now -= costs[n]
    keep = {n: torch.tensor(sorted(set(range(n_out(mods[n]))) - removed[n]), dtype=torch.long)
            for n in scores}
    return keep, {n: len(r) for n, r in removed.items()}

def l1norm_importance(model, names):
    """Filter L1-norm (Li et al., 2017): sum of |weights| per output channel, L2-normalised per layer."""
    mods = dict(model.named_modules())
    out = {}
    for n in names:
        w = mods[n].weight.detach().flatten(1).abs().sum(1)
        out[n] = (w / (w.norm() + 1e-12)).cpu()
    return out


def select_keep_uniform(model, scores, costs, pbs_now, pbs_target, min_keep):
    """Remove the same fraction of channels from every candidate layer (lowest scores first)."""
    mods = dict(model.named_modules())
    pool = sum(costs[n] * n_out(mods[n]) for n in scores)
    frac = min(1.0, max(0.0, (pbs_now - pbs_target) / max(pool, 1)))
    keep, removed = {}, {}
    for n, v in scores.items():
        k = max(0, min(math.ceil(frac * n_out(mods[n])), n_out(mods[n]) - min_keep[n]))
        drop = set(torch.argsort(v)[:k].tolist())
        keep[n] = torch.tensor(sorted(set(range(n_out(mods[n]))) - drop), dtype=torch.long)
        removed[n] = k
    return keep, removed

# ----------------------------------------------------------------------------
# Model surgery: build the narrower network and copy the surviving weights
# ----------------------------------------------------------------------------
@torch.no_grad()
def surgery(old, keep, s):
    layers = prunable_layers(old)
    convs = [n for n, m in layers if isinstance(m, qnn.QuantConv2d)]
    fcs = [n for n, m in layers if isinstance(m, qnn.QuantLinear)]
    last_fc = [n for n, m in old.named_modules() if isinstance(m, qnn.QuantLinear)][-1]

    it = iter(convs)
    new_cfg = [x if x in ("A", "M") else len(keep[next(it)]) for x in old.cfg]
    new_cls = [len(keep[n]) for n in fcs]
    new = build_model(s, cfg=new_cfg, cls_cfg=new_cls).to(next(old.parameters()).device)

    # (out_idx, in_idx) for every conv / linear, out_idx for every BatchNorm
    io = {}
    prev = torch.arange(3)
    for n in convs:
        io[n] = (keep[n], prev)
        prev = keep[n]
    spatial = old.flatten_dim // old.cfg[[i for i, x in enumerate(old.cfg) if x not in ("A", "M")][-1]]
    prev = torch.cat([c * spatial + torch.arange(spatial) for c in keep[convs[-1]].tolist()])
    for n in fcs:
        io[n] = (keep[n], prev)
        prev = keep[n]
    io[last_fc] = (None, prev)
    bn_out = {bn: keep[own] for bn, own in owners(old, (nn.BatchNorm1d, nn.BatchNorm2d)).items()}

    # Iterate over the OLD keys: a freshly built Brevitas model does not list its learned
    # activation scales (scaling_impl.value) until they are initialised, but it accepts them.
    old_sd, ref_sd = old.state_dict(), new.state_dict()
    new_sd = {}
    for k, t in old_sd.items():
        v = ref_sd.get(k, t)
        prefix, pname = k.rsplit(".", 1) if "." in k else ("", k)
        if prefix in io and pname in ("weight", "bias"):
            out_idx, in_idx = io[prefix]
            if out_idx is not None:
                t = t.index_select(0, out_idx.to(t.device))
            if pname == "weight" and in_idx is not None:
                t = t.index_select(1, in_idx.to(t.device))
        elif prefix in bn_out and pname in ("weight", "bias", "running_mean", "running_var"):
            t = t.index_select(0, bn_out[prefix].to(t.device))
        if t.shape != v.shape:
            raise RuntimeError(f"Surgery shape mismatch for {k}: {tuple(t.shape)} vs {tuple(v.shape)}")
        new_sd[k] = t.clone()
    new.load_state_dict(new_sd, strict=True)
    return new.to(next(old.parameters()).device)


# ----------------------------------------------------------------------------
# Fine-tuning with learning-rate rewinding
# ----------------------------------------------------------------------------
def fine_tune(model, epochs, args, loaders, device, teacher=None):
    train_l, val_l, _ = loaders
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    rest = utils.RestTimer() if hasattr(utils, "RestTimer") else None
    best_acc, best_state, curves = -1.0, None, dict(train_acc=[], val_acc=[])
    t0 = time.time()
    for ep in range(epochs):
        kd = dict(teacher=teacher, kd_temp=args.kd_temp, kd_alpha=args.kd_alpha) if teacher is not None else {}
        _, t_acc, _ = utils.training_phase(model, train_l, criterion, optimizer, device, clip_weights=True, **kd)
        _, v_acc = utils.evaluate(model, val_l, criterion, device)
        scheduler.step()
        curves["train_acc"].append(float(t_acc))
        curves["val_acc"].append(float(v_acc))
        if v_acc > best_acc:
            best_acc, best_state = v_acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"    ft[{ep + 1:03d}/{epochs}] t_acc={t_acc:.4f} v_acc={v_acc:.4f}", flush=True)
        if rest is not None:
            rest()
    if best_state is not None:
        model.load_state_dict(best_state)
    t_rest = rest.total_rest if rest is not None else 0.0
    return dict(epochs=epochs, best_val_acc=float(best_acc), time_sec=round(time.time() - t0 - t_rest, 1),
                curves=curves)


# ----------------------------------------------------------------------------
# Profiling (same quantities as fhe_eval.py, without compilation)
# ----------------------------------------------------------------------------
def profile_model(model, s, val_l, device):
    sample = next(iter(val_l))[0][:64].to(device)
    reg = accum.layer_registry(model, sample)
    prof = accum.profile(reg, narrow=s["narrow"])
    emp = accum.empirical_bits(model, reg, val_l, max_batches=10)
    rq = accum.requant_pbs(model, sample)
    for row in prof:
        row["B_emp"] = emp.get(row["name"])
    return dict(
        B_cert_global=max(r["B_cert_max"] for r in prof),
        B_emp_global=max((r["B_emp"] for r in prof if r["B_emp"]), default=None),
        pbs_convfc=sum(r["pbs"] for r in prof),
        pbs_requant=sum(rq.values()),
        pbs_total=sum(r["pbs"] for r in prof) + sum(rq.values()),
        layers=prof,
    )


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    return str(o)


def save_json(rec, path):
    with open(path, "w") as f:
        json.dump(jsonable(rec), f, indent=2)


# ----------------------------------------------------------------------------
# Main pruning loop for one checkpoint
# ----------------------------------------------------------------------------
def prune_one(path, args, device):
    s = parse_setting(path)
    tag = f"{s['base']}_SP{int(round(args.target_pbs_ratio * 100))}_a{args.alpha:g}"
    if args.criterion != "taylor":
        tag += f"_{args.criterion}"
    if args.alloc != "global":
        tag += f"_{args.alloc}"
    if args.only_layers:
        tag += "_only" + "-".join(x.split(".")[-1] for x in args.only_layers.split(","))
    only = set(args.only_layers.split(",")) if args.only_layers else None
    
    out_ckpt = os.path.join(args.save_dir, f"{tag}.pth")
    out_json = os.path.join(args.results_dir, f"{tag}.json")
    if os.path.exists(out_json) and os.path.exists(out_ckpt) and not args.overwrite:
        print(f"[SKIP] {tag} (already done)")
        return

    print(f"\n=== Structured pruning: {s['base']} -> {tag} ===", flush=True)
    utils.set_seed(args.seed)
    loaders = utils.get_data_loaders(batch_size=args.batch)
    train_l, val_l, test_l = loaders
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model = build_model(s)
    accum.load_state(model, path)
    model.to(device)
    sample = next(iter(val_l))[0][:4].to(device)

    teacher = None
    if args.kd == "dense":
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    _, val0 = utils.evaluate(model, val_l, criterion, device)
    _, test0 = utils.evaluate(model, test_l, criterion, device)
    _, pbs0, act0 = pbs_cost(model, sample)
    mods = dict(model.named_modules())
    min_keep = {n: max(args.min_ch, math.ceil(args.min_ch_frac * n_out(m))) for n, m in prunable_layers(model)}
    pbs_final_target = pbs0 * args.target_pbs_ratio

    rec = dict(
        tag=tag, method=f"{s['prefix']}_struct", source_ckpt=path, out_ckpt=out_ckpt,
        setting=dict(w=s["w"], a=s["a"], inp=s["inp"], narrow=s["narrow"]),
        args=vars(args), time_start=time.strftime("%Y-%m-%d %H:%M"),
        initial=dict(**arch_of(model), params=count_params(model), val_acc=val0, test_acc=test0,
                     pbs_theory=pbs0, pbs_per_act=act0),
        min_keep=min_keep, rounds=[],
    )
    print(f"  initial: val={val0:.4f} test={test0:.4f} PBS={pbs0} target={int(pbs_final_target)}", flush=True)

    t_all, pbs_now, rnd = time.time(), pbs0, 0
    while pbs_now > pbs_final_target and rnd < args.max_rounds:
        rnd += 1
        t_r = time.time()
        round_target = max(pbs_final_target, pbs_now * (1.0 - args.step))
        costs, _, _ = pbs_cost(model, sample)
        mods = dict(model.named_modules())
        if args.criterion == "taylor":
            imp = taylor_importance(model, train_l, device, args.taylor_batches)
        else:
            imp = l1norm_importance(model, list(min_keep))
        scores = {n: imp[n] / (costs[n] ** args.alpha) for n in imp if only is None or n in only}
        if args.alloc == "uniform":
            keep, n_removed = select_keep_uniform(model, scores, costs, pbs_now, round_target, min_keep)
        else:
            keep, n_removed = select_keep(model, scores, costs, pbs_now, round_target, min_keep)
        for n in imp:  # layers outside --only_layers stay untouched
            keep.setdefault(n, torch.arange(n_out(mods[n]), dtype=torch.long))
            n_removed.setdefault(n, 0)
        
        if sum(n_removed.values()) == 0:
            print("  No channel can be removed under the min_keep constraints; stopping.")
            rec["stop_reason"] = "min_keep"
            break

        prev_model = copy.deepcopy(model)  # kept for a possible rollback
        model = surgery(model, keep, s)
        _, pbs_new, act_new = pbs_cost(model, sample)
        print(f"  round {rnd}: PBS {pbs_now} -> {pbs_new} (target {int(round_target)}) | "
              f"cfg={model.cfg} cls={model.cls_cfg}", flush=True)
        ft = fine_tune(model, args.ft_epochs, args, loaders, device, teacher)
        _, test_r = utils.evaluate(model, test_l, criterion, device)

        # Safety guard: reject the round if validation accuracy dropped too much vs. the dense model
        drop = val0 - ft["best_val_acc"]
        accepted = not (args.max_acc_drop >= 0 and drop > args.max_acc_drop)
        rec["rounds"].append(dict(
            round=rnd, accepted=accepted, pbs_before=pbs_now, pbs_after=pbs_new, pbs_round_target=round_target,
            removed=n_removed, **arch_of(model), params=count_params(model),
            val_acc=ft["best_val_acc"], val_drop=drop, test_acc=test_r, finetune=ft,
            time_sec=round(time.time() - t_r, 1),
        ))
        if not accepted:
            print(f"  round {rnd} REJECTED: val drop {100 * drop:.2f}% > {100 * args.max_acc_drop:.2f}%; "
                  f"rolling back to the previous round and stopping.", flush=True)
            model = prev_model
            rec["stop_reason"] = "max_acc_drop"
            save_json(rec, out_json)
            break
        del prev_model
        pbs_now = pbs_new
        print(f"  round {rnd} done: val={ft['best_val_acc']:.4f} (drop {100 * drop:+.2f}%) "
              f"test={test_r:.4f}", flush=True)
        save_json(rec, out_json)  # partial progress

        # Snapshot after every accepted round (one point of the accuracy-PBS curve)
        snap = os.path.join(args.save_dir, f"{tag}_r{rnd}_pbs{pbs_now}.pth")
        torch.save(dict(state_dict=model.state_dict(), **arch_of(model), setting=rec["setting"],
                        source_ckpt=path, round=rnd), snap)

    rec.setdefault("stop_reason", "target_reached" if pbs_now <= pbs_final_target else "max_rounds")
    final_ft = None
    if args.final_epochs > 0 and any(r["accepted"] for r in rec["rounds"]):
        print(f"  final fine-tune ({args.final_epochs} epochs)", flush=True)
        final_ft = fine_tune(model, args.final_epochs, args, loaders, device, teacher)

    _, val_f = utils.evaluate(model, val_l, criterion, device)
    _, test_f = utils.evaluate(model, test_l, criterion, device)
    _, pbs_f, act_f = pbs_cost(model, sample)
    prof = profile_model(model, s, val_l, device)

    torch.save(dict(state_dict=model.state_dict(), **arch_of(model), setting=rec["setting"],
                    source_ckpt=path), out_ckpt)

    rec["final"] = dict(
        **arch_of(model), params=count_params(model), val_acc=val_f, test_acc=test_f,
        pbs_theory=pbs_f, pbs_per_act=act_f,
        pbs_reduction=1.0 - pbs_f / pbs0, params_reduction=1.0 - count_params(model) / rec["initial"]["params"],
        acc_drop_test=test0 - test_f, final_finetune=final_ft,
        # Cross-check: cost model of this script vs. accum.py counting (should be equal)
        pbs_check_accum=dict(accum_total=prof["pbs_total"], match=prof["pbs_total"] == pbs_f),
        profile=prof,
    )
    rec["total_time_sec"] = round(time.time() - t_all, 1)
    rec["time_end"] = time.strftime("%Y-%m-%d %H:%M")
    save_json(rec, out_json)
    print(f"  FINAL: test={test_f:.4f} (drop {test0 - test_f:+.4f}) | PBS {pbs0} -> {pbs_f} "
          f"({100 * (1 - pbs_f / pbs0):.1f}% less) | B_cert={prof['B_cert_global']} B_emp={prof['B_emp_global']}")
    print(f"  Saved {out_ckpt} and {out_json}")


def main():
    p = argparse.ArgumentParser(description="PBS-aware real structured pruning (Taylor + LR rewinding)")
    p.add_argument("--ckpt_dir", type=str, default=None, help="Prune every checkpoint of this folder")
    p.add_argument("--ckpt", type=str, default=None, help="Prune a single checkpoint")
    p.add_argument("--pattern", type=str, default="*_VGG9_W*_A*_IN*_NR*.pth")
    p.add_argument("--save_dir", type=str, default="checkpoints_struct")
    p.add_argument("--results_dir", type=str, default="results_struct")
    # Schedule
    p.add_argument("--target_pbs_ratio", type=float, default=0.3, help="Final PBS / initial PBS")
    p.add_argument("--step", type=float, default=0.15, help="Fraction of the remaining PBS removed per round")
    p.add_argument("--max_rounds", type=int, default=20)
    p.add_argument("--alpha", type=float, default=1.0, help="score = taylor / cost^alpha (0 = plain Taylor)")
    p.add_argument("--criterion", choices=["taylor", "l1norm"], default="taylor",
                   help="Channel importance: Taylor (ours) or filter L1-norm (Li et al., 2017)")
    p.add_argument("--alloc", choices=["global", "uniform"], default="global",
                   help="global: greedy across layers; uniform: same fraction removed from every candidate layer")
    p.add_argument("--only_layers", type=str, default=None,
                   help="Comma-separated layers that may be pruned, e.g. features.0,features.3")
    p.add_argument("--min_ch", type=int, default=4, help="Minimum channels kept per layer")
    p.add_argument("--min_ch_frac", type=float, default=0.25, help="Minimum fraction of channels kept per layer")
    p.add_argument("--taylor_batches", type=int, default=50)
    p.add_argument("--max_acc_drop", type=float, default=0.03,
                   help="Reject a round (rollback + stop) if val accuracy drops more than this vs. the dense "
                        "model; negative value disables the guard")
    # Fine-tuning (learning-rate rewinding every round)
    p.add_argument("--ft_epochs", type=int, default=30)
    p.add_argument("--final_epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--kd", choices=["none", "dense"], default="dense", help="Distil from the unpruned model")
    p.add_argument("--kd_temp", type=float, default=4.0)
    p.add_argument("--kd_alpha", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not args.ckpt and not args.ckpt_dir:
        p.error("give --ckpt or --ckpt_dir")
    paths = [args.ckpt] if args.ckpt else sorted(glob.glob(os.path.join(args.ckpt_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"No checkpoints found in {args.ckpt_dir} matching {args.pattern}")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | {len(paths)} checkpoint(s)")
    for path in paths:
        prune_one(path, args, device)


if __name__ == "__main__":
    main()
