"""Accumulator restriction (step 3 of the VGG9 CIFAR-10 FHE pipeline).

Takes EITHER a dense checkpoint from phase 0 (checkpoints_A/, checkpoints_B/) OR a pruned checkpoint
from phase 1 (checkpoints_struct/, which stores its own cfg / cls_cfg) and lowers the certified
accumulator bit-width of every constrained layer to a target B. Three constraints share exactly the
same per-layer target, so they differ only in HOW the target is enforced:

  l1    : per-neuron L1 projection (accum.project_l1). Every neuron is certified: B_cert <= B.
  fanin : per-neuron top-K fan-in (accum.project_fanin), K = accum.fanin_K(B). Certified.
  zama  : Zama-style layer-wise L1 unstructured pruning (Stoian et al. 2023 / Concrete-ML MNIST MLP):
          keep the K * out_channels largest-magnitude weights of the whole layer. Only the MEAN fan-in
          is bounded, so individual neurons may exceed the target (not certified).

The target is reached gradually over --stages (linear schedule from the current B_cert to B), each
stage fine-tuned with a fresh AdamW + cosine schedule (learning-rate rewinding, as in real_pruning.py)
and optional self-distillation from the input model. The first conv layer is left unconstrained by
default (its 8-bit input makes any useful target destroy it); this applies to all three constraints.

Outputs: checkpoints_accum/<tag>.pth (state_dict + cfg + cls_cfg, loadable by fhe_eval.py) and
results_accum/<tag>.json (per-stage history, per-layer profile, certification status).

Examples:
  python accum_restrict.py --ckpt checkpoints_struct/A_VGG9_W2_A2_IN8_NR1_SP30_a1.pth --constraint l1 --b_target 10
  python accum_restrict.py --ckpt checkpoints_A/A_VGG9_W2_A2_IN8_NR1.pth --constraint zama --b_target 10
"""
import argparse
import copy
import math
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

import accum
import utils_cifar10 as utils
from real_pruning import (arch_of, build_model, count_params, parse_setting, pbs_cost, profile_model,
                          save_json)


# ----------------------------------------------------------------------------
# Constraints
# ----------------------------------------------------------------------------
@torch.no_grad()
def zama_masks(reg, budgets, narrow):
    """Layer-wise magnitude masks: keep K * out_ch weights of the whole layer (torch l1_unstructured)."""
    masks = {}
    for r in reg:
        if r["name"] not in budgets:
            continue
        W = r["module"].weight.data
        K = accum.fanin_K(r, budgets[r["name"]], narrow)
        keep = K * r["out_ch"]
        if keep >= W.numel():
            continue
        mask = torch.zeros(W.numel(), device=W.device)
        if keep > 0:
            mask[torch.topk(W.abs().flatten(), keep).indices] = 1.0
        masks[r["name"]] = mask.view_as(W)
    return masks


def make_post_step(state, args, narrow):
    """Called after every optimizer step: re-impose the constraint of the current stage."""
    @torch.no_grad()
    def post_step():
        for r in state["reg"]:
            n = r["name"]
            if n not in state["Bs"]:
                continue
            if args.constraint == "l1":
                accum.project_l1(r, state["Bs"][n], exact=False, margin=args.margin)
            elif args.constraint == "fanin":
                accum.project_fanin(r, state["Bs"][n], narrow)
            elif n in state["masks"]:
                r["module"].weight.data.mul_(state["masks"][n])
    return post_step


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Accumulator restriction: per-neuron L1 / top-K fan-in / Zama layer-wise")
    p.add_argument("--ckpt", required=True, type=str, help="Dense (phase 0) or pruned (phase 1) checkpoint")
    p.add_argument("--constraint", default="l1", choices=["l1", "fanin", "zama"])
    p.add_argument("--b_target", type=int, required=True, help="Target accumulator bit-width (certified for l1/fanin)")
    p.add_argument("--constrain_first", type=int, default=0, help="1 = also constrain the first conv layer")
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--epochs_per_stage", type=int, default=10)
    p.add_argument("--final_epochs", type=int, default=30, help="Extra fine-tune at the final target")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--margin", type=float, default=0.02, help="L1 budget margin during training (l1 only)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--kd", choices=["none", "self"], default="self", help="Distil from the input model")
    p.add_argument("--kd_temp", type=float, default=4.0)
    p.add_argument("--kd_alpha", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="checkpoints_accum")
    p.add_argument("--results_dir", type=str, default="results_accum")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    s = parse_setting(args.ckpt)
    narrow = s["narrow"]
    tag = f"{s['base']}_{args.constraint}_B{args.b_target}"
    out_ckpt = os.path.join(args.save_dir, f"{tag}.pth")
    out_json = os.path.join(args.results_dir, f"{tag}.json")
    if os.path.exists(out_ckpt) and os.path.exists(out_json) and not args.overwrite:
        print(f"[SKIP] {tag} (already done)")
        return
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.set_seed(args.seed)
    train_l, val_l, test_l = utils.get_data_loaders(batch_size=args.batch)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Pruned checkpoints carry their architecture; dense ones use the default VGG9 configuration
    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck.get("cfg") if isinstance(ck, dict) else None
    cls_cfg = ck.get("cls_cfg") if isinstance(ck, dict) else None
    model = build_model(s, cfg=cfg, cls_cfg=cls_cfg)
    accum.load_state(model, args.ckpt)
    model.to(device)
    print(f"=== Accumulator restriction: {s['base']} -> {tag} | cfg={model.cfg} cls={model.cls_cfg} ===", flush=True)

    teacher = None
    if args.kd == "self":
        teacher = copy.deepcopy(model).eval()
        for q in teacher.parameters():
            q.requires_grad_(False)

    sample = next(iter(val_l))[0][:64].to(device)
    reg = accum.layer_registry(model, sample)
    prof0 = accum.profile(reg, narrow)
    B_dense = {r["name"]: r["B_cert_max"] for r in prof0}
    first = reg[0]["name"]

    # Per-layer final target; layers already at or below the target are left untouched
    B_final = {n: min(args.b_target, b) for n, b in B_dense.items()}
    if not args.constrain_first:
        B_final[first] = B_dense[first]
    active = [n for n in B_dense if B_final[n] < B_dense[n]]
    K_final = {r["name"]: accum.fanin_K(r, B_final[r["name"]], narrow) for r in reg if r["name"] in active}
    for n, K in K_final.items():
        if K == 0:
            print(f"  WARNING: {n} has K=0 at B={B_final[n]}: the constraint removes every weight of this layer")

    _, val0 = utils.evaluate(model, val_l, criterion, device)
    _, test0 = utils.evaluate(model, test_l, criterion, device)
    _, pbs0, _ = pbs_cost(model, sample[:4])
    print(f"  initial: val={val0:.4f} test={test0:.4f} PBS={pbs0} B_cert={B_dense}", flush=True)
    print(f"  targets (active layers): { {n: B_final[n] for n in active} } | K={K_final}", flush=True)

    rec = dict(
        tag=tag, method=f"{s['prefix']}_accum_{args.constraint}", source_ckpt=args.ckpt, out_ckpt=out_ckpt,
        setting=dict(w=s["w"], a=s["a"], inp=s["inp"], narrow=narrow), args=vars(args),
        time_start=time.strftime("%Y-%m-%d %H:%M"),
        initial=dict(**arch_of(model), params=count_params(model), val_acc=val0, test_acc=test0, pbs_theory=pbs0,
                     layers=[dict(name=r["name"], B_cert_max=r["B_cert_max"], K_max=r["K_max"],
                                  L1_max=r["L1_max"], fan_in=r["fan_in"]) for r in prof0]),
        B_dense=B_dense, B_final=B_final, active_layers=active, K_final=K_final, stages=[],
    )

    if not active:
        print("  Nothing to do: every layer is already at or below the target.")

    state = dict(reg=reg, Bs={}, masks={})
    post_step = make_post_step(state, args, narrow)
    kd = dict(teacher=teacher, kd_temp=args.kd_temp, kd_alpha=args.kd_alpha) if teacher is not None else {}

    # Stage schedule: linear from B_dense to B_final, then an extra fine-tune at the final target
    plan = [(st, args.epochs_per_stage) for st in range(1, args.stages + 1)]
    if args.final_epochs > 0:
        plan.append((args.stages, args.final_epochs))

    t_all = time.time()
    best_val, best_state = -1.0, None
    for i, (st, n_ep) in enumerate(plan if active else []):
        is_last = i == len(plan) - 1
        Bs = {n: math.ceil(B_dense[n] - (B_dense[n] - B_final[n]) * st / args.stages) for n in active}
        state["reg"] = accum.layer_registry(model, sample)
        state["Bs"] = Bs
        if args.constraint == "zama":
            state["masks"] = zama_masks(state["reg"], Bs, narrow)
        post_step()  # impose the new stage target before training
        print(f"\n  [stage {i + 1}/{len(plan)}] targets={Bs} epochs={n_ep}", flush=True)

        # Learning-rate rewinding: fresh optimizer and cosine schedule for every stage
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_ep, eta_min=1e-6)
        hist, t_s = [], time.time()
        for ep in range(n_ep):
            if args.constraint != "zama":
                state["reg"] = accum.layer_registry(model, sample)  # activation scales may drift
            _, t_acc, _ = utils.training_phase(model, train_l, criterion, optimizer, device, clip_weights=True,
                                               post_step=post_step, **kd)
            scheduler.step()
            _, v_acc = utils.evaluate(model, val_l, criterion, device)
            hist.append(dict(epoch=ep + 1, train_acc=float(t_acc), val_acc=float(v_acc)))
            print(f"    ep {ep + 1:03d}/{n_ep} t_acc={t_acc:.4f} v_acc={v_acc:.4f}", flush=True)
            # Best validation model of the final-target stage (all of its epochs satisfy the final target)
            if is_last and v_acc > best_val:
                best_val = v_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        rec["stages"].append(dict(stage=i + 1, targets=Bs, epochs=n_ep, history=hist,
                                  best_val=max(h["val_acc"] for h in hist), time_sec=round(time.time() - t_s, 1)))
        save_json(rec, out_json)  # partial progress

    if best_state is not None:
        model.load_state_dict(best_state)

    model.to(device)  # Brevitas may restore some buffers on CPU from the CPU state_dict
    _, val_pre = utils.evaluate(model, val_l, criterion, device)
    print(f"  val before certification: {val_pre:.4f}", flush=True)
    # Safety snapshot before certification, so training is never lost
    torch.save(dict(state_dict=model.state_dict(), **arch_of(model), setting=rec["setting"],
                    source_ckpt=args.ckpt, constraint=args.constraint, b_target=args.b_target),
                    out_ckpt.replace(".pth", "_precert.pth"))
    
    # Final check: exact integer projection for l1 (certification), plain profile otherwise
    reg = accum.layer_registry(model, sample)
    cert_error = None
    if args.constraint == "l1" and active:
        try:
            accum.certify(reg, {n: B_final[n] for n in active}, narrow=narrow)
        except RuntimeError as e:
            cert_error = str(e)[:500]
            print(f"  CERTIFICATION FAILED: {cert_error}")
        reg = accum.layer_registry(model, sample)
    prof = accum.profile(reg, narrow)
    meets = {r["name"]: r["B_cert_max"] <= B_final[r["name"]] for r in prof if r["name"] in active}

    _, val_f = utils.evaluate(model, val_l, criterion, device)
    _, test_f = utils.evaluate(model, test_l, criterion, device)
    full = profile_model(model, s, val_l, device)

    torch.save(dict(state_dict=model.state_dict(), **arch_of(model), setting=rec["setting"],
                    source_ckpt=args.ckpt, constraint=args.constraint, b_target=args.b_target), out_ckpt)

    rec["final"] = dict(
        **arch_of(model), val_acc=val_f, test_acc=test_f, acc_drop_test=test0 - test_f,
        certified=bool(active) and all(meets.values()) and cert_error is None, cert_error=cert_error,
        meets_target=meets,
        layers=[dict(name=r["name"], B_target=B_final[r["name"]], B_cert_max=r["B_cert_max"], K_max=r["K_max"],
                     K_mean=r["K_mean"], L1_max=r["L1_max"], int_sparsity=r["int_sparsity"]) for r in prof],
        profile=full,
    )
    rec["total_time_sec"] = round(time.time() - t_all, 1)
    rec["time_end"] = time.strftime("%Y-%m-%d %H:%M")
    save_json(rec, out_json)

    print(f"\n  FINAL [{tag}]: test={test_f:.4f} (drop {test0 - test_f:+.4f}) | certified={rec['final']['certified']} "
          f"| B_cert={full['B_cert_global']} B_emp={full['B_emp_global']}")
    for L in rec["final"]["layers"]:
        flag = "" if L["name"] not in active else ("  OK" if meets[L["name"]] else "  EXCEEDS")
        print(f"    {L['name']:<16} B_cert={L['B_cert_max']:3d} target={L['B_target']:3d} "
              f"K_max={L['K_max']:5d} sparsity={L['int_sparsity']:.3f}{flag}")
    print(f"  Saved {out_ckpt} and {out_json}")


if __name__ == "__main__":
    main()
