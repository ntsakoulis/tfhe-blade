"""Accumulator overflow test: can real or crafted inputs exceed the bit-widths of the compiled FHE circuit?

Concrete sizes every accumulator from the values it sees on the calibration set. An input that drives an
accumulator outside that range overflows silently in FHE (wrong result, no error). This script measures
how often that happens, without FHE execution:

  1. The compiled bit-width of every conv / linear accumulator is read from an fhe_eval.py JSON
     ("circuit_layers" -> acc_bits, acc_min); use the JSON of the SAME checkpoint.
  2. The integer accumulator of every layer is recomputed in PyTorch (same convention as accum.acc_stats:
     output / (input_scale * weight_scale)); these values match the ranges reported by the compiler.
  3. Several input sets are passed through the model and every value outside the representable range of
     the compiled bit-width is counted as an overflow.

Input sets (all inside the valid pixel range [0, 1] per channel, after normalisation):
  natural    : the full CIFAR-10 test set
  constant   : white, black and mid-grey images
  noise      : uniform random pixels
  patterns   : checkerboards and stripes
  conv1_worst: for every first-layer neuron, a tiled 3x3 patch at +max where its weight is +1 and at
               -max where it is -1 (and the opposite sign), i.e. its certified worst case
  adversarial: projected sign-gradient ascent from test images that maximises the largest |accumulator|
               of one target layer (one run per layer)

A certified model (B_cert <= compiled bits for every layer) cannot overflow on any input.

Example:
  python overflow_test.py --ckpt checkpoints_B/B_VGG9_W2_A2_IN8_NR1.pth \
                          --fhe_json results_fhe_B/Profile_B_M1_W2A2_IN8_RndNone.json
"""
import argparse
import json
import os
import time

import torch

import accum
import utils_cifar10 as utils
from real_pruning import build_model, parse_setting, save_json


# ----------------------------------------------------------------------------
# Compiled bit-widths
# ----------------------------------------------------------------------------
def compiled_limits(fhe_json):
    """{layer: (bits, lo, hi)} from the circuit_layers block of an fhe_eval.py JSON."""
    with open(fhe_json) as f:
        rec = json.load(f)
    if "circuit_layers" not in rec:
        raise SystemExit(f"{fhe_json} has no 'circuit_layers' (run fhe_eval.py with compilation)")
    lim = {}
    for tag, d in rec["circuit_layers"].items():
        parts = tag.strip("/").split("/")          # "/features/features.11/Conv" -> features.11
        if len(parts) < 2 or d.get("acc_bits", -1) <= 0 or not parts[-1].split(".")[0] in ("Conv", "Gemm", "MatMul"):
            continue
        b = int(d["acc_bits"])
        signed = d.get("acc_min") is None or d["acc_min"] < 0
        lo, hi = (-(2 ** (b - 1)), 2 ** (b - 1) - 1) if signed else (0, 2 ** b - 1)
        lim[parts[-2]] = (b, lo, hi)
    return lim, rec


# ----------------------------------------------------------------------------
# Integer accumulators through forward hooks
# ----------------------------------------------------------------------------
class AccProbe:
    def __init__(self, model, reg):
        self.store, self.hooks = {}, []
        for r in reg:
            _, s = accum.int_weight(r["module"])
            div = r["in_scale"] * float(s.flatten()[0])
            self.hooks.append(r["module"].register_forward_hook(self._mk(r["name"], div)))

    def _mk(self, name, div):
        def h(m, inp, out):
            o = out.value if hasattr(out, "value") else out
            self.store[name] = o / div
        return h

    def remove(self):
        for h in self.hooks:
            h.remove()


class Counter:
    """Accumulates per-layer overflow statistics for one input set."""

    def __init__(self, names):
        self.n, self.any = 0, 0
        self.per = {n: dict(images=0, values=0, min=float("inf"), max=float("-inf")) for n in names}

    @torch.no_grad()
    def update(self, store, lim):
        flag = None
        for name, a in store.items():
            if name not in lim:
                continue
            a = torch.round(a.detach()).flatten(1)
            _, lo, hi = lim[name]
            bad = (a < lo) | (a > hi)
            img = bad.any(1)
            p = self.per[name]
            p["images"] += int(img.sum())
            p["values"] += int(bad.sum())
            p["min"] = min(p["min"], float(a.min()))
            p["max"] = max(p["max"], float(a.max()))
            flag = img if flag is None else (flag | img)
        bs = next(iter(store.values())).shape[0]
        self.n += bs
        self.any += int(flag.sum()) if flag is not None else 0

    def result(self):
        return dict(n=self.n, images_with_overflow=self.any,
                    frac_with_overflow=self.any / max(self.n, 1), layers=self.per)


# ----------------------------------------------------------------------------
# Input sets
# ----------------------------------------------------------------------------
def pixel_bounds(loader, n_batches=40):
    """Per-channel normalised values of pixel 0 and pixel 1 (min / max over real images)."""
    lo, hi = None, None
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        mn, mx = x.amin(dim=(0, 2, 3)), x.amax(dim=(0, 2, 3))
        lo = mn if lo is None else torch.minimum(lo, mn)
        hi = mx if hi is None else torch.maximum(hi, mx)
    return lo.view(1, 3, 1, 1), hi.view(1, 3, 1, 1)


def from_unit(u, lo, hi):
    """Map pixels in [0, 1] to the normalised input range."""
    return lo + u * (hi - lo)


def synthetic_sets(lo, hi, n_noise, conv1_w, seed):
    g = torch.Generator().manual_seed(seed)
    sets = {}
    sets["constant"] = from_unit(torch.tensor([1.0, 0.0, 0.5]).view(3, 1, 1, 1).expand(3, 3, 32, 32), lo, hi)
    sets["noise"] = from_unit(torch.rand(n_noise, 3, 32, 32, generator=g), lo, hi)

    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    pats = []
    for p in (1, 2, 4, 8):
        pats.append(((yy // p + xx // p) % 2).float())          # checkerboard
        pats.append(((xx // p) % 2).float())                      # vertical stripes
        pats.append(((yy // p) % 2).float())                      # horizontal stripes
    pats = torch.stack(pats).unsqueeze(1).expand(-1, 3, 32, 32)
    sets["patterns"] = from_unit(torch.cat([pats, 1 - pats]), lo, hi)

    # Certified worst case of every first-layer neuron: tile its weight-sign pattern over the image
    sign = torch.sign(conv1_w)                                    # (C_out, 3, k, k)
    k = sign.shape[-1]
    reps = (32 + k - 1) // k
    tiled = sign.repeat(1, 1, reps, reps)[:, :, :32, :32]
    u = (tiled > 0).float() + 0.5 * (tiled == 0).float()
    sets["conv1_worst"] = from_unit(torch.cat([u, 1 - u]), lo, hi)
    return sets


def adversarial(model, probe, target, x0, lo, hi, steps, alpha):
    """Sign-gradient ascent on the largest |accumulator| of one layer, projected to the pixel range."""
    x = x0.clone()
    step = alpha * (hi - lo)
    for _ in range(steps):
        x.requires_grad_(True)
        model(x)
        obj = probe.store[target].abs().flatten(1).amax(1).sum()
        grad, = torch.autograd.grad(obj, x)
        with torch.no_grad():
            x = torch.max(torch.min(x + step * grad.sign(), hi), lo)
    return x.detach()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Accumulator overflow test against the compiled FHE bit-widths")
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--fhe_json", required=True, type=str, help="fhe_eval.py JSON of the same checkpoint")
    p.add_argument("--n_noise", type=int, default=2000)
    p.add_argument("--adv_images", type=int, default=128, help="Test images used as starting points per layer")
    p.add_argument("--adv_steps", type=int, default=40)
    p.add_argument("--adv_alpha", type=float, default=0.01, help="Step as a fraction of the pixel range")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results_overflow")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.set_seed(args.seed)
    s = parse_setting(args.ckpt)

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck.get("cfg") if isinstance(ck, dict) else None
    cls_cfg = ck.get("cls_cfg") if isinstance(ck, dict) else None
    model = build_model(s, cfg=cfg, cls_cfg=cls_cfg)
    accum.load_state(model, args.ckpt)
    model.to(device).eval()
    for q in model.parameters():
        q.requires_grad_(False)

    lim, fhe_rec = compiled_limits(args.fhe_json)
    _, val_l, test_l = utils.get_data_loaders(batch_size=args.batch, augmentation=False)
    sample = next(iter(val_l))[0][:64].to(device)
    reg = accum.layer_registry(model, sample)
    prof = {r["name"]: r for r in accum.profile(reg, s["narrow"])}
    names = [r["name"] for r in reg if r["name"] in lim]
    missing = [r["name"] for r in reg if r["name"] not in lim]
    if missing:
        print(f"WARNING: no compiled bit-width for {missing} (not checked)")

    print(f"=== Overflow test: {s['base']} | compiled bits from {args.fhe_json} ===")
    print(f"{'layer':<16}{'B_cert':>7}{'compiled':>9}{'certified safe':>16}")
    layers_info = {}
    for n in names:
        b, lo_, hi_ = lim[n]
        safe = prof[n]["B_cert_max"] <= b
        layers_info[n] = dict(B_cert=prof[n]["B_cert_max"], compiled_bits=b, lo=lo_, hi=hi_, certified_safe=safe)
        print(f"{n:<16}{prof[n]['B_cert_max']:>7}{b:>9}{'yes' if safe else 'NO':>16}")

    probe = AccProbe(model, reg)
    lo, hi = pixel_bounds(test_l)
    lo, hi = lo.to(device), hi.to(device)
    results = {}
    t0 = time.time()

    def run(name, tensors):
        c = Counter(names)
        with torch.no_grad():
            for i in range(0, tensors.shape[0], args.batch):
                model(tensors[i:i + args.batch].to(device))
                c.update(probe.store, lim)
        results[name] = c.result()

    # 1. Natural test images
    c = Counter(names)
    with torch.no_grad():
        for x, _ in test_l:
            model(x.to(device))
            c.update(probe.store, lim)
    results["natural"] = c.result()

    # 2-5. Synthetic inputs
    conv1_w = accum.int_weight(reg[0]["module"])[0].view_as(reg[0]["module"].weight).cpu()
    for name, t in synthetic_sets(lo.cpu(), hi.cpu(), args.n_noise, conv1_w, args.seed).items():
        run(name, t)

    # 6. Adversarial search, one run per target layer
    x0 = torch.cat([x for x, _ in test_l])[:args.adv_images].to(device)
    adv = {}
    for target in names:
        xa = adversarial(model, probe, target, x0, lo, hi, args.adv_steps, args.adv_alpha)
        c = Counter(names)
        with torch.no_grad():
            model(xa)
            c.update(probe.store, lim)
        adv[target] = c.result()
    results["adversarial"] = dict(
        n=sum(r["n"] for r in adv.values()),
        images_with_overflow=sum(r["images_with_overflow"] for r in adv.values()),
        per_target=adv)
    results["adversarial"]["frac_with_overflow"] = (results["adversarial"]["images_with_overflow"]
                                                    / max(results["adversarial"]["n"], 1))
    probe.remove()

    # Report
    print(f"\n{'input set':<14}{'images':>8}{'overflow':>10}{'%':>8}   layers with overflow (images)")
    for name, r in results.items():
        if name == "adversarial":
            hit = {t: v["layers"][t]["images"] for t, v in r["per_target"].items() if v["layers"][t]["images"]}
        else:
            hit = {n: v["images"] for n, v in r["layers"].items() if v["images"]}
        print(f"{name:<14}{r['n']:>8}{r['images_with_overflow']:>10}{100 * r['frac_with_overflow']:>7.2f}%   {hit or '-'}")

    print(f"\n{'layer':<16}{'compiled range':>18}{'natural range':>18}{'worst seen':>18}")
    for n in names:
        lo_, hi_ = lim[n][1], lim[n][2]
        nat = results["natural"]["layers"][n]
        mins = [nat["min"]] + [results[k]["layers"][n]["min"] for k in results if k not in ("natural", "adversarial")]
        maxs = [nat["max"]] + [results[k]["layers"][n]["max"] for k in results if k not in ("natural", "adversarial")]
        for v in results["adversarial"]["per_target"].values():
            mins.append(v["layers"][n]["min"])
            maxs.append(v["layers"][n]["max"])
        comp = f"[{lo_}, {hi_}]"
        natr = f"[{nat['min']:.0f}, {nat['max']:.0f}]"
        worst = f"[{min(mins):.0f}, {max(maxs):.0f}]"
        print(f"{n:<16}{comp:>18}{natr:>18}{worst:>18}")

    tag = os.path.splitext(os.path.basename(args.fhe_json))[0]
    out = os.path.join(args.out, f"{tag}_overflow.json")
    save_json(dict(ckpt=args.ckpt, fhe_json=args.fhe_json, rounding_bits=fhe_rec.get("rounding_bits"),
                   args=vars(args), layers=layers_info, results=results,
                   time_sec=round(time.time() - t0, 1)), out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
