import math
import numpy as np
import torch
import torch.nn as nn
import brevitas.nn as qnn

QLAYERS = (qnn.QuantConv2d, qnn.QuantLinear)

@torch.no_grad()
def layer_registry(model, sample_batch):
    model.eval()
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, QLAYERS)]
    info, hooks = {}, []

    def mk(name):
        def h(m, inp, out):
            x = inp[0]
            o = out.value if hasattr(out, "value") else out
            bw, sg, sc = getattr(x, "bit_width", None), getattr(x, "signed", None), getattr(x, "scale", None)
            info[name] = dict(in_bits=int(bw.item()) if bw is not None else None,
                              in_signed=bool(sg) if sg is not None else None,
                              in_scale=float(sc.flatten()[0]) if sc is not None else None,
                              out_numel=int(o[0].numel()))
        return h

    for n, m in layers:
        hooks.append(m.register_forward_hook(mk(n)))
    model(sample_batch.to(next(model.parameters()).device))
    for h in hooks:
        h.remove()

    reg = []
    for i, (n, m) in enumerate(layers):
        d = info[n]
        first, last = (i == 0), (i == len(layers) - 1)
        b = d["in_bits"]
        if first:
            # BUG FIX: Signed input (CIFAR is normalized), range [-2^(b-1), 2^(b-1)-1]
            nonneg, in_max = False, 2 ** (b - 1)
        elif d["in_signed"]:
            nonneg, in_max = True, 2 ** (b - 1) - 1
        else:
            nonneg, in_max = True, 2 ** b - 1
            
        qw = m.quant_weight()
        w_bits = int(qw.bit_width.item())
        fan_in = int(m.weight[0].numel())
        
        reg.append(dict(name=n, module=m, kind="conv" if isinstance(m, qnn.QuantConv2d) else "fc",
                        fan_in=fan_in, out_ch=int(m.weight.shape[0]),
                        pbs=0 if last else d["out_numel"], in_bits=b, in_signed=d["in_signed"],
                        nonneg=nonneg, in_max=float(in_max), in_scale=d["in_scale"],
                        w_bits=w_bits, last=last))
    return reg

def int_weight(m):
    qw = m.quant_weight()
    s = qw.scale.detach()
    q = torch.round(qw.value.detach() / s)
    return q.reshape(q.shape[0], -1), s

def int_bias(m, in_scale, w_scale):
    if m.bias is None:
        return torch.zeros(m.weight.shape[0], device=m.weight.device)
    sb = in_scale * float(w_scale.flatten()[0])
    return torch.clamp(torch.round(m.bias.detach() / sb), -128, 127)

def neuron_bits(q, qb, in_max, nonneg):
    if nonneg:
        pos = in_max * q.clamp(min=0).sum(1) + qb.clamp(min=0)
        neg = in_max * (-q).clamp(min=0).sum(1) + (-qb).clamp(min=0)
    else:
        pos = neg = in_max * q.abs().sum(1) + qb.abs()
    b_pos = torch.ceil(torch.log2(pos + 1))
    b_neg = torch.where(neg > 0, torch.ceil(torch.log2(neg.clamp(min=1))), torch.zeros_like(neg))
    return (1 + torch.maximum(b_pos, b_neg)).long()

def w_max_int(w_bits, narrow=True):
    return 2 ** (w_bits - 1) - (1 if narrow else 0)

def naive_bits(K, in_max, w_max):
    return 1 + math.ceil(math.log2(max(K, 1) * in_max * w_max + 1))

def _cap_rows(v, budget):
    vs, idx = torch.sort(v, dim=1, descending=True)
    prev = torch.cumsum(vs, 1) - vs
    kept = torch.minimum(vs, (budget[:, None] - prev).clamp(min=0))
    return torch.zeros_like(v).scatter(1, idx, kept)

def _l1_budgets(B, qb, in_max, nonneg):
    top = 2 ** (B - 1)
    if nonneg:
        bp = torch.floor((top - 1 - qb.clamp(min=0)) / in_max).clamp(min=0)
        bn = torch.floor((top - (-qb).clamp(min=0)) / in_max).clamp(min=0)
        return bp, bn
    b = torch.floor((top - 1 - qb.abs()) / in_max).clamp(min=0)
    return b, b

@torch.no_grad()
def project_l1(r, B, exact=False, margin=0.0):
    """Integer-aware L1 projection for C-AAP.

    The budget is enforced on the quantized integer weights (what the FHE
    accumulator actually sees). Weights whose integer value is kept also keep
    their latent float value, so training can still move them; only the weights
    that must shrink are snapped to their reduced integer value.
    """
    m = r["module"]
    W = m.weight.data
    n = W.shape[0]
    q, s = int_weight(m)          # clamped integer weights, shape (n, fan_in)
    u = W.reshape(n, -1) / s      # latent weights in integer units
    qb = int_bias(m, r["in_scale"], s)
    bp, bn = _l1_budgets(B, qb, r["in_max"], r["nonneg"])
    if not exact and margin > 0:
        bp, bn = torch.floor(bp * (1 - margin)), torch.floor(bn * (1 - margin))

    def cap(qmag, umag, budget):
        # Greedy cap on integer magnitudes; entries with the largest latent magnitude are kept first
        order = torch.argsort(umag, dim=1, descending=True)
        qs = torch.gather(qmag, 1, order)
        prev = torch.cumsum(qs, 1) - qs
        kept = torch.minimum(qs, (budget[:, None] - prev).clamp(min=0))
        return torch.zeros_like(qmag).scatter(1, order, kept)

    if r["nonneg"]:
        q_new = cap(q.clamp(min=0), u.clamp(min=0), bp) - cap((-q).clamp(min=0), (-u).clamp(min=0), bn)
    else:
        q_new = cap(q.abs(), u.abs(), bp) * torch.sign(q)

    u_new = q_new if exact else torch.where(q_new != q, q_new, u)
    W.copy_((u_new * s).reshape_as(W))


def fanin_K(r, B, narrow=True):
    """ABLATION: Top-K per neuron math."""
    m = r["module"]
    _, s = int_weight(m)
    qb = int_bias(m, r["in_scale"], s).abs().max().item()
    K = int((2 ** (B - 1) - 1 - qb) // (r["in_max"] * w_max_int(r["w_bits"], narrow)))
    return max(0, min(K, r["fan_in"]))

@torch.no_grad()
def project_fanin(r, B, narrow=True):
    """ABLATION: Projects to Top-K fan-in."""
    m = r["module"]
    W = m.weight.data
    n = W.shape[0]
    K = fanin_K(r, B, narrow)
    Wf = W.reshape(n, -1)
    if K >= Wf.shape[1]:
        return
    mask = torch.zeros_like(Wf)
    if K > 0:
        mask.scatter_(1, torch.topk(Wf.abs(), K, dim=1).indices, 1.0)
    Wf.mul_(mask)

@torch.no_grad()
def layer_stats(r, narrow=True):
    m = r["module"]
    q, s = int_weight(m)
    qb = int_bias(m, r["in_scale"], s)
    bits_i = neuron_bits(q, qb, r["in_max"], r["nonneg"])
    bits = bits_i.float()

    # Integer weight statistics
    qf = q.float()

    # Per-neuron fan-in (non-zero integer weights) and L1 norm (L1 drives B_cert)
    nnz = (q != 0).sum(1).float()
    l1 = q.abs().sum(1).float()
    qb_abs = qb.abs().float()
    wm = w_max_int(r["w_bits"], narrow)

    # Number of neurons per certified bit-width, e.g. {11: 40, 12: 150, 13: 66}
    vals, cnts = torch.unique(bits_i, return_counts=True)
    hist = {int(b): int(c) for b, c in zip(vals.tolist(), cnts.tolist())}

    return dict(
        name=r["name"], kind=r["kind"], fan_in=r["fan_in"], out_ch=r["out_ch"], pbs=r["pbs"],
        in_max=r["in_max"], nonneg=r["nonneg"],
        # Certified accumulator bit-width across neurons
        B_cert_min=int(bits.min()), B_cert_max=int(bits.max()), B_cert_mean=float(bits.mean()),
        B_cert_std=float(bits.std()), B_cert_var=float(bits.var()),
        B_cert_hist=hist,
        # Integer weight statistics
        W_min=float(qf.min()), W_max=float(qf.max()), W_mean=float(qf.mean()),
        W_std=float(qf.std()), W_var=float(qf.var()),
        # Fan-in / sparsity
        K_min=int(nnz.min()), K_mean=float(nnz.mean()), K_max=int(nnz.max()),
        int_sparsity=float(1 - nnz.sum() / q.numel()),
        # Per-neuron L1 norm
        L1_min=float(l1.min()), L1_mean=float(l1.mean()), L1_max=float(l1.max()), L1_std=float(l1.std()),
        # Integer bias
        bias_absmax=float(qb_abs.max()), bias_absmean=float(qb_abs.mean()),
        # Naive (fan-in based) bounds
        B_naive_Kmean=naive_bits(max(1, round(float(nnz.mean()))), r["in_max"], wm),
        B_naive_Kmax=naive_bits(int(nnz.max()), r["in_max"], wm),
    )


def profile(reg, narrow=True):
    return [layer_stats(r, narrow) for r in reg]

@torch.no_grad()
def certify(reg, budgets, max_iter=5, narrow=True):
    for _ in range(max_iter):
        for r in reg:
            if r["name"] in budgets:
                project_l1(r, budgets[r["name"]], exact=True)
        bad = [(r["name"], layer_stats(r, narrow)["B_cert_max"], budgets[r["name"]])
               for r in reg if r["name"] in budgets
               and layer_stats(r, narrow)["B_cert_max"] > budgets[r["name"]]]
        if not bad:
            return profile(reg, narrow)
    raise RuntimeError(f"Certification failed: {bad}")

@torch.no_grad()
def empirical_bits(model, reg, loader, max_batches=10):
    model.eval()
    obs, hooks = {}, []

    def mk(r):
        def h(m, inp, out):
            o = out.value if hasattr(out, "value") else out
            _, s = int_weight(m)
            acc = torch.round(o / (r["in_scale"] * float(s.flatten()[0])))
            obs[r["name"]] = max(obs.get(r["name"], 0.0), float(acc.abs().max()))
        return h

    for r in reg:
        hooks.append(r["module"].register_forward_hook(mk(r)))
    dev = next(model.parameters()).device
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        model(x.to(dev))
    for h in hooks:
        h.remove()
    return {k: (1 + math.ceil(math.log2(v + 1)) if v > 0 else 1) for k, v in obs.items()}

def load_state(model, path):
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    for k in [k for k in sd if k.endswith("weight_orig")]:
        base = k[: -len("_orig")]
        mask = sd.pop(base + "_mask", None)
        w = sd.pop(k)
        sd[base] = w * mask if mask is not None else w
    sd = {k: v for k, v in sd.items() if not k.endswith("weight_mask")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"WARNING: Missing : {missing}, Unexpected : {unexpected}")
    return model




@torch.no_grad()
def requant_pbs(model, sample_batch):
    """Extra table lookups from QuantIdentity re-quantizers (after pooling / before flatten)."""
    model.eval()
    counts, hooks = {}, []
    def mk(name):
        def h(m, inp, out):
            o = out.value if hasattr(out, "value") else out
            counts[name] = int(o[0].numel())
        return h
    for n, m in model.named_modules():
        if isinstance(m, qnn.QuantIdentity) and n != "quant_input":
            hooks.append(m.register_forward_hook(mk(n)))
    model(sample_batch.to(next(model.parameters()).device))
    for h in hooks:
        h.remove()
    return counts

@torch.no_grad()
def acc_stats(model, reg, loader, max_batches=10, sample_per_batch=200000):
    """Distribution of the integer accumulator of every conv / linear layer on real inputs.

    Returns {layer: dict(min, max, mean, std, var, abs_p99, abs_p999, abs_max, bits_max, bits_p999, n)}.
    bits_* use the same signed convention as empirical_bits (1 sign bit + magnitude bits).
    """
    model.eval()
    st, hooks = {}, []

    def mk(r):
        def h(m, inp, out):
            o = out.value if hasattr(out, "value") else out
            _, s = int_weight(m)
            a = torch.round(o / (r["in_scale"] * float(s.flatten()[0]))).double().flatten()
            d = st.setdefault(r["name"], dict(n=0, s=0.0, s2=0.0, mn=float("inf"), mx=float("-inf"), smp=[]))
            d["n"] += a.numel()
            d["s"] += float(a.sum())
            d["s2"] += float((a * a).sum())
            d["mn"] = min(d["mn"], float(a.min()))
            d["mx"] = max(d["mx"], float(a.max()))
            k = min(sample_per_batch, a.numel())
            idx = torch.randint(0, a.numel(), (k,), device=a.device)
            d["smp"].append(a[idx].abs().float().cpu())
        return h

    for r in reg:
        hooks.append(r["module"].register_forward_hook(mk(r)))
    dev = next(model.parameters()).device
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        model(x.to(dev))
    for h in hooks:
        h.remove()

    def bits(v):
        return 1 + math.ceil(math.log2(v + 1)) if v > 0 else 1

    res = {}
    for name, d in st.items():
        mean = d["s"] / d["n"]
        var = max(d["s2"] / d["n"] - mean * mean, 0.0)
        smp = torch.cat(d["smp"])
        p99, p999 = (float(torch.quantile(smp, q)) for q in (0.99, 0.999))
        amax = max(abs(d["mn"]), abs(d["mx"]))
        res[name] = dict(n=int(d["n"]), min=d["mn"], max=d["mx"], mean=mean, std=math.sqrt(var), var=var,
                         abs_p99=p99, abs_p999=p999, abs_max=amax,
                         bits_max=bits(amax), bits_p999=bits(p999))
    return res