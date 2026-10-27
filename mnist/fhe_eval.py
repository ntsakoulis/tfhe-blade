import argparse
import json
import os
import time
import numpy as np
import torch

import accum

import arc_mnist as arc
import utils_mnist as utils

p = argparse.ArgumentParser(description="MNIST: evaluate PyTorch models and compile/simulate/execute in Concrete-ML.")
p.add_argument("--ckpt", required=True, type=str)
p.add_argument("--method", required=True, type=str)
p.add_argument("--tag", required=True, type=str)
p.add_argument("--w_bits", type=int, default=2)
# Added explicit a_bits argument
p.add_argument("--a_bits", type=int, default=2)
p.add_argument("--in_bit_width", type=int, default=2)
p.add_argument("--narrow_range", type=int, default=1)
p.add_argument("--unsigned_requant", type=int, default=0)
p.add_argument("--B_target", type=int, default=None)
# Rounding is a string so that "None" can be passed safely from bash scripts
p.add_argument("--rounding", type=str, default="None")
p.add_argument("--calib_n", type=int, default=500)
p.add_argument("--sim_n", type=int, default=10000)
p.add_argument("--calib_adv", type=int, default=0,
               help="Adversarial images per layer added to the calibration set (0 = off)")
p.add_argument("--adv_steps", type=int, default=40)
p.add_argument("--adv_alpha", type=float, default=0.01)
# Number of real FHE executions. Samples are picked round-robin over classes:
#   1 -> one sample, 3 -> three different classes, 10 -> exactly one per class,
#   >10 -> a second sample per class, and so on. Use 0 to skip FHE execution.
p.add_argument("--exec_n", type=int, default=0,
               help="Number of FHE execution samples (10 = one per class, 0 = skip)")
# One untimed execution before measuring, to exclude one-time overhead
# (key loading, thread pool start-up) from the latency numbers
p.add_argument("--no_warmup", action="store_true",
               help="Disable the untimed warm-up FHE run")
p.add_argument("--no_compile", action="store_true")
p.add_argument("--out", default="results_fhe")
p.add_argument("--dump_graph", action="store_true", help="Write the compiled graph to a text file")
p.add_argument("--exec_rest", type=int, default=5,
               help="Seconds of rest between timed FHE executions (CPU cool-down)")
a = p.parse_args()

os.makedirs(a.out, exist_ok=True)
utils.set_seed(0)


def jsonable(o):
    """Recursively converts numpy/torch types to native Python types for JSON serialization."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    return str(o)


def select_class_balanced(labels, n, n_classes=10):
    """Pick n indices round-robin over classes (class 0, 1, ..., 9, 0, 1, ...).

    n=10 gives exactly one sample per class. If a class has no more samples,
    it is skipped, so fewer than n indices may be returned in degenerate cases.
    """
    per_class = {c: list(np.where(labels == c)[0]) for c in range(n_classes)}
    idx, rnd = [], 0
    max_rounds = max((len(v) for v in per_class.values()), default=0)
    while len(idx) < n and rnd < max_rounds:
        for c in range(n_classes):
            if len(idx) >= n:
                break
            if rnd < len(per_class[c]):
                idx.append(int(per_class[c][rnd]))
        rnd += 1
    return idx

def circuit_layer_stats(circuit):
    """Per-layer statistics of the compiled circuit.

    Linear nodes (conv / matmul / avgpool) carry the ONNX layer name as tag; table lookups (PBS)
    are untagged, so each one is attributed to the nearest tagged node feeding it.
    """
    g = circuit.graph

    def layer_of(node, depth=0):
        # "/features.0/Conv.conv" -> "/features.0/Conv"
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
        dt = n.output.dtype
        if type(dt).__name__ != "Integer":
            continue
        bw = max(n.properties.get("original_bit_width", dt.bit_width), n.properties.get("bit_width_hint", -1))
        lo, hi = (None, None) if n.bounds is None else (as_int(n.bounds[0]), as_int(n.bounds[1]))
        is_tlu = getattr(n, "converted_to_table_lookup", False)
        if not (is_tlu or n.tag):
            continue
        tag = layer_of(n) or "(input)"
        d = out.setdefault(tag, dict(module=tag.strip("/").split("/")[0], acc_bits=-1, acc_min=None,
                                     acc_max=None, pbs=0, tlu_in_bits=-1, ops=[]))
        if is_tlu:
            d["pbs"] += int(np.prod(n.output.shape)) if n.output.shape else 1
            pin = max((p.properties.get("original_bit_width", p.output.dtype.bit_width)
                       for p in g.ordered_preds_of(n) if type(p.output.dtype).__name__ == "Integer"), default=-1)
            d["tlu_in_bits"] = max(d["tlu_in_bits"], pin)
        else:
            name = n.properties.get("name", "")
            if name and name not in d["ops"]:
                d["ops"].append(name)
            if name in ("conv2d", "matmul", "dot", "add", "subtract", "sum", "multiply"):
                d["acc_bits"] = max(d["acc_bits"], bw)
                if lo is not None:
                    d["acc_min"] = lo if d["acc_min"] is None else min(d["acc_min"], lo)
                    d["acc_max"] = hi if d["acc_max"] is None else max(d["acc_max"], hi)
    return out


def add_adversarial_calibration(model, calib, val_l, n_per_layer, steps, alpha):
    """Append inputs that maximise every layer's |accumulator| to the calibration set.

    Concrete sizes each accumulator from the calibration values, so these
    worst-case-seeking inputs push the compiled bit-widths towards B_cert.
    Start points come from the validation set (never the test set).
    """
    import copy
    from overflow_test import AccProbe, adversarial, pixel_bounds
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = copy.deepcopy(model).to(dev).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    reg_d = accum.layer_registry(m, next(iter(val_l))[0][:64].to(dev))
    probe = AccProbe(m, reg_d)
    lo, hi = pixel_bounds(val_l)
    lo, hi = lo.to(dev), hi.to(dev)
    xs = torch.cat([x for x, _ in val_l])
    adv = []
    for i, r in enumerate(reg_d):
        x0 = xs[i * n_per_layer:(i + 1) * n_per_layer].to(dev)  # different start points per layer
        adv.append(adversarial(m, probe, r["name"], x0, lo, hi, steps, alpha).cpu())
    probe.remove()
    base = torch.as_tensor(calib)
    out = torch.cat([base, torch.cat(adv).to(base.dtype)])
    return out.numpy() if isinstance(calib, np.ndarray) else out


# Parse rounding argument safely
rnd_val = None if a.rounding.lower() == "none" else int(a.rounding)

_, val_l, test_l = utils.get_data_loaders(batch_size=128, augmentation=False)

# Pruned checkpoints store their own architecture (cfg, cls_cfg); dense ones do not
ck = torch.load(a.ckpt, map_location="cpu")
arch_cfg = ck.get("cfg") if isinstance(ck, dict) else None
arch_cls = ck.get("cls_cfg") if isinstance(ck, dict) else None
model = arc.nnet_mnist(w_bit_width=a.w_bits, act_bit_width=a.a_bits, in_bit_width=a.in_bit_width,
                         narrow_range=bool(a.narrow_range), unsigned_requant=bool(a.unsigned_requant),
                         cfg=arch_cfg, cls_cfg=arch_cls)
accum.load_state(model, a.ckpt)
model.eval()

# Perform mathematical (worst-case) and empirical profiling
sample = next(iter(val_l))[0][:64]
reg = accum.layer_registry(model, sample)
prof = accum.profile(reg, narrow=bool(a.narrow_range))
emp = accum.empirical_bits(model, reg, val_l, max_batches=10)
rq = accum.requant_pbs(model, sample)
acc = accum.acc_stats(model, reg, val_l, max_batches=10)

for row in prof:
    row["B_emp"] = emp.get(row["name"])
    row["acc_dist"] = acc.get(row["name"])

# Evaluate PyTorch (cleartext) accuracy
_, acc_test = utils.evaluate(model, test_l, torch.nn.CrossEntropyLoss(), "cpu")

# Initialize the record dictionary
rec = dict(
    tag=a.tag,
    method=a.method,
    ckpt=a.ckpt,
    B_target=a.B_target,
    rounding_bits=rnd_val,
    setting=dict(w=a.w_bits, a=a.a_bits, inp=a.in_bit_width, narrow=bool(a.narrow_range)),
    time=time.strftime("%Y-%m-%d %H:%M"),
    cpu_count=os.cpu_count(),  # for reproducibility of latency numbers
    torch_threads=torch.get_num_threads(),
    acc_torch_test=acc_test,
    B_cert_global=max(r["B_cert_max"] for r in prof),
    # default=None avoids a ValueError if every B_emp is None/0
    B_emp_global=max((r["B_emp"] for r in prof if r["B_emp"]), default=None),
    pbs_convfc=sum(r["pbs"] for r in prof),
    pbs_requant=sum(rq.values()),
    pbs_total=sum(r["pbs"] for r in prof) + sum(rq.values()),
    layers=prof
)

print(f"[{a.tag}] acc={acc_test:.4f}  B_cert={rec['B_cert_global']}  "
      f"B_emp={rec['B_emp_global']}  Rounding={rnd_val}")

if not a.no_compile:
    from concrete.fhe import Configuration, ParameterSelectionStrategy
    from concrete.ml.torch.compile import compile_brevitas_qat_model

    # Prepare calibration and simulation datasets
    calib = utils.get_calibration_set(val_l, num_samples=a.calib_n)
    if a.calib_adv > 0:
        calib = add_adversarial_calibration(model, calib, val_l, a.calib_adv, a.adv_steps, a.adv_alpha)
        rec["calib_adv"] = dict(per_layer=a.calib_adv, steps=a.adv_steps, alpha=a.adv_alpha,
                                total=int(len(calib)))
        print(f"  CALIBRATION -> {len(calib)} samples (incl. adversarial)")
    xs, ys = [], []
    for x, y in test_l:
        xs.append(x)
        ys.append(y)
        if sum(t.shape[0] for t in xs) >= a.sim_n:
            break
    X = np.ascontiguousarray(torch.cat(xs)[:a.sim_n].numpy(), dtype=np.float32)
    Y = torch.cat(ys)[:a.sim_n].numpy()

    cfg = Configuration(parameter_selection_strategy=ParameterSelectionStrategy.MULTI)
    rnd_dict = {"n_bits": rnd_val, "method": "APPROXIMATE"} if rnd_val else None

    # ------------------------------------------------------------------
    # Stage 1: compilation + simulation
    # ------------------------------------------------------------------
    qm, c = None, None
    compiled_ok = False
    try:
        t0 = time.time()
        qm = compile_brevitas_qat_model(model, calib, configuration=cfg, rounding_threshold_bits=rnd_dict)
        rec["compile_s"] = round(time.time() - t0, 1)
        if a.dump_graph:  # compiled graph, to locate the widest integer nodes
            with open(os.path.join(a.out, f"{a.tag}_graph.txt"), "w") as f:
                f.write(qm.fhe_circuit.graph.format(maximum_constant_length=8))
        # Extract circuit statistics
        c = qm.fhe_circuit
        st = c.statistics
        st = st() if callable(st) else st
        rec["B_meas"] = int(c.graph.maximum_integer_bit_width())
        rec["pbs"] = st.get("programmable_bootstrap_count") if isinstance(st, dict) else "N/A"
        rec["complexity"] = getattr(c, "complexity", "N/A")
        # Key / ciphertext sizes (bytes) and operation counts, available without key generation
        if isinstance(st, dict):
            rec["circuit_stats"] = {k: v for k, v in st.items() if isinstance(v, (int, float))}
            # ===== NEW: ciphertext expansion from the compiler statistics (no key generation needed) =====
            in_ct = st.get("size_of_inputs")
            out_ct = st.get("size_of_outputs")
            in_plain = int(np.prod(X.shape[1:])) * a.in_bit_width / 8  # one quantized image, in bytes
            if in_ct:
                rec["ciphertext"] = dict(input_bytes=int(in_ct), output_bytes=int(out_ct) if out_ct else None,
                                         plain_input_bytes=in_plain,
                                         expansion_ratio=round(in_ct / in_plain, 1))
                print(f"  CIPHERTEXT -> input {in_ct / 2**20:.2f} MB (x{rec['ciphertext']['expansion_ratio']} "
                      f"vs {int(in_plain)} B plaintext) | output "
                      f"{(out_ct or 0) / 2**10:.1f} KB")
            # ===== END NEW =====
        # Per-layer view of the compiled circuit (bit-width, value range, PBS)
        try:
            rec["circuit_layers"] = circuit_layer_stats(c)
        except Exception as e:
            rec["circuit_layers_error"] = str(e)[:300]

        # Theory vs. compiled circuit: PBS count check
        if isinstance(rec["pbs"], int):
            rec["pbs_check"] = dict(
                theory_convfc=rec["pbs_convfc"],
                theory_total=rec["pbs_total"],
                compiled=rec["pbs"],
                diff=rec["pbs"] - rec["pbs_total"],
                match=rec["pbs"] == rec["pbs_total"],
            )
            status = "MATCH" if rec["pbs_check"]["match"] else f"MISMATCH (diff={rec['pbs_check']['diff']:+d})"
            print(f"  PBS CHECK -> theory={rec['pbs_total']} | compiled={rec['pbs']} | {status}")
        
        # FHE simulation vs. clear (quantized) execution
        p_dis, p_sim = [], []
        for i in range(0, len(X), 100):
            xb = X[i:i + 100]
            p_dis.append(qm.forward(xb, fhe="disable").argmax(1))
            p_sim.append(qm.forward(xb, fhe="simulate").argmax(1))

        p_dis, p_sim = np.concatenate(p_dis), np.concatenate(p_sim)
        # Cleartext Brevitas predictions on the same subset (batched to bound memory)
        p_tor = []
        with torch.no_grad():
            for i in range(0, len(X), 100):
                o = model(torch.from_numpy(X[i:i + 100]))
                o = o.value if hasattr(o, "value") else o
                p_tor.append(o.argmax(1).numpy())
        p_tor = np.concatenate(p_tor)
        rec["sim"] = dict(n=int(len(Y)),
                          acc_torch=float((p_tor == Y).mean()),
                          acc_disable=float((p_dis == Y).mean()),
                          acc_sim=float((p_sim == Y).mean()),
                          disagree_disable_vs_torch=float((p_dis != p_tor).mean()),
                          disagree_sim_vs_disable=float((p_sim != p_dis).mean()))

        print(f"  COMPILE OK -> Time: {rec['compile_s']}s | B_meas: {rec['B_meas']} | "
              f"PBS Count: {rec['pbs']} | Complexity: {rec['complexity']}")
        print(f"  SIMULATION (n={rec['sim']['n']}) -> Acc_torch: {rec['sim']['acc_torch']:.4f} | "
              f"Acc_disable: {rec['sim']['acc_disable']:.4f} | Acc_sim: {rec['sim']['acc_sim']:.4f} | "
              f"Disagree dis/torch: {rec['sim']['disagree_disable_vs_torch']:.4f} | "
              f"Disagree sim/dis: {rec['sim']['disagree_sim_vs_disable']:.4f}")
        compiled_ok = True

    except Exception as e:
        rec["compile_error"] = str(e)[:1000]
        print(f"  COMPILE/SIM FAILED: {str(e)[:200]}")

    # ------------------------------------------------------------------
    # Stage 2: real FHE execution
    # ------------------------------------------------------------------
    if compiled_ok and a.exec_n > 0:
        try:
            exec_idx = select_class_balanced(Y, a.exec_n)
            X_exec, Y_exec = X[exec_idx], Y[exec_idx]
            rec["exec_n"] = len(exec_idx)
            rec["exec_classes"] = Y_exec.tolist()
            print(f"\n  --- Measuring FHE Execution Latency & Key Sizes "
                  f"({len(exec_idx)} samples, classes: {Y_exec.tolist()}) ---")

            # Key generation (client side)
            print("  -> Generating FHE keys (this might take a while for B>10)... ", end="", flush=True)
            t_key = time.time()
            c.keygen()
            rec["keygen_s"] = round(time.time() - t_key, 2)
            print(f"Done in {rec['keygen_s']}s")

            # Evaluation key size (network overhead metric)
            try:
                eval_keys_serialized = c.client.evaluation_keys.serialize()
                rec["eval_key_size_mb"] = round(len(eval_keys_serialized) / (1024 * 1024), 2)
                print(f"  -> Evaluation Keys Size: {rec['eval_key_size_mb']} MB")
            except Exception:
                rec["eval_key_size_mb"] = None
                print("  -> Evaluation Keys Size: Could not determine automatically.")



            # Untimed warm-up run
            if not a.no_warmup:
                print("  -> Warm-up run (not timed)... ", end="", flush=True)
                qm.forward(X_exec[:1], fhe="execute")
                print("Done")

            # Timed execution, stage by stage: quantize + encrypt (client), run (server), decrypt (client)
            exec_times, fhe_preds, stages = [], [], []
            for i in range(len(X_exec)):
                if i > 0 and a.exec_rest > 0:
                    print(f"    -> Rest {a.exec_rest}s (cool-down)... ", end="", flush=True)
                    time.sleep(a.exec_rest)
                    print("Done")
                x_single = X_exec[i:i + 1]
                print(f"    -> FHE inference {i + 1}/{len(X_exec)} (class {Y_exec[i]})... ", end="", flush=True)

                t0 = time.time()
                q_in = qm.quantize_input(x_single)
                enc = c.encrypt(q_in)
                t1 = time.time()
                res = c.run(enc)
                t2 = time.time()
                q_out = c.decrypt(res)
                out = qm.dequantize_output(q_out)
                t3 = time.time()

                enc_v = enc[0] if isinstance(enc, tuple) else enc
                res_v = res[0] if isinstance(res, tuple) else res
                st_i = dict(encrypt_s=round(t1 - t0, 3), run_s=round(t2 - t1, 2), decrypt_s=round(t3 - t2, 3),
                            total_s=round(t3 - t0, 2))
                try:
                    st_i["input_ct_bytes"] = len(enc_v.serialize())
                    st_i["output_ct_bytes"] = len(res_v.serialize())
                except Exception:
                    pass
                stages.append(st_i)
                exec_times.append(t3 - t0)
                fhe_preds.append(int(np.asarray(out).argmax(1)[0]))
                print(f"Done in {st_i['total_s']:.2f}s (encrypt {st_i['encrypt_s']:.2f}s | "
                      f"run {st_i['run_s']:.2f}s | decrypt {st_i['decrypt_s']:.2f}s)", flush=True)

            fhe_preds = np.array(fhe_preds)

            # Correctness check: real FHE output vs. simulation on the same samples
            sim_preds = qm.forward(X_exec, fhe="simulate").argmax(1)

            rec["exec_stages"] = stages
            rec["exec_rest_s"] = a.exec_rest
            rec["exec_times_s"] = [round(t, 2) for t in exec_times]
            rec["latency_mean_s"] = round(float(np.mean(exec_times)), 2)
            rec["latency_std_s"] = round(float(np.std(exec_times)), 2)
            rec["run_mean_s"] = round(float(np.mean([s["run_s"] for s in stages])), 2)
            rec["warmup_used"] = not a.no_warmup
            rec["fhe_matches_sim"] = bool((fhe_preds == sim_preds).all())
            rec["fhe_acc_exec"] = float((fhe_preds == Y_exec).mean())

            print(f"  [FINAL FHE METRICS] Latency: {rec['latency_mean_s']}s ± {rec['latency_std_s']}s "
                  f"per image (server run: {rec['run_mean_s']}s, n={len(exec_times)}) | "
                  f"KeyGen: {rec['keygen_s']}s | FHE==Sim: {rec['fhe_matches_sim']}")
                

        except Exception as e:
            rec["exec_error"] = str(e)[:1000]
            print(f"  FHE EXECUTION FAILED: {str(e)[:200]}")

# Save all results to JSON
path = os.path.join(a.out, f"{a.tag}.json")
with open(path, "w") as f:
    json.dump(jsonable(rec), f, indent=2)
print(f"Saved results to {path}")