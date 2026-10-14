"""Cheap merge selection: score merge candidates by held-out val loss on the 4 expert domains, in memory.

The base model is loaded ONCE (same kernels as training). For each candidate the dense merged ΔW of the
250 adapted modules is computed exactly as merge_experts.py does (same sparsify/combine, same DARE seeds),
applied in place as W' = bf16(W + ΔW) (= what folding writes to disk), the 4 expert val sets are scored, and
W is restored. Only the chosen candidate is then folded to disk (scripts/merge_submit.sh experts ...) and run
on the 4 benchmarks, so the benchmark test sets are never used for selection.

  python -m src.merge.merge_select --config configs/merge/select.yaml [--set val_max_seq_len=32768 ...]

Candidates (`candidates:` list in the config):
  base                       ΔW = 0
  expert:<name>              that expert alone (its own adapter, re-scored under this protocol)
  adapter:<name>             a single LoRA from `extra_adapters` (e.g. the mixed control)
  {method, global_scale, density, drop_p, weights, name}   a K-expert merge (methods as in merge_experts.py)
Outputs in `out_dir`: rows.jsonl (one row per candidate, appended as it finishes -> resumable), table.csv,
table.md. Per domain d: val loss L_d and retention R_d = (L_d(base) - L_d(cand)) / (L_d(base) - L_d(expert_d)),
the fraction of expert d's own val-loss gain that the candidate keeps (1.0 = as good as the expert on d).
Ranking: mean retention over domains (ties broken by mean val loss).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.merge.lora_io import LoraAdapter, stable_seed  # noqa: E402
from src.merge.merge_experts import METHODS, combine, sparsify  # noqa: E402
from src.train import data as D  # noqa: E402
from src.train.config import load_config  # noqa: E402


def cand_name(c) -> str:
    if isinstance(c, str):
        return c
    if c.get("name"):
        return c["name"]
    m = c.get("method", "linear")
    sp = METHODS[m][0]
    n = f"{m}_l{float(c.get('global_scale', 1.0)):g}"
    if sp == "magnitude":
        n += f"_d{float(c.get('density', 0.2)):g}"
    if sp == "dare":
        n += f"_p{float(c.get('drop_p', 0.9)):g}"
    if c.get("weights"):
        n += "_w" + "-".join(f"{float(v):g}" for v in c["weights"].values())
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true", help="CPU, no model/data: build every candidate's ΔW for 3 "
                    "modules and print norms (config + merge-math check on a login node)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    tcfg = load_config(cfg["train_config"])
    tcfg["model"]["gradient_checkpointing"] = False
    max_len = int(cfg.get("val_max_seq_len", 65536))
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    done = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            done[r["name"]] = r

    names = list(cfg["experts"].keys())
    adapters = {n: LoraAdapter(p) for n, p in cfg["experts"].items()}
    extra = {}
    for n, p in (cfg.get("extra_adapters") or {}).items():
        if (Path(p) / "adapter_model.safetensors").exists():
            extra[n] = LoraAdapter(p)
        else:
            print(f"[select] extra adapter {n} missing ({p}); its candidate is skipped")
    cands = [c for c in cfg["candidates"]
             if not (isinstance(c, str) and c.startswith("adapter:") and c.split(":", 1)[1] not in extra)]
    todo = [c for c in cands if cand_name(c) not in done]
    print(f"[select] {len(cands)} candidates, {len(done)} already scored, {len(todo)} to run; val max_len={max_len}")
    if args.dry_run:
        return dry_run(cfg, cands, names, adapters, extra)

    # ---------------- val data (same caches/packing as training, fixed max_len for every candidate)
    dcfg = tcfg["data"]
    common = dict(model_path=tcfg["model"]["path"], cache_dir=dcfg["cache_dir"],
                  preserve_thinking=bool(dcfg.get("preserve_thinking", True)),
                  num_proc=int(dcfg.get("num_proc", 16)), build=True, log=print)
    val_sets, val_packs = {}, {}
    for vname, path in dcfg["val"].items():
        val_sets[vname] = D.build_or_load_cache(vname, path, **common)
        vp, _ = D.make_pieces([val_sets[vname]], max_len, dcfg.get("overlong", "truncate"),
                              max_tokens=int(dcfg.get("val_max_tokens_per_set", 524288)))
        val_packs[vname] = D.pack_pieces(vp, max_len, None, int(dcfg.get("pack_window", 512)))
    domain_of = {n: n for n in names}   # val set name == expert name (e_swe, e_tb, e_asi, e_hle)
    assert set(domain_of) <= set(val_sets), f"val sets {list(val_sets)} must cover experts {names}"

    if todo:
        from src.train import modeling as Mdl

        device = torch.device("cuda", 0)
        torch.backends.cuda.matmul.allow_tf32 = True
        Mdl.setup_attention(tcfg["model"].get("attn_backend", "auto"), device)
        Mdl.patch_gated_deltanet(tcfg["model"].get("gdn_backend", "fla"))
        model = Mdl.load_model(tcfg, device)
        model.eval()
        text_model, lm_head = model.model.language_model, model.lm_head
        params = dict(model.named_parameters())
        keys = sorted(set().union(*[set(a.keys) for a in list(adapters.values()) + list(extra.values())]))
        missing = [k for k in keys if k not in params]
        if missing:
            raise KeyError(f"{len(missing)} adapter keys not found in model parameters, e.g. {missing[:3]}")
        orig = {k: params[k].detach().clone() for k in keys}
        print(f"[select] {len(keys)} adapted weights backed up "
              f"({sum(v.numel() for v in orig.values()) * 2 / 2**30:.2f} GiB)")
        ce_chunk = int(tcfg["train"].get("ce_chunk_tokens", 4096))

        def to_dev(pack, sets):
            ids, lab, pos, cu = D.build_pack_arrays(pack, sets)
            cu_cpu = torch.from_numpy(cu.astype(np.int64))
            return (torch.from_numpy(ids).to(device), torch.from_numpy(lab).to(device),
                    torch.from_numpy(pos).to(device), torch.from_numpy(cu).to(device), cu_cpu)

        def evaluate() -> dict:
            res = {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for vname, packs in val_packs.items():
                    s, n = 0.0, 0
                    for pack in packs:
                        ids, lab, pos, cu, cu_cpu = to_dev(pack, [val_sets[vname]])
                        ls, k = Mdl.forward_pack(text_model, lm_head, ids, lab, pos, cu, cu_cpu, ce_chunk)
                        s += float(ls)
                        n += int(k)
                    res[vname] = {"loss": s / max(1, n), "targets": n}
            return res

        delta_fn = make_delta_fn(cfg, names, adapters, extra, device, lambda k: orig[k].shape)

        for c in todo:
            name = cand_name(c)
            t0 = time.time()
            f = delta_fn(c)
            dnorm, wnorm = 0.0, 0.0
            with torch.no_grad():
                for k in keys:
                    d = f(k)
                    if d is None:
                        params[k].data.copy_(orig[k])
                        continue
                    params[k].data.copy_((orig[k].float() + d).to(orig[k].dtype))
                    dnorm += float(d.square().sum())
                    wnorm += float(orig[k].float().square().sum())
            t1 = time.time()
            res = evaluate()
            row = {"name": name, "spec": c, "val": res, "rel_delta_norm": (dnorm / wnorm) ** 0.5 if wnorm else 0.0,
                   "merge_s": round(t1 - t0, 1), "eval_s": round(time.time() - t1, 1), "max_len": max_len,
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with open(rows_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            done[name] = row
            print(f"[select] {name}: " + " ".join(f"{v}={r['loss']:.4f}" for v, r in res.items()) +
                  f"  (merge {row['merge_s']}s, eval {row['eval_s']}s)", flush=True)
        with torch.no_grad():
            for k in keys:
                params[k].data.copy_(orig[k])

    write_tables(out_dir, [done[cand_name(c)] for c in cands if cand_name(c) in done], domain_of)


def make_delta_fn(cfg, names, adapters, extra, device, shape_of):
    """candidate -> (key -> fp32 ΔW on device, or None = untouched); same math/seeds as merge_experts.py."""
    def delta_fn(c):
        """key -> fp32 ΔW on device (None = untouched) for candidate c."""
        if c == "base":
            return lambda key: None
        if isinstance(c, str):
            kind, n = c.split(":", 1)
            a = adapters[n] if kind == "expert" else extra[n]
            return lambda key: a.delta(key, device) if key in a.A else None
        method = c.get("method", "linear")
        sp_mode, comb_mode = METHODS[method]
        w = [float((c.get("weights") or {}).get(n, cfg.get("weights", {}).get(n, 1.0))) for n in names]
        lam = float(c.get("global_scale", 1.0))
        dens = float(c.get("density", cfg.get("density", 0.2)))
        drop_p = float(c.get("drop_p", cfg.get("drop_p", 0.9)))
        seed = int(c.get("seed", cfg.get("seed", 0)))

        def f(key):
            shape = shape_of(key)
            sp = []
            for n in names:
                a = adapters[n]
                t = a.delta(key, device) if key in a.A else torch.zeros(shape, device=device)
                gen = None
                if sp_mode == "dare":
                    gen = torch.Generator(device=device)
                    gen.manual_seed(stable_seed(seed, key, n))
                sp.append(sparsify(t, sp_mode, dens, drop_p, gen))
            return combine(sp, w, comb_mode, lam)
        return f

    return delta_fn


def dry_run(cfg, cands, names, adapters, extra):
    keys = sorted(set().union(*[set(a.keys) for a in adapters.values()]))
    probe = [keys[0], keys[len(keys) // 2], keys[-1]]
    shapes = {k: next(a for a in adapters.values() if k in a.A).delta(k, "cpu").shape for k in probe}
    delta_fn = make_delta_fn(cfg, names, adapters, extra, torch.device("cpu"), lambda k: shapes[k])
    for c in cands:
        f = delta_fn(c)
        norms = []
        for k in probe:
            d = f(k)
            norms.append(0.0 if d is None else float(d.norm()))
        print(f"[dry] {cand_name(c):32s} |ΔW| on {len(probe)} probe modules: " + " ".join(f"{x:.4f}" for x in norms))


def write_tables(out_dir: Path, rows: list[dict], domain_of: dict):
    by = {r["name"]: r for r in rows}
    if "base" not in by:
        print("[select] no base row: retention not computed")
        return
    doms = list(domain_of)
    L = lambda r, d: r["val"][d]["loss"]  # noqa: E731
    out = []
    for r in rows:
        rec = {"name": r["name"], "rel_delta_norm": round(r["rel_delta_norm"], 5)}
        rets = []
        for d in doms:
            rec[f"L_{d}"] = round(L(r, d), 4)
            ex = by.get(f"expert:{domain_of[d]}")
            if ex is not None and L(by["base"], d) - L(ex, d) > 1e-9:
                ret = (L(by["base"], d) - L(r, d)) / (L(by["base"], d) - L(ex, d))
                rec[f"R_{d}"] = round(ret, 3)
                rets.append(ret)
        rec["L_mean"] = round(sum(L(r, d) for d in doms) / len(doms), 4)
        rec["R_mean"] = round(sum(rets) / len(rets), 3) if len(rets) == len(doms) else None
        rec["R_min"] = round(min(rets), 3) if len(rets) == len(doms) else None
        out.append(rec)
    out.sort(key=lambda x: (-(x["R_mean"] if x["R_mean"] is not None else -1e9), x["L_mean"]))
    cols = list(out[0].keys())
    with open(out_dir / "table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join("" if x.get(c) is None else str(x.get(c)) for c in cols) + " |" for x in out]
    (out_dir / "table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
