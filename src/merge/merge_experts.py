"""Merge K expert LoRAs in dense ΔW space and fold the result into the base model; plus conflict analysis.

  python -m src.merge.merge_experts --config configs/merge/ties.yaml [--set global_scale=0.75 ...]
  python -m src.merge.merge_experts --config configs/merge/ties.yaml --analyze-only

Per adapted module (key = base weight name), with expert task vectors τ_t = (α_t/r_t) B_t A_t  (dense, fp32):
  1. sparsify   none | magnitude (keep top-`density` |τ_t| per module = TIES trim) | dare (drop w.p. `drop_p`,
                rescale survivors by 1/(1-p))
  2. combine    sum:   τ = λ Σ_t w_t τ̂_t                                   (task arithmetic / linear)
                ties:  γ = sign(Σ_t w_t τ̂_t);  τ = λ Σ_t w_t τ̂_t [sign τ̂_t = γ] / Σ_t w_t [sign τ̂_t = γ]
                       (sign election + disjoint weighted mean)
  methods: linear=(none,sum)  ties=(magnitude,ties)  dare_linear=(dare,sum)  dare_ties=(dare,ties)
  3. optional   SVD truncation of τ back to rank `svd_rank` (Σ_t r_t = 128 for linear; TIES/DARE make τ
                full-rank -> the merge is only exact in dense space) and optional export as a PEFT adapter.
  4. fold       W' = W + τ  (fp32 add -> bf16), shard by shard (lora_io.write_folded_model).
Conflict analysis (per module and expert pair): cosine(τ_i, τ_j), sign-conflict fraction on the union of
top-`analysis_density` entries, Frobenius norms; TIES-style multi-expert disagreement rate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.merge.lora_io import (LoraAdapter, pick_device, save_lowrank_adapter, stable_seed,  # noqa: E402
                               write_folded_model)
from src.train.config import load_config  # noqa: E402

METHODS = {"linear": ("none", "sum"), "task_arithmetic": ("none", "sum"), "ties": ("magnitude", "ties"),
           "dare_linear": ("dare", "sum"), "dare_ties": ("dare", "ties")}


# ----------------------------------------------------------------------------- merge primitives
def topk_mask(x: torch.Tensor, density: float) -> torch.Tensor:
    if density >= 1.0:
        return torch.ones_like(x, dtype=torch.bool)
    n = x.numel()
    k = max(1, int(round(density * n)))
    a = x.abs().flatten()
    thr = torch.kthvalue(a.float().cpu() if a.device.type == "mps" else a, n - k + 1).values
    return x.abs() >= thr


def sparsify(tau: torch.Tensor, mode: str, density: float, drop_p: float, gen: torch.Generator | None):
    if mode == "none":
        return tau
    if mode == "magnitude":
        return tau * topk_mask(tau, density)
    if mode == "dare":
        if drop_p <= 0:
            return tau
        keep = torch.rand(tau.shape, generator=gen, device=tau.device) >= drop_p
        return tau * keep / (1.0 - drop_p)
    raise ValueError(mode)


def combine(taus: list[torch.Tensor], weights: list[float], mode: str, lam: float) -> torch.Tensor:
    if mode == "sum":
        out = torch.zeros_like(taus[0])
        for w, t in zip(weights, taus):
            out.add_(t, alpha=w)
        return out * lam
    if mode == "ties":
        wsum = torch.zeros_like(taus[0])
        for w, t in zip(weights, taus):
            wsum.add_(t, alpha=w)
        gamma = torch.sign(wsum)
        num = torch.zeros_like(taus[0])
        den = torch.zeros_like(taus[0])
        for w, t in zip(weights, taus):
            agree = (torch.sign(t) == gamma) & (t != 0)
            num.add_(torch.where(agree, t * w, 0.0))
            den.add_(agree.to(t.dtype) * abs(w))
        return lam * torch.where(den > 0, num / den.clamp(min=1e-12), 0.0)
    raise ValueError(mode)


def svd_truncate(delta: torch.Tensor, k: int):
    U, S, Vh = torch.linalg.svd(delta.float(), full_matrices=False)
    energy = (S[:k].square().sum() / S.square().sum().clamp(min=1e-30)).item()
    sq = S[:k].sqrt()
    B = U[:, :k] * sq
    A = sq[:, None] * Vh[:k]
    return B, A, energy


# ----------------------------------------------------------------------------- analysis
def module_type(key: str) -> tuple[str, int]:
    m = re.search(r"layers\.(\d+)\.(.+)\.weight$", key)
    layer = int(m.group(1)) if m else -1
    name = m.group(2) if m else key
    return name, layer


def analyze_module(key: str, names: list[str], taus: list[torch.Tensor], density: float) -> tuple[dict, list[dict]]:
    mtype, layer = module_type(key)
    norms = [t.norm().item() for t in taus]
    masks = [topk_mask(t, density) for t in taus]
    signs = [torch.sign(t) for t in taus]
    pair_rows = []
    for i, j in combinations(range(len(taus)), 2):
        ti, tj = taus[i], taus[j]
        cos = (ti.flatten() @ tj.flatten()).item() / max(1e-30, norms[i] * norms[j])
        both = masks[i] & masks[j]
        nb = int(both.sum().item())
        conflict = ((signs[i] != signs[j]) & both).sum().item() / max(1, nb)
        allc = (signs[i] * signs[j] < 0).float().mean().item()
        pair_rows.append({"module": key, "type": mtype, "layer": layer, "a": names[i], "b": names[j],
                          "cosine": cos, "sign_conflict_topk_overlap": conflict, "topk_overlap_frac": nb / ti.numel(),
                          "sign_conflict_all": allc})
    # multi-expert disagreement after TIES trimming: positions kept by >=2 experts with mixed signs
    trimmed = torch.stack([s * m for s, m in zip(signs, masks)])
    n_kept = (trimmed != 0).sum(0)
    multi = n_kept >= 2
    mixed = multi & (trimmed.max(0).values > 0) & (trimmed.min(0).values < 0)
    row = {"module": key, "type": mtype, "layer": layer, "numel": taus[0].numel(),
           **{f"norm_{n}": v for n, v in zip(names, norms)},
           "ties_multi_kept_frac": multi.float().mean().item(),
           "ties_mixed_sign_frac_of_multi": mixed.sum().item() / max(1, multi.sum().item())}
    return row, pair_rows


def write_analysis(out_dir: Path, names, mod_rows, pair_rows, log=print):
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, rows in (("per_module.csv", mod_rows), ("pairwise.csv", pair_rows)):
        if rows:
            with open(out_dir / fname, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    # aggregates
    def agg(rows, keyf, fields):
        groups: dict = {}
        for r in rows:
            groups.setdefault(keyf(r), []).append(r)
        return {str(k): {f: sum(r[f] for r in v) / len(v) for f in fields} | {"n": len(v)}
                for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}
    pf = ["cosine", "sign_conflict_topk_overlap", "sign_conflict_all"]
    summary = {
        "experts": names,
        "pair_mean": agg(pair_rows, lambda r: f"{r['a']}~{r['b']}", pf),
        "by_module_type": agg(pair_rows, lambda r: r["type"], pf),
        "by_layer": agg(pair_rows, lambda r: r["layer"], pf),
        "ties_by_module_type": agg(mod_rows, lambda r: r["type"],
                                   ["ties_multi_kept_frac", "ties_mixed_sign_frac_of_multi"]),
        "norms_by_module_type": agg(mod_rows, lambda r: r["type"], [f"norm_{n}" for n in names]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        k = len(names)
        for metric, title in (("cosine", "mean cosine(ΔW_i, ΔW_j) over modules"),
                              ("sign_conflict_topk_overlap", "sign-conflict fraction (top-k overlap)")):
            M = np.full((k, k), np.nan)  # diagonal left blank so off-diagonal contrast is visible
            for i, j in combinations(range(k), 2):
                v = summary["pair_mean"][f"{names[i]}~{names[j]}"][metric]
                M[i, j] = M[j, i] = v
            fig, ax = plt.subplots(figsize=(4.8, 4.2))
            if metric == "cosine":  # diverging around 0
                vmax = max(1e-6, float(np.nanmax(np.abs(M))))
                im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            else:  # sequential, 0.5 = chance level for random signs
                im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(0.5, float(np.nanmax(M))))
            ax.set_xticks(range(k), names, rotation=30)
            ax.set_yticks(range(k), names)
            for i in range(k):
                for j in range(k):
                    if i != j:
                        ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color="#1f1f1e", fontsize=8)
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(out_dir / f"heatmap_{metric}.png", dpi=150)
            plt.close(fig)
        # per-layer cosine by module type
        fig, ax = plt.subplots(figsize=(8, 4))
        types = sorted({r["type"] for r in pair_rows})
        for t in types:
            layers = sorted({r["layer"] for r in pair_rows if r["type"] == t})
            ys = [np.mean([r["cosine"] for r in pair_rows if r["type"] == t and r["layer"] == L]) for L in layers]
            ax.plot(layers, ys, marker="o", ms=3, label=t)
        ax.set_xlabel("layer")
        ax.set_ylabel("mean pairwise cosine(ΔW_i, ΔW_j)")
        ax.set_title("Expert task-vector similarity by layer and module type")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "cosine_by_layer.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # plotting is best effort
        log(f"[analysis] plotting skipped: {e}")
    log(f"[analysis] wrote {out_dir}")
    return summary


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    dev = pick_device(cfg.get("device", "auto"))
    names = list(cfg["experts"].keys())
    adapters = [LoraAdapter(cfg["experts"][n]) for n in names]
    weights = [float(cfg.get("weights", {}).get(n, 1.0)) for n in names]
    method = cfg.get("method", "linear")
    sp_mode, comb_mode = METHODS[method]
    lam = float(cfg.get("global_scale", 1.0))
    density = float(cfg.get("density", 0.2))
    drop_p = float(cfg.get("drop_p", 0.9))
    seed = int(cfg.get("seed", 0))
    svd_rank = cfg.get("svd_rank")
    fold_svd = bool(cfg.get("fold_svd", False))
    do_analysis = bool(cfg.get("analysis", True)) or args.analyze_only
    an_density = float(cfg.get("analysis_density", 0.2))
    name = cfg.get("name") or f"{method}_l{lam:g}" + (f"_d{density:g}" if sp_mode == "magnitude" else "") + \
        (f"_p{drop_p:g}" if sp_mode == "dare" else "")
    keysets = [set(a.keys) for a in adapters]
    keys = sorted(set.union(*keysets))
    if any(ks != keysets[0] for ks in keysets):
        print("[merge] WARNING: experts adapt different module sets; missing modules count as ΔW=0")
    print(f"[merge] {name}: method={method} ({sp_mode},{comb_mode}) λ={lam} weights={dict(zip(names, weights))} "
          f"density={density} drop_p={drop_p} modules={len(keys)} device={dev}")

    mod_rows, pair_rows, svd_energy, lowrank = [], [], {}, {}

    def taus_for(key, shape):
        out = []
        for a in adapters:
            out.append(a.delta(key, dev) if key in a.A else torch.zeros(shape, device=dev))
        return out

    def merged_delta(key: str, w: torch.Tensor) -> torch.Tensor:
        taus = taus_for(key, w.shape)
        if do_analysis:
            r, pr = analyze_module(key, names, taus, an_density)
            mod_rows.append(r)
            pair_rows.extend(pr)
        sp = []
        for t_i, (n, t) in enumerate(zip(names, taus)):
            gen = None
            if sp_mode == "dare":
                gen = torch.Generator(device=dev)
                gen.manual_seed(stable_seed(seed, key, n))
            sp.append(sparsify(t, sp_mode, density, drop_p, gen))
        d = combine(sp, weights, comb_mode, lam)
        if svd_rank:
            B, A, e = svd_truncate(d, int(svd_rank))
            svd_energy[key] = e
            if cfg.get("export_adapter"):
                lowrank[key] = (B.cpu(), A.cpu())
            if fold_svd:
                d = B @ A
        return d

    t0 = time.time()
    analysis_dir = Path(cfg.get("analysis_dir", "/scratch/eh3323/apex/analysis")) / \
        ("conflicts_" + "_".join(names) if args.analyze_only else name)
    if args.analyze_only:
        from safetensors import safe_open
        from src.merge.lora_io import base_weight_map

        wmap = base_weight_map(cfg["base"])
        for i, key in enumerate(keys):
            with safe_open(str(Path(cfg["base"]) / wmap[key]), "pt") as f:
                shape = f.get_slice(key).get_shape()
            r, pr = analyze_module(key, names, taus_for(key, tuple(shape)), an_density)
            mod_rows.append(r)
            pair_rows.extend(pr)
            if i % 25 == 0:
                print(f"[analysis] {i}/{len(keys)} {time.time() - t0:.0f}s")
        write_analysis(analysis_dir, names, mod_rows, pair_rows)
        return

    out_dir = Path(cfg.get("out_root", "/scratch/eh3323/apex/models/merged")) / name
    info = {"type": "expert_merge", "name": name, "method": method, "sparsify": sp_mode, "combine": comb_mode,
            "global_scale": lam, "weights": dict(zip(names, weights)), "density": density, "drop_p": drop_p,
            "seed": seed, "svd_rank": svd_rank, "fold_svd": fold_svd,
            "experts": {n: {"path": str(a.path), "r": a.r, "scale": a.scale} for n, a in zip(names, adapters)},
            "config": cfg}
    if cfg.get("fold", True):
        write_folded_model(cfg["base"], out_dir, keys, merged_delta, info,
                           link_unchanged=bool(cfg.get("link_unchanged", True)))
    else:
        from safetensors import safe_open
        from src.merge.lora_io import base_weight_map

        wmap = base_weight_map(cfg["base"])
        for key in keys:
            with safe_open(str(Path(cfg["base"]) / wmap[key]), "pt") as f:
                merged_delta(key, f.get_tensor(key))
    if svd_energy:
        vals = list(svd_energy.values())
        print(f"[merge] SVD rank {svd_rank}: retained energy mean={sum(vals)/len(vals):.4f} min={min(vals):.4f}")
        (analysis_dir).mkdir(parents=True, exist_ok=True)
        (analysis_dir / "svd_energy.json").write_text(json.dumps(svd_energy, indent=1))
    if lowrank:
        save_lowrank_adapter(out_dir.with_name(out_dir.name + "_svd_adapter"), adapters[0].cfg, lowrank, int(svd_rank))
    if do_analysis and mod_rows:
        write_analysis(analysis_dir, names, mod_rows, pair_rows)
    print(f"[merge] done in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
