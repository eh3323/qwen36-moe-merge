#!/usr/bin/env python
"""Train/val loss curves from logs/<run_id>/metrics.jsonl -> curves/<run_id>.png + curves/source/<run_id>_{train,val}.csv
plus an overlay of all runs (curves/all_runs_val.png, curves/all_runs_train.png).

  python scripts/plot_curves.py                    # all runs under logs/
  python scripts/plot_curves.py --runs e_tb_r32 mixed_r32 --logs logs --out curves

Resumed runs append to metrics.jsonl; duplicate steps are de-duplicated keeping the LAST record.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# fixed categorical order (never cycled): blue, orange, aqua, yellow, magenta, green, violet, red
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def load(path: Path):
    train, val, ckpt = {}, {}, {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t = r.get("type")
        if t == "train":
            train[r["step"]] = r
        elif t == "val":
            val[(r["set"], r["step"])] = r
        elif t == "ckpt":
            ckpt[r["step"]] = r
    return ([train[k] for k in sorted(train)], [val[k] for k in sorted(val, key=lambda x: (x[0], x[1]))],
            [ckpt[k] for k in sorted(ckpt)])


def ema(xs, a=0.9):
    out, m = [], None
    for x in xs:
        m = x if m is None else a * m + (1 - a) * x
        out.append(m)
    return out


def style(ax, title, xlabel="optimizer step", ylabel="loss (CE / assistant token)"):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="curves")
    ap.add_argument("--runs", nargs="*")
    args = ap.parse_args()
    logs, out = Path(args.logs), Path(args.out)
    (out / "source").mkdir(parents=True, exist_ok=True)
    runs = args.runs or sorted(p.parent.name for p in logs.glob("*/metrics.jsonl"))
    data = {}
    for rid in runs:
        p = logs / rid / "metrics.jsonl"
        if not p.exists():
            print(f"skip {rid}: no {p}")
            continue
        train, val, ckpt = load(p)
        data[rid] = (train, val, ckpt)
        write_csv(out / "source" / f"{rid}_train.csv", train,
                  ["step", "epoch", "loss", "lr", "grad_norm", "tokens", "label_tokens", "cum_tokens",
                   "tokens_per_s", "tokens_per_s_per_gpu", "step_time", "peak_mem_gb"])
        write_csv(out / "source" / f"{rid}_val.csv", val, ["step", "set", "loss", "tokens"])

        fig, ax = plt.subplots(figsize=(8, 4.5))
        if train:
            xs = [r["step"] for r in train]
            ys = [r["loss"] for r in train]
            ax.plot(xs, ys, color=PALETTE[0], alpha=0.3, linewidth=1, label="train (per step)")
            ax.plot(xs, ema(ys), color=PALETTE[0], linewidth=2, label="train (EMA 0.9)")
        for i, vs in enumerate(sorted({r["set"] for r in val})):
            pts = [r for r in val if r["set"] == vs]
            ax.plot([r["step"] for r in pts], [r["loss"] for r in pts], color=PALETTE[1 + i % 7], linewidth=2,
                    marker="o", markersize=5, label=f"val: {vs}")
        for j, c in enumerate(ckpt):
            ax.axvline(c["step"], color=MUTED, linestyle=":", linewidth=0.8,
                       label="checkpoint" if j == 0 else None)
        style(ax, f"{rid}: train / val loss")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"{rid}.png", dpi=150)
        plt.close(fig)
        print(f"wrote {out / (rid + '.png')}")

    if len(data) > 1:
        for kind in ("train", "val"):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for i, (rid, (train, val, _)) in enumerate(data.items()):
                c = PALETTE[i % len(PALETTE)]
                if kind == "train" and train:
                    ax.plot([r["step"] for r in train], ema([r["loss"] for r in train]), color=c, linewidth=2,
                            label=rid)
                elif kind == "val" and val:
                    # in-domain val set = the one matching the run's prefix; fall back to the mean over sets
                    steps = sorted({r["step"] for r in val})
                    own = [s for s in {r["set"] for r in val} if rid.startswith(s)]
                    ys = [sum(r["loss"] for r in val if r["step"] == s and (not own or r["set"] == own[0])) /
                          max(1, len([r for r in val if r["step"] == s and (not own or r["set"] == own[0])]))
                          for s in steps]
                    ax.plot(steps, ys, color=c, linewidth=2, marker="o", markersize=5,
                            label=f"{rid} ({own[0] if own else 'mean of val sets'})")
            style(ax, f"All runs: {kind} loss" + (" (EMA 0.9)" if kind == "train" else " (in-domain val set)"))
            ax.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            fig.savefig(out / f"all_runs_{kind}.png", dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    main()
