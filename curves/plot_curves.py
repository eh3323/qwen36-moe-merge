#!/usr/bin/env python3
"""Plot loss curves and the merge-method ablation from the committed source data (stdlib + matplotlib only).

Inputs (all in curves/source/, produced from the cluster run logs; see logs/train/<run_id>/metrics.jsonl):
  runs.csv            one row per training run (experiment name, in-domain val set, GPU, seed, ...)
  <run_id>.csv        one row per optimizer step: train_loss, lr, tokens, ..., val_<set> at eval steps
                      (step 0 = evaluation before the first update), checkpoint name where one was saved
  merge_select.csv    18 merge candidates scored on the 4 expert val sets (src/merge/merge_select.py)
Outputs:
  curves/<run_id>.png       train loss / val losses / learning rate vs optimizer step, checkpoints marked
  curves/merge_ablation.png mean val loss and mean retention per merge candidate, final choice highlighted

  python3 curves/plot_curves.py                  # all runs listed in curves/source/runs.csv
  python3 curves/plot_curves.py --runs e_tb_v2_r32
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent

# light chart surface + text/axis tokens
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3de"
TRAIN_RAW, TRAIN_EMA = "#c3c2b7", "#3d3c38"
# categorical slots in fixed order (validated: adjacent CVD dE >= 9.1, normal-vision dE >= 19.6 on the light surface).
# Colour follows the val set (entity), the same in every chart.
VAL_COLOR = {"e_swe": "#2a78d6", "e_tb": "#eb6834", "e_asi": "#1baf7a", "e_hle": "#eda100",
             "e_swe_v2": "#e87ba4", "e_tb_v2_turns": "#e87ba4"}
VAL_ORDER = ["e_swe", "e_swe_v2", "e_tb", "e_tb_v2_turns", "e_asi", "e_hle"]
VAL_NAME = {"e_swe": "SWE val (v1 data)", "e_swe_v2": "SWE-v2 val (Open-SWE-Traces)", "e_tb": "TB val (v1 data)",
            "e_tb_v2_turns": "TB-v2 per-turn val", "e_asi": "ASI val", "e_hle": "HLE val"}
# merge families: 3 slots (all-pairs safe for a dot plot) + neutral for reference points; marker shape doubles the cue
FAMILY = {"Linear": ("#2a78d6", "o"), "TIES": ("#eb6834", "s"), "DARE": ("#1baf7a", "^"),
          "Base": (MUTED, "D"), "Single expert": (MUTED, "D"), "Mixed control": (MUTED, "D")}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "text.color": INK, "legend.frameon": False,
})


def fnum(s):
    return float(s) if s not in ("", None) else None


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def style(ax, ylabel, xlabel=None):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.tick_params(labelsize=8, length=3)


def ema(xs, a=0.9):
    out, m = [], None
    for x in xs:
        m = x if m is None else a * m + (1 - a) * x
        out.append(m)
    return out


def spread(ys, gap):
    """Nudge label y positions apart (keep order) so end labels do not overlap."""
    idx = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [ys[i] for i in idx]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + gap)
    shift = (sum(pos) - sum(ys[i] for i in idx)) / len(pos)  # re-centre the block
    out = [0.0] * len(ys)
    for k, i in enumerate(idx):
        out[i] = pos[k] - shift
    return out


# ----------------------------------------------------------------------------- per-run figure
def plot_run(meta: dict, rows: list[dict], out: Path):
    rid = meta["run_id"]
    train = [r for r in rows if r["train_loss"]]
    steps = [int(r["step"]) for r in train]
    loss = [float(r["train_loss"]) for r in train]
    lr = [float(r["lr"]) for r in train]
    sets = sorted([c[4:] for c in rows[0] if c.startswith("val_")],
                  key=lambda s: VAL_ORDER.index(s) if s in VAL_ORDER else 99)
    own = [s for s in meta["in_domain_val"].split("|") if s]
    ck = [(int(r["step"]), r["checkpoint"], r["checkpoint_on_disk"] == "1") for r in rows if r["checkpoint"]]
    final = [c for c in ck if c[1] == "adapter_final"][0]
    periodic = [c for c in ck if c[1] != "adapter_final"]
    ep2 = next((int(r["step"]) for r in train if r["epoch"] == "1"), None)
    last = steps[-1]

    fig = plt.figure(figsize=(11.5, 9.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 3.0, 1.2], hspace=0.26, left=0.075, right=0.785, top=0.855,
                          bottom=0.065)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # (1) training loss
    ax1.plot(steps, loss, color=TRAIN_RAW, linewidth=1.0, label="train loss, per step")
    ax1.plot(steps, ema(loss), color=TRAIN_EMA, linewidth=2.0, label="train loss, EMA (α = 0.9)")
    style(ax1, "train loss\n(CE per assistant token)")
    span = max(loss) - min(loss)
    ax1.set_ylim(min(loss) - 0.14 * span, max(loss) + 0.06 * span)  # room for the checkpoint rug at the bottom
    ax1.set_autoscaley_on(False)
    tr = ax1.get_xaxis_transform()
    ax1.vlines([c[0] for c in periodic if not c[2]], 0, 0.05, transform=tr, color=MUTED, linewidth=1.2)
    ax1.vlines([c[0] for c in periodic if c[2]], 0, 0.09, transform=tr, color=INK2, linewidth=2.0)
    handles, labels = ax1.get_legend_handles_labels()
    handles += [Line2D([], [], color=MUTED, linewidth=1.2), Line2D([], [], color=INK2, linewidth=2.0),
                Line2D([], [], color=INK, linewidth=1.0)]
    labels += [f"checkpoint saved (every {periodic[1][0] - periodic[0][0] if len(periodic) > 1 else '?'} steps)",
               "checkpoint kept on disk (last 2)", f"adapter_final (step {final[0]}, final adapter)"]
    ax1.legend(handles, labels, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1)

    # (2) validation losses
    ends = []
    for s in sets:
        pts = [(int(r["step"]), float(r[f"val_{s}"])) for r in rows if r[f"val_{s}"]]
        mine = s in own
        ax2.plot([p[0] for p in pts], [p[1] for p in pts], color=VAL_COLOR.get(s, INK2),
                 linewidth=2.6 if mine else 1.6, marker="o", markersize=6 if mine else 4.5,
                 markeredgecolor=SURFACE, markeredgewidth=1.0,
                 label=f"{VAL_NAME.get(s, s)}" + ("  (in-domain)" if mine else ""), zorder=3 if mine else 2)
        ends.append((s, pts[0][1], pts[-1][1], pts[-1][0]))
    style(ax2, "val loss\n(CE per assistant token)")
    ax2.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.2)
    lo, hi = ax2.get_ylim()
    ys = spread([e[2] for e in ends], (hi - lo) * 0.055)
    for (s, y0, y1, x1), yl in zip(ends, ys):
        ax2.annotate(f"{s}  {y0:.3f} → {y1:.3f}", xy=(x1, y1), xytext=(last * 1.035 + 1, yl),
                     textcoords="data", fontsize=7.5, color=INK2, va="center", annotation_clip=False,
                     arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8, shrinkA=0, shrinkB=3))
    ax2.text(1.01, 1.0, "val loss at step 0 → adapter_final", transform=ax2.transAxes, fontsize=7.5,
             color=MUTED, va="bottom")
    if "e_hle" not in sets:
        ax2.text(1.01, 1.07, "no HLE val curve: the HLE val split\nwas built after this run started",
                 transform=ax2.transAxes, fontsize=7.5, color=MUTED, va="bottom")

    # (3) learning rate
    ax3.plot(steps, lr, color=TRAIN_EMA, linewidth=1.6)
    style(ax3, "learning rate", f"optimizer step  (≈262k tokens per step; final checkpoint adapter_final at step {last})")
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax3.yaxis.get_offset_text().set_fontsize(7.5)

    for ax in (ax1, ax2, ax3):
        ax.axvline(final[0], color=INK, linewidth=1.0, zorder=1)
        if ep2:
            ax.axvline(ep2 - 0.5, color=GRID, linewidth=1.0, zorder=0)
        ax.set_xlim(-last * 0.015, last * 1.015)
    if ep2:
        ax1.text(ep2, 0.97, " epoch 2 →", transform=ax1.get_xaxis_transform(), fontsize=7.5, color=MUTED,
                 va="top")
    ax1.annotate(f"adapter_final\nstep {final[0]}", xy=(final[0], 1.0), xycoords=("data", "axes fraction"),
                 xytext=(4, 2), textcoords="offset points", fontsize=7.5, color=INK, va="bottom", ha="left",
                 annotation_clip=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    fig.text(0.075, 0.962, f"{meta['experiment']}  —  train / val loss", fontsize=12.5, fontweight="bold",
             color=INK)
    fig.text(0.075, 0.937,
             f"run {rid}  ·  LoRA r{meta['lora_r']} on Qwen3.6-35B-A3B  ·  {meta['gpu_count']}× {meta['gpu']}  ·  "
             f"seed {meta['seed']}  ·  Slurm job {meta['slurm_job']}", fontsize=8.5, color=INK2)
    fig.text(0.075, 0.918,
             f"{meta['steps']} optimizer steps / 2 epochs  ·  {int(meta['tokens_trained']) / 1e6:.1f}M tokens  ·  "
             f"peak lr {float(meta['peak_lr']):g}, warmup 3%, cosine to 10%  ·  max_seq_len "
             f"{int(meta['max_seq_len']) // 1024}K  ·  step 0 = eval before the first update", fontsize=8.5, color=INK2)
    fig.text(0.075, 0.899, f"source: curves/source/{rid}.csv (from logs/train/{rid}/metrics.jsonl)",
             fontsize=7.5, color=MUTED)
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- merge ablation figure
def label_of(r: dict) -> str:
    n, m = r["name"], r["method"]
    if r["kind"] == "reference":
        return {"base": "Base (no adapter)", "adapter:mixed": "Mixed control (single LoRA, union data)"}.get(
            n, {"expert:e_swe": "Expert E-SWE v1 alone", "expert:e_tb": "Expert E-TB v1 alone",
                "expert:e_asi": "Expert E-ASI alone", "expert:e_hle": "Expert E-HLE alone"}.get(n, n))
    lam = f"λ={float(r['lambda']):g}"
    if m == "linear":
        return f"Linear  {lam}" + ("  (= plain average)" if float(r["lambda"]) == 0.25 else "")
    if m == "ties":
        return f"TIES  {lam}  density={float(r['density']):g}"
    return f"{'DARE-TIES' if m == 'dare_ties' else 'DARE-Linear'}  {lam}  drop p={float(r['drop_p']):g}"


def plot_merge(rows: list[dict], out: Path):
    rows = sorted(rows, key=lambda r: int(r["rank"]))
    n = len(rows)
    ys = list(range(n))[::-1]
    sel = [r for r in rows if r["selected"] == "1"][0]
    base = [r for r in rows if r["name"] == "base"][0]

    fig, (axl, axr) = plt.subplots(1, 2, sharey=True, figsize=(12.5, 8.4),
                                   gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.07, left=0.23, right=0.975,
                                                    top=0.79, bottom=0.17))
    for ax, key in ((axl, "val_mean"), (axr, "R_mean")):
        for y, r in zip(ys, rows):
            c, mk = FAMILY[r["family"]]
            is_sel = r is sel
            ax.plot([float(r[key])], [y], marker=mk, linestyle="none", color=c,
                    markersize=11 if is_sel else 7.5, markeredgecolor=INK if is_sel else SURFACE,
                    markeredgewidth=1.8 if is_sel else 1.0, zorder=3)
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.axhspan(ys[rows.index(sel)] - 0.45, ys[rows.index(sel)] + 0.45, color="#f1f0ec", zorder=0)

    axl.axvline(float(base["val_mean"]), color=MUTED, linewidth=0.9, zorder=1)
    axl.text(float(base["val_mean"]), n - 0.35, "base ", fontsize=7.5, color=MUTED, va="bottom", ha="right")
    axl.set_xlabel("mean val loss over the 4 expert val sets\n(CE per assistant token; lower is better)")
    axl.set_title("Mean validation loss", loc="left", fontsize=10)
    axr.axvline(0.0, color=MUTED, linewidth=0.9, zorder=1)
    axr.axvline(1.0, color=MUTED, linewidth=0.9, zorder=1)
    axr.text(0.0, n - 0.35, " base", fontsize=7.5, color=MUTED, va="bottom")
    axr.text(1.0, n - 0.35, " each expert on its own domain", fontsize=7.5, color=MUTED, va="bottom", ha="right")
    axr.set_xlabel("mean retention R = (L_base − L) / (L_base − L_expert_d), averaged over\nthe 4 domains "
                   "(1 = keeps each expert's full gain; higher is better)")
    axr.set_title("Mean retention of expert gains  (ranking criterion)", loc="left", fontsize=10)
    axl.set_yticks(ys)
    axl.set_yticklabels([label_of(r) for r in rows], fontsize=8.5)
    for t, r in zip(axl.get_yticklabels(), rows):
        if r is sel:
            t.set_fontweight("bold")
            t.set_color(INK)
        elif r["kind"] == "reference":
            t.set_color(MUTED)
    ysel = ys[rows.index(sel)]
    axl.annotate(f"selected  {float(sel['val_mean']):.4f}", xy=(float(sel["val_mean"]), ysel), xytext=(10, 0),
                 textcoords="offset points", fontsize=8, fontweight="bold", va="center", color=INK)
    axr.annotate(f"selected  {float(sel['R_mean']):.3f}", xy=(float(sel["R_mean"]), ysel), xytext=(-12, 0),
                 textcoords="offset points", fontsize=8, fontweight="bold", va="center", ha="right", color=INK)
    axl.set_ylim(-0.7, n - 0.1)

    handles = [Line2D([], [], marker=FAMILY[f][1], color=FAMILY[f][0], linestyle="none", markersize=7.5,
                      markeredgecolor=SURFACE, label=lab)
               for f, lab in (("Linear", "Linear (task arithmetic)"), ("TIES", "TIES"),
                              ("DARE", "DARE (TIES / Linear combine)"), ("Base", "reference: base / single expert / mixed control"))]
    handles.append(Line2D([], [], marker="s", color=FAMILY["TIES"][0], linestyle="none", markersize=10,
                          markeredgecolor=INK, markeredgewidth=1.8, label="selected merge"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.225, 0.878), ncol=3, fontsize=8)

    fig.text(0.015, 0.955, "Merge-method ablation: 18 candidates scored by held-out val loss", fontsize=12.5,
             fontweight="bold")
    fig.text(0.015, 0.925,
             "Inputs: adapter_final checkpoints of E-SWE v1, E-TB v1, E-ASI, E-HLE (LoRA r32). Dense ΔW merged in memory "
             "and scored on the 4 expert val splits (max_len 64K), 1× H200, Slurm job 19807941, 2026-10-14.",
             fontsize=8, color=INK2)
    fig.text(0.015, 0.905,
             "Rows sorted by the job's ranking: mean retention rounded to 3 dp (desc), ties broken by mean val loss. "
             "The mixed control is a reference, not a merge.", fontsize=8, color=INK2)
    d02 = [r for r in rows if r["name"] == "ties_l1_d0.2"]
    note = ""
    if d02:
        d = d02[0]
        note = (f"TIES density 0.5 and 0.2 tie at R = {float(sel['R_mean_3dp']):.3f} (unrounded {float(sel['R_mean']):.5f} vs "
                f"{float(d['R_mean']):.5f}); the tie was broken by mean val loss ({float(sel['val_mean']):.5f} vs "
                f"{float(d['val_mean']):.5f}). ")
    fig.text(0.015, 0.052, note, fontsize=7.8, color=INK2)
    fig.text(0.015, 0.034, "Final models ties_l1_d0.5_v3 / _v4 reuse TIES λ=1, density 0.5 with E-SWE v2 (and E-TB v2 "
             "in v4) swapped in for the v1 experts; they were not re-scored in this ablation.", fontsize=7.8, color=INK2)
    fig.text(0.015, 0.012, "source: curves/source/merge_select.csv (from analysis/merge_select/rows.jsonl)",
             fontsize=7.5, color=MUTED)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE / "source"))
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--runs", nargs="*")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    metas = {r["run_id"]: r for r in read_csv(src / "runs.csv")}
    for rid in args.runs or list(metas):
        plot_run(metas[rid], read_csv(src / f"{rid}.csv"), out / f"{rid}.png")
        print(f"wrote {out / (rid + '.png')}")
    if (src / "merge_select.csv").exists() and not args.runs:
        plot_merge(read_csv(src / "merge_select.csv"), out / "merge_ablation.png")
        print(f"wrote {out / 'merge_ablation.png'}")


if __name__ == "__main__":
    main()
