"""LoRA SFT for Qwen3.6-35B-A3B (qwen3_5_moe) — lean custom loop, resumable, multi-GPU (data parallel).

Launch (one process per GPU; works under torchrun or `srun` with RANK/LOCAL_RANK/WORLD_SIZE exported):
  python -m src.train.sft_lora --config configs/train/e_swe.yaml [--set train.max_steps=30 ...]
  python -m src.train.sft_lora --config configs/train/e_swe.yaml --prepare-only     # tokenize+cache, stats
  python -m src.train.sft_lora --config configs/train/e_swe.yaml --dry-run-render 2 # CPU render/mask check

Parallelism: every rank holds the full bf16 model (67 GB fits on H200/B200 with activation checkpointing);
only the LoRA grads (~38M fp32 params at r=32) are all-reduced once per optimizer step (no DDP wrapper, so
no hook/reentrancy issues with checkpointing and no comm inside the backward of each micro-step).
Batching: each micro-step is one packed sequence of <= max_seq_len tokens (bs=1, cu_seqlens isolation);
packs_per_step = global_batch_tokens / max_seq_len, split across ranks (grad accumulation = packs/world).
Loss = sum CE over assistant targets in the global step / #assistant targets in the global step.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.train.config import load_config  # noqa: E402
from src.train import data as D  # noqa: E402

STOP = {"flag": False, "signal": None}


def _on_signal(signum, frame):
    STOP["flag"] = True
    STOP["signal"] = signal.Signals(signum).name


# ----------------------------------------------------------------------------- utils
class Logger:
    def __init__(self, rank: int, log_dir: Path | None):
        self.rank = rank
        self.log_dir = log_dir
        self.metrics_f = None
        if rank == 0 and log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_f = open(log_dir / "metrics.jsonl", "a", buffering=1)

    def __call__(self, *a):
        if self.rank == 0:
            print(time.strftime("%H:%M:%S"), *a, flush=True)

    def metric(self, rec: dict):
        if self.metrics_f is not None:
            rec = {"time": time.time(), **rec}
            self.metrics_f.write(json.dumps(rec) + "\n")


def code_fingerprint() -> str:
    h = hashlib.sha256()
    for p in sorted(glob.glob(str(REPO / "src/train/*.py")) + glob.glob(str(REPO / "configs/train/*.yaml"))):
        h.update(Path(p).name.encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        pass
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    f = REPO / ".git_commit"
    return f.read_text().strip() if f.exists() else None


def versions() -> dict:
    v = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "peft", "fla", "triton", "kernels", "safetensors", "numpy"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "?")
        except Exception:
            v[mod] = None
    try:
        import torch

        v["cuda"] = torch.version.cuda
        v["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    try:
        v["nvidia_driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).split("\n")[0]
    except Exception:
        v["nvidia_driver"] = None
    return v


def lr_lambda_factory(total: int, warmup: int, min_ratio: float):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return f


# ----------------------------------------------------------------------------- data setup
def load_sets(cfg, rank: int, log) -> tuple[list, dict, dict]:
    dcfg = cfg["data"]
    if dcfg.get("synthetic"):
        sc = dcfg["synthetic"]
        tr = [D.synthetic_set(int(sc.get("n_samples", 512)), int(sc.get("min_len", 2048)),
                              int(sc.get("max_len", 16384)), seed=int(cfg["train"]["seed"]))]
        return tr, {}, {"synthetic": sc}
    common = dict(model_path=cfg["model"]["path"], cache_dir=dcfg["cache_dir"],
                  preserve_thinking=bool(dcfg.get("preserve_thinking", True)),
                  num_proc=int(dcfg.get("num_proc", 16)), build=(rank == 0), log=log)
    train_sets = [D.build_or_load_cache(name, path, **common) for name, path in dcfg["train"].items()]
    val_sets = {}
    for name, path in (dcfg.get("val") or {}).items():
        if os.path.exists(path):
            val_sets[name] = D.build_or_load_cache(name, path, **common)
        else:
            log(f"[data] WARNING val set {name} missing: {path}")
    manifest = {"train": {}, "val": {}}
    for s in train_sets:
        manifest["train"][s.name] = {k: s.meta.get(k) for k in (
            "source_file", "source_sha256", "n_records", "n_kept", "n_template_errors", "n_no_label",
            "n_turn_count_mismatch", "total_tokens", "label_tokens", "len_p50", "len_p90", "len_max",
            "tokenizer_fingerprint", "preserve_thinking")}
    for n, s in val_sets.items():
        manifest["val"][n] = {k: s.meta.get(k) for k in ("source_file", "source_sha256", "n_kept",
                                                         "total_tokens", "label_tokens")}
    # data-agent manifests (if any) are hashed too
    for mf in sorted(glob.glob(str(REPO / "data/manifests/*"))):
        if os.path.isfile(mf):
            manifest.setdefault("data_manifests", {})[os.path.basename(mf)] = D.sha256_file(mf)
    return train_sets, val_sets, manifest


# ----------------------------------------------------------------------------- dry-run render (CPU)
def dry_run_render(cfg, n: int, sample_file: str | None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model"]["path"])
    path = sample_file or next(iter(cfg["data"]["train"].values()))
    D._init_worker(cfg["model"]["path"], bool(cfg["data"].get("preserve_thinking", True)))
    with open(path) as f:
        lines = [ln for ln in f if ln.strip()][:n]
    for ln in lines:
        r = D.tokenize_record(ln)
        if "error" in r:
            print("TEMPLATE ERROR", r)
            continue
        ids, mask = r["ids"], r["mask"]
        print(f"=== id={r['id']} tokens={len(ids)} label_tokens={int(mask.sum())} "
              f"assistant_turns(tokens)={r['n_turns']} assistant_msgs={r['n_asst_msgs']}")
        # print with supervised spans wrapped in [[ ]]
        out, cur, buf = [], None, []
        for t, m in zip(ids.tolist(), mask.tolist()):
            if m != cur and buf:
                txt = tok.decode(buf)
                out.append(f"[[{txt}]]" if cur else txt)
                buf = []
            cur = m
            buf.append(t)
        if buf:
            txt = tok.decode(buf)
            out.append(f"[[{txt}]]" if cur else txt)
        s = "".join(out)
        print(s if len(s) < 6000 else s[:3000] + "\n...\n" + s[-3000:])


# ----------------------------------------------------------------------------- checkpoints
def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    cands = sorted(p for p in ckpt_dir.glob("step_*") if p.is_dir() and (p / "state.json").exists()
                   and not p.name.endswith(".tmp"))
    return cands[-1] if cands else None


def save_checkpoint(model, opt, sched, ckpt_dir: Path, step: int, state: dict, keep_last: int, log):
    d = ckpt_dir / f"step_{step:06d}"
    tmp = ckpt_dir / f"step_{step:06d}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save_pretrained(str(tmp))
    import torch

    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, tmp / "trainer_state.pt")
    (tmp / "state.json").write_text(json.dumps(state, indent=1))
    shutil.rmtree(d, ignore_errors=True)
    os.replace(tmp, d)
    old = sorted(p for p in ckpt_dir.glob("step_*") if p.is_dir() and not p.name.endswith(".tmp"))
    for p in old[:-keep_last] if keep_last > 0 else []:
        shutil.rmtree(p, ignore_errors=True)
    log(f"[ckpt] saved {d}")
    return d


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="key.sub=value (YAML)")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--dry-run-render", type=int, default=0)
    ap.add_argument("--sample-file", default=None)
    ap.add_argument("--print-run-dir", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.print_run_dir:
        print(Path(cfg["output"]["root"]) / cfg["output"]["run_id"])
        return

    if args.dry_run_render:
        return dry_run_render(cfg, args.dry_run_render, args.sample_file)

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    tcfg, dcfg, ocfg = cfg["train"], cfg["data"], cfg["output"]
    run_id = ocfg["run_id"]
    run_dir = Path(ocfg["root"]) / run_id
    log_dir = run_dir / "logs"
    ckpt_dir = run_dir / "checkpoints"
    log = Logger(rank, None if args.prepare_only else log_dir)
    seed = int(tcfg["seed"])
    max_len = int(dcfg["max_seq_len"])
    window = int(dcfg.get("pack_window", 512))
    seg_start = time.time()
    if (not args.prepare_only and (run_dir / "EXIT_STATUS").exists()
            and (run_dir / "EXIT_STATUS").read_text().strip() == "done" and (run_dir / "adapter_final").exists()):
        log(f"[train] {run_id} already finished ({run_dir}/adapter_final); nothing to do")
        return

    # ---------------- data (rank 0 builds caches; others wait on the files) ----------------
    train_sets, val_sets, manifest = load_sets(cfg, rank, log)
    pieces, piece_stats = D.make_pieces(train_sets, max_len, dcfg.get("overlong", "truncate"))
    epochs = int(tcfg["epochs"])
    plans = [D.epoch_plan(pieces, max_len, seed, e, window) for e in range(epochs)]
    packs_per_step = max(1, int(round(int(tcfg["global_batch_tokens"]) / max_len)))
    packs_per_step = int(math.ceil(packs_per_step / world) * world)
    steps_per_epoch = [D.steps_from_packs(len(p), packs_per_step) for p in plans]
    total_steps = sum(steps_per_epoch)
    if tcfg.get("max_steps"):
        total_steps = min(total_steps, int(tcfg["max_steps"]))
    tot_tok = piece_stats["tokens"]
    lab_tok = sum(s.label_tokens(i, L) for si, i, L in pieces for s in [train_sets[si]])
    pack_eff = tot_tok / max(1, len(plans[0]) * max_len)
    data_summary = {**piece_stats, "label_tokens": lab_tok, "packs_per_epoch": [len(p) for p in plans],
                    "pack_efficiency": round(pack_eff, 4), "packs_per_step": packs_per_step,
                    "grad_accum": packs_per_step // world, "tokens_per_step_capacity": packs_per_step * max_len,
                    "steps_per_epoch": steps_per_epoch, "total_steps": total_steps}
    log(f"[data] {json.dumps(data_summary)}")
    if args.prepare_only:
        if rank == 0:
            print(json.dumps({"manifest": manifest, "plan": data_summary}, indent=1))
        return

    val_packs = {}
    for name, s in val_sets.items():
        vp, _ = D.make_pieces([s], max_len, dcfg.get("overlong", "truncate"),
                              max_tokens=int(dcfg.get("val_max_tokens_per_set", 524288)))
        val_packs[name] = D.pack_pieces(vp, max_len, None, window)

    # ---------------- distributed + model ----------------
    import torch
    import torch.distributed as dist

    from src.train import modeling as Mdl

    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device())
    if world > 1:
        import datetime

        dist.init_process_group("nccl", device_id=device, timeout=datetime.timedelta(minutes=60))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGUSR1, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    mcfg = cfg["model"]
    Mdl.setup_attention(mcfg.get("attn_backend", "auto"), device, log)
    Mdl.patch_gated_deltanet(mcfg.get("gdn_backend", "fla"), log)
    gdn_test = Mdl.selftest_gdn(device, log) if mcfg.get("gdn_backend", "fla") == "fla" else {}
    t_load = time.time()
    model = Mdl.load_model(cfg, device, log)
    model, lora_modules, lora_regex = Mdl.apply_lora(model, cfg["lora"], log)
    load_s = time.time() - t_load
    model.train()
    text_model, lm_head = Mdl.get_text_parts(model)
    tparams = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in tparams)
    if rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "lora_targets.txt").write_text("\n".join(lora_modules) + "\n")

    opt = torch.optim.AdamW(tparams, lr=float(tcfg["lr"]), betas=tuple(tcfg.get("adam_betas", (0.9, 0.999))),
                            eps=float(tcfg.get("adam_eps", 1e-8)), weight_decay=float(tcfg.get("weight_decay", 0.0)),
                            fused=True)
    warmup = max(1, int(round(float(tcfg.get("warmup_ratio", 0.03)) * total_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(total_steps, warmup, float(tcfg.get("min_lr_ratio", 0.1))))

    # ---------------- resume ----------------
    global_step, prev_state = 0, {}
    ck = latest_checkpoint(ckpt_dir)
    if ck is not None and not tcfg.get("no_resume", False):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        sd = load_file(str(ck / "adapter_model.safetensors"), device=str(device))
        res = set_peft_model_state_dict(model, sd)
        unexpected = getattr(res, "unexpected_keys", [])
        if unexpected:
            raise RuntimeError(f"unexpected keys when resuming: {unexpected[:5]}")
        for p in tparams:
            p.data = p.data.float()
        ts = torch.load(ck / "trainer_state.pt", map_location=device, weights_only=False)
        opt.load_state_dict(ts["optimizer"])
        sched.load_state_dict(ts["scheduler"])
        prev_state = json.loads((ck / "state.json").read_text())
        global_step = int(prev_state["global_step"])
        log(f"[resume] from {ck} at global_step={global_step}")

    # ---------------- metadata ----------------
    meta_path = log_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if (rank == 0 and meta_path.exists()) else {}
    gpu_name = torch.cuda.get_device_name(device)
    segment = {"job_id": os.environ.get("SLURM_JOB_ID"), "restart_count": os.environ.get("SLURM_RESTART_COUNT"),
               "host": socket.gethostname(), "world_size": world, "gpu": gpu_name,
               "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(seg_start)), "start_step": global_step,
               "model_load_s": round(load_s, 1)}

    def write_meta(status: str, extra: dict | None = None):
        if rank != 0:
            return
        segs = meta.get("segments", [])
        seg = dict(segment, end=time.strftime("%Y-%m-%dT%H:%M:%S"), end_step=global_step,
                   gpu_hours=round((time.time() - seg_start) * world / 3600, 4))
        all_segs = segs + [seg]
        out = {
            "run_id": run_id, "status": status,
            "git_commit": git_commit(), "code_fingerprint": code_fingerprint(),
            "config_path": cfg.get("_config_path"), "config": cfg,
            "seed": seed, "world_size": world, "gpu_type": gpu_name,
            "gpu_count": world, "versions": versions(), "kernels": dict(Mdl.KERNEL_INFO),
            "gdn_selftest": gdn_test, "data_manifest": manifest, "data_plan": data_summary,
            "lora": {"regex": lora_regex, "n_modules": len(lora_modules), "trainable_params": n_trainable},
            "start": all_segs[0]["start"], "end": seg["end"],
            "gpu_hours_total": round(sum(s["gpu_hours"] for s in all_segs), 4),
            "segments": all_segs, **(extra or {}),
        }
        tmp = meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=1, default=str))
        os.replace(tmp, meta_path)
        (run_dir / "EXIT_STATUS").write_text(status + "\n")

    write_meta("running")

    # ---------------- helpers ----------------
    ce_chunk = int(tcfg.get("ce_chunk_tokens", 4096))

    def to_dev(pack, sets):
        ids, lab, pos, cu = D.build_pack_arrays(pack, sets)
        cu_cpu = torch.from_numpy(cu.astype(np.int64))
        return (torch.from_numpy(ids).pin_memory().to(device, non_blocking=True),
                torch.from_numpy(lab).pin_memory().to(device, non_blocking=True),
                torch.from_numpy(pos).pin_memory().to(device, non_blocking=True),
                torch.from_numpy(cu).to(device, non_blocking=True), cu_cpu)

    def allreduce_(t, op=None):
        if world > 1:
            dist.all_reduce(t, op=op or dist.ReduceOp.SUM)
        return t

    def evaluate(step: int):
        if not val_packs:
            return {}
        model.eval()
        res = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for name, packs in val_packs.items():
                acc = torch.zeros(2, device=device, dtype=torch.float64)
                for pack in packs[rank::world]:
                    ids, lab, pos, cu, cu_cpu = to_dev(pack, [val_sets[name]])
                    ls, n = Mdl.forward_pack(text_model, lm_head, ids, lab, pos, cu, cu_cpu, ce_chunk)
                    acc[0] += ls.double()
                    acc[1] += n
                allreduce_(acc)
                vl = (acc[0] / acc[1].clamp(min=1)).item()
                res[name] = vl
                log.metric({"type": "val", "step": step, "set": name, "loss": vl, "tokens": int(acc[1].item())})
                log(f"[val] step={step} {name} loss={vl:.4f} targets={int(acc[1].item())}")
        model.train()
        return res

    eval_every = int(tcfg.get("eval_every", 0) or 0)
    save_every = int(tcfg.get("save_every", 0) or 0)
    keep_last = int(tcfg.get("keep_last", 2))
    clip = float(tcfg.get("grad_clip", 1.0))
    cum_steps = np.cumsum([0] + steps_per_epoch)
    tok_s_hist, peak_hist = [], []
    cum_tokens = int(prev_state.get("cum_tokens", 0))
    log(f"[train] run={run_id} world={world} gpu={gpu_name} total_steps={total_steps} warmup={warmup} "
        f"packs/step={packs_per_step} accum={packs_per_step // world} trainable={n_trainable/1e6:.2f}M")

    if global_step == 0 and tcfg.get("eval_at_start", True):
        evaluate(0)

    def ckpt_state():
        return {"global_step": global_step, "cum_tokens": cum_tokens, "run_id": run_id,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ---------------- train loop ----------------
    while global_step < total_steps:
        epoch = int(np.searchsorted(cum_steps, global_step, side="right") - 1)
        s_in_ep = global_step - int(cum_steps[epoch])
        group = plans[epoch][s_in_ep * packs_per_step:(s_in_ep + 1) * packs_per_step]
        n_lab = sum(D.pack_label_tokens(p, train_sets) for p in group)
        n_tok = sum(D.pack_tokens(p) for p in group)
        mine = group[rank::world]
        torch.manual_seed(seed * 1_000_003 + global_step * 1009 + rank)  # dropout RNG: resume-deterministic
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t0 = time.time()
        loss_acc = torch.zeros((), device=device, dtype=torch.float64)
        for pack in mine:
            ids, lab, pos, cu, cu_cpu = to_dev(pack, train_sets)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ls, _ = Mdl.forward_pack(text_model, lm_head, ids, lab, pos, cu, cu_cpu, ce_chunk)
            (ls / max(1, n_lab)).backward()
            loss_acc += ls.detach().double()
        for p in tparams:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        if world > 1:
            from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

            grads = [p.grad for p in tparams]
            flat = _flatten_dense_tensors(grads)
            dist.all_reduce(flat)
            for g, f in zip(grads, _unflatten_dense_tensors(flat, grads)):
                g.copy_(f)
        allreduce_(loss_acc)
        gnorm = torch.nn.utils.clip_grad_norm_(tparams, clip)
        lr_now = sched.get_last_lr()[0]
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        dt = time.time() - t0
        peak = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30,
                             torch.cuda.max_memory_reserved(device) / 2**30], device=device)
        allreduce_(peak, dist.ReduceOp.MAX if world > 1 else None)
        global_step += 1
        cum_tokens += n_tok
        loss = loss_acc.item() / max(1, n_lab)
        tps = n_tok / dt
        tok_s_hist.append(tps)
        peak_hist.append(peak[0].item())
        rec = {"type": "train", "step": global_step, "epoch": epoch, "loss": loss, "lr": lr_now,
               "grad_norm": float(gnorm), "tokens": n_tok, "label_tokens": n_lab, "n_packs": len(group),
               "step_time": dt, "tokens_per_s": tps, "tokens_per_s_per_gpu": tps / world,
               "peak_mem_gb": peak[0].item(), "peak_reserved_gb": peak[1].item(), "cum_tokens": cum_tokens}
        log.metric(rec)
        if global_step % int(tcfg.get("log_every", 1)) == 0 or global_step <= 3:
            log(f"[train] step {global_step}/{total_steps} ep={epoch} loss={loss:.4f} lr={lr_now:.2e} "
                f"gnorm={float(gnorm):.3f} tok={n_tok} {tps:,.0f} tok/s ({tps/world:,.0f}/gpu) "
                f"{dt:.1f}s peak={peak[0].item():.1f}GB")

        # stop request (signal on any rank, or STOP file)
        stop = torch.tensor([1.0 if (STOP["flag"] or (run_dir / "STOP").exists()) else 0.0], device=device)
        allreduce_(stop, dist.ReduceOp.MAX if world > 1 else None)
        stopping = stop.item() > 0 and global_step < total_steps

        if eval_every and global_step % eval_every == 0 and global_step < total_steps and not stopping:
            evaluate(global_step)
        if rank == 0 and ((save_every and global_step % save_every == 0) or stopping) and global_step < total_steps:
            d = save_checkpoint(model, opt, sched, ckpt_dir, global_step, ckpt_state(), keep_last, log)
            log.metric({"type": "ckpt", "step": global_step, "path": str(d)})
            write_meta("running")
        if world > 1:
            dist.barrier()
        if stopping:
            log(f"[train] stop requested ({STOP['signal'] or 'STOP file'}); checkpointed at step {global_step}")
            write_meta("interrupted")
            if world > 1:
                dist.destroy_process_group()
            sys.exit(0)  # EXIT_STATUS=interrupted tells the sbatch wrapper to requeue

    # ---------------- final ----------------
    final_val = evaluate(global_step)
    summary = {}
    if tok_s_hist:
        warm = tok_s_hist[3:] if len(tok_s_hist) > 5 else tok_s_hist
        summary = {"steps_measured": len(warm), "median_tokens_per_s": float(np.median(warm)),
                   "median_tokens_per_s_per_gpu": float(np.median(warm)) / world,
                   "max_peak_mem_gb": float(max(peak_hist)), "world_size": world, "gpu": gpu_name,
                   "max_seq_len": max_len, "attention": Mdl.KERNEL_INFO.get("attention"),
                   "experts": Mdl.KERNEL_INFO.get("experts"), "gdn": Mdl.KERNEL_INFO.get("gdn")}
        log(f"THROUGHPUT_SUMMARY {json.dumps(summary)}")
    if rank == 0:
        if not tcfg.get("skip_final_save", False):
            fin = run_dir / "adapter_final"
            shutil.rmtree(fin, ignore_errors=True)
            model.save_pretrained(str(fin))
            (fin / "state.json").write_text(json.dumps(ckpt_state(), indent=1))
            log.metric({"type": "ckpt", "step": global_step, "path": str(fin), "final": True})
            log(f"[ckpt] final adapter -> {fin}")
        (log_dir / "throughput.json").write_text(json.dumps(summary, indent=1))
        write_meta("done", {"final_val": final_val, "throughput": summary})
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
