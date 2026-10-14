"""Shared I/O for LoRA adapters and shard-wise folding of dense deltas into the base HF checkpoint.

Works directly on safetensors shards (no model instantiation): every tensor name, the multimodal wrapper
(`model.visual.*`), the MTP head (`mtp.*`), config.json, tokenizer, chat_template.jinja and the
preprocessor configs are carried over unchanged, so vLLM loads the merged dir exactly like the base model.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import zlib
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

PEFT_PREFIX = "base_model.model."


def pick_device(dev: str = "auto") -> torch.device:
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


class LoraAdapter:
    """ΔW(module) = scale * B @ A, with module keyed by its base-checkpoint weight name."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.cfg = json.loads((self.path / "adapter_config.json").read_text())
        if self.cfg.get("rank_pattern") or self.cfg.get("alpha_pattern"):
            raise NotImplementedError("rank_pattern/alpha_pattern adapters not supported")
        r = int(self.cfg["r"])
        alpha = float(self.cfg["lora_alpha"])
        self.r = r
        self.scale = alpha / (r ** 0.5) if self.cfg.get("use_rslora") else alpha / r
        sd = load_file(str(self.path / "adapter_model.safetensors"))
        self.A: dict[str, torch.Tensor] = {}
        self.B: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            name = k[len(PEFT_PREFIX):] if k.startswith(PEFT_PREFIX) else k
            if ".lora_A." in name:
                self.A[name.split(".lora_A.")[0] + ".weight"] = v
            elif ".lora_B." in name:
                self.B[name.split(".lora_B.")[0] + ".weight"] = v
            else:
                raise ValueError(f"unexpected adapter tensor {k} (only LoRA A/B supported)")
        if set(self.A) != set(self.B):
            raise ValueError("adapter has unmatched lora_A/lora_B tensors")

    @property
    def keys(self) -> list[str]:
        return sorted(self.A)

    def delta(self, key: str, device, dtype=torch.float32) -> torch.Tensor:
        A = self.A[key].to(device=device, dtype=dtype)
        B = self.B[key].to(device=device, dtype=dtype)
        return (B @ A) * self.scale

    def factors(self, key: str, device, dtype=torch.float32):
        return (self.B[key].to(device=device, dtype=dtype) * self.scale,
                self.A[key].to(device=device, dtype=dtype))


def base_weight_map(base_dir: str | os.PathLike) -> dict[str, str]:
    idx = Path(base_dir) / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    single = Path(base_dir) / "model.safetensors"
    with safe_open(str(single), "pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def write_folded_model(
    base_dir: str | os.PathLike,
    out_dir: str | os.PathLike,
    target_keys: list[str],
    delta_fn: Callable[[str, torch.Tensor], torch.Tensor],
    info: dict,
    link_unchanged: bool = True,
    log=print,
) -> dict:
    """Copy base dir to out_dir, replacing W by W + delta_fn(key, W) for target_keys (shard by shard).

    delta_fn returns an fp32 delta on any device; addition is done in fp32 then cast back to the base dtype.
    Shards that contain no target key are symlinked (saves ~disk) when link_unchanged is True.
    """
    base_dir, out_dir = Path(base_dir).resolve(), Path(out_dir)
    wmap = base_weight_map(base_dir)
    missing = [k for k in target_keys if k not in wmap]
    if missing:
        raise KeyError(f"{len(missing)} adapter modules not in base checkpoint, e.g. {missing[:3]}")
    tmp = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    # 1) all non-weight files (config, tokenizer, chat template, preprocessor/video configs, generation config...)
    for p in base_dir.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors"):
            shutil.copy2(p, tmp / p.name)
    # 2) shards
    tset = set(target_keys)
    by_shard: dict[str, list[str]] = {}
    for k, s in wmap.items():
        by_shard.setdefault(s, []).append(k)
    t0, done, stats = time.time(), 0, {"n_modified": 0, "max_rel_delta": 0.0}
    for shard, keys in sorted(by_shard.items()):
        mod = [k for k in keys if k in tset]
        if not mod and link_unchanged:
            os.symlink(base_dir / shard, tmp / shard)
            continue
        tensors = {}
        with safe_open(str(base_dir / shard), "pt") as f:
            meta = f.metadata() or {"format": "pt"}
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        for k in mod:
            w = tensors[k]
            d = delta_fn(k, w)
            if d.shape != w.shape:
                raise ValueError(f"delta shape {tuple(d.shape)} != weight shape {tuple(w.shape)} for {k}")
            new = (w.to(d.device, torch.float32) + d).to(w.dtype).cpu()
            rel = (d.norm() / w.float().norm().clamp(min=1e-12)).item()
            stats["max_rel_delta"] = max(stats["max_rel_delta"], rel)
            tensors[k] = new
            done += 1
        save_file(tensors, str(tmp / shard), metadata=meta)
        log(f"[fold] {shard}: {len(mod)} modified ({done}/{len(tset)}) {time.time() - t0:.0f}s")
    stats["n_modified"] = done
    if done != len(tset):
        raise RuntimeError(f"modified {done} tensors but expected {len(tset)}")
    info = {**info, **stats, "base_dir": str(base_dir), "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": round(time.time() - t0, 1)}
    (tmp / "merge_info.json").write_text(json.dumps(info, indent=1, default=str))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)
    log(f"[fold] wrote {out_dir} ({done} tensors modified, max ||ΔW||/||W|| = {stats['max_rel_delta']:.3e})")
    return info


def save_lowrank_adapter(out_dir: str | os.PathLike, template_cfg: dict, factors: dict[str, tuple], rank: int):
    """Export {base_key: (B[out,k], A[k,in])} as a PEFT LoRA adapter with alpha=r (scale 1)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = {}
    for key, (B, A) in factors.items():
        mod = key[: -len(".weight")]
        sd[f"{PEFT_PREFIX}{mod}.lora_A.weight"] = A.contiguous().to(torch.bfloat16).cpu()
        sd[f"{PEFT_PREFIX}{mod}.lora_B.weight"] = B.contiguous().to(torch.bfloat16).cpu()
    save_file(sd, str(out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    cfg = dict(template_cfg)
    cfg.update({"r": rank, "lora_alpha": rank, "use_rslora": False, "lora_dropout": 0.0})
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
