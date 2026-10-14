"""Fold ONE LoRA adapter into the base weights -> full HF model dir servable by vLLM like the base model.

  python -m src.merge.merge_lora_to_full --adapter $APEX/runs/e_tb_r32/adapter_final \
      --out $APEX/models/merged/e_tb_r32 [--scale 1.0] [--base $APEX/models/Qwen3.6-35B-A3B] [--no-link]

W' = W + scale * (alpha/r) * B @ A for every adapted module (fp32 add, cast back to bf16). Everything else
(vision tower, MTP head, routed experts, configs, tokenizer, chat template, preprocessor configs) is copied
or symlinked unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.merge.lora_io import LoraAdapter, pick_device, write_folded_model  # noqa: E402

DEFAULT_BASE = "/scratch/eh3323/apex/models/Qwen3.6-35B-A3B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--scale", type=float, default=1.0, help="extra multiplier on the LoRA delta")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-link", action="store_true", help="copy (not symlink) shards without adapted tensors")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = LoraAdapter(args.adapter)
    print(f"[merge] adapter {args.adapter}: {len(ad.keys)} modules, r={ad.r}, scale={ad.scale} x {args.scale}")

    def delta_fn(key, w):
        return ad.delta(key, dev) * args.scale

    write_folded_model(args.base, args.out, ad.keys, delta_fn,
                       info={"type": "single_adapter", "adapter": str(Path(args.adapter).resolve()),
                             "adapter_config": ad.cfg, "extra_scale": args.scale},
                       link_unchanged=not args.no_link)


if __name__ == "__main__":
    main()
