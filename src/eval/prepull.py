#!/usr/bin/env python3
"""Pre-pull plan for the container images of TB2 (89) and SWE-Pro HARD-51 (51).

Subcommands
-----------
list      print "<docker_image>\t<sif_path>" for every image of the chosen sets
estimate  query registries for compressed image sizes (no pulls, no blobs):
            * Docker Hub images: hub.docker.com REST API (`full_size`); this does
              NOT count against Docker Hub's anonymous pull rate limit.
            * ghcr.io images: registry v2 manifest GET with an anonymous token.
          Writes a TSV and prints totals. SIF size ~= compressed size (both are
          compressed filesystems), so the total is a good disk estimate.

The SIF file name reproduces Harbor's singularity backend naming exactly
(environments/singularity/singularity.py `_convert_docker_to_sif`):
    <cache>/<image with "/" and ":" replaced by "_">.sif
so Harbor finds the pre-built file and never pulls at trial time.

Stdlib only. Usage (login node is fine for list/estimate):
    python -I src/eval/prepull.py estimate --sets tb2,swe51 --out $APEX/work/prepull_sizes.tsv
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import urllib.request
from pathlib import Path

BENCH = Path("/scratch/eh3323/apex/data/benchmarks")
TB2_DIR = BENCH / "terminal_bench2_repo"
SWE_DIR = BENCH / "swebench_pro_repo/v2"
DEFAULT_CACHE = Path("/scratch/eh3323/apex/cache/sif")


def _docker_image(task_toml: Path) -> str | None:
    m = re.search(r'^docker_image\s*=\s*"([^"]+)"', task_toml.read_text(), re.M)
    return m.group(1) if m else None


def images(sets: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if "tb2" in sets:
        for t in sorted(TB2_DIR.glob("*/task.toml")):
            img = _docker_image(t)
            if img:
                out.append(("tb2", img))
    if "swe51" in sets or "swe642" in sets:
        ids = (
            (SWE_DIR / "hard51_ids.txt").read_text().split()
            if "swe642" not in sets
            else sorted(p.name for p in (SWE_DIR / "tasks").iterdir())
        )
        for i in ids:
            img = _docker_image(SWE_DIR / "tasks" / i / "task.toml")
            if img:
                out.append(("swe", img))
    return out


def sif_path(image: str, cache: Path) -> Path:
    if ":" not in image.split("/")[-1]:
        image = f"{image}:latest"
    return cache / (image.replace("/", "_").replace(":", "_") + ".sif")


def _get(url: str, headers: dict | None = None, timeout: int = 30) -> tuple[dict, dict]:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()), dict(r.headers)


def size_dockerhub(image: str) -> int | None:
    repo, tag = image.rsplit(":", 1)
    if "/" not in repo:
        repo = f"library/{repo}"
    data, _ = _get(f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag}")
    if data.get("full_size"):
        return int(data["full_size"])
    for im in data.get("images", []):
        if im.get("architecture") == "amd64":
            return int(im.get("size") or 0)
    return None


_GHCR_TOKENS: dict[str, str] = {}


def size_ghcr(image: str) -> int | None:
    ref = image[len("ghcr.io/"):]
    repo, tag = ref.rsplit(":", 1)
    if repo not in _GHCR_TOKENS:
        tok, _ = _get(f"https://ghcr.io/token?scope=repository:{repo}:pull")
        _GHCR_TOKENS[repo] = tok["token"]
    h = {
        "Authorization": f"Bearer {_GHCR_TOKENS[repo]}",
        "Accept": ", ".join([
            "application/vnd.oci.image.index.v1+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
            "application/vnd.oci.image.manifest.v1+json",
            "application/vnd.docker.distribution.manifest.v2+json",
        ]),
    }
    man, _ = _get(f"https://ghcr.io/v2/{repo}/manifests/{tag}", h)
    if "manifests" in man:  # index -> pick linux/amd64
        dig = next(m["digest"] for m in man["manifests"]
                   if m.get("platform", {}).get("architecture") == "amd64")
        man, _ = _get(f"https://ghcr.io/v2/{repo}/manifests/{dig}", h)
    return sum(int(layer["size"]) for layer in man.get("layers", []))


def estimate_one(item: tuple[str, str]) -> tuple[str, str, int | None, str]:
    bench, image = item
    try:
        size = size_ghcr(image) if image.startswith("ghcr.io/") else size_dockerhub(image)
        return bench, image, size, ""
    except Exception as e:  # report, do not abort the whole estimate
        return bench, image, None, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "estimate"])
    ap.add_argument("--sets", default="tb2,swe51", help="comma list of tb2, swe51, swe642")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    items = images(a.sets.split(","))
    cache = Path(a.cache)
    if a.cmd == "list":
        for _, img in items:
            print(f"{img}\t{sif_path(img, cache)}")
        return 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(estimate_one, items))
    lines = ["bench\timage\tcompressed_bytes\terror"]
    tot: dict[str, int] = {}
    n_err = 0
    for bench, img, size, err in rows:
        lines.append(f"{bench}\t{img}\t{size if size is not None else ''}\t{err}")
        if size is None:
            n_err += 1
        else:
            tot[bench] = tot.get(bench, 0) + size
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")
    for b, s in tot.items():
        n = sum(1 for r in rows if r[0] == b and r[2] is not None)
        print(f"{b}: {n} images, {s/1e9:.1f} GB compressed (SIF on disk ~ same)")
    print(f"total: {sum(tot.values())/1e9:.1f} GB; lookup errors: {n_err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
