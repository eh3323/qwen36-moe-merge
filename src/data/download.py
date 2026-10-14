"""Download pinned raw sources into $APEX/data/raw/<key>/ (snapshot_download, allow-listed files).

Nemotron-Math-v2 and OpenScienceReasoning-2 are NOT downloaded: they are streamed with early stop
by build_hle.py (see `stream_file` in configs/data/sources.yaml).

usage: python -m src.data.download [--only key1,key2] [--list]
"""
from __future__ import annotations

import argparse
import json
import time

from huggingface_hub import HfApi, snapshot_download

from .common import CFG, apex_path, log, raw_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    keys = [k for k in CFG["sources"] if "allow" in CFG["sources"][k]]
    if a.only:
        keys = [k for k in keys if k in a.only.split(",")]
    if a.list:
        print("\n".join(keys))
        return
    api = HfApi()
    record = {}
    for k in keys:
        s = CFG["sources"][k]
        rtype = "model" if s.get("kind") == "model" else "dataset"
        t0 = time.time()
        log(f"[download] {k}: {s['repo']}@{s['revision'][:10]} allow={s['allow']}")
        try:
            path = snapshot_download(repo_id=s["repo"], repo_type=rtype, revision=s["revision"],
                                     allow_patterns=s["allow"], local_dir=str(raw_dir(k)),
                                     max_workers=a.workers)
        except Exception as e:   # gated (e.g. Idavidrein/gpqa) or transient: record and continue
            log(f"[download] {k}: FAILED {type(e).__name__}: {str(e)[:200]}")
            record[k] = {"repo": s["repo"], "error": f"{type(e).__name__}: {str(e)[:300]}"}
            continue
        try:
            sha = api.repo_info(s["repo"], repo_type=rtype, revision=s["revision"]).sha
        except Exception as e:  # pragma: no cover
            sha = f"unknown ({e})"
        files = sorted(p for p in raw_dir(k).rglob("*") if p.is_file() and ".cache" not in p.parts)
        record[k] = {"repo": s["repo"], "requested_revision": s["revision"], "resolved_sha": sha,
                     "files": {str(p.relative_to(raw_dir(k))): p.stat().st_size for p in files},
                     "seconds": round(time.time() - t0, 1)}
        log(f"[download] {k}: {len(files)} files in {record[k]['seconds']}s -> {path}")
    out = apex_path("raw") / "download_record.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(record)
    out.write_text(json.dumps(old, indent=1))
    log(f"[download] wrote {out}")


if __name__ == "__main__":
    main()
