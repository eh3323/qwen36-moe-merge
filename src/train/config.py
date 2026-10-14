"""YAML config loading with `inherit:` support and dotted CLI overrides (key.sub=value)."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml


def _deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and not v.get("_replace", False):
            out[k] = _deep_update(out[k], v)
        else:
            if isinstance(v, dict):
                v = {kk: vv for kk, vv in v.items() if kk != "_replace"}
            out[k] = copy.deepcopy(v)
    return out


def load_yaml_config(path: str | os.PathLike) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parents = cfg.pop("inherit", None)
    if parents:
        if isinstance(parents, str):
            parents = [parents]
        merged: dict = {}
        for p in parents:
            merged = _deep_update(merged, load_yaml_config(path.parent / p))
        cfg = _deep_update(merged, cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply `a.b.c=value` overrides; value parsed as YAML (so 1e-4, true, [1,2] work)."""
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    cfg = apply_overrides(load_yaml_config(path), overrides)
    # expand env vars in string leaves
    def _expand(x):
        if isinstance(x, dict):
            return {k: _expand(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_expand(v) for v in x]
        if isinstance(x, str):
            return os.path.expandvars(x)
        return x
    cfg = _expand(cfg)
    cfg["_config_path"] = str(path)
    return cfg
