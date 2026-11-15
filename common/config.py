"""Config loading, command-line overrides, seeding and run directories."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def base_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(REPO_ROOT / "config.yaml"), help="path to config.yaml")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set E3_1.max_samples=200 (repeatable; VALUE is parsed as YAML)",
    )
    return p


def _apply_override(cfg: dict, assignment: str) -> None:
    if "=" not in assignment:
        raise ValueError(f"--set expects KEY=VALUE, got {assignment!r}")
    key, raw = assignment.split("=", 1)
    node = cfg
    parts = key.strip().split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot set {key}: {part} is not a mapping")
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for assignment in overrides or []:
        _apply_override(cfg, assignment)
    cache = cfg.get("common", {}).get("hf_cache_dir")
    if cache:
        # Must happen before transformers / datasets / vllm are imported.
        os.environ.setdefault("HF_HOME", str(cache))
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_run_dir(cfg: dict, exp_key: str) -> Path:
    exp_cfg = cfg[exp_key]
    run_name = exp_cfg.get("run_name") or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_cfg["run_name"] = run_name
    run_dir = REPO_ROOT / cfg["common"]["output_root"] / exp_key / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_resolved_config(cfg: dict, run_dir: Path) -> None:
    """Keep the exact config of each run next to its results.

    On resume, the stored config is kept as config.yaml and the new one is
    written alongside so that silent changes between sessions are visible.
    """
    target = run_dir / "config.yaml"
    if target.exists():
        with open(target, encoding="utf-8") as f:
            old = yaml.safe_load(f)
        if old != cfg:
            stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(run_dir / f"config.resume_{stamp}.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            print(f"[warn] config differs from the one this run started with; see {run_dir}/config.resume_{stamp}.yaml")
        return
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    for mod in ("torch", "transformers", "vllm", "datasets"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return info


def dump_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
