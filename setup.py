#!/usr/bin/env python
"""Prepare a fresh Colab (L4) runtime for the Stage-3 experiment scripts.

This is an environment script, not a setuptools package file: run it directly.

    python setup.py                  # install packages, download model + datasets, check GPU
    python setup.py --smoke-test     # ... and generate one answer end to end
    python setup.py --skip-install   # only downloads / checks (e.g. after a runtime restart)

What is installed and downloaded is set in the `setup:` section of config.yaml.
To keep the ~16 GB of model weights across Colab sessions, mount Google Drive
first and set `common.hf_cache_dir` to a folder on it.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def sh(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_yaml() -> None:
    try:
        import yaml  # noqa: F401
    except ImportError:
        sh([sys.executable, "-m", "pip", "install", "-q", "pyyaml"])


def check_gpu() -> None:
    print("\n== GPU ==")
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"])
    else:
        print("[warn] nvidia-smi not found: no GPU visible. Select Runtime > Change runtime type > L4 GPU.")


def install(packages: list[str]) -> None:
    print("\n== Installing packages ==")
    sh([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    sh([sys.executable, "-m", "pip", "install", "-q", *packages])


def report_versions() -> None:
    print("\n== Versions ==")
    code = (
        "import importlib\n"
        "for m in ['torch','transformers','vllm','datasets','accelerate']:\n"
        "    try:\n"
        "        print(f'{m:14s}', importlib.import_module(m).__version__)\n"
        "    except Exception as e:\n"
        "        print(f'{m:14s}', 'NOT AVAILABLE', type(e).__name__, e)\n"
        "import torch\n"
        "print('cuda available', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\n"
    )
    # Fresh interpreter, so freshly installed versions are reported.
    subprocess.run([sys.executable, "-c", code])


def download_model(name: str) -> None:
    print(f"\n== Downloading model {name} ==")
    from huggingface_hub import snapshot_download

    path = snapshot_download(name, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.jinja"])
    print(f"model at {path}")


def download_datasets(cfg: dict, names: list[str]) -> None:
    from datasets import load_dataset

    for name in names:
        dcfg = cfg["common"]["datasets"][name]
        hf_id = dcfg["hf_id"]
        splits = dcfg.get("download_splits")
        print(f"\n== Downloading dataset {name} ({hf_id}, splits: {splits or 'all'}) ==")
        if splits:
            for split in splits:
                print(split, load_dataset(hf_id, split=split))
        else:
            print(load_dataset(hf_id))


def smoke_test(cfg: dict) -> None:
    print("\n== Smoke test ==")
    sys.path.insert(0, str(REPO_ROOT))
    from common.scienceqa import load_scienceqa
    from common.vlm import VLM, GenParams, Request

    items = load_scienceqa(cfg["common"]["datasets"]["scienceqa"]["hf_id"], "test", image_only=True, max_samples=4)
    vlm = VLM(cfg["common"]["model"])
    prompt = cfg["E3_1"]["prompts"]["direct"]
    reqs = [Request(prompt.format(question=it.question, context=it.context(), options=it.options_str()), it.image) for it in items]
    outs = vlm.generate(reqs, GenParams(max_new_tokens=16))
    for it, o in zip(items, outs):
        print(f"  {it.id}: gold={it.answer_letter} output={o.text!r}")
    print("smoke test OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Set up the Stage-3 experiment environment.")
    ap.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--smoke-test", action="store_true", help="load the model and answer 4 ScienceQA items")
    args = ap.parse_args()

    ensure_yaml()
    sys.path.insert(0, str(REPO_ROOT))
    from common.config import load_config  # sets HF_HOME from common.hf_cache_dir

    cfg = load_config(args.config)
    scfg = cfg.get("setup", {})

    check_gpu()
    if not args.skip_install:
        install(scfg.get("packages", []))
    report_versions()
    if not args.skip_model and scfg.get("download_model", True):
        download_model(cfg["common"]["model"]["name"])
    if not args.skip_data:
        download_datasets(cfg, scfg.get("download_datasets", []))
    (REPO_ROOT / cfg["common"]["output_root"]).mkdir(parents=True, exist_ok=True)
    if args.smoke_test:
        smoke_test(cfg)
    print("\nSetup complete.")


if __name__ == "__main__":
    main()
