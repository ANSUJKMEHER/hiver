"""Config loading.

One yaml file (`config.yaml`) holds every tunable: brand, paths, model names,
sample sizes, grounding k, escalation rules. No constants scattered in code.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: Path | str | None = None) -> dict:
    """Load config.yaml and resolve repo-relative paths to absolute."""
    p = Path(path) if path else CONFIG_PATH
    cfg = yaml.safe_load(p.read_text())

    # Resolve repo-relative data paths against the repo root so scripts work
    # regardless of the current working directory.
    for key in ("raw_path", "processed_path", "golden_path", "cache_dir"):
        rel = cfg["data"].get(key)
        if rel and not os.path.isabs(rel):
            cfg["data"][key] = str(REPO_ROOT / rel)

    return cfg
