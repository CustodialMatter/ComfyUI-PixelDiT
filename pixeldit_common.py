"""Shared helpers for ComfyUI-PixelDiT nodes.

Handles sys.path injection so the vendored upstream PixelDiT repo can be
imported, plus device/dtype helpers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

NODE_DIR = Path(__file__).resolve().parent
UPSTREAM_DIR = NODE_DIR / "upstream"
UPSTREAM_T2I_DIR = UPSTREAM_DIR / "t2i"

PINNED_COMMIT = "07015dd233bab8a057ac6edc7b67711da5ba896e"
UPSTREAM_URL = "https://github.com/NVlabs/PixelDiT.git"


def _require_upstream() -> None:
    if not UPSTREAM_T2I_DIR.is_dir():
        raise RuntimeError(
            f"PixelDiT upstream not found at {UPSTREAM_DIR}. "
            f"Run `python install.py` inside {NODE_DIR} or clone {UPSTREAM_URL} "
            f"(commit {PINNED_COMMIT}) into 'upstream/'."
        )


def ensure_sys_path() -> None:
    """Put upstream repo on sys.path. inference.py does the same two insertions."""
    _require_upstream()
    for p in (str(UPSTREAM_T2I_DIR), str(UPSTREAM_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def torch_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"unknown dtype: {name}")


def comfy_image_from_samples(samples: torch.Tensor) -> torch.Tensor:
    """PixelDiT outputs (B, 3, H, W) in [-1, 1]. ComfyUI IMAGE is (B, H, W, 3) in [0, 1]."""
    x = samples.detach().to(torch.float32)
    # NaN -> 0, +Inf/-Inf -> clamped edges. Without this, a single NaN anywhere
    # survives clamp + normalize and shows as a fully black PNG after uint8 cast.
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    x = x.clamp(-1.0, 1.0)
    x = (x + 1.0) * 0.5
    return x.permute(0, 2, 3, 1).contiguous().cpu()


def find_config(config_name: str) -> Path:
    """Resolve a config YAML by name. Accepts bare names or subpaths; searches t2i/configs/."""
    _require_upstream()
    configs_dir = UPSTREAM_T2I_DIR / "configs"
    candidate = Path(config_name)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    for p in (configs_dir / config_name, configs_dir / f"{config_name}.yaml"):
        if p.is_file():
            return p
    raise FileNotFoundError(f"Config '{config_name}' not found under {configs_dir}")


def list_configs() -> list[str]:
    if not UPSTREAM_T2I_DIR.is_dir():
        return []
    configs_dir = UPSTREAM_T2I_DIR / "configs"
    if not configs_dir.is_dir():
        return []
    return sorted(p.name for p in configs_dir.iterdir() if p.suffix in (".yaml", ".yml"))
