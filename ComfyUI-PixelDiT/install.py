r"""One-shot installer for ComfyUI-PixelDiT.

Usage (from inside ComfyUI-PixelDiT/):
    ..\..\..\python_embeded\python.exe install.py

Steps:
  1. Clone NVlabs/PixelDiT into ./upstream at a pinned commit (idempotent).
  2. Install only the *missing* runtime deps, and never touch torch/torchvision/
     torchaudio — ComfyUI's embedded Python ships a specific CUDA torch build
     and letting pip re-resolve those swaps it for a CPU wheel.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

PINNED_COMMIT = "07015dd233bab8a057ac6edc7b67711da5ba896e"
UPSTREAM_URL = "https://github.com/NVlabs/PixelDiT.git"

# Import name -> pip spec. We use import names so we can detect "already
# installed" without pip resolver overhead.
RUNTIME_DEPS: dict[str, str] = {
    "omegaconf": "omegaconf",
    "pyrallis": "pyrallis",
    "timm": "timm",
    "termcolor": "termcolor",
    # NOT listed (already in ComfyUI, or dangerous to touch):
    #   torch / torchvision / torchaudio — CUDA build would be overwritten
    #   diffusers / transformers / accelerate — already ComfyUI deps
}

# Safety net: if any pip call ever *does* transitively pull torch, route
# through the cu130 index so it picks a CUDA wheel, not a CPU one.
CU_EXTRA_INDEX = "https://download.pytorch.org/whl/cu130"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[install] $ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def clone_upstream(node_dir: Path) -> None:
    upstream = node_dir / "upstream"
    if upstream.is_dir() and (upstream / ".git").is_dir():
        print(f"[install] upstream already present at {upstream} — checking pinned commit")
        run(["git", "fetch", "--depth", "1", "origin", PINNED_COMMIT], cwd=upstream)
        run(["git", "checkout", PINNED_COMMIT], cwd=upstream)
        return
    run(["git", "clone", UPSTREAM_URL, str(upstream)])
    run(["git", "checkout", PINNED_COMMIT], cwd=upstream)


def install_missing_deps() -> None:
    missing: list[str] = []
    for import_name, pip_spec in RUNTIME_DEPS.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(pip_spec)
    if not missing:
        print("[install] all runtime deps already satisfied — not invoking pip")
        return
    print(f"[install] installing missing deps: {missing}")
    run([
        sys.executable, "-m", "pip", "install",
        "--extra-index-url", CU_EXTRA_INDEX,
        *missing,
    ])


def main() -> None:
    node_dir = Path(__file__).resolve().parent
    clone_upstream(node_dir)
    install_missing_deps()
    print("[install] done. Next steps:")
    print("  1. Download pixeldit_t2i_v1.pth -> ComfyUI/models/diffusion_models/")
    print("     https://huggingface.co/nvidia/PixelDiT-1300M-1024px/resolve/main/pixeldit_t2i_v1.pth")
    print("  2. (Optional) snapshot Gemma-2-2B-IT -> ComfyUI/models/text_encoders/gemma-2-2b-it/")
    print("     HF repo: Efficient-Large-Model/gemma-2-2b-it")
    print("  3. Restart ComfyUI.")


if __name__ == "__main__":
    main()
