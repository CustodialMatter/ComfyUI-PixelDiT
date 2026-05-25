"""Model loader node for PixelDiT.

Reads a YAML config from the vendored upstream repo, builds the PixelDiT
backbone via `build_model`, and loads a `.pth` checkpoint from
`ComfyUI/models/diffusion_models/`.

Output type: `PIXELDIT_MODEL` — a dict bundling the model, config, and dtype.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

import folder_paths  # provided by ComfyUI

from .pixeldit_common import (
    device as get_device,
    ensure_sys_path,
    find_config,
    list_configs,
    torch_dtype,
)


def _list_checkpoints() -> list[str]:
    names: list[str] = []
    for folder in ("diffusion_models", "checkpoints"):
        try:
            names.extend(folder_paths.get_filename_list(folder))
        except Exception:
            pass
    seen: set[str] = set()
    result: list[str] = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        if n.lower().endswith((".pth", ".safetensors", ".bin", ".ckpt")):
            result.append(n)
    return sorted(result) or ["(place pixeldit_t2i_v1.pth in models/diffusion_models/)"]


def _resolve_checkpoint_path(name: str) -> str:
    for folder in ("diffusion_models", "checkpoints"):
        try:
            p = folder_paths.get_full_path(folder, name)
            if p and os.path.isfile(p):
                return p
        except Exception:
            pass
    raise FileNotFoundError(
        f"Checkpoint '{name}' not found under ComfyUI/models/diffusion_models/ or /checkpoints/."
    )


class PixelDiTLoader:
    """Load a PixelDiT model from a config YAML + checkpoint."""

    @classmethod
    def INPUT_TYPES(cls):
        configs = list_configs() or ["PixelDiT_1024px_pixel_diffusion_stage3.yaml"]
        return {
            "required": {
                "config_name": (configs,),
                "ckpt_name": (_list_checkpoints(),),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("PIXELDIT_MODEL",)
    RETURN_NAMES = ("pixeldit_model",)
    FUNCTION = "load"
    CATEGORY = "PixelDiT/loaders"

    def load(self, config_name: str, ckpt_name: str, precision: str):
        ensure_sys_path()
        from dataclasses import asdict
        from omegaconf import OmegaConf  # type: ignore
        from diffusion.model.builder import build_model  # type: ignore
        from diffusion.utils.config import PixDiTConfig, model_init_config  # type: ignore

        config_path = find_config(config_name)
        ckpt_path = _resolve_checkpoint_path(ckpt_name)
        dtype = torch_dtype(precision)
        device = get_device()

        # Three-way dance to dodge two bugs:
        #   1. OmegaConf.structured(PixDiTConfig()) chokes on the Union[Dict, str]
        #      on ModelConfig.resume_from.
        #   2. pyrallis.parse() opens the YAML with the locale encoding (cp1252
        #      on Windows), which blows up on the UTF-8 smart quotes in the
        #      chi_prompt block.
        # asdict() materialises the defaults as a pure nested dict (the Union
        # field's default_factory returns a dict, so no Union remains), and
        # OmegaConf.load/merge handles UTF-8 natively.
        defaults = OmegaConf.create(asdict(PixDiTConfig()))
        yaml_cfg = OmegaConf.load(str(config_path))
        config = OmegaConf.merge(defaults, yaml_cfg)

        latent_size = int(config.model.image_size)
        init_kwargs = model_init_config(config, latent_size=latent_size)
        model = build_model(
            config.model.model,
            use_fp32_attention=bool(config.model.get("fp32_attention", False)),
            **init_kwargs,
        ).to(device)

        state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        if ckpt_path.endswith(".bin") and "state_dict" not in state:
            state = {"state_dict": state}
        sd = state.get("state_dict", state)
        sd.pop("pos_embed", None)

        # Auto-strip `module.` / `model.` prefixes if the checkpoint was saved
        # through DDP or a wrapper that the current model class doesn't use.
        expected = set(model.state_dict().keys())
        sample_ckpt_key = next(iter(sd))
        if sample_ckpt_key not in expected:
            for prefix in ("module.", "model.", "ema."):
                if all(k.startswith(prefix) for k in sd):
                    stripped = {k[len(prefix):]: v for k, v in sd.items()}
                    if any(k in expected for k in stripped):
                        print(f"[PixelDiT] stripping '{prefix}' prefix from state dict keys")
                        sd = stripped
                        break

        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(sd.keys())
        overlap = len(model_keys & ckpt_keys)
        print(
            f"[PixelDiT] state_dict: model expects {len(model_keys)} keys, "
            f"checkpoint has {len(ckpt_keys)} keys, {overlap} overlap"
        )

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[PixelDiT] missing keys: {len(missing)} (first 5: {missing[:5]})")
        if unexpected:
            print(f"[PixelDiT] unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

        if missing and len(missing) >= len(model_keys) * 0.9:
            raise RuntimeError(
                f"[PixelDiT] {len(missing)}/{len(model_keys)} keys missing after load — "
                f"checkpoint format likely doesn't match this model. Sample checkpoint key: "
                f"'{next(iter(ckpt_keys))}'. Sample expected key: '{next(iter(model_keys))}'"
            )

        # Sanity-check that weights are not all zero (indicates failed load or
        # a param that didn't land). We pick a deep layer to avoid embeddings.
        with torch.no_grad():
            params = list(model.named_parameters())
            for name, p in params:
                if p.dim() >= 2 and p.numel() > 1024:
                    print(
                        f"[PixelDiT] weight probe [{name}]: shape={tuple(p.shape)} "
                        f"mean={p.float().mean().item():+.5f} std={p.float().std().item():.5f} "
                        f"nonzero={p.float().abs().gt(1e-8).float().mean().item():.3f}"
                    )
                    break

        model.eval().to(dtype)

        bundle = {
            "model": model,
            "config": config,
            "dtype": dtype,
            "device": device,
            "latent_size": latent_size,
            "flow_shift": float(config.scheduler.flow_shift),
        }
        return (bundle,)


NODE_CLASS_MAPPINGS = {"PixelDiTLoader": PixelDiTLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"PixelDiTLoader": "PixelDiT — Load Model"}
