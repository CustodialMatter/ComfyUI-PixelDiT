"""Text-encoder nodes for PixelDiT.

PixelDiT uses Gemma-2-2B-IT (decoder-only) as its text encoder. The upstream
checkpoint is HF `Efficient-Large-Model/gemma-2-2b-it`. Two nodes:

  * `PixelDiTLoadTextEncoder` — loads tokenizer + the Gemma decoder to the
    requested device.
  * `PixelDiTEncodeText` — builds conditioning + unconditional embeddings,
    applying the `chi_prompt` system-prefix trick from the config if present.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

import folder_paths  # provided by ComfyUI

from .pixeldit_common import device as get_device, ensure_sys_path

_DEFAULT_HF_REPO = "Efficient-Large-Model/gemma-2-2b-it"


def _local_gemma_candidates() -> list[str]:
    """Subdirs under models/text_encoders/ that look like a Gemma HF snapshot."""
    roots: list[Path] = []
    try:
        for r in folder_paths.get_folder_paths("text_encoders"):
            roots.append(Path(r))
    except Exception:
        pass
    # common fallback
    models_root = Path(folder_paths.models_dir) if hasattr(folder_paths, "models_dir") else None
    if models_root is not None:
        roots.append(models_root / "text_encoders")

    seen: set[str] = set()
    out: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for sub in root.iterdir():
            if not sub.is_dir():
                continue
            if "gemma" not in sub.name.lower():
                continue
            if (sub / "config.json").is_file():
                rel = sub.name
                if rel not in seen:
                    seen.add(rel)
                    out.append(rel)
    return sorted(out)


def _resolve_text_encoder_path(choice: str) -> str:
    """If a local snapshot dir is picked, return its absolute path; else return the HF repo id."""
    if not choice or choice == _DEFAULT_HF_REPO:
        return _DEFAULT_HF_REPO
    try:
        for r in folder_paths.get_folder_paths("text_encoders"):
            p = Path(r) / choice
            if (p / "config.json").is_file():
                return str(p)
    except Exception:
        pass
    # last resort — treat as an HF repo id
    return choice


class PixelDiTLoadTextEncoder:
    """Load Gemma-2-2B-IT tokenizer + decoder for PixelDiT conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        locals_ = _local_gemma_candidates()
        options = [_DEFAULT_HF_REPO] + locals_
        return {
            "required": {
                "source": (options, {"default": _DEFAULT_HF_REPO}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "device_mode": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("PIXELDIT_TEXT_ENCODER",)
    RETURN_NAMES = ("text_encoder",)
    FUNCTION = "load"
    CATEGORY = "PixelDiT/loaders"

    def load(self, source: str, precision: str, device_mode: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import logging as transformers_logging

        transformers_logging.set_verbosity_error()

        resolved = _resolve_text_encoder_path(source)
        if device_mode == "auto":
            device = get_device()
        else:
            device = device_mode

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map[precision]

        tokenizer = AutoTokenizer.from_pretrained(resolved)
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(resolved, torch_dtype=dtype)
        decoder = model.get_decoder().to(device)
        decoder.eval()

        return ({
            "tokenizer": tokenizer,
            "encoder": decoder,
            "device": device,
            "dtype": dtype,
            "source": resolved,
        },)


class PixelDiTEncodeText:
    """Encode positive and negative prompts into PixelDiT conditioning embeddings."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("PIXELDIT_TEXT_ENCODER",),
                "pixeldit_model": ("PIXELDIT_MODEL",),
                "positive": ("STRING", {"multiline": True, "default": "A cute artist cat painting the phrase 'PixelDiT is awesome' on a bright canvas."}),
                "negative": ("STRING", {"multiline": True, "default": "low quality, worst quality, over-saturated, blurry, deformed, watermark"}),
                "use_chi_prompt": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("PIXELDIT_COND",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "PixelDiT/conditioning"

    @torch.inference_mode()
    def encode(self, text_encoder, pixeldit_model, positive: str, negative: str, use_chi_prompt: bool):
        ensure_sys_path()
        tokenizer = text_encoder["tokenizer"]
        encoder = text_encoder["encoder"]
        device = text_encoder["device"]

        config = pixeldit_model["config"]
        model_max_length = int(config.text_encoder.model_max_length)
        chi_prompt_cfg = getattr(config.text_encoder, "chi_prompt", None)

        if use_chi_prompt and chi_prompt_cfg:
            chi_prompt = "\n".join(list(chi_prompt_cfg))
            prompt_full = chi_prompt + positive
            num_chi_tokens = len(tokenizer.encode(chi_prompt))
            max_length_all = num_chi_tokens + model_max_length - 2  # magic 2: [bos], [_]
        else:
            prompt_full = positive
            max_length_all = model_max_length

        pos_tok = tokenizer(
            prompt_full,
            max_length=max_length_all,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        select_index = [0] + list(range(-model_max_length + 1, 0))
        pos_hidden = encoder(pos_tok.input_ids, pos_tok.attention_mask)[0][:, None]
        caption_embs = pos_hidden[:, :, select_index]
        emb_masks = pos_tok.attention_mask[:, select_index]

        neg_tok = tokenizer(
            negative or "",
            max_length=model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        null_caption_embs = encoder(neg_tok.input_ids, neg_tok.attention_mask)[0]

        return ({
            "caption_embs": caption_embs,
            "emb_masks": emb_masks,
            "null_caption_embs": null_caption_embs,
            "model_max_length": model_max_length,
        },)


NODE_CLASS_MAPPINGS = {
    "PixelDiTLoadTextEncoder": PixelDiTLoadTextEncoder,
    "PixelDiTEncodeText": PixelDiTEncodeText,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelDiTLoadTextEncoder": "PixelDiT — Load Text Encoder (Gemma-2)",
    "PixelDiTEncodeText": "PixelDiT — Encode Text",
}
