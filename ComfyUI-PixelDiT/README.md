# ComfyUI-PixelDiT

ComfyUI wrapper for [NVlabs/PixelDiT](https://github.com/NVlabs/PixelDiT) — NVIDIA's pixel-space Diffusion Transformer (CVPR 2026 Oral). **No VAE** — the model emits pixels directly, so sampler output lands straight in an `IMAGE`.

## ⚠️ License warning (read first)

The upstream PixelDiT model weights are released under NVIDIA's **Non-Commercial Source Code License (NSCLv1)** — *"research or evaluation purposes only"*. Any images you generate with it, and any derivative work, inherit that restriction. **Do not use PixelDiT outputs in commercial projects, client work, or products you sell.** This wrapper's own code is MIT, but that doesn't change the weights' license.

## What's included

| Node | Purpose |
|------|---------|
| `PixelDiT — Load Model` | Build the backbone from a config YAML and load a `.pth` checkpoint. |
| `PixelDiT — Load Text Encoder (Gemma-2)` | Load Gemma-2-2B-IT tokenizer + decoder. |
| `PixelDiT — Encode Text` | Encode positive + negative prompts (with optional CHI system prefix from the config). |
| `PixelDiT — Sampler (Flow DPM-Solver)` | Pixel-space sampling with classifier-free guidance. Output → `IMAGE`. |
| `PixelDiT — Resolution Picker` | Helper for the training-time aspect-ratio buckets (1024 / 512). |

## Install

From `ComfyUI/custom_nodes/ComfyUI-PixelDiT/`:

```bash
..\..\..\python_embeded\python.exe install.py
```

This clones upstream at a pinned commit into `./upstream/` and pip-installs the runtime deps (`omegaconf`, `diffusers`, `transformers`, `accelerate`, `timm`, `torchvision`, `termcolor`, `pyrallis`) into the embedded Python.

### Download weights

1. **PixelDiT checkpoint** → `ComfyUI/models/diffusion_models/pixeldit_t2i_v1.pth`
   ```
   https://huggingface.co/nvidia/PixelDiT-1300M-1024px/resolve/main/pixeldit_t2i_v1.pth
   ```
   (~5 GB)

2. **Text encoder (Gemma-2-2B-IT)** — you have two options:
   - **Auto-download on first run** (default): keep the `source` widget set to `Efficient-Large-Model/gemma-2-2b-it` and Transformers will pull it into your HF cache (~5 GB in bf16).
   - **Local snapshot** (recommended for offline setups): put a full HF snapshot at `ComfyUI/models/text_encoders/gemma-2-2b-it/` (must contain `config.json`, `tokenizer.json`, `*.safetensors`, etc.). The loader auto-detects subdirs matching `*gemma*` and adds them to the dropdown.

Restart ComfyUI after install.

## Usage

Load `workflows/pixeldit_t2i_1024.json` — it wires up the full pipeline. Key widgets:

- **config_name**: `PixelDiT_1024px_pixel_diffusion_stage3.yaml` for 1024px t2i. The 512px stage configs also work if you're experimenting.
- **precision**: `bf16` is the trained dtype; `fp16` may underflow attention, `fp32` doubles VRAM.
- **steps**: 50 is the default; 25 is noticeably faster with a modest quality hit. Don't go below ~15.
- **cfg_scale**: 2.75 (sample) and 3.5 (default) are both sensible. Higher values sharpen/over-saturate more than with latent diffusion because there's no VAE smoothing.
- **flow_shift_override**: `-1` = use the config's value (4.0 for the 1024px stage). Raising it pushes more sampling weight toward the noisier end of the trajectory.
- **interval_guidance**: `[0.0, 1.0]` = apply CFG over the full trajectory. Narrow this to restrict CFG to a time window — useful to tame saturation.
- **order**: `2` is the recommended DPM-Solver order. `3` is slightly sharper but can diverge at low step counts.

## Notes & gotchas

- **Pixel-space VRAM**: 1024×1024 pixel diffusion is significantly heavier than latent diffusion at the same param count. Expect noticeably slower per-step times than Flux on the same GPU. Drop to a 512px config if you hit OOM.
- **CHI prompt**: the Stage 3 config ships with a long system prefix that asks Gemma to "enhance" the user prompt inline before encoding. The `use_chi_prompt` toggle lets you turn it off if you want raw prompt-as-written behaviour.
- **No ControlNet / LoRA / IPAdapter** — the upstream architecture doesn't have adapters yet. If these matter to you, stick with Flux/SD3.
- **Gemma tokenizer padding** is forced to right-padding to match upstream; don't override it elsewhere in the graph.
- **Checkpoints search order** looks in `models/diffusion_models/` first, then `models/checkpoints/`.

## Updating the vendored upstream

Pinned commit is in `pixeldit_common.py` (`PINNED_COMMIT`). To update:

```bash
cd upstream
git fetch origin && git checkout <new-commit>
```

Then bump `PINNED_COMMIT` to match so future installs stay reproducible.

## License

- Wrapper code (everything under this directory except `upstream/`): MIT.
- Vendored upstream (`upstream/`) and the `.pth` weights: NVIDIA NSCLv1 — **non-commercial only**.
