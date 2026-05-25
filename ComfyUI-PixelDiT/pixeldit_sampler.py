"""Sampler node for PixelDiT.

Wraps upstream `DPMS` (flow DPM-Solver) and runs classifier-free guided
sampling in *pixel space* — so the output goes straight to a ComfyUI IMAGE,
no VAE decode.
"""

from __future__ import annotations

import torch

from .pixeldit_common import comfy_image_from_samples, ensure_sys_path


class PixelDiTSampler:
    """Run PixelDiT DPM-Solver sampling in pixel space."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixeldit_model": ("PIXELDIT_MODEL",),
                "conditioning": ("PIXELDIT_COND",),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "cfg_scale": ("FLOAT", {"default": 2.75, "min": 0.0, "max": 30.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "flow_shift_override": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1,
                                                    "tooltip": "-1 = use value from config. DO NOT set to 0: collapses the schedule and produces NaN."}),
                "interval_guidance_low": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "interval_guidance_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "order": ("INT", {"default": 2, "min": 1, "max": 3,
                                   "tooltip": "DPM-Solver order (2 recommended)"}),
                "debug_nan_trace": ("BOOLEAN", {"default": False,
                                                  "tooltip": "Attach forward hooks to bisect where NaN first appears"}),
                "attention_backend": (["auto", "math", "mem_efficient", "flash"], {"default": "auto",
                                                  "tooltip": "SDPA backend. 'math' forces fp32-capable path (slowest, most stable)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "PixelDiT/sampling"

    @torch.inference_mode()
    def sample(
        self,
        pixeldit_model,
        conditioning,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        seed: int,
        flow_shift_override: float,
        interval_guidance_low: float,
        interval_guidance_high: float,
        order: int,
        debug_nan_trace: bool = False,
        attention_backend: str = "auto",
    ):
        ensure_sys_path()
        from diffusion import DPMS  # type: ignore

        model = pixeldit_model["model"]
        device = pixeldit_model["device"]
        dtype = pixeldit_model["dtype"]
        # Guard against flow_shift=0 (including tiny values that underflow the
        # schedule's (shift*s)/(1+(shift-1)*s) transformation). shift<~0.01
        # collapses every sigma to 0 -> lambda=-inf -> NaN across all steps.
        if flow_shift_override < 0:
            flow_shift = pixeldit_model["flow_shift"]
        elif flow_shift_override < 0.01:
            print(f"[PixelDiT] flow_shift_override={flow_shift_override} would collapse schedule; using config value instead")
            flow_shift = pixeldit_model["flow_shift"]
        else:
            flow_shift = float(flow_shift_override)

        caption_embs = conditioning["caption_embs"]
        emb_masks = conditioning["emb_masks"]
        null_caption_embs = conditioning["null_caption_embs"]

        # Ensure conditioning tensors are on model device / dtype. The encoder may
        # have produced them in bf16 already, but double-check.
        caption_embs = caption_embs.to(device=device, dtype=dtype)
        emb_masks = emb_masks.to(device=device)
        null_caption_embs = null_caption_embs.to(device=device, dtype=dtype)

        # PixelDiT expects per-sample hw + aspect ratio for position embedding scaling.
        hw = torch.tensor([[float(height), float(width)]], dtype=torch.float, device=device)
        ar = (hw[:, 0] / hw[:, 1]).unsqueeze(1)
        null_y = null_caption_embs.repeat(caption_embs.shape[0], 1, 1)[:, None]

        # pixel-space noise — keep fp32 to match upstream. Casting to bf16 here
        # underflows the DPM-Solver's incremental updates and produces NaN/black.
        generator = torch.Generator(device=device).manual_seed(int(seed))
        z = torch.randn(
            caption_embs.shape[0], 3, height, width,
            device=device, dtype=torch.float32, generator=generator,
        )

        interval = [max(0.0, min(1.0, interval_guidance_low)),
                    max(0.0, min(1.0, interval_guidance_high))]
        if interval[0] > interval[1]:
            interval = [interval[1], interval[0]]

        model_kwargs = dict(
            data_info={"img_hw": hw, "aspect_ratio": ar},
            mask=emb_masks,
        )

        def _stats(name: str, t: torch.Tensor) -> None:
            f = t.detach().to(torch.float32)
            print(
                f"[PixelDiT][probe] {name}: shape={tuple(t.shape)} dtype={t.dtype} "
                f"min={f.min().item():+.4f} max={f.max().item():+.4f} "
                f"mean={f.mean().item():+.4f} std={f.std().item():.4f} "
                f"nan={torch.isnan(f).any().item()} inf={torch.isinf(f).any().item()}"
            )

        _stats("z (noise)", z)
        _stats("caption_embs", caption_embs)
        _stats("null_y", null_y)
        _stats("emb_masks", emb_masks.to(torch.float32))

        # One-shot forward BEFORE the solver runs, to isolate whether the model
        # itself produces NaN or whether the DPM-Solver arithmetic blows up.
        # Use t=999 (matches time_uniform_flow first step: t_continuous=0.999, then *1000).
        with torch.no_grad():
            t_probe = torch.tensor([999.0], device=device, dtype=torch.float32)
            try:
                out_probe = model.forward_with_dpmsolver(
                    z.to(dtype), t_probe, caption_embs, **model_kwargs,
                )
                _stats("raw_model_forward @ t=999 (cond)", out_probe)
            except Exception as e:
                print(f"[PixelDiT][probe] raw_model_forward raised: {type(e).__name__}: {e}")

        hook_handles: list = []
        if debug_nan_trace:
            import torch.nn as nn
            first_bad = {"name": None}
            def make_hook(name: str):
                def _hook(module, inputs, output):
                    if first_bad["name"] is not None:
                        return
                    tensors = output if isinstance(output, (list, tuple)) else (output,)
                    for o in tensors:
                        if isinstance(o, torch.Tensor) and (torch.isnan(o).any() or torch.isinf(o).any()):
                            first_bad["name"] = name
                            f = o.detach().to(torch.float32)
                            print(
                                f"[PixelDiT][NaN-trace] FIRST bad output in '{name}' "
                                f"({type(module).__name__}): shape={tuple(o.shape)} "
                                f"nan={torch.isnan(f).any().item()} inf={torch.isinf(f).any().item()} "
                                f"range=[{f[~torch.isnan(f) & ~torch.isinf(f)].min().item() if (~torch.isnan(f) & ~torch.isinf(f)).any() else 'all-nan'}, ...]"
                            )
                            return
                return _hook
            for mod_name, module in model.named_modules():
                if len(list(module.children())) == 0:  # leaf modules only
                    hook_handles.append(module.register_forward_hook(make_hook(mod_name)))
            print(f"[PixelDiT][NaN-trace] attached {len(hook_handles)} forward hooks")

        # Per-call tracer around the model: logs t + in/out NaN on every
        # forward inside the DPM solver. This pinpoints the *step* at which
        # NaN enters — the solver arithmetic is fine in isolation, so if the
        # input x becomes NaN, it was produced by the previous update step.
        call_counter = {"n": 0}
        def traced_forward(x, timestep, y, mask=None, **kwargs):
            i = call_counter["n"]
            call_counter["n"] += 1
            x_nan = torch.isnan(x).any().item()
            x_inf = torch.isinf(x).any().item()
            out = model.forward_with_dpmsolver(x, timestep, y, mask=mask, **kwargs)
            o_nan = torch.isnan(out).any().item()
            o_inf = torch.isinf(out).any().item()
            # Only log the first ~6 calls and any call that flips clean->NaN,
            # to avoid flooding the console at 50 steps * 2 branches.
            should_log = i < 6 or (not x_nan and o_nan)
            if should_log:
                t_val = float(timestep.flatten()[0].item())
                xf = x.detach().to(torch.float32)
                of = out.detach().to(torch.float32)
                print(
                    f"[PixelDiT][trace] call#{i:03d} t={t_val:.4f} "
                    f"x(nan={x_nan},inf={x_inf},max={xf.abs().max().item():.2e}) "
                    f"out(nan={o_nan},inf={o_inf},max={of.abs().max().item() if not o_nan else float('nan'):.2e})"
                )
            return out

        dpm_solver = DPMS(
            traced_forward,
            condition=caption_embs,
            uncondition=null_y,
            guidance_type="classifier-free",
            cfg_scale=cfg_scale,
            model_type="flow",
            model_kwargs=model_kwargs,
            schedule="FLOW",
            interval_guidance=interval,
        )

        # SDPA backend selector. upstream's `fp32_attention` config flag is dead
        # code; the attention call just uses whatever tensor dtype arrives. 'math'
        # forces the reference kernel which promotes to fp32 internally and is
        # our fallback when bf16 flash/mem-efficient kernels produce NaN.
        from torch.nn.attention import SDPBackend, sdpa_kernel
        backend_map = {
            "auto": [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                     SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION],
            "math": [SDPBackend.MATH],
            "mem_efficient": [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
            "flash": [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH],
        }
        backends = backend_map[attention_backend]
        print(f"[PixelDiT] SDPA backend selection: {attention_backend} -> {[b.name for b in backends]}")

        try:
            with sdpa_kernel(backends):
                samples = dpm_solver.sample(
                    z,
                    steps=int(steps),
                    order=int(order),
                    skip_type="time_uniform_flow",
                    method="multistep",
                    flow_shift=flow_shift,
                )
        finally:
            for h in hook_handles:
                h.remove()

        # Diagnostic probe: a one-liner summary of the sampler output. If this
        # says nan=True or the range is degenerate, something in conditioning
        # or the forward pass is collapsing before you see the image.
        s32 = samples.detach().to(torch.float32)
        print(
            f"[PixelDiT] samples shape={tuple(samples.shape)} dtype={samples.dtype} "
            f"min={s32.min().item():.4f} max={s32.max().item():.4f} "
            f"mean={s32.mean().item():.4f} std={s32.std().item():.4f} "
            f"nan={torch.isnan(s32).any().item()} inf={torch.isinf(s32).any().item()}"
        )

        torch.cuda.empty_cache()
        return (comfy_image_from_samples(samples),)


NODE_CLASS_MAPPINGS = {"PixelDiTSampler": PixelDiTSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"PixelDiTSampler": "PixelDiT — Sampler (Flow DPM-Solver)"}
