"""ComfyUI-PixelDiT — pixel-space Diffusion Transformer wrapper.

Registers loader, text encoder, encoder, sampler, and resolution-picker nodes.
See README.md for install and download instructions.
"""

from .pixeldit_loader import (
    NODE_CLASS_MAPPINGS as _LOADER_MAP,
    NODE_DISPLAY_NAME_MAPPINGS as _LOADER_NAMES,
)
from .pixeldit_text_encoder import (
    NODE_CLASS_MAPPINGS as _TE_MAP,
    NODE_DISPLAY_NAME_MAPPINGS as _TE_NAMES,
)
from .pixeldit_sampler import (
    NODE_CLASS_MAPPINGS as _SAMP_MAP,
    NODE_DISPLAY_NAME_MAPPINGS as _SAMP_NAMES,
)
from .pixeldit_resolution import (
    NODE_CLASS_MAPPINGS as _RES_MAP,
    NODE_DISPLAY_NAME_MAPPINGS as _RES_NAMES,
)

NODE_CLASS_MAPPINGS = {**_LOADER_MAP, **_TE_MAP, **_SAMP_MAP, **_RES_MAP}
NODE_DISPLAY_NAME_MAPPINGS = {**_LOADER_NAMES, **_TE_NAMES, **_SAMP_NAMES, **_RES_NAMES}

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
