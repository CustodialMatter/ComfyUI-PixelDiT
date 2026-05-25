"""Resolution presets matching PixelDiT's training aspect-ratio buckets."""

from __future__ import annotations

# Mirrors diffusion.data.datasets.utils.ASPECT_RATIO_1024_TEST (16px-aligned).
# Keep this list in sync with upstream if they add buckets.
PRESETS_1024 = {
    "1:1  (1024x1024)": (1024, 1024),
    "4:3  (1152x896)":  (896, 1152),
    "3:4  (896x1152)":  (1152, 896),
    "3:2  (1216x832)":  (832, 1216),
    "2:3  (832x1216)":  (1216, 832),
    "16:9 (1344x768)":  (768, 1344),
    "9:16 (768x1344)":  (1344, 768),
    "21:9 (1536x640)":  (640, 1536),
}

PRESETS_512 = {
    "1:1  (512x512)": (512, 512),
    "4:3  (576x448)": (448, 576),
    "3:4  (448x576)": (576, 448),
    "16:9 (672x384)": (384, 672),
    "9:16 (384x672)": (672, 384),
}


class PixelDiTResolutionPicker:
    """Pick a (width, height) pair matching PixelDiT's trained aspect buckets."""

    @classmethod
    def INPUT_TYPES(cls):
        keys = list(PRESETS_1024.keys()) + list(PRESETS_512.keys())
        return {
            "required": {
                "preset": (keys, {"default": "1:1  (1024x1024)"}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "pick"
    CATEGORY = "PixelDiT/utils"

    def pick(self, preset: str):
        if preset in PRESETS_1024:
            h, w = PRESETS_1024[preset]
        elif preset in PRESETS_512:
            h, w = PRESETS_512[preset]
        else:
            h, w = 1024, 1024
        return (int(w), int(h))


NODE_CLASS_MAPPINGS = {"PixelDiTResolutionPicker": PixelDiTResolutionPicker}
NODE_DISPLAY_NAME_MAPPINGS = {"PixelDiTResolutionPicker": "PixelDiT — Resolution Picker"}
