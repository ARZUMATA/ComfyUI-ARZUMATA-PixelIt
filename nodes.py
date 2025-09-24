from .pixelit import Pixelit as pxlt
import torch

# Import combined palette list
from .palettes import PALETTE_LIST, PALETTE_NAMES

class PixelIt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "downscale_ratio": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "target_block_size_resize": ("BOOLEAN", {"default": False}),
                "target_block_size": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "auto_collapse_blocks": ("BOOLEAN", {"default": False}),
                "compensate_collapse": ("BOOLEAN", {"default": False}),
                "block_detection_tolerance": ("FLOAT", {
                    "default": 15.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "detect_jpg": ("BOOLEAN", {"default": True},),
                "grid": ("BOOLEAN", {"default": False}),
                "grid_alpha": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "reduce_colors": ("BOOLEAN", {"default": False}),
                "colors": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1}),
                "convert_grayscale": ("BOOLEAN", {"default": False}),
                "convert_palette": ("BOOLEAN", {"default": False}),
                "palette_preset": (["none"] + PALETTE_NAMES, {"default": "none"},),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "pixelit"
    CATEGORY = "pixelit"

    def pixelit(self, 
                image: torch.Tensor,
                downscale_ratio: int,
                target_block_size_resize: bool,
                target_block_size: int,
                auto_collapse_blocks: bool,
                compensate_collapse: bool,
                block_detection_tolerance: float,
                detect_jpg: bool,
                grid: bool,
                grid_alpha: int,
                reduce_colors: bool,
                colors: int,
                convert_grayscale: bool,
                convert_palette: bool,
                palette_preset: str,
                ):
        
        # Build config
        config = {
            'downscale_ratio': downscale_ratio,
            'target_block_size_resize': target_block_size_resize,
            'target_block_size': target_block_size,
            'auto_collapse_blocks': auto_collapse_blocks,
            'compensate_collapse': compensate_collapse,
            'block_detection_tolerance': block_detection_tolerance,
            'detect_jpg': detect_jpg,
            'grid': grid,
            'grid_alpha': grid_alpha,
            'reduce_colors': reduce_colors,
            'colors': colors,
            'convert_grayscale': convert_grayscale,
            'convert_palette': convert_palette,
            'palette_preset': palette_preset,
        }
        
        # Add palette if selected
        if palette_preset != "none" and convert_palette:
            preset_index = PALETTE_NAMES.index(palette_preset)
            config['palette'] = PALETTE_LIST[preset_index]  # Use preset colors
            # config['reduce_colors'] = True  # Enforce color reduction
            # config['colors'] = len(config['palette'])  # Use exact palette size

        # Initialize the pixelator with config 
        px = pxlt(config)

        # Process each image in the batch
        results = []
        for img_tensor in image:
            # Add batch dim: (H, W, C) -> (1, H, W, C)
            img_tensor = img_tensor.unsqueeze(0)
            # Pixelate (assumes pixelate_as_tensor returns (1, H, W, C) in [0,1])
            result = px.pixelate_as_tensor(img_tensor)
            results.append(result.squeeze(0))  # Remove batch dim

        output = torch.stack(results, dim=0)
        return (output,)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "PixelIt": PixelIt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelIt": "PixelIt 🎨",
}