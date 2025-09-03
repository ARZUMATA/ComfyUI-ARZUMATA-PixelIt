from .pixelit import Pixelit as pxlt
from .pixelit import PALETTE_NAMES, PALETTE_LIST
import torch

class PixelIt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "block_size": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "target_block_size_resize": ("BOOLEAN", {"default": False}),
                "target_block_size": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "grid": ("BOOLEAN", {"default": False}),
                "grid_alpha": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "reduce_colors": ("BOOLEAN", {"default": False}),
                "colors": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1}),
                "palette_preset": (["none"] + PALETTE_NAMES, {"default": "none"},),
                "convert_grayscale": ("BOOLEAN", {"default": False}),
                "convert_palette": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "pixelit"
    CATEGORY = "pixelit"

    def pixelit(self, 
                image: torch.Tensor,
                convert_grayscale: bool,
                convert_palette: bool,
                reduce_colors: bool,
                colors: int,
                palette_preset: str,
                block_size: int,
                target_block_size_resize: bool,
                target_block_size: int,
                grid: bool,
                grid_alpha: int):
        
        # Build config
        config = {
            'convert_grayscale': convert_grayscale,
            'convert_palette': convert_palette,
            'reduce_colors': reduce_colors,
            'colors': colors,
            'block_size': block_size,
            'target_block_size_resize': target_block_size_resize,
            'target_block_size': target_block_size,
            'grid': grid,
            'grid_alpha': grid_alpha,
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