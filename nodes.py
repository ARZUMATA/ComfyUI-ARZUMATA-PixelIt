from .pixelit import Pixelit as pxlt
class PixelIt:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": 
                    {
                        "image": ("IMAGE",),
                        "convert_grayscale": ("BOOLEAN", {"default": False}),
                        "convert_palette": ("BOOLEAN", {"default": False}),
                        "reduce_colors": ("BOOLEAN", {"default": False}),
                        "colors": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1}),
                        "block_size": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                        "target_block_size_resize": ("BOOLEAN", {"default": False}),
                        "target_block_size": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                        "grid": ("BOOLEAN", {"default": False}),
                        "grid_alpha": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                    },
                # "optional": {
                #     "palette": ("BOOLEAN", {"default": False}),
                # }
                }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "pixelit"
    CATEGORY = "pixelit"

    def pixelit(self, 
        image,
        convert_grayscale,
        convert_palette,
        reduce_colors,
        colors,
        block_size,
        target_block_size_resize,
        target_block_size,
        grid,
        grid_alpha,
        ):
        px = pxlt({
            'image_input': '.\\test2.png',
            'image_output': '.\output.png',
            'convert_grayscale': convert_grayscale,
            'convert_palette': convert_palette,
            'reduce_colors': reduce_colors,
            'colors': colors,
            'block_size': block_size,
            'target_block_size_resize': target_block_size_resize,
            'target_block_size': target_block_size,
            'grid': grid,
            'grid_alpha': grid_alpha,
        })

        image = px.pixelate_as_tensor(image)

        return (image,)



NODE_CLASS_MAPPINGS = {
    "PixelIt": PixelIt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelIt": "PixelIt node",
}