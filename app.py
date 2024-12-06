from pixelit import Pixelit

px = Pixelit({
    'image_input': '.\\test.png',
    'image_output': '.\output.png',
    'convert_grayscale': False,
    'convert_palette': False,
    'reduce_colors': False,
    'colors': 32,
    'block_size': 24,
    'target_block_size_resize': False,
    'target_block_size': 24,
    'grid': True,
    'grid_alpha': 255,
})

px.pixelate_as_rgb()
px.image_to_tensor()