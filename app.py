from pixelit import Pixelit

px = Pixelit({
    'image_input': '.\\0001.jpg',
    'image_output': '.\output.png',
    'downscale_ratio': 2,
    'target_block_size_resize': True,
    'target_block_size': 1,
    'auto_collapse_blocks': True,
    'compensate_collapse': True,
    'block_detection_tolerance': 1,
    'detect_jpg': True,
    'grid': False,
    'grid_alpha': 255,
    'reduce_colors': False,
    'colors': 32,
    'convert_grayscale': False,
    'convert_palette': False,
    'palette_preset': 'none'
})

px.pixelate_as_rgb()
# px.image_to_tensor()