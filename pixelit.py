import torch
from PIL import Image
import numpy as np
from PIL import ImageDraw, ImageOps
from PIL import ImageFilter

# Import combined palette list
from .palettes import PALETTE_LIST, PALETTE_NAMES

class Pixelit:
    def __init__(self, config=None):
        if config is None:
            config = {}

        # Input image
        self.image_input = config.get('image_input', None)

        # Output image
        self.image_output = config.get('image_output', None)

        # Loaded image
        self.image_loaded = None

        # Target image
        self.image_target = None
        
        # Do grayscale conversion
        self.convert_grayscale = config.get('convert_grayscale', False)

        # Do palette conversion
        self.convert_palette = config.get('convert_palette', False)

        # Reduce colors
        self.reduce_colors = config.get('reduce_colors', False)

        # Colors amount
        self.colors = config.get('colors', None)

        # Add downscale ratio for image to bne 1/n size
        self.downscale_ratio = config.get('downscale_ratio', 8)

        # Rezize image to match target block size
        self.target_block_size_resize = config.get('target_block_size_resize', False)

        # Add target_block_size parameter
        self.target_block_size = config.get('target_block_size', 8)

        # Draw grid
        self.grid = config.get('grid', False)

        # Grid alpha
        self.grid_alpha = config.get('grid_alpha', 255)

        # If block_size is provided, calculate scale based on image dimensions
        if self.downscale_ratio:
            # Scale will be set when image is loaded
            self.scale = None

        # Max width and height
        self.max_height = config.get('maxHeight')
        self.max_width = config.get('maxWidth')

        self.currentPalette = 0

        # Store color stats
        self.end_color_stats = {}

        #  Collapse blocks to single pixel
        self.auto_collapse_blocks = config.get('auto_collapse_blocks', False)

        # Compensate downscale
        self.compensate_collapse = config.get('compensate_collapse', False)

        # Block detection sensitivity
        self.block_detection_tolerance = config.get('block_detection_tolerance', 15.0)
        self.block_detection_tolerance = max(0.0, min(50.0, self.block_detection_tolerance))  # Clamp 0–50

        # Handle palette preset if provided and conversion is enabled
        palette_preset = config.get('palette_preset', None)
        if self.convert_palette and palette_preset:
            self.set_palette_by_name(palette_preset)
        
        # If we want to detect JPG and increase tolerance due to artifacts
        self.detect_jpg = config.get('detect_jpg', False)

    def set_palette_index(self, index):
        """Set the active palette by index."""
        if 0 <= index < len(PALETTE_LIST):
            self.currentPalette = index
        else:
            raise ValueError(f"Palette index must be between 0 and {len(PALETTE_LIST) - 1}")
        return self
    
    def set_palette_by_name(self, name):
        """Set the active palette by its name."""
        try:
            index = PALETTE_NAMES.index(name)
            self.currentPalette = index
            return self
        except ValueError:
            raise ValueError(f"Palette '{name}' not found. Available: {PALETTE_NAMES}")
        
    def set_scale(self):
        """Set scale using downscale_ratio."""
        self.scale = 1.0 / max(1, self.downscale_ratio)
        return self

    def convert_to_grayscale(self):
        # Convert to grayscale
        self.image_target = self.image_target.convert("L")

    def set_color_amount(self):
        # Reduce color amount
        self.image_target = self.image_target.convert('RGB')  # Convert to RGB first
        self.image_target = self.image_target.quantize(
            colors=self.colors,
            method=Image.Quantize.MEDIANCUT,
                # MEDIANCUT = 0
                # MAXCOVERAGE = 1 
                # FASTOCTREE = 2
                # LIBIMAGEQUANT = 3
            kmeans=1,
            dither=Image.Dither.FLOYDSTEINBERG)
        self.image_target = self.image_target.convert('RGB')  # Convert back to RGB after quantization

    def scale_with_block_size(self):
        """
        Resize the image by turning each pixel into an NxN block using `downscale_ratio`.
        
        Using PyTorch GPU acceleration for massive speedup:
        - Converts image to tensor
        - Expands each pixel to (N, N) block using repeat_interleave
        - Keeps color integrity and avoids blending
        - Supports CPU fallback if GPU not available

        Uses `self.downscale_ratio` as the block size (e.g. 8 → each pixel = 8x8)
        Only applied if `self.target_block_size_resize` is True.
        
        Returns self for method chaining.
        """

        if not self.target_block_size_resize:
            return self

        # Get original image
        img = self.image_target
        width, height = img.size
        block_size = self.downscale_ratio  # Each pixel becomes NxN

        # Update block size if needed
        if self.target_block_size != self.downscale_ratio:
            self.downscale_ratio = self.target_block_size
            block_size = self.target_block_size

        # Convert to tensor: (H, W, C) -> (C, H, W), normalized to [0,1]
        img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_tensor = img_tensor.to(device)

        # Expand: repeat pixels along H and W using block_size
        # (C, H, W) -> (C, H, 1, W, 1) -> (C, H, N, W, N) -> (C, H*N, W*N)
        expanded = img_tensor.unsqueeze(2).unsqueeze(4)  # Add dummy dims
        expanded = expanded.repeat(1, 1, block_size, 1, block_size)  # Repeat blocks
        output_tensor = expanded.reshape(3, height * block_size, width * block_size)

        # Move back to CPU and convert to PIL
        output_np = output_tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        self.image_target = Image.fromarray(output_np, 'RGB')

        return self

    def pixelate_as_tensor(self, image_tensor):
        self.load_image_as_tensor(image_tensor)
        self.pixelate()
        return self.image_to_tensor()

    def pixelate_as_rgb(self):
        self.load_image(self.image_input)
        self.pixelate()
        self.save_image(self.image_output)

    def pixelate(self):
        self.collapse_to_pixel_grid()

        # Now apply standard scale
        self.set_scale()  # uses downscale_ratio to shrink
        self.image_target = self.image_target.resize(
            (int(self.image_target.width * self.scale), int(self.image_target.height * self.scale)),
            Image.Resampling.NEAREST
        )

        if self.reduce_colors and self.colors:
            self.set_color_amount()

        if self.convert_grayscale:
            self.convert_to_grayscale()

        if self.convert_palette:
            self.convert_to_palette()

        # Upscale using block size
        if self.target_block_size_resize:
            self.scale_with_block_size()

        if self.grid:
            self.draw_grid()

        return self

    def load_image_as_tensor(self, image_tensor):
        # Convert tensor from B,H,W,C to RGB PIL Image
        # Assuming first batch only
        image_array = image_tensor[0].numpy()

        # Ensure [0,1]
        image_array = np.clip(image_tensor[0].numpy(), 0, 1)
        
        # Convert to uint8 if needed and ensure RGB
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Create PIL image from numpy array
        self.image_loaded = Image.fromarray(image_array, 'RGB')
        self.image_target = self.image_loaded.copy()
        return self

    def image_to_tensor(self):
        # Assign the target image to 'i'
        i = self.image_target 

        # Apply EXIF orientation correction to 'i'
        i = ImageOps.exif_transpose(i)

        # Convert the image to RGB mode
        image = i.convert("RGB")

        # Convert the image array to float32 and normalize it by dividing by 255.0
        image = np.array(image).astype(np.float32) / 255.0

        # Create a PyTorch tensor from the numpy array and add an extra dimension at the beginning
        image = torch.from_numpy(image)[None,]
        return image

    def load_image(self, image_path):
        self.image_loaded = Image.open(image_path)
        self.image_target = self.image_loaded.copy()
        return self

    def save_image(self, output_path="pixelart.png"):
        self.image_target.convert('RGB').save(output_path)
        return self

    def draw_grid(self, color=(0, 0, 0), rgb=True):
        # Get image dimensions
        width, height = self.image_target.size
        
        # Use block_size directly for grid cells
        cell_width = self.downscale_ratio
        cell_height = self.downscale_ratio

        # If alpha is provided and color is in RGB, convert it to RGBA
        if not rgb or (rgb and self.grid_alpha != 255):
            color = tuple(list(color) + [self.grid_alpha])
        
        # Create a new image with same size for grid overlay
        grid_overlay = Image.new('RGBA', self.image_target.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(grid_overlay)
        
        # Draw vertical lines
        for x in range(0, width, cell_width):
            draw.line([(x, 0), (x, height)], fill=color, width=1)
        
        # Draw horizontal lines
        for y in range(0, height, cell_height):
            draw.line([(0, y), (width, y)], fill=color, width=1)
        
        # Composite the grid over the pixelated image
        result = Image.alpha_composite(self.image_target.convert('RGBA'), grid_overlay)
        
        # Convert back to RGB if specified
        if rgb:
            result = result.convert('RGB')
        
        self.image_target = result
        return self

    def convert_to_palette(self):
        # Convert image to numpy array for efficient processing
        img_array = np.array(self.image_target)
        height, width, _ = img_array.shape

        # Process each pixel
        for y in range(height):
            for x in range(width):
                # Get RGB values of current pixel
                # The [:3] slice notation in current_color = img_array[y, x][:3] specifically extracts the first 3 values from the pixel data array, which represent the RGB (Red, Green, Blue) color channels.
                # For example, if a pixel has values [255, 128, 0, 255] (orange with full opacity), [:3] would give us [255, 128, 0], which is just the RGB part we need for color matching.
                current_color = img_array[y, x][:3]
                
                # # Find the closest color in palette
                final_color = self.similar_color(current_color)
                
                # # Update pixel with new color
                img_array[y, x] = final_color

        # Convert back to PIL Image
        self.image_target = Image.fromarray(img_array)
        return self

    def similar_color(self, actual_color):
        # Initialize 'selected_color' with the first color in the current palette
        selected_color = PALETTE_LIST[self.currentPalette][0]

        # Calculate the similarity between 'actual_color' and the first color in the palette
        current_sim = self.color_sim(actual_color, PALETTE_LIST[self.currentPalette][0])
        
        # Iterate over each color in the current palette
        for color in PALETTE_LIST[self.currentPalette]:
            # Calculate the similarity between 'actual_color' and the next color in the palette
            next_color = self.color_sim(actual_color, color)

            # If the new similarity is less than or equal to the current similarity,
            if next_color <= current_sim:
                # Update 'selected_color' to the next color and update 'current_sim'
                selected_color = color
                current_sim = next_color
                
        # Return the color in the palette that is most similar to 'actual_color'
        return selected_color

    def color_sim(self, rgb_color, compare_color):
        # Compute Euclidean distance using safe numeric type
        diff = np.array(rgb_color, dtype=np.float64) - np.array(compare_color, dtype=np.float64)
        return np.sqrt(np.sum(diff ** 2))

    def collapse_to_pixel_grid(self):
        """
        Collapses uniform pixel blocks (e.g. 8x8) into single pixels to recover 
        the original low-resolution pixel art grid.
        
        Uses GPU-accelerated tensor operations for speed.
        Only applied if `auto_collapse_blocks` is True.
        """
        if not self.collapse_to_pixel_grid or not self.target_block_size_resize:
            return self

        # Try to detect block size and downscale image to 1x1 px block size
        if self.collapse_to_pixel_grid:
            # Auto-detect JPG compression
            if self.detect_jpg and self.is_likely_jpg():
                print("Detected: Likely JPG-compressed image")
                
                # Auto-adjust tolerance if not user-set
                self.block_detection_tolerance = max(self.block_detection_tolerance, 18.0)
                print(f"Increasing block_detection_tolerance to {self.block_detection_tolerance} for noise tolerance")

                # Apply light denoising
                self.image_target = self.image_target.filter(ImageFilter.MedianFilter(3))
                print("Applied light denoising to reduce JPG artifacts")

        # Detect block size (CPU - lightweight)
        detected_size = self.detect_block_size()
        print(f"\nAuto-detected block size: {detected_size}")

        if detected_size > 1 and self.compensate_collapse:
            self.target_block_size_resize = True
            self.target_block_size *= detected_size

        width, height = self.image_target.size

        # Ensure dimensions are divisible by detected_size
        if width % detected_size != 0 or height % detected_size != 0:
            print(f"Image size not divisible by {detected_size}, cropping...")
            new_w = (width // detected_size) * detected_size
            new_h = (height // detected_size) * detected_size
            self.image_target = self.image_target.crop((0, 0, new_w, new_h))
            width, height = new_w, new_h

        # Convert to tensor: (H, W, C) -> (C, H, W)
        np_img = np.array(self.image_target, dtype=np.float32)  # [H, W, C]
        tensor = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()  # [C, H, W]

        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor = tensor.to(device)

        # Reshape into blocks: (C, H, W) -> (C, H//B, B, W//B, B)
        b = detected_size
        c, h, w = tensor.shape
        small_tensor = tensor.view(c, h // b, b, w // b, b)

        # Take top-left pixel of each block -> (C, H//B, W//B)
        # Equivalent to nearest downsample: use [0, 0] of each block
        downsampled = small_tensor[:, :, 0, :, 0]  # (C, h//b, w//b)

        # Move back to CPU and convert to PIL
        result_np = downsampled.byte().permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        self.image_target = Image.fromarray(result_np, 'RGB')

        return self

    def detect_block_size(self, max_block: int = 64, min_block: int = 1, debug: bool = False) -> int:
        """
        Hybrid block size detection:
        - Tests common block sizes (8,16,32,...) in priority order.
        - Probes multiple small regions across image.
        - Scores each size by how many regions pass.
        - Avoids large blocks unless they are truly dominant.
        - Uses `self.block_detection_tolerance` for color variation threshold.
        """
        
        if self.image_target is None:
            raise ValueError("No image loaded.")

        img = self.image_target.convert("RGB")
        width, height = img.size
        img_array = np.array(img, dtype=np.float64)

        tolerance = self.block_detection_tolerance
        candidates = {}

        # Prioritize realistic pixel art block sizes
        candidate_sizes = [s for s in [8, 16, 32, 4, 24, 48, 12, 2, 1, 64]
                        if min_block <= s <= min(max_block, width, height)]
        
        # Remove duplicates and sort by preference, not size
        seen = set()
        ordered_sizes = []
        for s in candidate_sizes:
            if s not in seen and s <= min(width, height):
                ordered_sizes.append(s)
                seen.add(s)

        # Sample points: avoid only corners — use grid
        def make_grid_samples(grid=3, margin=10):
            dx = max(1, (width - 2 * margin) // (grid - 1)) if grid > 1 else 0
            dy = max(1, (height - 2 * margin) // (grid - 1)) if grid > 1 else 0
            points = []
            for i in range(grid):
                for j in range(grid):
                    x = margin + i * dx
                    y = margin + j * dy
                    if x < width and y < height:
                        points.append((x, y))
            return points

        sample_points = make_grid_samples(5)  # 5x5 grid = 25 points
        if debug:
            print(f"Testing {len(ordered_sizes)} sizes at {len(sample_points)} grid points")

        # Test each block size
        for size in ordered_sizes:
            valid_blocks = 0
            total_tests = 0

            for x, y in sample_points:
                # Align to block grid
                gx = (x // size) * size
                gy = (y // size) * size
                x_end, y_end = gx + size, gy + size

                if x_end > width or y_end > height:
                    continue

                # Extract block and test uniformity
                block = img_array[gy:y_end, gx:gx + size].reshape(-1, 3)
                base_color = block[0]
                distances = np.sqrt(np.sum((block - base_color) ** 2, axis=1))
                if np.max(distances) <= tolerance:
                    valid_blocks += 1
                total_tests += 1

            if total_tests > 0:
                score = valid_blocks / total_tests
                if score > 0.5:  # At least 50% of blocks are solid
                    candidates[size] = score
                    if debug:
                        print(f"Size {size:2d}: {valid_blocks}/{total_tests} valid ({score:.1%})")

        # Choose best candidate
        if candidates:
            # Prefer smaller sizes if scores are close (avoid over-pixelation)
            best_size = min(candidates.keys(), key=lambda s: (abs(s - 16), s))  # Bias toward 16
            if debug:
                print(f"Detected block size: {best_size} (scored {candidates[best_size]:.2f})")
            return best_size

        if debug:
            print("No valid block size found. Falling back to 1")
        return 1
    
    def is_likely_jpg(self, max_blocks: int = 64, edge_threshold: int = 20) -> bool:
        """
        Detects if image is likely JPEG-compressed by checking for:
        - 8x8 blocking artifacts (sudden color changes on 8-pixel grid)
        - Higher edge energy on 8-grid vs random offsets
        
        Returns True if JPG-like blocking is detected.
        """
        if self.image_target is None:
            return False

        img = self.image_target.convert("L")  # Grayscale for speed
        width, height = img.size
        img_array = np.array(img, dtype=np.float32)

        if width < 16 or height < 16:
            return False

        block_size = 8
        energy_on_grid = 0.0
        energy_off_grid = 0.0
        sample_count = 0

        # Sample vertical and horizontal edges on and off 8x8 grid
        for _ in range(max_blocks):
            # Random block-aligned position
            x = np.random.randint(1, width - 2)
            y = np.random.randint(1, height - 2)

            # Compute horizontal and vertical gradients (edge strength)
            dx = abs(img_array[y, x+1] - img_array[y, x-1]) / 2.0
            dy = abs(img_array[y+1, x] - img_array[y-1, x]) / 2.0
            edge_energy = dx + dy

            # Classify: on-grid vs off-grid
            on_grid = (x % block_size == 0) or (y % block_size == 0)

            if on_grid:
                energy_on_grid += edge_energy
            else:
                energy_off_grid += edge_energy

            sample_count += 1

        # Avoid divide by zero
        energy_off_grid = energy_off_grid or 1e-6

        # If grid-aligned edges are significantly stronger → likely JPG
        ratio = energy_on_grid / energy_off_grid
        return ratio > 1.5  # Tuned: 1.5+ suggests structured blocking