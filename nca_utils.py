# nca_utils.py
"""Utility functions for Neural Cellular Automata."""

import numpy as np
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import requests
import io
import tensorflow as tf
from PIL import Image

# Check for Resampling attribute in PIL.Image for compatibility
if hasattr(Image, 'Resampling'):
    RESAMPLING_METHOD = Image.Resampling.LANCZOS
else:
    RESAMPLING_METHOD = Image.ANTIALIAS

def np2pil(a, zoom_factor=1):
    """
    Convert a NumPy array (float32 or uint8) to a PIL Image.

    Args:
        a (np.ndarray): Image array in [H, W, C] format.
                       If dtype is float, values are assumed to be in [0.0, 1.0].
        zoom_factor (int): Factor to resize the image by.

    Returns:
        PIL.Image: The converted PIL image.
    """
    if a is None: # Handle None input gracefully
        return PIL.Image.new('RGB', (zoom_factor, zoom_factor), color='lightgrey') # Default tiny placeholder
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1) * 255)
    img = PIL.Image.fromarray(a)
    if zoom_factor > 1 and img.width > 0 and img.height > 0 :
        try:
            img = img.resize((img.width * zoom_factor, img.height * zoom_factor), RESAMPLING_METHOD)
        except ValueError: # Handle cases like zero-sized image after processing
            return PIL.Image.new('RGB', (zoom_factor, zoom_factor), color='lightgrey')
    return img

def load_image_from_url(url, max_size=64):
    """
    Load an image from a URL, resize it, pad to square, and return as float32.

    Args:
        url (str): The URL pointing to the image file.
        max_size (int): The maximum width/height for the resized image.

    Returns:
        np.ndarray: RGBA image array of shape [H, W, 4] in float32.
    """
    r = requests.get(url)
    r.raise_for_status() 
    img = PIL.Image.open(io.BytesIO(r.content)).convert("RGBA")

    original_size = img.size
    if original_size[0] == 0 or original_size[1] == 0:
        raise ValueError("Loaded image has zero dimension.")
    ratio = min(max_size / original_size[0], max_size / original_size[1])
    new_size = (max(1, int(original_size[0] * ratio)), max(1, int(original_size[1] * ratio))) # Ensure not zero
    img = img.resize(new_size, RESAMPLING_METHOD)

    square_img = PIL.Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    upper_left = ((max_size - new_size[0]) // 2, (max_size - new_size[1]) // 2)
    square_img.paste(img, upper_left)

    img_np = np.float32(square_img) / 255.0
    img_np[..., :3] *= img_np[..., 3:] 
    return img_np

def load_image_from_file(file_stream, target_dim=64):
    """
    Load an image from a file stream and resize it to the target_dim x target_dim.
    The image will be stretched/shrunk to fit the target dimensions, no padding.
    Returns as float32 RGBA.
    """
    img = PIL.Image.open(file_stream).convert("RGBA")

    original_size = img.size
    if original_size[0] == 0 or original_size[1] == 0:
        raise ValueError("Loaded image has zero dimension.")
    
    # Resize directly to target_dim x target_dim, stretching if aspect ratio differs
    img_resized = img.resize((target_dim, target_dim), RESAMPLING_METHOD)
    
    img_np = np.float32(img_resized) / 255.0
    img_np[..., :3] *= img_np[..., 3:]
    return img_np


def load_emoji(emoji, max_size=64):
    code = hex(ord(emoji))[2:].lower()
    url = (f"https://github.com/googlefonts/noto-emoji/blob/main/png/128/"
           f"emoji_u{code}.png?raw=true")
    return load_image_from_url(url, max_size)

def to_alpha(x):
    return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)

def to_rgb(x):
    rgb, a = x[..., :3], to_alpha(x)
    return 1.0 - a + rgb 

def get_living_mask(x_input):
    """
    Determine which cells are "alive" by checking alpha > 0.1
    in a 3x3 max-pooling sense. Handles both batched and single images.
    """
    # Ensure x is a TensorFlow tensor
    x = tf.convert_to_tensor(x_input, dtype=tf.float32)
    
    # Extract alpha channel, keeping rank for pooling
    alpha = x[..., 3:4] # Shape: [B, H, W, 1] or [H, W, 1]

    # Add batch dimension if it's a single image for max_pool2d
    is_single_image = (len(alpha.shape) == 3)
    if is_single_image:
        alpha = tf.expand_dims(alpha, 0) # Shape: [1, H, W, 1]
    
    # Max pooling expects input of rank 4
    if len(alpha.shape) != 4:
        # This case should ideally not be hit if logic above is correct
        # Fallback or error for unexpected shape
        tf.print("Warning: get_living_mask received unexpected alpha shape:", alpha.shape)
        # Attempt to reshape or return a default mask
        # For simplicity, return a mask of all False if shape is wrong
        return tf.zeros_like(alpha, dtype=tf.bool) if not is_single_image else tf.zeros_like(x_input[..., 3:4], dtype=tf.bool)


    pool_out = tf.nn.max_pool2d(alpha, ksize=3, strides=1, padding='SAME') # Shape: [B, H, W, 1]
    
    living_mask = pool_out > 0.1 # Shape: [B, H, W, 1]

    # Remove batch dimension if it was added
    if is_single_image:
        living_mask = tf.squeeze(living_mask, axis=0) # Shape: [H, W, 1]
        
    return living_mask


def make_circle_masks(n, h, w):
    x_coords = tf.linspace(-1.0, 1.0, w)[None, None, :]
    y_coords = tf.linspace(-1.0, 1.0, h)[None, :, None]
    center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
    r = tf.random.uniform([n, 1, 1], 0.1, 0.4)
    # Adjusted broadcasting for x and y relative to center and r
    x_rel = (x_coords - center[0]) / r 
    y_rel = (y_coords - center[1]) / r
    mask = tf.cast(x_rel * x_rel + y_rel * y_rel < 1.0, tf.float32)
    return mask # Shape [n, h, w]

class SamplePool:
    def __init__(self, **slots):
        self._slot_names = list(slots.keys())
        self._size = None
        for k, v_arr in slots.items():
            v_arr = np.asarray(v_arr)
            if self._size is None:
                self._size = len(v_arr)
            assert self._size == len(v_arr), f"Slots mismatch: {k} has {len(v_arr)}, expected {self._size}"
            setattr(self, k, v_arr)

    def sample(self, n_samples):
        if self._size == 0:
            raise ValueError("Cannot sample from an empty pool.")
        
        use_replacement = n_samples > self._size
        chosen_indices = np.random.choice(self._size, n_samples, replace=use_replacement)
        
        batch_data = {k: getattr(self, k)[chosen_indices] for k in self._slot_names}
        return SamplePool(**batch_data)

    def get_indices(self, n_samples):
        if self._size == 0:
            return np.array([], dtype=int)
        use_replacement = n_samples > self._size
        return np.random.choice(self._size, n_samples, replace=use_replacement)

    def __len__(self):
        return self._size


class VideoWriter:
    def __init__(self, filename, duration=0.05):
        self.frames = []
        self.filename = filename
        self.duration = duration

    def add(self, img_data):
        if isinstance(img_data, np.ndarray):
            pil_img = np2pil(img_data)
        elif isinstance(img_data, Image.Image):
            pil_img = img_data
        else:
            raise TypeError("VideoWriter.add expects NumPy array or PIL Image.")
        self.frames.append(pil_img)

    def close(self):
        if self.frames:
            self.frames[0].save(
                self.filename,
                save_all=True,
                append_images=self.frames[1:],
                duration=int(self.duration * 1000),
                loop=0
            )
    def __enter__(self):
        return self
    def __exit__(self, *kw):
        self.close()

def get_model_summary(model_instance):
    if model_instance is None:
        return "Model not available."
    summary_list = []
    try:
        model_instance.summary(print_fn=lambda x: summary_list.append(x))
        return "\n".join(summary_list)
    except Exception as e_detail:
        return f"Failed to retrieve model summary: {str(e_detail)}"


def format_training_time(time_seconds):
    if time_seconds is None:
        return "N/A"
    minutes_val, seconds_val = divmod(time_seconds, 60)
    return f"{int(minutes_val)}m {int(seconds_val)}s"