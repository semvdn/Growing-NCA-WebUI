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
    """
    if a is None:
        return PIL.Image.new('RGB', (zoom_factor, zoom_factor), color='lightgrey')
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1) * 255)
    img = PIL.Image.fromarray(a)
    if zoom_factor > 1 and img.width > 0 and img.height > 0 :
        try:
            img = img.resize((img.width * zoom_factor, img.height * zoom_factor), RESAMPLING_METHOD)
        except ValueError:
            return PIL.Image.new('RGB', (zoom_factor, zoom_factor), color='lightgrey')
    return img

def load_image_from_url(url, max_size=64):
    """
    Load an image from a URL, resize it, pad to square, and return as float32.
    """
    r = requests.get(url)
    r.raise_for_status() 
    img = PIL.Image.open(io.BytesIO(r.content)).convert("RGBA")

    img.thumbnail((max_size, max_size), RESAMPLING_METHOD)

    square_img = PIL.Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    upper_left = ((max_size - img.width) // 2, (max_size - img.height) // 2)
    square_img.paste(img, upper_left)

    img_np = np.float32(square_img) / 255.0
    img_np[..., :3] *= img_np[..., 3:] 
    return img_np

def load_image_from_file(file_stream, target_dim=64):
    """
    (Legacy) Load an image from a file stream and resize it to stretch/shrink to the target dimensions.
    """
    img = PIL.Image.open(file_stream).convert("RGBA")
    img_resized = img.resize((target_dim, target_dim), RESAMPLING_METHOD)
    
    img_np = np.float32(img_resized) / 255.0
    img_np[..., :3] *= img_np[..., 3:]
    return img_np

def load_and_pad_image_from_file(file_stream, target_size, final_grid_dim):
    """
    Loads an image, resizes it to fit within target_size while maintaining aspect ratio,
    and then pads it to the center of a final_grid_dim square.
    """
    img = PIL.Image.open(file_stream).convert("RGBA")

    # Resize to fit within the TARGET_SIZE box
    img.thumbnail((target_size, target_size), RESAMPLING_METHOD)

    # Create a new transparent canvas for the full padded grid
    padded_img = PIL.Image.new("RGBA", (final_grid_dim, final_grid_dim), (0, 0, 0, 0))
    
    # Paste the resized image into the center
    paste_pos = ((final_grid_dim - img.width) // 2, (final_grid_dim - img.height) // 2)
    padded_img.paste(img, paste_pos)

    # Convert to numpy and premultiply alpha
    img_np = np.float32(padded_img) / 255.0
    img_np[..., :3] *= img_np[..., 3:]
    return img_np

def to_alpha(x):
    return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)

def to_rgb(x):
    rgb, a = x[..., :3], to_alpha(x)
    return 1.0 - a + rgb 

def get_living_mask(x_input):
    x = tf.convert_to_tensor(x_input, dtype=tf.float32)
    alpha = x[..., 3:4]
    is_single_image = (len(alpha.shape) == 3)
    if is_single_image:
        alpha = tf.expand_dims(alpha, 0)
    
    pool_out = tf.nn.max_pool2d(alpha, ksize=3, strides=1, padding='SAME')
    living_mask = pool_out > 0.1

    if is_single_image:
        living_mask = tf.squeeze(living_mask, axis=0)
    return living_mask

def make_circle_masks(n, h, w):
    x_coords = tf.linspace(-1.0, 1.0, w)[None, None, :]
    y_coords = tf.linspace(-1.0, 1.0, h)[None, :, None]
    center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
    r = tf.random.uniform([n, 1, 1], 0.1, 0.4)
    x_rel = (x_coords - center[0]) / r 
    y_rel = (y_coords - center[1]) / r
    mask = tf.cast(x_rel * x_rel + y_rel * y_rel < 1.0, tf.float32)
    return mask

class SamplePool:
    def __init__(self, **slots):
        self._slot_names = list(slots.keys())
        self._size = None
        for k, v_arr in slots.items():
            v_arr = np.asarray(v_arr)
            if self._size is None: self._size = len(v_arr)
            assert self._size == len(v_arr)
            setattr(self, k, v_arr)

    def sample(self, n_samples):
        idx = np.random.choice(self._size, n_samples, replace=(n_samples > self._size))
        return SamplePool(**{k: getattr(self, k)[idx] for k in self._slot_names})

    def get_indices(self, n_samples):
        return np.random.choice(self._size, n_samples, replace=(n_samples > self._size))

    def __len__(self):
        return self._size

class VideoWriter:
    def __init__(self, filename, duration=0.05):
        self.frames = []
        self.filename = filename
        self.duration = duration
    def add(self, img_data):
        self.frames.append(np2pil(img_data) if isinstance(img_data, np.ndarray) else img_data)
    def close(self):
        if self.frames:
            self.frames[0].save(self.filename, save_all=True, append_images=self.frames[1:], duration=int(self.duration * 1000), loop=0)
    def __enter__(self): return self
    def __exit__(self, *kw): self.close()

def get_model_summary(model_instance):
    if model_instance is None: return "Model not available."
    summary_list = []
    try:
        model_instance.summary(print_fn=lambda x: summary_list.append(x))
        return "\n".join(summary_list)
    except Exception as e:
        return f"Failed to retrieve model summary: {str(e)}"

def format_training_time(time_seconds):
    if time_seconds is None: return "N/A"
    minutes, seconds = divmod(time_seconds, 60)
    return f"{int(minutes)}m {int(seconds)}s"