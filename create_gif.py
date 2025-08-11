# create_gif.py
"""
A command-line utility to generate animated GIFs from trained NCA models.

This script loads a pre-trained .h5 weights file, simulates the model's growth
from a single seed, and saves the result as a GIF with crisp, pixel-perfect scaling.

Example Usage:
python create_gif.py --model_path demo/h5_models/lizard_regenerating_8000.weights.h5 --output_path media/lizard_growth.gif --steps 300
"""
import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

# --- Leverage existing application modules ---
from nca_globals import CHANNEL_N
from nca_model import CAModel
from nca_utils import to_rgb, np2pil

# --- MODIFICATION: Check for the modern Resampling attribute in Pillow ---
# This ensures compatibility with both old and new versions of the library.
if hasattr(Image, 'Resampling'):
    NEAREST_NEIGHBOR = Image.Resampling.NEAREST
else:
    NEAREST_NEIGHBOR = Image.NEAREST


def create_nca_gif(model_path, output_path, steps, size, zoom, duration):
    """Loads a model, runs the simulation, and saves a GIF."""
    print(f"Loading model from: {model_path}")
    try:
        ca = CAModel()
        ca.load_weights(model_path)
    except Exception as e:
        print(f"Error: Could not load model file at '{model_path}'.")
        print(f"Details: {e}")
        return

    seed = np.zeros([1, size, size, CHANNEL_N], np.float32)
    seed[:, size // 2, size // 2, 3:] = 1.0
    x = tf.convert_to_tensor(seed)

    frames = []
    print(f"Simulating {steps} steps to generate GIF frames...")

    for _ in tqdm(range(steps), desc="Generating Frames"):
        x = ca(x)
        rgb_frame = to_rgb(x)[0].numpy()

        # --- MODIFICATION FOR CRISP PIXELS ---
        # 1. Convert the numpy array to a small PIL image without any scaling.
        small_frame = np2pil(rgb_frame, zoom_factor=1)
        
        # 2. If zooming is needed, perform the resize using Nearest Neighbor scaling.
        if zoom > 1:
            large_frame = small_frame.resize(
                (size * zoom, size * zoom), 
                resample=NEAREST_NEIGHBOR
            )
            frames.append(large_frame)
        else:
            frames.append(small_frame)
        # --- END MODIFICATION ---

    print(f"Saving GIF to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    print("GIF created successfully!")


def main():
    parser = argparse.ArgumentParser(description="Generate GIFs from trained NCA models.")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the .h5 weights file of the trained model.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the output GIF file.')
    parser.add_argument('--steps', type=int, default=300,
                        help='Number of simulation steps (frames) in the GIF.')
    parser.add_argument('--size', type=int, default=96,
                        help='Grid size for the simulation (should match training, e.g., 96 for 64x64 padded models).')
    parser.add_argument('--zoom', type=int, default=4,
                        help='Integer zoom factor to apply to the final GIF.')
    parser.add_argument('--duration', type=int, default=50,
                        help='Duration of each frame in milliseconds (e.g., 50ms = 20 FPS).')
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPUs, memory growth enabled.")
        except RuntimeError as e:
            print(f"Error setting up GPU: {e}")

    create_nca_gif(
        model_path=args.model_path,
        output_path=args.output_path,
        steps=args.steps,
        size=args.size,
        zoom=args.zoom,
        duration=args.duration
    )

if __name__ == '__main__':
    main()