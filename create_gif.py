# create_gif.py
"""
A command-line utility to generate animated GIFs from trained NCA models.

This script can simulate simple growth or apply multiple, randomly-placed
damage events to demonstrate a model's regeneration capabilities.

Example Usage (Growth):
python create_gif.py --model_path demo/h5_models/lizard_growing_8000.weights.h5 --output_path media/lizard_growth.gif

Example Usage (Regeneration):
python create_gif.py --model_path demo/h5_models/lizard_regenerating_8000.weights.h5 --output_path media/lizard_regen.gif --steps 500 --num_damage 3 --damage_interval 100
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

# Check for the modern Resampling attribute in Pillow
if hasattr(Image, 'Resampling'):
    NEAREST_NEIGHBOR = Image.Resampling.NEAREST
else:
    NEAREST_NEIGHBOR = Image.NEAREST


def create_damage_mask(size, radius_factor, center):
    """Creates a circular damage mask at a specific center point."""
    y, x = np.ogrid[:size, :size]
    cy, cx = center
    radius = size * radius_factor
    mask = (x - cx)**2 + (y - cy)**2 > radius**2  # True for pixels outside the circle
    # Add batch and channel dimensions for broadcasting
    mask = mask[np.newaxis, ..., np.newaxis]
    return tf.cast(mask, tf.float32)


def create_nca_gif(model_path, output_path, steps, size, zoom, duration, damage_steps, damage_radius):
    """Loads a model, runs the simulation with optional damage, and saves a GIF."""
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

    # Convert to a set for efficient lookup
    damage_steps_set = set(damage_steps)

    frames = []
    print(f"Simulating {steps} steps to generate GIF frames...")

    for i in tqdm(range(steps), desc="Generating Frames"):
        # Apply damage if the current step is in our list
        if i in damage_steps_set:
            # --- NEW: Apply damage at a random off-center location ---
            # Define a central area to apply damage, avoiding the very edges.
            center_min = int(size * 0.25)
            center_max = int(size * 0.75)
            rand_cx = np.random.randint(center_min, center_max)
            rand_cy = np.random.randint(center_min, center_max)
            
            print(f"\nApplying damage at step {i}, centered at ({rand_cx}, {rand_cy})...")
            damage_mask = create_damage_mask(size, damage_radius, center=(rand_cx, rand_cy))
            x = x * damage_mask

        x = ca(x)

        rgb_frame = to_rgb(x)[0].numpy()
        small_frame = np2pil(rgb_frame, zoom_factor=1)
        
        if zoom > 1:
            large_frame = small_frame.resize(
                (size * zoom, size * zoom), 
                resample=NEAREST_NEIGHBOR
            )
            frames.append(large_frame)
        else:
            frames.append(small_frame)

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
                        help='Grid size for the simulation (e.g., 96 for 64x64 padded models).')
    parser.add_argument('--zoom', type=int, default=4,
                        help='Integer zoom factor to apply to the final GIF.')
    parser.add_argument('--duration', type=int, default=50,
                        help='Duration of each frame in ms (e.g., 50ms = 20 FPS).')
    
    # --- NEW ARGUMENTS FOR MULTIPLE DAMAGE EVENTS ---
    parser.add_argument('--num_damage', type=int, default=0,
                        help='Number of times to apply damage. Disabled by default.')
    parser.add_argument('--damage_interval', type=int, default=100,
                        help='Number of steps between each damage event.')
    parser.add_argument('--damage_radius', type=float, default=0.3,
                        help='Radius of the damage circle as a fraction of grid size.')
    
    args = parser.parse_args()

    # Calculate the list of steps where damage should be applied
    damage_steps_list = []
    if args.num_damage > 0:
        initial_growth_period = 100  # Wait for the pattern to form before first damage
        for i in range(args.num_damage):
            step = initial_growth_period + (i * args.damage_interval)
            if step < args.steps:
                damage_steps_list.append(step)
        print(f"Damage will be applied at steps: {damage_steps_list}")

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
        duration=args.duration,
        damage_steps=damage_steps_list,
        damage_radius=args.damage_radius
    )

if __name__ == '__main__':
    main()