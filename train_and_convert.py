# train_and_convert.py
"""
Trains and converts Neural CA models for the WebGL demo by leveraging the
existing application modules (NCATrainer, model_converter, etc.).

This script takes image file paths as input to define the training targets.

Example Usage:
python train_and_convert.py --steps 8000 --image_paths path/to/lizard.png
"""
import os
import json
import argparse
import numpy as np
import tensorflow as tf

# --- Leverage existing application modules ---
from nca_globals import (TARGET_SIZE, TARGET_PADDING, DEFAULT_FIRE_RATE,
                         DEFAULT_BATCH_SIZE, DEFAULT_POOL_SIZE)
from nca_utils import load_image_from_file
from nca_trainer import NCATrainer
from model_converter import export_to_webgl_format

def get_model_name_from_path(file_path):
    """Generates a clean model name from a file path."""
    return os.path.splitext(os.path.basename(file_path))[0]

def train_and_convert(image_path, regimen, total_steps, checkpoint_steps, output_dir):
    """
    Handles the full training and conversion pipeline for a single model.
    """
    model_name = get_model_name_from_path(image_path)
    print(f"--- Training: Model='{model_name}', Regimen='{regimen}' ---")

    full_grid_dim = TARGET_SIZE + 2 * TARGET_PADDING
    try:
        with open(image_path, 'rb') as f:
            target_rgba = load_image_from_file(f, target_dim=full_grid_dim)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}. Skipping.")
        return
    except Exception as e:
        print(f"Error loading image {image_path}: {e}. Skipping.")
        return

    regimen_map = {"growing": 0, "persistent": 1, "regenerating": 2}
    reg_n = regimen_map[regimen]
    damage_n = [0, 0, 3][reg_n]

    config = {
        "fire_rate": DEFAULT_FIRE_RATE,
        "batch_size": DEFAULT_BATCH_SIZE,
        "pool_size": DEFAULT_POOL_SIZE,
        "experiment_type": regimen.capitalize(),
        "learning_rate": 2e-3,
        "damage_n": damage_n,
        "run_dir": None,
        'enable_entropy': False,
        'entropy_strength': 0.0,
    }

    trainer = NCATrainer(target_img_rgba_processed=target_rgba, config=config)
    print(f"NCATrainer initialized for '{model_name}'. Starting training...")

    for i in range(total_steps + 1):
        _, loss = trainer.run_training_step()
        if loss is None:
            print("Warning: Loss is None, training may have stalled. Continuing...")
            continue
            
        if i in checkpoint_steps:
            print(f"\nCheckpoint at step {i}, log10(loss): {np.log10(loss):.3f}")
            ca_model = trainer.get_model()
            webgl_model_list = export_to_webgl_format(ca_model)
            
            filename = f"{model_name}_{regimen}_{i}.json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(webgl_model_list, f)
            print(f"Saved WebGL model to {filepath}")

        if i % 20 == 0:
            print(f'\rStep: {i}, log10(loss): {np.log10(loss):.3f}', end='', flush=True)

    print(f"\n--- Training complete for Model='{model_name}', Regimen='{regimen}' ---\n")

# --- NEW FUNCTION: generate_checkpoints ---
def generate_checkpoints(total_steps, start_step=50, max_spacing=500):
    """
    Generates a list of checkpoint steps with increasing spacing, starting after
    a specific step and adhering to a maximum spacing.
    """
    if total_steps < start_step:
        return [total_steps] if total_steps > 0 else []

    checkpoints = []
    current_step = start_step
    # The gap starts small and grows with each checkpoint
    gap = 75 
    
    while current_step <= total_steps:
        checkpoints.append(current_step)
        current_step += gap
        # Increase the gap for the next iteration, but cap it at the max_spacing
        gap = min(gap + 25, max_spacing)

    # Always include the final step for the fully trained model
    if not checkpoints or checkpoints[-1] < total_steps:
        # To avoid tiny gaps at the end, if the last checkpoint is too close, just replace it.
        if checkpoints and total_steps - checkpoints[-1] < 50:
             checkpoints[-1] = total_steps
        else:
            checkpoints.append(total_steps)

    return sorted(list(np.unique(checkpoints)))

def main():
    parser = argparse.ArgumentParser(description="Train and convert NCA models for the WebGL demo using the app's core modules.")
    parser.add_argument('--image_paths', type=str, nargs='+', required=True,
                        help='One or more file paths to the target images.')
    parser.add_argument('--steps', type=int, default=8000,
                        help='Total number of training steps for each model.')
    parser.add_argument('--output_dir', type=str, default='demo/webgl_models',
                        help='Directory to save the converted WebGL model files.')
    args = parser.parse_args()
    
    manifest_path = os.path.join(os.path.dirname(args.output_dir), 'models.json')
    if os.path.exists(manifest_path):
        print(f"Found existing manifest at '{manifest_path}'. It will be updated.")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    else:
        print("No existing manifest found. A new one will be created.")
        manifest = {}
    
    # --- MODIFICATION: Use the new checkpoint generation function ---
    all_checkpoints = generate_checkpoints(total_steps=args.steps, start_step=50, max_spacing=500)

    print(f"Starting training process for {len(args.image_paths)} image(s).")
    print(f"Total steps per model: {args.steps}. Saving checkpoints at steps: {all_checkpoints}")

    os.makedirs(args.output_dir, exist_ok=True)
    
    regimens = ["growing", "persistent", "regenerating"]

    for image_path in args.image_paths:
        model_name = get_model_name_from_path(image_path)
        if model_name not in manifest:
            manifest[model_name] = {"emoji": "", "regimens": {}}
        
        for regimen in regimens:
            train_and_convert(image_path, regimen, args.steps, all_checkpoints, args.output_dir)
            manifest[model_name]["regimens"][regimen.capitalize()] = {"checkpoints": all_checkpoints}

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, cls=NpEncoder)
    print(f"\nProcess complete. Updated manifest saved to: {manifest_path}")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPUs, memory growth enabled.")
        except RuntimeError as e:
            print(f"Error setting up GPU: {e}")

    main()