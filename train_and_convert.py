# train_and_convert.py
"""
Trains and converts Neural CA models for the WebGL demo and saves .h5 weights.
Includes features for:
1. Standard training of all regimens for a model.
2. Resuming training for a model across all regimens (--resume_model).
3. Training a single regimen starting from a specific pre-trained .h5 model
   (--single_regimen and --start_model).
"""
import os
import json
import argparse
import glob
import re
import numpy as np
import tensorflow as tf

# --- Leverage existing application modules ---
from nca_globals import (TARGET_SIZE, TARGET_PADDING, DEFAULT_FIRE_RATE,
                         DEFAULT_BATCH_SIZE, DEFAULT_POOL_SIZE)
from nca_utils import load_and_pad_image_from_file
from nca_trainer import NCATrainer
from model_converter import export_to_webgl_format

# --- Constants for Stuck Run Detection ---
STUCK_CHECK_STEP = 1500
MASS_THRESHOLD = 5.0
MAX_ATTEMPTS = 3

def get_model_name_from_path(file_path):
    """Generates a clean model name from a file path."""
    return os.path.splitext(os.path.basename(file_path))[0]

def find_latest_checkpoint(model_name, regimen, h5_dir):
    """Scans for the latest .h5 checkpoint for a given model and regimen."""
    latest_step, latest_file = -1, None
    search_pattern = os.path.join(h5_dir, f"{model_name}_{regimen}_*.weights.h5")
    for f in glob.glob(search_pattern):
        match = re.search(r'_(\d+)\.weights\.h5$', f)
        if match:
            step = int(match.group(1))
            if step > latest_step:
                latest_step, latest_file = step, f
    return latest_file, latest_step if latest_file else 0

def train_and_convert(image_path, regimen, total_steps, checkpoint_steps, output_dirs, resume_path=None, resume_step=0):
    # This function remains largely the same, as the logic is handled in main()
    model_name = get_model_name_from_path(image_path)
    print(f"--- Training: Model='{model_name}', Regimen='{regimen}' ---")

    try:
        with open(image_path, 'rb') as f: target_rgba = load_and_pad_image_from_file(f, target_size=TARGET_SIZE, final_grid_dim=(TARGET_SIZE + 2 * TARGET_PADDING))
    except Exception as e:
        print(f"Error loading image {image_path}: {e}. Skipping.")
        return 'error'

    regimen_map = {"growing": 0, "persistent": 1, "regenerating": 2}
    config = {
        "fire_rate": DEFAULT_FIRE_RATE, "batch_size": DEFAULT_BATCH_SIZE,
        "pool_size": DEFAULT_POOL_SIZE, "experiment_type": regimen.capitalize(),
        "learning_rate": 2e-3, "damage_n": [0, 0, 3][regimen_map[regimen]],
        "run_dir": None, 'enable_entropy': False, 'entropy_strength': 0.0,
    }

    trainer = NCATrainer(target_img_rgba_processed=target_rgba, config=config)
    
    if resume_path:
        print(f"Loading weights from '{resume_path}' to resume training...")
        trainer.load_weights(resume_path)
        trainer.current_step = resume_step
        print(f"Weights loaded. Trainer step counter set to {trainer.current_step}.")
    else:
        print("Starting training from scratch for this regimen.")

    print(f"NCATrainer initialized. Training from step {resume_step + 1} up to {total_steps}...")

    num_steps_to_run = total_steps - resume_step
    if num_steps_to_run <= 0:
        print("Target step count already reached or exceeded. Nothing to train for this regimen.")
        return 'success'

    for i in range(num_steps_to_run):
        _, loss = trainer.run_training_step()
        current_step = trainer.current_step
        if loss is None: continue
            
        if current_step == STUCK_CHECK_STEP and resume_step < STUCK_CHECK_STEP:
            mass = np.sum(trainer.last_preview_state[..., 3])
            print(f"\nPerforming growth check at step {current_step}: Current mass = {mass:.2f}")
            if mass < MASS_THRESHOLD:
                print(f"  FAILURE: Mass is below threshold of {MASS_THRESHOLD}. Declaring run as STUCK.")
                return 'stuck'
            else:
                 print(f"  SUCCESS: Mass is above threshold. Continuing training.")

        if current_step in checkpoint_steps:
            print(f"\nCheckpoint at step {current_step}, log10(loss): {np.log10(loss):.3f}")
            ca_model = trainer.get_model()
            base_filename = f"{model_name}_{regimen}_{current_step}"
            
            webgl_path = os.path.join(output_dirs['webgl'], f"{base_filename}.json")
            with open(webgl_path, 'w') as f: json.dump(export_to_webgl_format(ca_model), f)
            print(f"Saved WebGL model to {webgl_path}")

            h5_path = os.path.join(output_dirs['h5'], f"{base_filename}.weights.h5")
            ca_model.save_weights(h5_path)
            print(f"Saved H5 weights to {h5_path}")

        if i % 20 == 0:
            print(f'\rStep: {current_step}, log10(loss): {np.log10(loss):.3f}', end='', flush=True)

    print(f"\n--- Training complete for Model='{model_name}', Regimen='{regimen}' ---\n")
    return 'success'

def generate_checkpoints(total_steps, num_checkpoints):
    # This function remains unchanged
    if total_steps < 50: return [total_steps] if total_steps > 0 else []
    log_points = np.logspace(np.log10(50), np.log10(total_steps), num_checkpoints - 1, base=10.0)
    initial_checkpoints = np.unique(np.concatenate(([50], np.round(log_points / 50) * 50, [total_steps]))).astype(int)
    final_checkpoints = []
    last_cp = 0
    for cp in initial_checkpoints:
        if cp <= last_cp: continue
        gap = cp - last_cp
        if gap > 500:
            num_fillers = int(np.ceil(gap / 500.0)) - 1
            filler_points = np.linspace(last_cp, cp, num_fillers + 2)[1:-1]
            for f in np.round(filler_points / 50) * 50:
                if f > last_cp and f not in final_checkpoints: final_checkpoints.append(int(f))
        if cp not in final_checkpoints: final_checkpoints.append(int(cp))
        last_cp = cp
    return sorted(list(set(final_checkpoints)))

def main():
    parser = argparse.ArgumentParser(description="Train, convert, and save NCA models for the WebGL demo.")
    parser.add_argument('--image_paths', type=str, nargs='+', required=True, help='One or more file paths to target images.')
    parser.add_argument('--steps', type=int, default=8000, help='Total target number of training steps for each model.')
    parser.add_argument('--checkpoints', type=int, default=15, help='Approximate number of checkpoints to generate.')
    parser.add_argument('--output_dir_base', type=str, default='demo', help='Base directory for saving models.')
    
    # --- MODIFIED ARGUMENTS ---
    parser.add_argument('--resume_model', type=str, default=None, help='Model name to resume training for (scans for latest checkpoints). Mutually exclusive with --single_regimen.')
    parser.add_argument('--single_regimen', type=str, default=None, choices=['growing', 'persistent', 'regenerating'], help='Train only this single regimen. Requires --start_model.')
    parser.add_argument('--start_model', type=str, default=None, help='Path to an .h5 weights file to be the starting point for a single regimen run. Requires --single_regimen.')
    args = parser.parse_args()

    # --- VALIDATION LOGIC ---
    if args.resume_model and args.single_regimen:
        print("Error: --resume_model and --single_regimen cannot be used together.")
        return
    if args.single_regimen and not args.start_model:
        print("Error: --single_regimen requires --start_model to be specified.")
        return
    if args.start_model and not args.single_regimen:
        print("Warning: --start_model is specified but --single_regimen is not. Ignoring --start_model.")
    if (args.resume_model or args.start_model) and len(args.image_paths) > 1:
        print("Error: Resuming training can only be done with a single image path.")
        return

    output_dirs = {'webgl': os.path.join(args.output_dir_base, 'webgl_models'), 'h5': os.path.join(args.output_dir_base, 'h5_models')}
    manifest_path = os.path.join(args.output_dir_base, 'models.json')
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f: manifest = json.load(f)
    else: manifest = {}
    
    print(f"Starting training process for {len(args.image_paths)} image(s).")
    os.makedirs(output_dirs['webgl'], exist_ok=True)
    os.makedirs(output_dirs['h5'], exist_ok=True)
    
    # --- DETERMINE REGIMENS TO RUN ---
    all_regimens = ["growing", "persistent", "regenerating"]
    regimens_to_run = [args.single_regimen] if args.single_regimen else all_regimens

    for image_path in args.image_paths:
        model_name = get_model_name_from_path(image_path)
        if model_name not in manifest:
            manifest[model_name] = {"emoji": "", "regimens": {}}
        
        for regimen in regimens_to_run:
            for attempt in range(MAX_ATTEMPTS):
                print(f"\nStarting run for '{model_name}' regimen '{regimen}', attempt {attempt + 1}/{MAX_ATTEMPTS}.")
                
                resume_path, resume_step = None, 0
                # --- DETERMINE RESUME POINT ---
                if args.resume_model and args.resume_model == model_name:
                    resume_path, resume_step = find_latest_checkpoint(model_name, regimen, output_dirs['h5'])
                elif args.single_regimen and args.start_model and regimen == args.single_regimen:
                    try:
                        filename = os.path.basename(args.start_model)
                        step_str = re.search(r'_(\d+)\.weights\.h5$', filename).group(1)
                        resume_step = int(step_str)
                        resume_path = args.start_model
                    except (AttributeError, IndexError, ValueError):
                        print(f"Error: Could not parse step number from --start_model filename '{args.start_model}'. Aborting.")
                        return

                full_checkpoints = generate_checkpoints(args.steps, args.checkpoints)
                run_checkpoints = [cp for cp in full_checkpoints if cp > resume_step]
                if args.steps not in run_checkpoints and args.steps > resume_step: run_checkpoints.append(args.steps)
                
                print(f"  Target step count: {args.steps}. Checkpoints for this run: {run_checkpoints}")
                
                manifest[model_name]["regimens"][regimen.capitalize()] = {"checkpoints": full_checkpoints}
                status = train_and_convert(image_path, regimen, args.steps, run_checkpoints, output_dirs, resume_path, resume_step)

                if status == 'success':
                    print(f"Successfully completed training for '{model_name}' regimen '{regimen}'.")
                    break
                elif status == 'stuck':
                    print(f"Attempt {attempt + 1} failed due to being stuck. Retrying...")
                else:
                    print(f"An error occurred. Aborting attempts for this regimen.")
                    break
            else:
                print(f"TRAINING FAILED for '{model_name}' regimen '{regimen}' after {MAX_ATTEMPTS} attempts.")

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, cls=NpEncoder)
    print(f"\nProcess complete. Updated manifest saved to: {manifest_path}")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPUs, memory growth enabled.")
        except RuntimeError as e: print(f"Error setting up GPU: {e}")
    main()