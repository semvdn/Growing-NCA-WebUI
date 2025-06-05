# app.py
"""Flask Web Application for Neural Cellular Automata."""

import os
import io
import json # New: For saving metadata to JSON files
import time
import threading
import traceback
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import PIL.Image
import base64

from nca_globals import (TARGET_SIZE, TARGET_PADDING, DEFAULT_FIRE_RATE, 
                         DEFAULT_BATCH_SIZE, DEFAULT_POOL_SIZE, CHANNEL_N,
                         DEFAULT_RUNNER_SLEEP_DURATION, DRAW_CANVAS_DISPLAY_SIZE) 
from nca_utils import load_image_from_file, np2pil, to_rgb, get_model_summary, format_training_time # load_emoji removed
from nca_model import CAModel
from nca_trainer import NCATrainer
from nca_runner import NCARunner


# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = os.urandom(24) 
UPLOAD_FOLDER = 'uploads' 
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# --- Global state & Threading Primitives ---
current_nca_trainer = None
trainer_target_image_rgba = None # For "file": TARGET_SIZE content; For "drawn_defines_padded_grid": (TARGET_SIZE+2*PAD) image
trainer_target_source_kind = None # "file" or "drawn_defines_padded_grid"
trainer_actual_target_shape = None # Shape of the final target used by trainer (e.g. (72,72,4))
trainer_target_image_name = "unknown_image" # Stores the name of the image used for training target (file name or drawn name)
trainer_target_image_loaded_or_drawn = "unknown" # "loaded" or "drawn"

current_training_run_id = None # New: Unique ID for the current training session
current_training_run_dir = None # New: Path to the dedicated directory for the current training session

current_nca_runner = None
runner_sleep_duration = DEFAULT_RUNNER_SLEEP_DURATION

training_thread = None
stop_training_event = threading.Event()
train_thread_lock = threading.Lock() 

running_thread = None
stop_running_event = threading.Event()
run_thread_lock = threading.Lock() 

# --- Helper Functions --- (get_preview_image_response, ensure_trainer_stopped, ensure_runner_stopped same as before)
def get_preview_image_response(state_array_or_rgba, zoom_factor=4, default_width_px=256, default_height_px=256):
    img = None
    if state_array_or_rgba is None:
        img = PIL.Image.new('RGB', (int(default_width_px), int(default_height_px)), color = 'grey')
    else:
        np_array = np.asarray(state_array_or_rgba)
        if np_array.ndim == 3 and np_array.shape[-1] == CHANNEL_N: 
            rgb_array = to_rgb(tf.constant(np_array[None,...]))[0].numpy()
            img = np2pil(rgb_array, zoom_factor=zoom_factor)
        elif np_array.ndim == 3 and np_array.shape[-1] == 4: 
            alpha = np.clip(np_array[..., 3:], 0, 1)
            rgb_array = 1.0 - alpha + np_array[..., :3]
            img = np2pil(rgb_array, zoom_factor=zoom_factor)
        else: 
            tf.print(f"get_preview_image_response: Unexpected array shape {np_array.shape if hasattr(np_array, 'shape') else 'Not an array'}")
            img = PIL.Image.new('RGB', (int(default_width_px), int(default_height_px)), color = 'lightgrey')
            
    img_io = io.BytesIO()
    if img.width == 0 or img.height == 0 : 
         img = PIL.Image.new('RGB', (zoom_factor, zoom_factor), color = 'pink') 
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', as_attachment=False, download_name=f'preview_{time.time()}.png')

def ensure_trainer_stopped():
    global training_thread, stop_training_event
    if training_thread and training_thread.is_alive():
        tf.print("Signalling training thread to stop...")
        stop_training_event.set()
        training_thread.join(timeout=7) 
        if training_thread.is_alive():
            tf.print("Warning: Training thread did not stop in time after signal.")
        training_thread = None
    stop_training_event.clear() 

def ensure_runner_stopped(): 
    global running_thread, stop_running_event
    if running_thread and running_thread.is_alive():
        tf.print("Signalling running thread to stop...")
        stop_running_event.set()
        running_thread.join(timeout=7)
        if running_thread.is_alive():
            tf.print("Warning: Running thread did not stop in time after signal.")
        running_thread = None
    stop_running_event.clear()

# --- Web Routes ---
@app.route('/')
def index_route(): 
    return render_template('index.html')

@app.route('/upload_drawn_pattern_target', methods=['POST'])
def upload_drawn_pattern_target():
    global trainer_target_image_rgba, trainer_actual_target_shape, trainer_target_source_kind, trainer_target_image_name, trainer_target_image_loaded_or_drawn
    try:
        data = request.json
        data_url = data.get('image_data_url')
        drawn_image_name = data.get('drawn_image_name', 'drawn_pattern') # Get the name from the frontend
        if not data_url or not data_url.startswith('data:image/png;base64,'):
            return jsonify({"success": False, "message": "Invalid image data URL."}), 400

        header, encoded = data_url.split(',', 1)
        image_data = base64.b64decode(encoded)
        img_pil = PIL.Image.open(io.BytesIO(image_data)).convert("RGBA")

        # For drawn patterns, they define the FINAL grid size the NCA operates on.
        # So, resize the drawn image (from DRAW_CANVAS_DISPLAY_SIZE) to this final grid size.
        final_grid_dim_h = TARGET_SIZE + 2 * TARGET_PADDING
        final_grid_dim_w = TARGET_SIZE + 2 * TARGET_PADDING
        
        # Resize maintaining aspect to fit within final_grid_dim box
        original_size = img_pil.size
        if original_size[0] == 0 or original_size[1] == 0:
             raise ValueError("Drawn image has zero dimension before resize.")
        
        # Fit into the (final_grid_dim_w, final_grid_dim_h) box, maintaining aspect ratio
        # This means the drawing might not fill the entire box if it's not square, 
        # and will be padded with transparency.
        # Or, we can stretch it if that's preferred. Let's try fitting with padding.
        
        ratio = min(final_grid_dim_w / original_size[0], final_grid_dim_h / original_size[1])
        new_size = (max(1, int(original_size[0] * ratio)), max(1, int(original_size[1] * ratio)))
        img_pil_resized = img_pil.resize(new_size, PIL.Image.Resampling.LANCZOS if hasattr(PIL.Image, 'Resampling') else PIL.Image.ANTIALIAS)

        # Create a new square image (final_grid_dim x final_grid_dim) and paste resized image onto it
        # This becomes the direct target for the trainer, no further padding by trainer.
        final_target_pil = PIL.Image.new("RGBA", (final_grid_dim_w, final_grid_dim_h), (0, 0, 0, 0)) # Transparent bg
        upper_left = ((final_grid_dim_w - new_size[0]) // 2, (final_grid_dim_h - new_size[1]) // 2)
        final_target_pil.paste(img_pil_resized, upper_left)
        
        img_np = np.float32(final_target_pil) / 255.0
        img_np[..., :3] *= img_np[..., 3:]
        
        trainer_target_image_rgba = img_np
        trainer_target_source_kind = "drawn_defines_padded_grid" # New kind
        trainer_actual_target_shape = trainer_target_image_rgba.shape # This is now (72,72,4) e.g.
        trainer_target_image_name = drawn_image_name # Store the user-provided name
        trainer_target_image_loaded_or_drawn = "drawn" # Indicate it was drawn
        
        tf.print(f"Trainer target from DRAWN pattern (defines full grid). Final shape: {trainer_actual_target_shape}")
        return jsonify({"success": True, "message": "Drawn pattern set as trainer target (defines full grid)."}), 200
    except Exception as e_detail:
        tf.print(f"Error in /upload_drawn_pattern_target: {e_detail}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": f"Error processing drawn pattern: {str(e_detail)}"}), 500


@app.route('/load_target_from_file', methods=['POST'])
def load_target_from_file_route():
    global trainer_target_image_rgba, trainer_actual_target_shape, trainer_target_source_kind, trainer_target_image_name, trainer_target_image_loaded_or_drawn
    try:
        if 'image_file' not in request.files or request.files['image_file'].filename == '':
            return jsonify({"success": False, "message": "No file provided for trainer target."}), 400
            
        file = request.files['image_file']
        filename = secure_filename(file.filename)
        trainer_target_image_name = filename # Store the filename for metadata
        trainer_target_image_loaded_or_drawn = "loaded" # Indicate it was loaded
        # load_image_from_file resizes to TARGET_SIZE content area
        trainer_target_image_rgba = load_image_from_file(file.stream, max_size=TARGET_SIZE)
        trainer_target_source_kind = "file"
        message = f"Trainer Target: File '{filename}' loaded as content."

        # For file targets, trainer_actual_target_shape will be after NCATrainer pads it.
        # We can store the content shape here for reference.
        # trainer_actual_target_shape will be set in initialize_trainer for files.
        # For now, just indicate success.

        tf.print(f"Trainer target from FILE loaded, content shape {trainer_target_image_rgba.shape}")
        # The height/width returned here should be of the *content* for files.
        return jsonify({"success": True, "message": message,
                        "target_height": trainer_target_image_rgba.shape[0],
                        "target_width": trainer_target_image_rgba.shape[1]})
    except Exception as e_detail:
        tf.print(f"Error in /load_target_from_file: {e_detail}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": f"Error loading trainer target from file: {str(e_detail)}"}), 500

@app.route('/get_trainer_target_preview') 
def get_trainer_target_preview_route():
    # This endpoint now previews trainer_target_image_rgba directly.
    # If it's a "drawn_defines_padded_grid", it's already the final size.
    # If it's "file", it's TARGET_SIZE content; NCATrainer will pad it.
    # The preview should ideally show what NCATrainer will use.
    
    preview_image_data = None
    effective_target_h, effective_target_w = DRAW_CANVAS_DISPLAY_SIZE, DRAW_CANVAS_DISPLAY_SIZE
    zoom = 1

    if trainer_target_image_rgba is not None:
        if trainer_target_source_kind == "drawn_defines_padded_grid":
            preview_image_data = trainer_target_image_rgba # Already final size
            if preview_image_data.shape[0] > 0:
                zoom = DRAW_CANVAS_DISPLAY_SIZE // preview_image_data.shape[0]

        elif trainer_target_source_kind == "file":
            # For files, preview the content padded as the trainer would
            p = TARGET_PADDING
            padded_content = tf.pad(trainer_target_image_rgba, [(p,p),(p,p),(0,0)]).numpy()
            preview_image_data = padded_content
            if preview_image_data.shape[0] > 0:
                zoom = DRAW_CANVAS_DISPLAY_SIZE // preview_image_data.shape[0]
        else: # No target yet or unknown kind
             preview_image_data = None # Will show placeholder
    
    return get_preview_image_response(preview_image_data, zoom_factor=max(1,zoom), 
                                      default_width_px=DRAW_CANVAS_DISPLAY_SIZE, 
                                      default_height_px=DRAW_CANVAS_DISPLAY_SIZE) 


@app.route('/initialize_trainer', methods=['POST'])
def initialize_trainer_route():
    global current_nca_trainer, trainer_target_image_rgba, trainer_target_source_kind, trainer_actual_target_shape, current_training_run_id, current_training_run_dir
    
    if trainer_target_image_rgba is None:
        return jsonify({"success": False, "message": "Please set a target (draw or file) for the trainer first."}), 400
    try:
        with train_thread_lock:
            ensure_trainer_stopped()
            
            # Generate a unique run ID and create a dedicated directory for this training run
            current_training_run_id = f"{trainer_target_image_name}_{int(time.time())}"
            current_training_run_dir = os.path.join(app.config['MODEL_FOLDER'], current_training_run_id)
            os.makedirs(current_training_run_dir, exist_ok=True)
            tf.print(f"New training run directory created: {current_training_run_dir}")

            data = request.json
            config = {
                "fire_rate": float(data.get("fire_rate", DEFAULT_FIRE_RATE)),
                "batch_size": int(data.get("batch_size", DEFAULT_BATCH_SIZE)),
                "pool_size": int(data.get("pool_size", DEFAULT_POOL_SIZE)),
                "experiment_type": data.get("experiment_type", "Growing"),
                "target_padding": TARGET_PADDING, # NCATrainer uses this for 'file' source
                "learning_rate": float(data.get("learning_rate", 2e-3)),
                "target_source_kind": trainer_target_source_kind, # Pass the source kind
                "model_folder_path": app.config['MODEL_FOLDER'], # Pass the general model folder path
                "run_dir": current_training_run_dir # New: Pass the specific run directory
            }
            if config["experiment_type"] == "Regenerating":
                config["damage_n"] = int(data.get("damage_n", 3))
            else:
                config["damage_n"] = 0
            
            # trainer_target_image_rgba is passed to NCATrainer
            # NCATrainer's __init__ now handles padding conditionally
            current_nca_trainer = NCATrainer(target_img_rgba_processed=trainer_target_image_rgba, config=config)
            trainer_actual_target_shape = current_nca_trainer.pad_target.shape # Store the true operational target shape
            tf.print(f"NCATrainer initialized. Operational target shape: {trainer_actual_target_shape}")
        
        model_summary_str = get_model_summary(current_nca_trainer.get_model())
        return jsonify({
            "success": True,
            "message": "NCA Trainer Initialized. Ready to train.",
            "model_summary": model_summary_str,
            "initial_state_preview_url": "/get_live_trainer_preview"
        })
    except Exception as e_detail:
        tf.print(f"Error in /initialize_trainer: {e_detail}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": f"Error initializing trainer: {str(e_detail)}"}), 500

# ... (training_loop_task_function, start_training, stop_training - same)
def training_loop_task_function(): 
    tf.print("Training thread started.")
    while not stop_training_event.is_set():
        try:
            with train_thread_lock: 
                if not current_nca_trainer: 
                    tf.print("Trainer gone, stopping training loop.")
                    break
                preview_state, loss = current_nca_trainer.run_training_step() 
                if preview_state is None and loss is None and current_nca_trainer.current_step > 0 : 
                    tf.print("Training step returned None, possibly empty pool or batch. Stopping loop.")
                    break 
            time.sleep(0.01) 
        except Exception as e_detail:
            tf.print(f"Error in training loop: {e_detail}\n{traceback.format_exc()}")
            break 
    tf.print("Training thread ended.")
@app.route('/start_training', methods=['POST'])
def start_training_route():
    global training_thread
    with train_thread_lock: 
        if not current_nca_trainer:
            return jsonify({"success": False, "message": "Initialize Trainer first."}), 400
        if training_thread and training_thread.is_alive():
            return jsonify({"success": False, "message": "Training already running."}), 400
        stop_training_event.clear()
        if current_nca_trainer: # Resume timer when training starts
            current_nca_trainer.resume_training_timer()
        training_thread = threading.Thread(target=training_loop_task_function, daemon=True)
        training_thread.start()
        tf.print("Training started via /start_training.")
    return jsonify({"success": True, "message": "Training started."})
@app.route('/stop_training', methods=['POST'])
def stop_training_route():
    with train_thread_lock:
        ensure_trainer_stopped()
        if current_nca_trainer: # Pause timer when training stops
            current_nca_trainer.pause_training_timer()
    tf.print("Training stopped via /stop_training. Trainer instance preserved for saving.")
    return jsonify({"success": True, "message": "Training stopped. Trainer state preserved for saving."})


@app.route('/get_training_status')
def get_training_status_route():
    with train_thread_lock: 
        if not current_nca_trainer:
            return jsonify({
                "status_message": "Trainer Not Initialized", 
                "is_training": False, 
                "preview_url": "/get_live_trainer_preview" 
            }), 200 

        status_data = current_nca_trainer.get_status()
        is_currently_training = training_thread.is_alive() if training_thread else False
        formatted_time = format_training_time(status_data["training_time_seconds"])
    
    status_msg = f"Step: {status_data['step']}, Loss: {status_data['loss']} (log10: {status_data['log_loss']}), Time: {formatted_time}"
    
    return jsonify({
        "step": status_data["step"],
        "loss": status_data["loss"],
        "log_loss": status_data["log_loss"],
        "training_time": formatted_time, # Send formatted time
        "training_time_seconds": status_data["training_time_seconds"], # And raw seconds
        "is_training": is_currently_training,
        "status_message": status_msg, # Combined status message
        "preview_url": "/get_live_trainer_preview" 
    })

@app.route('/get_live_trainer_preview') 
def get_live_trainer_preview_route():
    preview_state_to_show = None
    # Base placeholder size on DRAW_CANVAS_DISPLAY_SIZE for consistency
    default_w, default_h = DRAW_CANVAS_DISPLAY_SIZE, DRAW_CANVAS_DISPLAY_SIZE 
    zoom = 1
    with train_thread_lock: 
        if current_nca_trainer: 
            if current_nca_trainer.last_preview_state is not None:
                preview_state_to_show = current_nca_trainer.last_preview_state 
            elif current_nca_trainer.pool and len(current_nca_trainer.pool) > 0: # Initial seed from pool
                preview_state_to_show = current_nca_trainer.pool.x[0].copy() 
            
            # Determine zoom factor based on actual operational grid size of the trainer
            if current_nca_trainer.pad_target is not None:
                 op_grid_h = current_nca_trainer.pad_target.shape[0]
                 if op_grid_h > 0 : zoom = DRAW_CANVAS_DISPLAY_SIZE // op_grid_h
            elif trainer_actual_target_shape: # Fallback to stored shape if pad_target not yet set
                 op_grid_h = trainer_actual_target_shape[0]
                 if op_grid_h > 0 : zoom = DRAW_CANVAS_DISPLAY_SIZE // op_grid_h

    return get_preview_image_response(preview_state_to_show, zoom_factor=max(1, zoom), 
                                      default_width_px=default_w, default_height_px=default_h)

# ... (save_trainer_model - same)
@app.route('/save_trainer_model', methods=['POST'])
def save_trainer_model_route():
    ca_model_to_save = None
    
    with train_thread_lock:
        if current_nca_trainer and current_nca_trainer.ca:
            ca_model_to_save = current_nca_trainer.ca
            if current_training_run_dir is None:
                return jsonify({"success": False, "message": "No active training run to save checkpoint for. Initialize trainer first."}), 400
            
            # Use the current run's directory for saving
            model_save_base_path = current_training_run_dir
            
            # Generate a unique filename for this checkpoint within the run directory
            # Include current step and timestamp for uniqueness and ordering
            current_step = current_nca_trainer.current_step
            checkpoint_name = f"checkpoint_step_{current_step}_{int(time.time())}"
            
            model_filename = f"{checkpoint_name}.weights.h5"
            json_filename = f"{checkpoint_name}.json"
            
            model_save_path = os.path.join(model_save_base_path, model_filename)
            json_save_path = os.path.join(model_save_base_path, json_filename)

        else:
            return jsonify({"success": False, "message": "No active NCA trainer instance to save."}), 400
    
    try:
        # Model weights file
        ca_model_to_save.save_weights(model_save_path)
        tf.print(f"Trainer model weights saved to {model_save_path}")

        # Metadata JSON file
        metadata = {
            "model_name": model_filename,
            "trained_on_image": trainer_target_image_name, # Original name
            "image_source_kind": trainer_target_image_loaded_or_drawn, # "loaded" or "drawn"
            "training_steps": current_nca_trainer.current_step,
            "experiment_type": current_nca_trainer.config.get('experiment_type','generic'),
            "final_loss": float(current_nca_trainer.last_loss), # Convert float32 to Python float
            "total_training_time_seconds": float(current_nca_trainer.get_status()["training_time_seconds"]), # Convert float32 to Python float
            "save_timestamp": time.time(),
            "save_datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "run_id": current_training_run_id # New: Add run ID to metadata
        }
        with open(json_save_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        tf.print(f"Metadata saved to {json_save_path}")

        # If the image was drawn, save the final initialized image as a PNG in the run directory
        if trainer_target_image_loaded_or_drawn == "drawn" and trainer_target_image_rgba is not None:
            # Use the original image name for the drawn target file
            drawn_image_filename = f"{secure_filename(trainer_target_image_name).replace('.', '_')}_initial_target.png"
            drawn_image_save_path = os.path.join(model_save_base_path, drawn_image_filename)
            
            # Convert RGBA numpy array to PIL Image and save
            img_pil_to_save = PIL.Image.fromarray((trainer_target_image_rgba * 255).astype(np.uint8))
            img_pil_to_save.save(drawn_image_save_path)
            tf.print(f"Drawn initial target image saved to {drawn_image_save_path}")
            metadata["initial_target_image_file"] = drawn_image_filename # Add to metadata
            # Re-save metadata with the new image file entry
            with open(json_save_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            tf.print(f"Metadata updated with initial target image path: {json_save_path}")


        return jsonify({"success": True, "message": f"Checkpoint saved to '{current_training_run_id}' directory."})
    except Exception as e:
        tf.print(f"Error save_trainer_model: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": f"Error: {e}"}), 500



@app.route('/load_trainer_model', methods=['POST'])
def load_trainer_model_route():
    global current_nca_trainer, trainer_target_image_rgba, trainer_target_source_kind, trainer_actual_target_shape, trainer_target_image_name, trainer_target_image_loaded_or_drawn
    with train_thread_lock:
        ensure_trainer_stopped() # Stop any active training before loading
        current_nca_trainer = None # Clear existing trainer

        if 'model_file' not in request.files or request.files['model_file'].filename == '':
            return jsonify({"success": False, "message": "No model file provided."}), 400

        file = request.files['model_file']
        filename = secure_filename(file.filename)
        if not filename.endswith(".weights.h5"): # Enforce the strict extension
            return jsonify({"success": False, "message": "Invalid model file type. Must end with .weights.h5"}), 400

        model_file_path = os.path.join(app.config['MODEL_FOLDER'], filename)
        try:
            file.save(model_file_path) # Save the uploaded model file

            metadata = {}
            json_file_path = model_file_path.replace('.weights.h5', '.json')
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as f:
                    metadata = json.load(f)
                tf.print(f"Trainer: Loaded metadata from {json_file_path}")

            # Try to load the original target image if specified in metadata
            loaded_target_rgba = None
            target_image_name_from_meta = metadata.get("trained_on_image")
            image_source_kind_from_meta = metadata.get("image_source_kind", "unknown")
            
            if target_image_name_from_meta and target_image_name_from_meta != "unknown_image":
                # Determine the path based on whether it was a loaded file or a drawn image
                if image_source_kind_from_meta == "loaded":
                    # For loaded files, they are in UPLOAD_FOLDER
                    target_image_path = os.path.join(app.config['UPLOAD_FOLDER'], target_image_name_from_meta)
                    if os.path.exists(target_image_path):
                        with open(target_image_path, 'rb') as img_f:
                            loaded_target_rgba = load_image_from_file(img_f, max_size=TARGET_SIZE)
                        trainer_target_source_kind = "file"
                        trainer_target_image_name = target_image_name_from_meta
                        trainer_target_image_loaded_or_drawn = "loaded"
                        tf.print(f"Trainer: Re-loaded target image '{target_image_name_from_meta}' for training.")
                    else:
                        tf.print(f"Trainer: Warning: Original loaded target image '{target_image_name_from_meta}' not found at {target_image_path}. Please set a new target.")
                elif image_source_kind_from_meta == "drawn":
                    # For drawn images, they are saved in the model's subdirectory
                    # Need to infer the model's subdirectory from the loaded model file path
                    model_dir_name = os.path.basename(os.path.dirname(model_file_path))
                    drawn_image_filename_in_subdir = metadata.get("initial_target_image_file")
                    if drawn_image_filename_in_subdir:
                        drawn_image_path = os.path.join(app.config['MODEL_FOLDER'], model_dir_name, drawn_image_filename_in_subdir)
                        if os.path.exists(drawn_image_path):
                            with open(drawn_image_path, 'rb') as img_f:
                                # For drawn images, they are already padded, so load as is
                                img_pil = PIL.Image.open(img_f).convert("RGBA")
                                loaded_target_rgba = np.float32(img_pil) / 255.0
                                loaded_target_rgba[..., :3] *= loaded_target_rgba[..., 3:] # Apply alpha
                            trainer_target_source_kind = "drawn_defines_padded_grid"
                            trainer_target_image_name = target_image_name_from_meta
                            trainer_target_image_loaded_or_drawn = "drawn"
                            tf.print(f"Trainer: Re-loaded drawn target image '{target_image_name_from_meta}' from model subdir.")
                        else:
                            tf.print(f"Trainer: Warning: Original drawn target image '{drawn_image_filename_in_subdir}' not found at {drawn_image_path}. Please set a new target.")
                    else:
                        tf.print(f"Trainer: Warning: Metadata for drawn image '{target_image_name_from_meta}' missing 'initial_target_image_file'. Please set a new target.")
                else:
                    tf.print(f"Trainer: Unknown image source kind '{image_source_kind_from_meta}' in metadata. Please set a new target.")
            else:
                tf.print("Trainer: No original target image specified in metadata or it was 'unknown_image'. Please set a new target.")

            if loaded_target_rgba is None:
                return jsonify({"success": False, "message": "Could not load original target image. Please set a new target after loading model."}), 400

            trainer_target_image_rgba = loaded_target_rgba # Set global target

            # Initialize trainer with loaded weights and restored state
            config = {
                "fire_rate": float(metadata.get("fire_rate", DEFAULT_FIRE_RATE)), # Get from metadata or default
                "batch_size": int(metadata.get("batch_size", DEFAULT_BATCH_SIZE)),
                "pool_size": int(metadata.get("pool_size", DEFAULT_POOL_SIZE)),
                "experiment_type": metadata.get("experiment_type", "Growing"),
                "target_padding": TARGET_PADDING,
                "learning_rate": float(metadata.get("learning_rate", 2e-3)),
                "target_source_kind": trainer_target_source_kind,
                "model_folder_path": app.config['MODEL_FOLDER']
            }
            if config["experiment_type"] == "Regenerating":
                config["damage_n"] = int(metadata.get("damage_n", 3))
            else:
                config["damage_n"] = 0

            current_nca_trainer = NCATrainer(target_img_rgba_processed=trainer_target_image_rgba, config=config)
            current_nca_trainer.ca.load_weights(model_file_path) # Load weights into the new trainer's model

            # Restore trainer state from metadata
            current_nca_trainer.current_step = metadata.get("training_steps", 0)
            current_nca_trainer.best_loss = metadata.get("final_loss", float('inf'))
            current_nca_trainer.total_training_time_paused = metadata.get("total_training_time_seconds", 0.0)
            current_nca_trainer.training_start_time = time.time() - current_nca_trainer.total_training_time_paused # Adjust start time for resume
            current_nca_trainer.last_loss = metadata.get("final_loss", None) # Set last loss for status

            trainer_actual_target_shape = current_nca_trainer.pad_target.shape
            tf.print(f"NCATrainer loaded and state restored. Operational target shape: {trainer_actual_target_shape}")

            model_summary_str = get_model_summary(current_nca_trainer.get_model())
            return jsonify({
                "success": True,
                "message": f"Trainer model '{filename}' loaded. Training state restored.",
                "model_summary": model_summary_str,
                "initial_state_preview_url": "/get_live_trainer_preview"
            })

        except Exception as e:
            tf.print(f"Error load_trainer_model: {e}\n{traceback.format_exc()}")
            current_nca_trainer = None
            return jsonify({"success": False, "message": f"Error loading trainer model: {str(e)}"}), 500


# --- Runner Routes ---
@app.route('/load_current_training_model_for_runner', methods=['POST'])
def load_current_training_model_for_runner_route():
    global current_nca_runner, current_nca_trainer, trainer_actual_target_shape, runner_sleep_duration, current_training_run_dir

    runner_h, runner_w = TARGET_SIZE + 2 * TARGET_PADDING, TARGET_SIZE + 2 * TARGET_PADDING # Default grid size for runner
    model_to_load = None
    model_fire_rate = DEFAULT_FIRE_RATE
    message = "Runner: "

    with train_thread_lock: # Acquire lock to safely access trainer state
        if current_nca_trainer and current_nca_trainer.ca:
            try:
                # Always try to load the best model from the current run's directory first
                if current_training_run_dir:
                    best_model_file_path = os.path.join(current_training_run_dir, "best_model.weights.h5")
                    if os.path.exists(best_model_file_path):
                        temp_ca_model = CAModel(channel_n=CHANNEL_N, fire_rate=DEFAULT_FIRE_RATE) # Fire rate will be updated
                        temp_ca_model.load_weights(best_model_file_path)
                        model_to_load = temp_ca_model
                        message += "Loaded best performing model from current run."
                        tf.print(f"Runner: Loaded best model from {best_model_file_path}")
                        
                        model_fire_rate = current_nca_trainer.ca.fire_rate # Use trainer's fire rate
                        model_to_load.fire_rate = model_fire_rate # Update fire rate of the loaded model
                        message += f" (Fire rate from trainer: {model_fire_rate})"
                    else:
                        tf.print(f"Runner: No 'best_model.weights.h5' found in current run directory: {current_training_run_dir}. Falling back to current training model.")
                else:
                    tf.print("Runner: No current training run directory set. Falling back to current training model.")

                # If no best model was loaded from the run directory, or if there's no run directory,
                # load the current state of the trainer's model directly.
                if model_to_load is None:
                    model_to_load = CAModel(channel_n=CHANNEL_N, fire_rate=current_nca_trainer.ca.fire_rate)
                    model_to_load.set_weights(current_nca_trainer.ca.get_weights())
                    model_fire_rate = current_nca_trainer.ca.fire_rate
                    message += "Loaded current training model."
                    tf.print(f"Runner: Loaded current training model. Fire rate: {model_fire_rate}.")
                
                if current_nca_trainer.pad_target is not None:
                    runner_h = current_nca_trainer.pad_target.shape[0]
                    runner_w = current_nca_trainer.pad_target.shape[1]
                elif trainer_actual_target_shape:
                    runner_h, runner_w = trainer_actual_target_shape[0], trainer_actual_target_shape[1]

            except Exception as e:
                tf.print(f"Error getting weights/info from trainer model: {e}\n{traceback.format_exc()}")
                return jsonify({"success": False, "message": f"Error accessing trainer model: {str(e)}"}), 500
        else:
            return jsonify({"success": False, "message": "No active training model available."}), 400

    with run_thread_lock:
        ensure_runner_stopped()
        current_nca_runner = None
        try:
            initial_runner_shape = (runner_h, runner_w, CHANNEL_N) # Use determined H, W
            current_nca_runner = NCARunner(ca_model_instance=model_to_load,
                                           initial_state_shape_tuple=initial_runner_shape)
            runner_sleep_duration = DEFAULT_RUNNER_SLEEP_DURATION

            model_summary_str = get_model_summary(current_nca_runner.ca)
            return jsonify({
                "success": True, "message": message, "model_summary": model_summary_str,
                "runner_preview_url": "/get_live_runner_preview"
            })
        except Exception as e_detail:
            tf.print(f"Error setting up runner with training model: {e_detail}\n{traceback.format_exc()}")
            current_nca_runner = None
            return jsonify({"success": False, "message": f"Error setting up runner: {str(e_detail)}"}), 500

# ... (load_model_for_runner - use trainer_actual_target_shape for H,W hint if available, otherwise default)
@app.route('/load_model_for_runner', methods=['POST'])
def load_model_for_runner_route():
    global current_nca_runner, trainer_actual_target_shape, runner_sleep_duration
    with run_thread_lock:
        ensure_runner_stopped()
        current_nca_runner = None
        model_file_path = None; message = ""
        
        if 'model_file' in request.files and request.files['model_file'].filename != '':
            file = request.files['model_file']; filename = secure_filename(file.filename)
            if filename.endswith(".weights.h5"): # Enforce .weights.h5 for runner loading too
                # Save uploaded file to a temporary location or directly process
                # For simplicity, let's assume it's saved to the main MODEL_FOLDER for now
                # A more robust solution might save it to a temp dir or process directly from stream
                model_file_path = os.path.join(app.config['MODEL_FOLDER'], filename)
                file.save(model_file_path)
                message = f"Runner: Uploaded model '{filename}' loaded."
            else: return jsonify({"success": False, "message": "Invalid model file type (.weights.h5 required)."}), 400
        else:
            # Recursively search for the latest .weights.h5 file in all subdirectories of MODEL_FOLDER
            latest_model_path = None
            latest_mtime = 0
            for root, _, files in os.walk(app.config['MODEL_FOLDER']):
                for f in files:
                    if f.endswith('.weights.h5'):
                        full_path = os.path.join(root, f)
                        mtime = os.path.getmtime(full_path)
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            latest_model_path = full_path
            
            if not latest_model_path:
                return jsonify({"success": False, "message": "No .weights.h5 models found in 'models' folder or its subdirectories."}), 404
            
            model_file_path = latest_model_path
            message = f"Runner: Loaded latest model '{os.path.basename(latest_model_path)}' from server."
        
        try:
            loaded_ca_model_for_runner = CAModel(channel_n=CHANNEL_N, fire_rate=DEFAULT_FIRE_RATE)
            loaded_ca_model_for_runner.load_weights(model_file_path)
            tf.print(f"Runner: Weights loaded from {model_file_path}")

            # Load companion metadata for display
            metadata = {}
            json_file_path = model_file_path.replace('.weights.h5', '.json')
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as f:
                    metadata = json.load(f)
                tf.print(f"Runner: Loaded metadata from {json_file_path}")

            initial_runner_h, initial_runner_w = TARGET_SIZE + 2 * TARGET_PADDING, TARGET_SIZE + 2 * TARGET_PADDING
            # Use trainer_actual_target_shape if trainer has been initialized, as it reflects the true grid size trainer used
            if trainer_actual_target_shape:
                initial_runner_h, initial_runner_w = trainer_actual_target_shape[0], trainer_actual_target_shape[1]
                message += f" Grid based on last trainer's grid size ({initial_runner_h}x{initial_runner_w})."
            else: message += f" Grid based on default ({initial_runner_h}x{initial_runner_w})."

            initial_runner_shape_to_use = (initial_runner_h, initial_runner_w, CHANNEL_N)
            current_nca_runner = NCARunner(ca_model_instance=loaded_ca_model_for_runner,
                                           initial_state_shape_tuple=initial_runner_shape_to_use)
            runner_sleep_duration = DEFAULT_RUNNER_SLEEP_DURATION
            model_summary_str = get_model_summary(current_nca_runner.ca)

            # Include metadata in the response for frontend display
            response_data = {
                "success": True, "message": message, "model_summary": model_summary_str,
                "runner_preview_url": "/get_live_runner_preview",
                "metadata": metadata # Include metadata here
            }
            return jsonify(response_data)
        except Exception as e:
            tf.print(f"Error load_model_for_runner: {e}\n{traceback.format_exc()}")
            current_nca_runner=None
            return jsonify({"success":False,"message":f"Error: {e}"}),500

# ... (running_loop_task_function, start_running, stop_running, set_runner_speed, 
#      get_runner_status, get_live_runner_preview, runner_action routes - same as previous correct version)
def running_loop_task_function(): 
    global runner_sleep_duration 
    tf.print("Runner thread started.")
    while not stop_running_event.is_set():
        loop_start_time = time.perf_counter()
        try:
            with run_thread_lock: 
                if not current_nca_runner: 
                    tf.print("Runner gone from running_loop, stopping.")
                    break 
                current_nca_runner.step() 
            
            current_sleep = runner_sleep_duration - (time.perf_counter() - loop_start_time)
            if current_sleep > 0: time.sleep(current_sleep)
            else: time.sleep(0.001)
        except Exception as e: tf.print(f"Error in running_loop: {e}\n{traceback.format_exc()}"); break 
    tf.print("Runner thread ended.")
@app.route('/start_running', methods=['POST'])
def start_running_loop_route(): 
    global running_thread
    with run_thread_lock: 
        if not current_nca_runner or not current_nca_runner.ca: return jsonify({"success":False,"message":"Load model for Runner first."}),400
        if running_thread and running_thread.is_alive(): return jsonify({"success":False,"message":"Runner loop active."}),400
        stop_running_event.clear() 
        running_thread = threading.Thread(target=running_loop_task_function, daemon=True)
        running_thread.start()
        tf.print("Runner loop started via /start_running.")
    return jsonify({"success": True, "message": "Runner loop started."})
@app.route('/stop_running', methods=['POST'])
def stop_running_loop_route(): 
    tf.print("/stop_running called by client")
    with run_thread_lock: 
        ensure_runner_stopped() 
    tf.print("Runner loop stop process finished from /stop_running.")
    return jsonify({"success": True, "message": "Runner loop stopped. State and history preserved."})
@app.route('/set_runner_speed', methods=['POST'])
def set_runner_speed_route():
    global runner_sleep_duration
    try:
        data = request.json; fps = float(data.get('fps', 20)) 
        if fps <= 0: fps = 0.1 
        runner_sleep_duration = 1.0 / fps
        tf.print(f"Runner speed set to {fps} FPS (sleep: {runner_sleep_duration:.4f}s)")
        return jsonify({"success": True, "message": f"Runner speed set to ~{fps:.1f} FPS."})
    except Exception as e: tf.print(f"Error setting speed: {e}"); return jsonify({"success":False,"message":"Invalid FPS."}),400
@app.route('/get_runner_status') 
def get_runner_status_route():
    is_loop_active_now = running_thread.is_alive() if running_thread else False
    hist_idx, total_hist_len = 0,0
    actual_fps_val = 0.0 # New: For actual FPS

    if current_nca_runner:
        with current_nca_runner._state_lock: # Access runner attributes safely
            hist_idx = current_nca_runner.history_index
            total_hist_len = len(current_nca_runner.history)
            actual_fps_val = current_nca_runner.actual_fps # Get actual_fps

    max_hist_step = max(0, total_hist_len -1 if total_hist_len > 0 else 0)
    status_msg = f"Runner Loop: {'Active' if is_loop_active_now else 'Paused/Stopped'}. Step: {hist_idx}/{max_hist_step}"
    if not current_nca_runner: status_msg = "Runner: No model loaded"

    return jsonify({
        "is_loop_active":is_loop_active_now,
        "preview_url":"/get_live_runner_preview", # This will be changed in Phase 2
        "history_step":hist_idx,
        "total_history":total_hist_len,
        "status_message":status_msg,
        "current_fps":1.0/runner_sleep_duration if runner_sleep_duration > 0 else "Max", # Target FPS
        "actual_fps": f"{actual_fps_val:.1f}" # New: Actual FPS
    })
@app.route('/get_live_runner_preview') 
def get_live_runner_preview_route():
    preview_state_to_show = None
    # Runner preview should use DRAW_CANVAS_DISPLAY_SIZE for its placeholder and zoom calculations
    default_w, default_h = DRAW_CANVAS_DISPLAY_SIZE, DRAW_CANVAS_DISPLAY_SIZE 
    zoom = 1
    if current_nca_runner :
        preview_state_to_show = current_nca_runner.get_current_state_for_display() 
        if preview_state_to_show is not None: 
            state_h = preview_state_to_show.shape[0] # Runner's actual state height
            if state_h > 0: zoom = DRAW_CANVAS_DISPLAY_SIZE // state_h
@app.route('/get_live_runner_raw_preview_data')
def get_live_runner_raw_preview_data_route():
    preview_state_to_show = None
    grid_h, grid_w = 0, 0

    if current_nca_runner:
        # Make sure to acquire the lock if get_current_state_for_display doesn't internally
        # current_nca_runner.get_current_state_for_display() already uses a lock
        preview_state_to_show = current_nca_runner.get_current_state_for_display()
        if preview_state_to_show is not None:
            grid_h, grid_w = preview_state_to_show.shape[0], preview_state_to_show.shape[1]

    if preview_state_to_show is None:
        return jsonify({"success": False, "message": "No runner state available", "height": 0, "width": 0, "pixels": []})

    # Assuming preview_state_to_show is [H, W, CHANNEL_N] (float32, 0.0-1.0)
    # and channels 0,1,2,3 are R,G,B,A for display
    if preview_state_to_show.shape[-1] < 4:
        # Should not happen if CHANNEL_N is sufficient and state is structured for RGBA output
        return jsonify({"success": False, "message": "Insufficient channels in state for RGBA", "height": grid_h, "width": grid_w, "pixels": []})

    rgba_data = preview_state_to_show[..., :4]  # Extract RGBA channels

    # Ensure alpha is applied if your rendering expects pre-multiplied alpha,
    # or if your RGB values are independent of alpha and you want standard alpha blending.
    # For direct ImageData, non-premultiplied is standard.
    # rgb_channels = rgba_data[..., :3]
    # alpha_channel = np.clip(rgba_data[..., 3:4], 0, 1)
    # display_rgb = rgb_channels # If not premultiplying: display_rgb = rgb_channels * alpha_channel + (1.0 - alpha_channel) * background_color
    # display_rgba = np.concatenate((display_rgb, alpha_channel), axis=-1)
    
    # For direct RGBA values (0-1) into Uint8ClampedArray (0-255)
    display_rgba_uint8 = np.uint8(np.clip(rgba_data, 0, 1) * 255)
    
    pixel_list = display_rgba_uint8.flatten().tolist()

    return jsonify({
        "success": True,
        "height": grid_h,
        "width": grid_w,
        "pixels": pixel_list
    })

# The old /get_live_runner_preview route can remain as is, or be deprecated later.
# For now, it's not actively used by the optimized runner.

@app.route('/runner_action', methods=['POST'])
def runner_action_route():
    response_msg="Action processed."; action_success=True; json_response_data={}
    with run_thread_lock:
        if not current_nca_runner: return jsonify({"success":False,"message":"Runner not active."}),400
        data=request.json; action_type=data.get('action')
        if action_type == 'rewind': _,idx=current_nca_runner.rewind();response_msg=f"Rewound to step {idx}."
        elif action_type == 'skip_forward': _,idx=current_nca_runner.fast_forward();response_msg=f"Skipped to step {idx}."
        elif action_type == 'modify_area':
            try:
                current_nca_runner.modify_area(data.get('tool_mode','erase'),float(data.get('norm_x')),float(data.get('norm_y')),
                                           float(data.get('brush_size_norm',0.05)),int(data.get('canvas_render_width')),
                                           int(data.get('canvas_render_height')),data.get('draw_color_hex','#000000'))
                response_msg=f"Area {data.get('tool_mode','modified')}d."
            except Exception as e:tf.print(f"Err modify_area:{e}\n{traceback.format_exc()}");action_success=False;response_msg=f"Err: {e}"
        elif action_type=='reset_runner':current_nca_runner.reset_state();response_msg="State reset."
        else: action_success=False;response_msg="Unknown action."
        hist_idx_after,total_hist_after=0,0
        if current_nca_runner:
            with current_nca_runner._state_lock: hist_idx_after=current_nca_runner.history_index;total_hist_after=len(current_nca_runner.history)
        json_response_data={"preview_url":"/get_live_runner_preview","history_step":hist_idx_after,"total_history":total_hist_after}
    json_response_data["success"]=action_success;json_response_data["message"]=response_msg
    return jsonify(json_response_data)


if __name__ == '__main__':
    tf.print(f"TensorFlow Version: {tf.__version__}")
    tf.print(f"Is TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
    tf.print(f"Is GPU available (tf.test.is_gpu_available): {tf.test.is_gpu_available()}")

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu_dev in physical_devices: tf.config.experimental.set_memory_growth(gpu_dev, True)
            tf.print(f"Found GPUs: {physical_devices}, memory growth enabled.")
            tf.print(f"Logical GPUs: {tf.config.experimental.list_logical_devices('GPU')}")
        except RuntimeError as e:
            tf.print(f"GPU Memory Growth Error: {e}")
    else:
        tf.print("No GPUs found by tf.config.experimental.list_physical_devices('GPU'). Running on CPU.")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)