# app.py
"""Flask Web Application for Neural Cellular Automata."""

import os
import io
import time
import threading
import traceback 
import numpy as np
import tensorflow as tf 
from flask import Flask, render_template, request, jsonify, send_file 
from werkzeug.utils import secure_filename
import PIL.Image 
import base64 # For handling data URL from drawn pattern

from nca_globals import (TARGET_SIZE, TARGET_PADDING, DEFAULT_FIRE_RATE, 
                         DEFAULT_BATCH_SIZE, DEFAULT_POOL_SIZE, CHANNEL_N,
                         DEFAULT_RUNNER_SLEEP_DURATION, DRAW_CANVAS_DISPLAY_SIZE) # Added
from nca_utils import load_emoji, load_image_from_file, np2pil, to_rgb, get_model_summary, format_training_time
from nca_model import CAModel
from nca_trainer import NCATrainer
from nca_runner import NCARunner


# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = os.urandom(24) 
UPLOAD_FOLDER = 'uploads' # For user-uploaded files and drawn patterns
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# --- Global state & Threading Primitives ---
current_nca_trainer = None
trainer_target_image_rgba = None # This will hold the RGBA numpy array for the trainer
trainer_target_source_kind = None # "file" or "drawn"
trainer_padded_target_shape = None 

current_nca_runner = None
runner_sleep_duration = DEFAULT_RUNNER_SLEEP_DURATION

training_thread = None
stop_training_event = threading.Event()
train_thread_lock = threading.Lock() 

running_thread = None
stop_running_event = threading.Event()
run_thread_lock = threading.Lock() 

# --- Helper Functions --- (get_preview_image_response, ensure_trainer_stopped, ensure_runner_stopped remain same)
def get_preview_image_response(state_array_or_rgba, zoom_factor=4, default_width_px=256, default_height_px=256):
    # ... (same as previous) ...
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
    # ... (same as previous) ...
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
    # ... (same as previous) ...
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
    global trainer_target_image_rgba, trainer_padded_target_shape, trainer_target_source_kind
    try:
        data = request.json
        data_url = data.get('image_data_url')
        if not data_url or not data_url.startswith('data:image/png;base64,'):
            return jsonify({"success": False, "message": "Invalid image data URL."}), 400

        header, encoded = data_url.split(',', 1)
        image_data = base64.b64decode(encoded)
        
        # Use PIL to open image from bytes, then convert to RGBA numpy array
        # This ensures it's handled like other images for consistency
        img_pil = PIL.Image.open(io.BytesIO(image_data)).convert("RGBA")

        # Resize the drawn PIL image to TARGET_SIZE before converting to np array and premultiplying alpha
        # This ensures the internal target is always TARGET_SIZExTARGET_SIZE
        original_size = img_pil.size
        if original_size[0] == 0 or original_size[1] == 0:
             raise ValueError("Drawn image has zero dimension before resize.")
        
        # Maintain aspect ratio when resizing to TARGET_SIZE bounding box
        ratio = min(TARGET_SIZE / original_size[0], TARGET_SIZE / original_size[1])
        new_size = (max(1, int(original_size[0] * ratio)), max(1, int(original_size[1] * ratio)))
        img_pil_resized = img_pil.resize(new_size, PIL.Image.Resampling.LANCZOS if hasattr(PIL.Image, 'Resampling') else PIL.Image.ANTIALIAS)

        # Create a new square image (TARGET_SIZE x TARGET_SIZE) and paste resized image onto it
        square_img_pil = PIL.Image.new("RGBA", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0, 0)) # Transparent background
        upper_left = ((TARGET_SIZE - new_size[0]) // 2, (TARGET_SIZE - new_size[1]) // 2)
        square_img_pil.paste(img_pil_resized, upper_left)

        # Convert to numpy array and premultiply alpha
        img_np = np.float32(square_img_pil) / 255.0
        img_np[..., :3] *= img_np[..., 3:] # Pre-multiply alpha
        
        trainer_target_image_rgba = img_np 
        trainer_target_source_kind = "drawn"
        
        p = TARGET_PADDING
        padded_target = tf.pad(trainer_target_image_rgba, [(p, p), (p, p), (0, 0)])
        trainer_padded_target_shape = padded_target.shape
        
        tf.print(f"Trainer target from DRAWN pattern loaded, internal size: {trainer_target_image_rgba.shape}, padded shape: {trainer_padded_target_shape}")
        return jsonify({"success": True, "message": "Drawn pattern set as trainer target."})
    except Exception as e_detail:
        tf.print(f"Error in /upload_drawn_pattern_target: {e_detail}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": f"Error processing drawn pattern: {str(e_detail)}"}), 500


@app.route('/load_target_from_file', methods=['POST']) # Specific endpoint for file upload
def load_target_from_file_route(): 
    global trainer_target_image_rgba, trainer_padded_target_shape, trainer_target_source_kind
    try:
        if 'image_file' not in request.files or request.files['image_file'].filename == '':
            return jsonify({"success": False, "message": "No file provided for trainer target."}), 400
            
        file = request.files['image_file']
        filename = secure_filename(file.filename)
        # This function already handles resize to TARGET_SIZE and premultiply alpha
        trainer_target_image_rgba = load_image_from_file(file.stream, max_size=TARGET_SIZE) 
        trainer_target_source_kind = "file"
        message = f"Trainer Target: File '{filename}' loaded."
        
        p = TARGET_PADDING
        padded_target = tf.pad(trainer_target_image_rgba, [(p, p), (p, p), (0, 0)])
        trainer_padded_target_shape = padded_target.shape 
        
        tf.print(f"Trainer target from FILE loaded, internal size {trainer_target_image_rgba.shape}, padded shape (H,W,4): {trainer_padded_target_shape}")
        return jsonify({"success": True, "message": message, 
                        "target_height": padded_target.shape[0], 
                        "target_width": padded_target.shape[1]})
    except Exception as e_detail:
        tf.print(f"Error in /load_target_from_file: {e_detail}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": f"Error loading trainer target from file: {str(e_detail)}"}), 500

@app.route('/get_trainer_target_preview') 
def get_trainer_target_preview_route():
    if trainer_target_image_rgba is None: # If no target set yet (neither file nor drawn)
        # For drawing, the client side canvas itself is the preview until confirmed.
        # This endpoint provides a preview of the *processed* target once loaded.
        # So, if no target is loaded yet, send a generic placeholder.
        # The display size for the preview image is DRAW_CANVAS_DISPLAY_SIZE
        return get_preview_image_response(None, default_width_px=DRAW_CANVAS_DISPLAY_SIZE, default_height_px=DRAW_CANVAS_DISPLAY_SIZE) 
    
    # If a target exists, pad it and send. Zoom factor will be applied by client based on its display size if needed.
    # Here, we send the padded target, and np2pil will scale it by zoom_factor=4 by default.
    # The client `<img>` tag for preview should have dimensions that match this.
    p = TARGET_PADDING
    padded_target_np = tf.pad(trainer_target_image_rgba, [(p,p),(p,p),(0,0)]).numpy()
    # The zoom factor in get_preview_image_response should align with client display.
    # If client displays at DRAW_CANVAS_DISPLAY_SIZE, and padded target is e.g. 72x72,
    # zoom_factor would be DRAW_CANVAS_DISPLAY_SIZE / 72.
    # Let's make zoom_factor dynamic or assume client handles final display scaling.
    # For now, keep default zoom, client CSS will handle final img size.
    zoom = DRAW_CANVAS_DISPLAY_SIZE // (TARGET_SIZE + 2 * TARGET_PADDING) if (TARGET_SIZE + 2 * TARGET_PADDING) > 0 else 1
    return get_preview_image_response(padded_target_np, zoom_factor=max(1, zoom))


@app.route('/initialize_trainer', methods=['POST']) 
def initialize_trainer_route():
    global current_nca_trainer, trainer_target_image_rgba
    if trainer_target_image_rgba is None: # This must be set by file upload or drawn pattern confirmation
        return jsonify({"success": False, "message": "Please load or draw & confirm a target pattern for the trainer first."}), 400
    try:
        with train_thread_lock: 
            ensure_trainer_stopped() 
            data = request.json
            config = {
                "fire_rate": float(data.get("fire_rate", DEFAULT_FIRE_RATE)),
                "batch_size": int(data.get("batch_size", DEFAULT_BATCH_SIZE)),
                "pool_size": int(data.get("pool_size", DEFAULT_POOL_SIZE)),
                "experiment_type": data.get("experiment_type", "Growing"),
                "target_padding": TARGET_PADDING,
                "learning_rate": float(data.get("learning_rate", 2e-3)) 
            }
            if config["experiment_type"] == "Regenerating":
                config["damage_n"] = int(data.get("damage_n", 3))
            else:
                config["damage_n"] = 0
            current_nca_trainer = NCATrainer(target_img_rgba=trainer_target_image_rgba, config=config)
            tf.print("NCATrainer initialized with current target.")
        
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

# ... (training_loop_task_function, start_training, stop_training, get_training_status, 
#      get_live_trainer_preview, save_trainer_model routes - largely unchanged from previous version)
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
        training_thread = threading.Thread(target=training_loop_task_function, daemon=True)
        training_thread.start()
        tf.print("Training started via /start_training.")
    return jsonify({"success": True, "message": "Training started."})
@app.route('/stop_training', methods=['POST'])
def stop_training_route():
    with train_thread_lock: 
        ensure_trainer_stopped() 
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
    return jsonify({
        "step": status_data["step"],
        "loss": status_data["loss"],
        "log_loss": status_data["log_loss"],
        "training_time": format_training_time(status_data["training_time_seconds"]),
        "is_training": is_currently_training,
        "status_message": f"Step: {status_data['step']}, Loss: {status_data['loss']}",
        "preview_url": "/get_live_trainer_preview" 
    })
@app.route('/get_live_trainer_preview') 
def get_live_trainer_preview_route():
    preview_state_to_show = None
    # Use DRAW_CANVAS_DISPLAY_SIZE for placeholder consistency
    default_w, default_h = DRAW_CANVAS_DISPLAY_SIZE, DRAW_CANVAS_DISPLAY_SIZE 
    zoom = 1
    with train_thread_lock: 
        if current_nca_trainer: 
            if current_nca_trainer.last_preview_state is not None:
                preview_state_to_show = current_nca_trainer.last_preview_state 
            elif current_nca_trainer.pool and len(current_nca_trainer.pool) > 0:
                preview_state_to_show = current_nca_trainer.pool.x[0].copy() 
            
            if trainer_padded_target_shape and preview_state_to_show is not None: # if actual state exists
                 state_h_padded = preview_state_to_show.shape[0]
                 if state_h_padded > 0 : zoom = DRAW_CANVAS_DISPLAY_SIZE // state_h_padded
        
    return get_preview_image_response(preview_state_to_show, zoom_factor=max(1, zoom), default_width_px=default_w, default_height_px=default_h)
@app.route('/save_trainer_model', methods=['POST']) 
def save_trainer_model_route():
    ca_model_to_save = None; filename_base = "nca_model"
    with train_thread_lock:
        if current_nca_trainer and current_nca_trainer.ca:
            ca_model_to_save = current_nca_trainer.ca
            exp_type = current_nca_trainer.config.get('experiment_type','generic')
            filename_base = f"trained_{exp_type}_{current_nca_trainer.current_step}steps"
        else: return jsonify({"success": False, "message": "No active NCA trainer instance to save."}), 400
    try:
        default_filename = f"{filename_base}_{int(time.time())}.h5"
        save_path = os.path.join(app.config['MODEL_FOLDER'], default_filename)
        ca_model_to_save.save_weights(save_path)
        tf.print(f"Trainer model saved to {save_path}")
        return jsonify({"success": True, "message": f"Trainer model saved: {default_filename}."})
    except Exception as e: tf.print(f"Error save_trainer_model: {e}\n{traceback.format_exc()}"); return jsonify({"success": False, "message": f"Error: {e}"}),500

@app.route('/load_current_training_model_for_runner', methods=['POST'])
def load_current_training_model_for_runner_route():
    global current_nca_runner, current_nca_trainer, trainer_padded_target_shape, runner_sleep_duration
    
    with train_thread_lock: # Access trainer safely
        if not current_nca_trainer or not current_nca_trainer.ca:
            return jsonify({"success": False, "message": "No active training model available to load into runner."}), 400
        
        # Get weights from the current trainer's model
        try:
            trainer_model_weights = current_nca_trainer.ca.get_weights()
            trainer_fire_rate = current_nca_trainer.ca.fire_rate # Get fire_rate from trainer's model
        except Exception as e:
            tf.print(f"Error getting weights from trainer model: {e}\n{traceback.format_exc()}")
            return jsonify({"success": False, "message": f"Error accessing trainer model: {str(e)}"}), 500

    # Now setup the runner (outside trainer lock, but inside runner lock)
    with run_thread_lock:
        ensure_runner_stopped()
        current_nca_runner = None 
        message = "Runner: Loaded current state of training model."
        try:
            # Create a new CAModel instance for the runner with the trainer's fire_rate
            runner_ca_model = CAModel(channel_n=CHANNEL_N, fire_rate=trainer_fire_rate)
            # Build the runner's model before setting weights (e.g., by a dummy call if not done in init)
            # The CAModel __init__ already does a dummy call, so it should be built.
            runner_ca_model.set_weights(trainer_model_weights)
            tf.print(f"Runner: Weights from trainer model set to new runner CAModel instance (Fire rate: {trainer_fire_rate}).")
            
            initial_runner_h, initial_runner_w = TARGET_SIZE + 2*TARGET_PADDING, TARGET_SIZE + 2*TARGET_PADDING
            if trainer_padded_target_shape: # Use H,W from current trainer target
                initial_runner_h, initial_runner_w = trainer_padded_target_shape[0], trainer_padded_target_shape[1]
                message += f" Grid based on current trainer target dimensions ({initial_runner_h}x{initial_runner_w})."
            else:
                message += f" Grid based on default dimensions ({initial_runner_h}x{initial_runner_w})."
            
            initial_runner_shape = (initial_runner_h, initial_runner_w, CHANNEL_N)
            current_nca_runner = NCARunner(ca_model_instance=runner_ca_model, 
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


@app.route('/load_model_for_runner', methods=['POST']) 
def load_model_for_runner_route():
    # ... (largely same, ensure runner_sleep_duration reset, using run_thread_lock correctly) ...
    global current_nca_runner, trainer_padded_target_shape, runner_sleep_duration
    with run_thread_lock: 
        ensure_runner_stopped() 
        current_nca_runner = None 
        model_file_path = None; message = ""
        if 'model_file' in request.files and request.files['model_file'].filename != '':
            file = request.files['model_file']
            filename = secure_filename(file.filename)
            if filename.endswith(".h5"):
                model_file_path = os.path.join(app.config['MODEL_FOLDER'], filename) 
                file.save(model_file_path)
                message = f"Runner: Uploaded model '{filename}' loaded."
            else: return jsonify({"success": False, "message": "Invalid model file type (.h5 required)."}), 400
        else: 
            h5_files = [f for f in os.listdir(app.config['MODEL_FOLDER']) if f.endswith('.h5')]
            if not h5_files: return jsonify({"success": False, "message": "No .h5 models in 'models' folder."}), 404
            h5_files.sort(key=lambda f: os.path.getmtime(os.path.join(app.config['MODEL_FOLDER'], f)), reverse=True)
            model_file_path = os.path.join(app.config['MODEL_FOLDER'], h5_files[0])
            message = f"Runner: Loaded latest model '{h5_files[0]}' from server."
        try:
            loaded_ca_model_for_runner = CAModel(channel_n=CHANNEL_N, fire_rate=DEFAULT_FIRE_RATE) 
            loaded_ca_model_for_runner.load_weights(model_file_path)
            tf.print(f"Runner: Weights loaded from {model_file_path}")
            
            initial_runner_h, initial_runner_w = TARGET_SIZE + 2 * TARGET_PADDING, TARGET_SIZE + 2 * TARGET_PADDING
            if trainer_padded_target_shape: 
                initial_runner_h, initial_runner_w = trainer_padded_target_shape[0], trainer_padded_target_shape[1]
                message += f" Grid based on trainer target ({initial_runner_h}x{initial_runner_w})."
            else: message += f" Grid based on default ({initial_runner_h}x{initial_runner_w})."
            
            initial_runner_shape_to_use = (initial_runner_h, initial_runner_w, CHANNEL_N)
            current_nca_runner = NCARunner(ca_model_instance=loaded_ca_model_for_runner, 
                                           initial_state_shape_tuple=initial_runner_shape_to_use)
            runner_sleep_duration = DEFAULT_RUNNER_SLEEP_DURATION
            model_summary_str = get_model_summary(current_nca_runner.ca) 
            return jsonify({"success": True, "message": message, "model_summary": model_summary_str, "runner_preview_url": "/get_live_runner_preview" })
        except Exception as e: tf.print(f"Error load_model_for_runner: {e}\n{traceback.format_exc()}"); current_nca_runner=None; return jsonify({"success":False,"message":f"Error: {e}"}),500

# ... (running_loop_task_function, start_running, stop_running, set_runner_speed, 
#      get_runner_status, get_live_runner_preview, runner_action routes - largely unchanged)
def running_loop_task_function(): 
    global runner_sleep_duration 
    tf.print("Runner thread started.")
    while not stop_running_event.is_set():
        loop_start_time = time.perf_counter()
        try:
            # The run_thread_lock is acquired by the caller of this loop (start_running)
            # or by individual actions. NCARunner methods are internally locked.
            # However, to prevent issues if current_nca_runner is set to None while loop is active:
            with run_thread_lock: # Added lock here as well for safety when accessing current_nca_runner
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
    if current_nca_runner: 
        with current_nca_runner._state_lock: 
            hist_idx = current_nca_runner.history_index
            total_hist_len = len(current_nca_runner.history)
    max_hist_step = max(0, total_hist_len -1 if total_hist_len > 0 else 0)
    status_msg = f"Runner Loop: {'Active' if is_loop_active_now else 'Paused/Stopped'}. Step: {hist_idx}/{max_hist_step}"
    if not current_nca_runner: status_msg = "Runner: No model loaded"
    return jsonify({"is_loop_active":is_loop_active_now, "preview_url":"/get_live_runner_preview", "history_step":hist_idx,
                    "total_history":total_hist_len, "status_message":status_msg, 
                    "current_fps":1.0/runner_sleep_duration if runner_sleep_duration > 0 else "Max"})

@app.route('/get_live_runner_preview') 
def get_live_runner_preview_route():
    preview_state_to_show = None
    default_w, default_h = DRAW_CANVAS_DISPLAY_SIZE, DRAW_CANVAS_DISPLAY_SIZE 
    zoom = 1
    if current_nca_runner :
        preview_state_to_show = current_nca_runner.get_current_state_for_display() 
        if preview_state_to_show is not None: 
            state_h = preview_state_to_show.shape[0]
            if state_h > 0: zoom = DRAW_CANVAS_DISPLAY_SIZE // state_h
    return get_preview_image_response(preview_state_to_show, zoom_factor=max(1, zoom), default_width_px=default_w, default_height_px=default_h)

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
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu_dev in physical_devices: tf.config.experimental.set_memory_growth(gpu_dev, True)
            tf.print(f"Found GPUs: {physical_devices}, memory growth enabled.")
        except RuntimeError as e: tf.print(f"GPU Memory Growth Error: {e}")
    else: tf.print("No GPUs found. Running on CPU.")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False) 