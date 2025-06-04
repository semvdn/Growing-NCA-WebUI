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

from nca_globals import (TARGET_SIZE, TARGET_PADDING, DEFAULT_FIRE_RATE, 
                         DEFAULT_BATCH_SIZE, DEFAULT_POOL_SIZE, CHANNEL_N,
                         DEFAULT_RUNNER_SLEEP_DURATION) # Added
from nca_utils import load_emoji, load_image_from_file, np2pil, to_rgb, get_model_summary, format_training_time
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
trainer_target_image_rgba = None 
trainer_padded_target_shape = None 

current_nca_runner = None
runner_sleep_duration = DEFAULT_RUNNER_SLEEP_DURATION # For FPS control

training_thread = None
stop_training_event = threading.Event()
train_thread_lock = threading.Lock() 

running_thread = None
stop_running_event = threading.Event()
run_thread_lock = threading.Lock() # This lock protects current_nca_runner instance and its methods

# --- Helper Functions (get_preview_image_response, ensure_trainer_stopped, ensure_runner_stopped remain same as previous) ---
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
    # Call this WITHOUT the train_thread_lock if the caller already holds it
    if training_thread and training_thread.is_alive():
        tf.print("Signalling training thread to stop...")
        stop_training_event.set()
        training_thread.join(timeout=7) 
        if training_thread.is_alive():
            tf.print("Warning: Training thread did not stop in time after signal.")
        training_thread = None
    stop_training_event.clear() 

def ensure_runner_stopped(): # Call this WITHOUT run_thread_lock if caller holds it
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

# ... (load_target_for_trainer, get_trainer_target_preview, initialize_trainer, training_loop_task_function,
#      start_training, stop_training, get_training_status, get_live_trainer_preview, save_trainer_model
#      routes remain largely the same as the previous fixed version, ensuring locks are used appropriately)
@app.route('/load_target', methods=['POST'])
def load_target_for_trainer(): 
    global trainer_target_image_rgba, trainer_padded_target_shape
    try:
        # ... (implementation from previous version) ...
        if 'emoji' in request.form and request.form['emoji']:
            emoji_char = request.form['emoji'][0] 
            trainer_target_image_rgba = load_emoji(emoji_char, max_size=TARGET_SIZE)
            message = f"Trainer Target: Emoji '{emoji_char}' loaded."
        elif 'image_file' in request.files and request.files['image_file'].filename != '':
            file = request.files['image_file']
            filename = secure_filename(file.filename)
            trainer_target_image_rgba = load_image_from_file(file.stream, max_size=TARGET_SIZE)
            message = f"Trainer Target: File '{filename}' loaded."
        else:
            return jsonify({"success": False, "message": "No emoji or file provided for trainer target."}), 400

        p = TARGET_PADDING
        padded_target = tf.pad(trainer_target_image_rgba, [(p, p), (p, p), (0, 0)])
        trainer_padded_target_shape = padded_target.shape 
        
        tf.print(f"Trainer target loaded, padded shape (H,W,4): {trainer_padded_target_shape}")
        return jsonify({"success": True, "message": message, 
                        "target_height": padded_target.shape[0], 
                        "target_width": padded_target.shape[1]})
    except Exception as e_detail:
        tf.print(f"Error in /load_target: {e_detail}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": f"Error loading trainer target: {str(e_detail)}"}), 500

@app.route('/get_trainer_target_preview') 
def get_trainer_target_preview_route():
    # ... (implementation from previous version) ...
    if trainer_target_image_rgba is None:
        return get_preview_image_response(None, default_width_px=TARGET_SIZE*4, default_height_px=TARGET_SIZE*4) 
    
    p = TARGET_PADDING
    padded_target_np = tf.pad(trainer_target_image_rgba, [(p,p),(p,p),(0,0)]).numpy()
    return get_preview_image_response(padded_target_np)

@app.route('/initialize_trainer', methods=['POST']) 
def initialize_trainer_route():
    global current_nca_trainer, trainer_target_image_rgba
    # ... (implementation from previous version, ensure ensure_trainer_stopped is called correctly) ...
    if trainer_target_image_rgba is None:
        return jsonify({"success": False, "message": "Please load a target image for the trainer first."}), 400
    try:
        with train_thread_lock: # Lock before calling ensure_trainer_stopped
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
            tf.print("NCATrainer initialized.")
        
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


def training_loop_task_function(): 
    # ... (implementation from previous version) ...
    tf.print("Training thread started.")
    while not stop_training_event.is_set():
        try:
            # Lock is acquired inside the loop for each step or check
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
    # ... (implementation from previous version) ...
    with train_thread_lock: 
        if not current_nca_trainer:
            return jsonify({"success": False, "message": "Initialize Trainer first."}), 400
        
        if training_thread and training_thread.is_alive():
            return jsonify({"success": False, "message": "Training already running."}), 400

        stop_training_event.clear() # Clear before starting!
        training_thread = threading.Thread(target=training_loop_task_function, daemon=True)
        training_thread.start()
        tf.print("Training started via /start_training.")
    return jsonify({"success": True, "message": "Training started."})


@app.route('/stop_training', methods=['POST'])
def stop_training_route():
    # ... (use the modified helper that only stops thread) ...
    with train_thread_lock: # Acquire lock before calling the helper
        ensure_trainer_stopped() # This now only stops the thread and clears event
    tf.print("Training stopped via /stop_training. Trainer instance preserved for saving.")
    return jsonify({"success": True, "message": "Training stopped. Trainer state preserved for saving."})

@app.route('/get_training_status')
def get_training_status_route():
    # ... (implementation from previous version) ...
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
    # ... (implementation from previous version) ...
    preview_state_to_show = None
    default_w, default_h = TARGET_SIZE * 4, TARGET_SIZE * 4 
    with train_thread_lock: 
        if current_nca_trainer: 
            if current_nca_trainer.last_preview_state is not None:
                preview_state_to_show = current_nca_trainer.last_preview_state 
            elif current_nca_trainer.pool and len(current_nca_trainer.pool) > 0:
                preview_state_to_show = current_nca_trainer.pool.x[0].copy() 
            if trainer_padded_target_shape:
                default_h = trainer_padded_target_shape[0] * 4
                default_w = trainer_padded_target_shape[1] * 4
    return get_preview_image_response(preview_state_to_show, default_width_px=default_w, default_height_px=default_h)

@app.route('/save_trainer_model', methods=['POST']) 
def save_trainer_model_route():
    # ... (implementation from previous version) ...
    ca_model_to_save = None
    filename_base = "nca_model"
    with train_thread_lock:
        if current_nca_trainer and current_nca_trainer.ca:
            ca_model_to_save = current_nca_trainer.ca
            exp_type = current_nca_trainer.config.get('experiment_type','generic')
            filename_base = f"trained_{exp_type}_{current_nca_trainer.current_step}steps"
        else:
            return jsonify({"success": False, "message": "No active NCA trainer instance to save model from."}), 400
    try:
        default_filename = f"{filename_base}_{int(time.time())}.h5"
        save_path = os.path.join(app.config['MODEL_FOLDER'], default_filename)
        ca_model_to_save.save_weights(save_path)
        tf.print(f"Trainer model saved to {save_path}")
        return jsonify({"success": True, "message": f"Trainer model saved on server as {default_filename}."})
    except Exception as e_detail:
        tf.print(f"Error in /save_trainer_model: {e_detail}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": f"Error saving trainer model: {str(e_detail)}"}), 500

@app.route('/load_model_for_runner', methods=['POST']) 
def load_model_for_runner_route():
    global current_nca_runner, trainer_padded_target_shape, runner_sleep_duration
    
    with run_thread_lock: # Ensure exclusive access for runner setup
        ensure_runner_stopped() # Stop existing runner thread if any
        current_nca_runner = None # Clear previous runner instance

        model_file_path = None
        message = ""
        if 'model_file' in request.files and request.files['model_file'].filename != '':
            file = request.files['model_file']
            filename = secure_filename(file.filename)
            if filename.endswith(".h5"):
                model_file_path = os.path.join(app.config['MODEL_FOLDER'], filename) 
                file.save(model_file_path)
                message = f"Runner: Uploaded model '{filename}' loaded."
            else:
                return jsonify({"success": False, "message": "Invalid model file type. Must be .h5"}), 400
        else: 
            h5_files = [f for f in os.listdir(app.config['MODEL_FOLDER']) if f.endswith('.h5')]
            if not h5_files:
                return jsonify({"success": False, "message": "No .h5 model file uploaded or found in 'models' folder."}), 404
            h5_files.sort(key=lambda f: os.path.getmtime(os.path.join(app.config['MODEL_FOLDER'], f)), reverse=True)
            model_file_path = os.path.join(app.config['MODEL_FOLDER'], h5_files[0])
            message = f"Runner: Loaded latest model '{h5_files[0]}' from server."

        try:
            loaded_ca_model_for_runner = CAModel(channel_n=CHANNEL_N, fire_rate=DEFAULT_FIRE_RATE) 
            loaded_ca_model_for_runner.load_weights(model_file_path)
            tf.print(f"Runner: Weights loaded from {model_file_path}")
            
            initial_runner_h, initial_runner_w = TARGET_SIZE + 2 * TARGET_PADDING, TARGET_SIZE + 2 * TARGET_PADDING # Defaults
            if trainer_padded_target_shape: 
                initial_runner_h, initial_runner_w = trainer_padded_target_shape[0], trainer_padded_target_shape[1]
                message += f" Grid based on trainer target ({initial_runner_h}x{initial_runner_w})."
            else:
                message += f" Grid based on default ({initial_runner_h}x{initial_runner_w})."
            
            initial_runner_shape_to_use = (initial_runner_h, initial_runner_w, CHANNEL_N)
            tf.print(f"Runner will be initialized with state shape: {initial_runner_shape_to_use}")
            
            current_nca_runner = NCARunner(ca_model_instance=loaded_ca_model_for_runner, 
                                           initial_state_shape_tuple=initial_runner_shape_to_use)
            # NCARunner init calls reset_state internally, which acquires its own lock
            tf.print(f"NCARunner instance created.")
            runner_sleep_duration = DEFAULT_RUNNER_SLEEP_DURATION # Reset to default speed

            model_summary_str = get_model_summary(current_nca_runner.ca) 
            return jsonify({
                "success": True, "message": message, "model_summary": model_summary_str,
                "runner_preview_url": "/get_live_runner_preview" 
            })
        except Exception as e_detail:
            tf.print(f"Error in /load_model_for_runner: {e_detail}\n{traceback.format_exc()}")
            current_nca_runner = None # Ensure runner is cleared on error
            return jsonify({"success": False, "message": f"Error loading model for runner: {str(e_detail)}"}), 500


def running_loop_task_function(): 
    global runner_sleep_duration # Allow modification
    tf.print("Runner thread started.")
    while not stop_running_event.is_set():
        loop_start_time = time.perf_counter()
        try:
            with run_thread_lock: 
                if not current_nca_runner: 
                    tf.print("Runner gone, stopping runner loop.")
                    break 
                current_nca_runner.step() # This uses NCARunner's internal lock for state
            
            # Dynamic sleep based on runner_sleep_duration
            # Ensure sleep is not negative if processing took longer
            current_sleep = runner_sleep_duration - (time.perf_counter() - loop_start_time)
            if current_sleep > 0:
                 time.sleep(current_sleep)
            else: # If processing was too slow, yield CPU briefly
                 time.sleep(0.001)

        except Exception as e_detail:
            tf.print(f"Error in running loop: {e_detail}\n{traceback.format_exc()}")
            break 
    tf.print("Runner thread ended.")


@app.route('/start_running', methods=['POST'])
def start_running_loop_route(): 
    global running_thread
    with run_thread_lock: 
        if not current_nca_runner or not current_nca_runner.ca:
            return jsonify({"success": False, "message": "Load a model for the Runner first."}), 400
        
        if running_thread and running_thread.is_alive():
            return jsonify({"success": False, "message": "Runner loop already active."}), 400

        stop_running_event.clear() # CRITICAL: Clear event before starting thread
        running_thread = threading.Thread(target=running_loop_task_function, daemon=True)
        running_thread.start()
        tf.print("Runner loop started via /start_running.")
    
    return jsonify({"success": True, "message": "Runner loop started."})

@app.route('/stop_running', methods=['POST'])
def stop_running_loop_route(): 
    tf.print("/stop_running called")
    with run_thread_lock: # Acquire lock before calling helper
        ensure_runner_stopped() # This helper now just stops thread and clears event
    tf.print("Runner loop stop process initiated.")
    return jsonify({"success": True, "message": "Runner loop stopping process initiated. State and history preserved."})

@app.route('/set_runner_speed', methods=['POST'])
def set_runner_speed_route():
    global runner_sleep_duration
    try:
        data = request.json
        fps = float(data.get('fps', 20)) # Default to 20 FPS if not provided
        if fps <= 0: fps = 0.1 # Avoid zero or negative FPS leading to issues
        runner_sleep_duration = 1.0 / fps
        tf.print(f"Runner speed set to {fps} FPS (sleep: {runner_sleep_duration:.4f}s)")
        return jsonify({"success": True, "message": f"Runner speed set to ~{fps:.1f} FPS."})
    except Exception as e:
        tf.print(f"Error setting runner speed: {e}")
        return jsonify({"success": False, "message": "Invalid FPS value."}), 400


@app.route('/get_runner_status') 
def get_runner_status_route():
    # No need to lock here for read-only access to thread status and runner basic info
    # current_nca_runner's internal state access (history_index, len(history)) is protected by its own lock
    is_loop_active_now = running_thread.is_alive() if running_thread else False
    
    hist_idx = 0
    total_hist_len = 0
    if current_nca_runner: # Check if runner exists before accessing its attributes
        with current_nca_runner._state_lock: # Use runner's internal lock for its data
            hist_idx = current_nca_runner.history_index
            total_hist_len = len(current_nca_runner.history)
    
    max_hist_step = max(0, total_hist_len -1 if total_hist_len > 0 else 0)
    status_msg = f"Runner Loop: {'Active' if is_loop_active_now else 'Paused/Stopped'}. Step: {hist_idx}/{max_hist_step}"

    if not current_nca_runner:
         status_msg = "Runner: No model loaded"

    return jsonify({
        "is_loop_active": is_loop_active_now, 
        "preview_url": "/get_live_runner_preview",
        "history_step": hist_idx,
        "total_history": total_hist_len, 
        "status_message": status_msg,
        "current_fps": 1.0/runner_sleep_duration if runner_sleep_duration > 0 else "Max"
    })

@app.route('/get_live_runner_preview') 
def get_live_runner_preview_route():
    preview_state_to_show = None
    default_w, default_h = 256, 256 
    # Accessing current_nca_runner here without lock is generally okay for read if it's just checking existence.
    # The method get_current_state_for_display() itself is internally locked in NCARunner.
    if current_nca_runner :
        preview_state_to_show = current_nca_runner.get_current_state_for_display() 
        if preview_state_to_show is not None: 
            default_h = preview_state_to_show.shape[0] * 4
            default_w = preview_state_to_show.shape[1] * 4
    return get_preview_image_response(preview_state_to_show, default_width_px=default_w, default_height_px=default_h)


@app.route('/runner_action', methods=['POST'])
def runner_action_route():
    response_msg = "Runner action processed."
    action_success = True
    json_response_data = {}

    with run_thread_lock: # All actions on runner instance are serialized
        if not current_nca_runner:
            return jsonify({"success": False, "message": "Runner not active."}), 400

        data = request.json
        action_type = data.get('action')
        
        if action_type == 'rewind':
            # NCARunner.rewind() is internally locked
            _, current_hist_idx = current_nca_runner.rewind()
            response_msg = f"Runner: Rewound to step {current_hist_idx}."
        elif action_type == 'skip_forward':
            _, current_hist_idx = current_nca_runner.fast_forward()
            response_msg = f"Runner: Skipped to step {current_hist_idx}."
        elif action_type == 'modify_area': # Unified endpoint
            try:
                tool_mode = data.get('tool_mode', 'erase')
                norm_x_coord = float(data.get('norm_x'))
                norm_y_coord = float(data.get('norm_y'))
                norm_brush_factor = float(data.get('brush_size_norm', 0.05)) 
                canvas_width = int(data.get('canvas_render_width'))
                canvas_height = int(data.get('canvas_render_height'))
                draw_color = data.get('draw_color_hex', '#000000')

                # NCARunner.modify_area() is internally locked
                current_nca_runner.modify_area(
                    tool_mode, norm_x_coord, norm_y_coord, 
                    norm_brush_factor, canvas_width, canvas_height, draw_color
                )
                response_msg = f"Runner: Area {tool_mode}d."
            except Exception as e_detail:
                tf.print(f"Error in /runner_action modify_area: {e_detail}\n{traceback.format_exc()}")
                action_success = False
                response_msg = f"Error during {data.get('tool_mode', 'modification')}: {str(e_detail)}"
        elif action_type == 'reset_runner':
            # NCARunner.reset_state() is internally locked
            current_nca_runner.reset_state()
            response_msg = "Runner: State reset to seed."
        else:
            action_success = False
            response_msg = "Runner: Unknown action."
        
        # Get fresh history info after action (under runner's internal lock)
        hist_idx_after_action, total_hist_after_action = 0, 0
        if current_nca_runner:
            with current_nca_runner._state_lock:
                hist_idx_after_action = current_nca_runner.history_index
                total_hist_after_action = len(current_nca_runner.history)

        json_response_data = {
            "preview_url": "/get_live_runner_preview", 
            "history_step": hist_idx_after_action,
            "total_history": total_hist_after_action
        }
            
    json_response_data["success"] = action_success
    json_response_data["message"] = response_msg
    return jsonify(json_response_data)


if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu_dev in physical_devices:
                tf.config.experimental.set_memory_growth(gpu_dev, True)
            tf.print(f"Found GPUs: {physical_devices}, memory growth enabled.")
        except RuntimeError as e_runtime:
            tf.print(f"GPU Memory Growth Error: {e_runtime}")
    else:
        tf.print("No GPUs found by TensorFlow. Running on CPU.")
        
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False) 