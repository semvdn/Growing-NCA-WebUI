# app_routes_trainer.py
"""
Flask routes for the Trainer section of the NCA Web UI.

This module uses a Flask Blueprint to organize all routes related to creating,
training, saving, and loading NCA models. It interacts with the shared
application state defined in `app_state.py`.
"""
import base64
import io
import json
import os
import shutil
import threading
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np
import PIL.Image
import tensorflow as tf
from flask import Blueprint, current_app, jsonify, request
from flask.wrappers import Response
from werkzeug.utils import secure_filename

import app_state
from app_utils import ensure_trainer_stopped, export_model_for_tfjs
from nca_globals import (DEFAULT_BATCH_SIZE, DEFAULT_ENTROPY_ENABLED,
                         DEFAULT_ENTROPY_STRENGTH, DEFAULT_FIRE_RATE,
                         DEFAULT_POOL_SIZE, TARGET_PADDING, TARGET_SIZE)
from nca_trainer import NCATrainer
from nca_utils import (format_training_time, get_model_summary,
                       load_image_from_file, to_rgb)

trainer_bp = Blueprint('trainer', __name__)


@trainer_bp.route('/upload_drawn_pattern_target', methods=['POST'])
def upload_drawn_pattern_target_route() -> Response:
    """Receives a drawn pattern from the frontend canvas to use as a training target."""
    try:
        data = request.json
        data_url = data.get('image_data_url')
        drawn_image_name = data.get('drawn_image_name', 'drawn_pattern')

        if not data_url or not data_url.startswith('data:image/png;base64,'):
            return jsonify({"success": False, "message": "Invalid image data URL."}), 400

        header, encoded = data_url.split(',', 1)
        image_data = base64.b64decode(encoded)
        img_pil = PIL.Image.open(io.BytesIO(image_data)).convert("RGBA")
        app_state.original_drawn_image_pil = img_pil.copy()

        final_grid_dim = TARGET_SIZE + 2 * TARGET_PADDING
        ratio = min(final_grid_dim / img_pil.width, final_grid_dim / img_pil.height)
        new_size = (max(1, int(img_pil.width * ratio)), max(1, int(img_pil.height * ratio)))
        img_pil_resized = img_pil.resize(new_size, PIL.Image.Resampling.LANCZOS)

        final_target_pil = PIL.Image.new("RGBA", (final_grid_dim, final_grid_dim), (0, 0, 0, 0))
        upper_left = ((final_grid_dim - new_size[0]) // 2, (final_grid_dim - new_size[1]) // 2)
        final_target_pil.paste(img_pil_resized, upper_left)

        img_np = np.float32(final_target_pil) / 255.0
        img_np[..., :3] *= img_np[..., 3:]

        app_state.trainer_target_image_rgba = img_np
        app_state.trainer_target_source_kind = "drawn_defines_padded_grid"
        app_state.trainer_actual_target_shape = img_np.shape
        app_state.trainer_target_image_name = drawn_image_name
        app_state.trainer_target_image_loaded_or_drawn = "drawn"

        tf.print(f"Trainer target from DRAWN pattern. Final shape: {app_state.trainer_actual_target_shape}")
        return jsonify({"success": True, "message": "Drawn pattern set as target."})

    except Exception as e:
        tf.print(f"Error in /upload_drawn_pattern_target: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": f"Error processing drawn pattern: {e}"}), 500


@trainer_bp.route('/load_target_from_file', methods=['POST'])
def load_target_from_file_route() -> Response:
    """Handles file uploads to be used as a training target."""
    try:
        if 'image_file' not in request.files or request.files['image_file'].filename == '':
            return jsonify({"success": False, "message": "No file provided."}), 400

        file = request.files['image_file']
        filename = secure_filename(file.filename)
        
        # Save the uploaded file to the general uploads folder
        file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))

        full_grid_dim = TARGET_SIZE + 2 * TARGET_PADDING
        
        # Reload from the saved path to ensure consistency
        with open(os.path.join(current_app.config['UPLOAD_FOLDER'], filename), 'rb') as f_stream:
            loaded_rgba = load_image_from_file(f_stream, target_dim=full_grid_dim)

        app_state.trainer_target_image_name = filename
        app_state.trainer_target_image_loaded_or_drawn = "loaded"
        app_state.trainer_target_image_rgba = loaded_rgba
        app_state.trainer_target_source_kind = "file"
        app_state.trainer_actual_target_shape = app_state.trainer_target_image_rgba.shape

        tf.print(f"Trainer target from FILE loaded, grid shape {app_state.trainer_actual_target_shape}")
        return jsonify({
            "success": True, "message": f"File '{filename}' loaded as target.",
            "target_height": app_state.trainer_actual_target_shape[0],
            "target_width": app_state.trainer_actual_target_shape[1]
        })
    except Exception as e:
        tf.print(f"Error in /load_target_from_file: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": f"Error loading target from file: {e}"}), 500


@trainer_bp.route('/get_trainer_target_raw_preview_data')
def get_trainer_target_raw_preview_data_route() -> Response:
    """Provides the raw pixel data of the current trainer target image."""
    if app_state.trainer_target_image_rgba is None:
        return jsonify({"success": False, "message": "No trainer target available"}), 404

    rgba_data = app_state.trainer_target_image_rgba[..., :4]
    display_rgba_uint8 = np.uint8(np.clip(rgba_data, 0, 1) * 255)

    return jsonify({
        "success": True,
        "height": app_state.trainer_target_image_rgba.shape[0],
        "width": app_state.trainer_target_image_rgba.shape[1],
        "pixels": display_rgba_uint8.flatten().tolist()
    })


@trainer_bp.route('/initialize_trainer', methods=['POST'])
def initialize_trainer_route() -> Response:
    """Initializes an NCATrainer instance and makes the run self-contained."""
    if app_state.trainer_target_image_rgba is None:
        return jsonify({"success": False, "message": "Please set a target first."}), 400

    try:
        with app_state.train_thread_lock:
            ensure_trainer_stopped()

            run_id = f"{app_state.trainer_target_image_name.rsplit('.', 1)[0]}_{int(time.time())}"
            run_dir = os.path.join(current_app.config['MODEL_FOLDER'], run_id)
            os.makedirs(run_dir, exist_ok=True)
            app_state.current_training_run_id = run_id
            app_state.current_training_run_dir = run_dir
            tf.print(f"New training run directory: {run_dir}")

            if app_state.trainer_target_image_loaded_or_drawn == "drawn":
                pixelated_target_path = os.path.join(run_dir, "target_pixelated.png")
                img_pil_pixelated = PIL.Image.fromarray((app_state.trainer_target_image_rgba * 255).astype(np.uint8))
                img_pil_pixelated.save(pixelated_target_path)
                if app_state.original_drawn_image_pil:
                    original_drawn_path = os.path.join(run_dir, "target_original_drawing.png")
                    app_state.original_drawn_image_pil.save(original_drawn_path)
            
            elif app_state.trainer_target_image_loaded_or_drawn == "loaded":
                source_path = os.path.join(current_app.config['UPLOAD_FOLDER'], app_state.trainer_target_image_name)
                if os.path.exists(source_path):
                    shutil.copy(source_path, os.path.join(run_dir, app_state.trainer_target_image_name))
                    tf.print(f"Copied target '{app_state.trainer_target_image_name}' to run directory for reproducibility.")

            data = request.json
            config = {
                "fire_rate": float(data.get("fire_rate", DEFAULT_FIRE_RATE)),
                "batch_size": int(data.get("batch_size", DEFAULT_BATCH_SIZE)),
                "pool_size": int(data.get("pool_size", DEFAULT_POOL_SIZE)),
                "experiment_type": data.get("experiment_type", "Growing"),
                "learning_rate": float(data.get("learning_rate", 2e-3)),
                "target_source_kind": app_state.trainer_target_source_kind,
                "run_dir": run_dir,
                "enable_entropy": bool(data.get("enable_entropy", DEFAULT_ENTROPY_ENABLED)),
                "entropy_strength": float(data.get("entropy_strength", DEFAULT_ENTROPY_STRENGTH)),
                "damage_n": int(data.get("damage_n", 3)) if data.get("experiment_type") == "Regenerating" else 0
            }

            app_state.current_nca_trainer = NCATrainer(target_img_rgba_processed=app_state.trainer_target_image_rgba, config=config)
            app_state.trainer_actual_target_shape = app_state.current_nca_trainer.pad_target.shape
            tf.print(f"NCATrainer initialized. Operational target shape: {app_state.trainer_actual_target_shape}")

        model_summary_str = get_model_summary(app_state.current_nca_trainer.get_model())
        return jsonify({
            "success": True,
            "message": "NCA Trainer Initialized. Ready to train.",
            "model_summary": model_summary_str
        })
    except Exception as e:
        tf.print(f"Error in /initialize_trainer: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": f"Error initializing trainer: {e}"}), 500


def training_loop_task_function() -> None:
    """The function executed by the background training thread."""
    tf.print("Training thread started.")
    while not app_state.stop_training_event.is_set():
        try:
            with app_state.train_thread_lock:
                if not app_state.current_nca_trainer:
                    tf.print("Trainer gone, stopping training loop.")
                    break
                app_state.current_nca_trainer.run_training_step()
            time.sleep(0.01)
        except Exception as e:
            tf.print(f"Error in training loop: {e}\n{traceback.format_exc()}")
            break
    tf.print("Training thread ended.")


@trainer_bp.route('/start_training', methods=['POST'])
def start_training_route() -> Response:
    """Starts the background training thread."""
    with app_state.train_thread_lock:
        if not app_state.current_nca_trainer:
            return jsonify({"success": False, "message": "Initialize Trainer first."}), 400
        if app_state.training_thread and app_state.training_thread.is_alive():
            return jsonify({"success": False, "message": "Training already running."}), 400
        app_state.stop_training_event.clear()
        app_state.current_nca_trainer.resume_training_timer()
        app_state.training_thread = threading.Thread(target=training_loop_task_function, daemon=True)
        app_state.training_thread.start()
        tf.print("Training started via /start_training.")
    return jsonify({"success": True, "message": "Training started."})


@trainer_bp.route('/stop_training', methods=['POST'])
def stop_training_route() -> Response:
    """Stops the background training thread."""
    with app_state.train_thread_lock:
        ensure_trainer_stopped()
        if app_state.current_nca_trainer:
            app_state.current_nca_trainer.pause_training_timer()
    tf.print("Training stopped via /stop_training.")
    return jsonify({"success": True, "message": "Training stopped. State preserved."})


@trainer_bp.route('/get_training_status')
def get_training_status_route() -> Response:
    """Fetches the current status of the trainer."""
    with app_state.train_thread_lock:
        if not app_state.current_nca_trainer:
            return jsonify({"status_message": "Trainer Not Initialized", "is_training": False})

        status_data = app_state.current_nca_trainer.get_status()
        is_currently_training = app_state.training_thread.is_alive() if app_state.training_thread else False
        formatted_time = format_training_time(status_data["training_time_seconds"])
        status_msg = f"Step: {status_data['step']}, Loss: {status_data['loss']} (log10: {status_data['log_loss']}), Time: {formatted_time}"

    return jsonify({
        "is_training": is_currently_training,
        "status_message": status_msg,
        **status_data
    })


@trainer_bp.route('/get_live_trainer_raw_preview_data')
def get_live_trainer_raw_preview_data_route() -> Response:
    """Provides raw pixel data of the live trainer's current preview state."""
    preview_state = None
    with app_state.train_thread_lock:
        if app_state.current_nca_trainer:
            if app_state.current_nca_trainer.last_preview_state is not None:
                preview_state = app_state.current_nca_trainer.last_preview_state
            elif app_state.current_nca_trainer.pool and len(app_state.current_nca_trainer.pool) > 0:
                preview_state = app_state.current_nca_trainer.pool.x[0]

    if preview_state is None:
        return jsonify({"success": False, "message": "No trainer state available"}), 404

    rgba_data = to_rgb(tf.constant(preview_state[None, ...]))[0].numpy()
    display_rgba_uint8 = np.uint8(np.clip(rgba_data, 0, 1) * 255)
    h, w, _ = display_rgba_uint8.shape
    final_rgba = np.dstack((display_rgba_uint8, np.full((h, w), 255, dtype=np.uint8)))

    return jsonify({
        "success": True, "height": h, "width": w,
        "pixels": final_rgba.flatten().tolist()
    })


@trainer_bp.route('/save_trainer_model', methods=['POST'])
def save_trainer_model_route() -> Response:
    """Saves the trainer's model weights, metadata, and a TF.js graph model."""
    if not app_state.current_nca_trainer or not app_state.current_training_run_dir:
        return jsonify({"success": False, "message": "No active training run to save."}), 400

    with app_state.train_thread_lock:
        ca_model = app_state.current_nca_trainer.ca
        run_dir = app_state.current_training_run_dir
        step = app_state.current_nca_trainer.current_step
        checkpoint_name = f"checkpoint_step_{step}_{int(time.time())}"

        weights_filename = f"{checkpoint_name}.weights.h5"
        metadata_filename = f"{checkpoint_name}.json"
        tfjs_filename = f"{checkpoint_name}.tfjs.json"

        weights_path = os.path.join(run_dir, weights_filename)
        metadata_path = os.path.join(run_dir, metadata_filename)
        tfjs_path = os.path.join(run_dir, tfjs_filename)

        metadata = {
            "model_weights_file": weights_filename, "tfjs_graph_file": tfjs_filename,
            "trained_on_image": app_state.trainer_target_image_name,
            "image_source_kind": app_state.trainer_target_image_loaded_or_drawn,
            "training_steps": step,
            "experiment_type": app_state.current_nca_trainer.config.get('experiment_type', 'generic'),
            "final_loss": float(app_state.current_nca_trainer.last_loss) if app_state.current_nca_trainer.last_loss is not None else -1.0,
            "total_training_time_seconds": float(app_state.current_nca_trainer.get_status()["training_time_seconds"]),
            "save_timestamp": time.time(),
            "save_datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "run_id": app_state.current_training_run_id,
            "enable_entropy": ca_model.enable_entropy,
            "entropy_strength": ca_model.entropy_strength,
        }

    try:
        ca_model.save_weights(weights_path)
        tf.print(f"Trainer model weights saved to {weights_path}")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        tf.print(f"Custom metadata saved to {metadata_path}")
        export_model_for_tfjs(ca_model, tfjs_path)
        tf.print(f"TensorFlow.js model graph saved to {tfjs_path}")
        return jsonify({"success": True, "message": f"Checkpoint saved to '{app_state.current_training_run_id}'."})
    except Exception as e:
        tf.print(f"Error during model saving: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": f"Error saving model artifacts: {e}"}), 500


def _load_target_image_from_metadata(metadata: Dict[str, Any], model_file_path: str) -> Optional[np.ndarray]:
    """
    Attempts to reload the target image based on metadata from a saved model.
    This function modifies the global `app_state` as a side effect.
    """
    target_image_name = metadata.get("trained_on_image", "unknown_image")
    source_kind = metadata.get("image_source_kind", "unknown")
    run_dir = os.path.dirname(model_file_path)

    if source_kind == "drawn":
        pixelated_filename = metadata.get("pixelated_target_image_file", "target_pixelated.png")
        pixelated_path = os.path.join(run_dir, pixelated_filename)

        if not os.path.exists(pixelated_path):
            tf.print(f"WARNING: Pixelated target file '{pixelated_filename}' not found.")
            return None

        with open(pixelated_path, 'rb') as f:
            img_pil = PIL.Image.open(f).convert("RGBA")
            loaded_target_rgba = np.float32(img_pil) / 255.0
            loaded_target_rgba[..., :3] *= loaded_target_rgba[..., 3:]

        original_drawn_filename = metadata.get("original_drawn_image_file", "target_original_drawing.png")
        if original_drawn_filename:
            original_path = os.path.join(run_dir, original_drawn_filename)
            if os.path.exists(original_path):
                app_state.original_drawn_image_pil = PIL.Image.open(original_path).convert("RGBA")

        app_state.trainer_target_source_kind = "drawn_defines_padded_grid"
        app_state.trainer_target_image_loaded_or_drawn = "drawn"
        app_state.trainer_target_image_name = target_image_name
        return loaded_target_rgba

    elif source_kind == "loaded":
        target_path = os.path.join(run_dir, target_image_name)
        if not os.path.exists(target_path):
            tf.print(f"INFO: Target not found in run directory, checking global /uploads folder.")
            target_path = os.path.join(current_app.config['UPLOAD_FOLDER'], target_image_name)
            if not os.path.exists(target_path):
                tf.print(f"WARNING: Original uploaded target '{target_image_name}' not found anywhere.")
                return None

        with open(target_path, 'rb') as f:
            loaded_target_rgba = load_image_from_file(f, target_dim=TARGET_SIZE + 2 * TARGET_PADDING)

        app_state.trainer_target_source_kind = "file"
        app_state.trainer_target_image_loaded_or_drawn = "loaded"
        app_state.trainer_target_image_name = target_image_name
        return loaded_target_rgba

    return None


@trainer_bp.route('/load_trainer_model', methods=['POST'])
def load_trainer_model_route() -> Response:
    """Loads a saved model and its training context to resume training."""
    with app_state.train_thread_lock:
        ensure_trainer_stopped()
        app_state.current_nca_trainer = None

        if 'model_file' not in request.files or not request.files['model_file'].filename:
            return jsonify({"success": False, "message": "No model file provided."}), 400

        file = request.files['model_file']
        if not file.filename.endswith(".weights.h5"):
            return jsonify({"success": False, "message": "Invalid file type. Must be .weights.h5"}), 400

        # The model file determines its run directory
        run_dir = os.path.join(current_app.config['MODEL_FOLDER'], secure_filename(os.path.dirname(file.filename)))
        model_path = os.path.join(run_dir, secure_filename(os.path.basename(file.filename)))
        
        # Save the file to its potential run directory (or a temp location if needed)
        # For simplicity, we assume the user uploads into the 'models' root or a run dir.
        # A more robust solution might handle temporary uploads differently.
        temp_model_path = os.path.join(current_app.config['MODEL_FOLDER'], secure_filename(file.filename))
        file.save(temp_model_path)


        json_path = temp_model_path.replace('.weights.h5', '.json')
        if not os.path.exists(json_path):
            return jsonify({"success": False, "message": f"Metadata file (.json) not found."}), 404

        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            loaded_target_rgba = _load_target_image_from_metadata(metadata, temp_model_path)
            if loaded_target_rgba is None:
                return jsonify({"success": False, "message": "Could not load original target image."}), 400

            app_state.trainer_target_image_rgba = loaded_target_rgba

            config = {
                "fire_rate": float(metadata.get("fire_rate", DEFAULT_FIRE_RATE)),
                "batch_size": int(metadata.get("batch_size", DEFAULT_BATCH_SIZE)),
                "pool_size": int(metadata.get("pool_size", DEFAULT_POOL_SIZE)),
                "experiment_type": metadata.get("experiment_type", "Growing"),
                "learning_rate": float(metadata.get("learning_rate", 2e-3)),
                "target_source_kind": app_state.trainer_target_source_kind,
                "enable_entropy": bool(metadata.get("enable_entropy", DEFAULT_ENTROPY_ENABLED)),
                "entropy_strength": float(metadata.get("entropy_strength", DEFAULT_ENTROPY_STRENGTH)),
                "damage_n": int(metadata.get("damage_n", 3)) if metadata.get("experiment_type") == "Regenerating" else 0,
                "run_dir": os.path.dirname(temp_model_path)
            }

            app_state.current_nca_trainer = NCATrainer(target_img_rgba_processed=loaded_target_rgba, config=config)
            app_state.current_nca_trainer.ca.load_weights(temp_model_path)
            app_state.current_nca_trainer.current_step = metadata.get("training_steps", 0)
            app_state.current_nca_trainer.best_loss = metadata.get("final_loss", float('inf'))
            app_state.current_nca_trainer.total_training_time_paused = metadata.get("total_training_time_seconds", 0.0)
            app_state.current_nca_trainer.last_loss = metadata.get("final_loss")
            app_state.trainer_actual_target_shape = app_state.current_nca_trainer.pad_target.shape

            tf.print(f"NCATrainer loaded and state restored. Target shape: {app_state.trainer_actual_target_shape}")
            model_summary_str = get_model_summary(app_state.current_nca_trainer.get_model())

            return jsonify({
                "success": True, "message": f"Model '{file.filename}' loaded.",
                "model_summary": model_summary_str,
                "metadata": metadata # Send metadata to frontend to restore UI state
            })

        except Exception as e:
            tf.print(f"Error load_trainer_model: {e}\n{traceback.format_exc()}")
            app_state.current_nca_trainer = None
            return jsonify({"success": False, "message": f"Error loading model: {e}"}), 500