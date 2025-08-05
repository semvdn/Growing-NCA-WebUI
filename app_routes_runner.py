# app_routes_runner.py
"""
Flask routes for the Runner section of the NCA Web UI.
"""
import os
import time
import traceback
import threading
import numpy as np
import tensorflow as tf
from flask import Blueprint, current_app, jsonify, request
from flask.wrappers import Response
from werkzeug.utils import secure_filename
import json
import app_state
from app_utils import ensure_runner_stopped
from nca_globals import (CHANNEL_N, DEFAULT_ENTROPY_ENABLED,
                         DEFAULT_ENTROPY_STRENGTH, DEFAULT_FIRE_RATE,
                         DEFAULT_RUNNER_SLEEP_DURATION,
                         RUNNER_SLEEP_DURATION_TRAINING_ACTIVE, TARGET_PADDING,
                         TARGET_SIZE)
from nca_model import CAModel
from nca_runner import NCARunner
from nca_utils import get_model_summary, to_rgb

runner_bp = Blueprint('runner', __name__)

@runner_bp.route('/load_current_training_model_for_runner', methods=['POST'])
def load_current_training_model_for_runner_route() -> Response:
    """Loads the currently active model from the trainer into the runner."""
    with app_state.train_thread_lock:
        if not app_state.current_nca_trainer or not app_state.current_nca_trainer.ca:
            return jsonify({"success": False, "message": "No active training model available."}), 400
        
        best_model_path = os.path.join(app_state.current_training_run_dir, "best_model.weights.h5") if app_state.current_training_run_dir else None
        
        try:
            model_to_load = CAModel(channel_n=CHANNEL_N, fire_rate=app_state.current_nca_trainer.ca.fire_rate)
            if best_model_path and os.path.exists(best_model_path):
                model_to_load.load_weights(best_model_path)
                message = "Loaded best performing model from current run."
            else:
                model_to_load.set_weights(app_state.current_nca_trainer.ca.get_weights())
                message = "Loaded current in-memory training model."
            
            runner_h, runner_w = app_state.trainer_actual_target_shape[:2] if app_state.trainer_actual_target_shape else (72, 72)
            runner_enable_entropy = app_state.current_nca_trainer.ca.enable_entropy
            runner_entropy_strength = app_state.current_nca_trainer.ca.entropy_strength
            model_to_load.enable_entropy = runner_enable_entropy
            model_to_load.entropy_strength = runner_entropy_strength
        except Exception as e:
            return jsonify({"success": False, "message": f"Error accessing trainer model: {e}"}), 500

    with app_state.run_thread_lock:
        ensure_runner_stopped()
        app_state.current_nca_runner = NCARunner(
            ca_model_instance=model_to_load,
            initial_state_shape_tuple=(runner_h, runner_w, CHANNEL_N),
            enable_entropy=runner_enable_entropy,
            entropy_strength=runner_entropy_strength
        )
        model_summary_str = get_model_summary(app_state.current_nca_runner.ca)
        return jsonify({
            "success": True, "message": message, "model_summary": model_summary_str,
            "metadata": {"enable_entropy": runner_enable_entropy, "entropy_strength": runner_entropy_strength}
        })


@runner_bp.route('/load_model_for_runner', methods=['POST'])
def load_model_for_runner_route() -> Response:
    """Loads a model for the runner, from an uploaded file or the latest on server."""
    model_file_path = None
    with app_state.run_thread_lock:
        ensure_runner_stopped()
        if 'model_file' in request.files and request.files['model_file'].filename:
            file = request.files['model_file']
            if not file.filename.endswith(".weights.h5"):
                return jsonify({"success": False, "message": "Invalid model file type."}), 400
            model_file_path = os.path.join(current_app.config['MODEL_FOLDER'], secure_filename(file.filename))
            file.save(model_file_path)
            message = f"Runner: Uploaded model '{file.filename}' loaded."
        else:
            latest_mtime = 0
            for root, _, files in os.walk(current_app.config['MODEL_FOLDER']):
                for f_name in files:
                    if f_name.endswith('.weights.h5'):
                        path = os.path.join(root, f_name)
                        mtime = os.path.getmtime(path)
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            model_file_path = path
            if not model_file_path:
                return jsonify({"success": False, "message": "No models found on server."}), 404
            message = f"Runner: Loaded latest model '{os.path.basename(model_file_path)}'."

        try:
            loaded_ca_model = CAModel(channel_n=CHANNEL_N, fire_rate=DEFAULT_FIRE_RATE)
            loaded_ca_model.load_weights(model_file_path)
            metadata = {}
            json_file_path = model_file_path.replace('.weights.h5', '.json')
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as f:
                    metadata = json.load(f)
            h, w = (app_state.trainer_actual_target_shape[:2] if app_state.trainer_actual_target_shape
                    else (TARGET_SIZE + 2 * TARGET_PADDING, TARGET_SIZE + 2 * TARGET_PADDING))
            enable_entropy = bool(metadata.get("enable_entropy", DEFAULT_ENTROPY_ENABLED))
            entropy_strength = float(metadata.get("entropy_strength", DEFAULT_ENTROPY_STRENGTH))
            loaded_ca_model.enable_entropy = enable_entropy
            loaded_ca_model.entropy_strength = entropy_strength
            app_state.current_nca_runner = NCARunner(
                ca_model_instance=loaded_ca_model,
                initial_state_shape_tuple=(h, w, CHANNEL_N),
                enable_entropy=enable_entropy,
                entropy_strength=entropy_strength
            )
            model_summary_str = get_model_summary(app_state.current_nca_runner.ca)
            return jsonify({
                "success": True, "message": message, "model_summary": model_summary_str,
                "metadata": {**metadata, "enable_entropy": enable_entropy, "entropy_strength": entropy_strength}
            })
        except Exception as e:
            tf.print(f"Error load_model_for_runner: {e}\n{traceback.format_exc()}")
            return jsonify({"success": False, "message": f"Error: {e}"}), 500


def running_loop_task_function() -> None:
    """The function executed by the background runner thread."""
    tf.print("Runner thread started.")
    while not app_state.stop_running_event.is_set():
        loop_start_time = time.perf_counter()
        try:
            with app_state.run_thread_lock:
                if not app_state.current_nca_runner:
                    break
                app_state.current_nca_runner.step()
            sleep_duration = RUNNER_SLEEP_DURATION_TRAINING_ACTIVE if app_state.training_thread and app_state.training_thread.is_alive() else app_state.runner_sleep_duration
            time_to_sleep = sleep_duration - (time.perf_counter() - loop_start_time)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
        except Exception as e:
            tf.print(f"Error in running_loop: {e}\n{traceback.format_exc()}")
            break
    tf.print("Runner thread ended.")


@runner_bp.route('/start_running', methods=['POST'])
def start_running_loop_route() -> Response:
    """Starts the background runner thread."""
    with app_state.run_thread_lock:
        if not app_state.current_nca_runner:
            return jsonify({"success": False, "message": "Load a model first."}), 400
        if app_state.running_thread and app_state.running_thread.is_alive():
            return jsonify({"success": False, "message": "Runner is already active."}), 400
        app_state.stop_running_event.clear()
        app_state.running_thread = threading.Thread(target=running_loop_task_function, daemon=True)
        app_state.running_thread.start()
    return jsonify({"success": True, "message": "Runner loop started."})


@runner_bp.route('/stop_running', methods=['POST'])
def stop_running_loop_route() -> Response:
    """Stops the background runner thread."""
    with app_state.run_thread_lock:
        ensure_runner_stopped()
    return jsonify({"success": True, "message": "Runner loop stopped."})


@runner_bp.route('/set_runner_speed', methods=['POST'])
def set_runner_speed_route() -> Response:
    """Sets the target FPS for the runner loop."""
    try:
        data = request.json
        fps = float(data.get('fps', 20))
        app_state.runner_sleep_duration = 1.0 / max(0.1, fps)
        return jsonify({"success": True, "message": f"Runner speed set to ~{fps:.1f} FPS."})
    except (ValueError, TypeError) as e:
        return jsonify({"success": False, "message": "Invalid FPS value."}), 400


@runner_bp.route('/get_runner_status')
def get_runner_status_route() -> Response:
    """Fetches the current status of the runner."""
    is_loop_active = app_state.running_thread.is_alive() if app_state.running_thread else False
    if not app_state.current_nca_runner:
        return jsonify({"is_loop_active": False, "status_message": "Runner: No model loaded"})

    with app_state.current_nca_runner._state_lock:
        hist_idx = app_state.current_nca_runner.history_index
        total_hist = len(app_state.current_nca_runner.history)
        actual_fps = app_state.current_nca_runner.actual_fps

    status_msg = f"Runner: {'Active' if is_loop_active else 'Paused'}. Step: {hist_idx}/{max(0, total_hist - 1)}"

    return jsonify({
        "is_loop_active": is_loop_active, "history_step": hist_idx, "total_history": total_hist,
        "status_message": status_msg, "actual_fps": f"{actual_fps:.1f}",
        "current_fps": 1.0 / app_state.runner_sleep_duration if app_state.runner_sleep_duration > 0 else "Max"
    })


@runner_bp.route('/get_live_runner_raw_preview_data')
def get_live_runner_raw_preview_data_route() -> Response:
    """Provides raw pixel data of the live runner's current state."""
    if not app_state.current_nca_runner:
        return jsonify({"success": False, "message": "No runner state available"}), 404
    preview_state = app_state.current_nca_runner.get_current_state_for_display()
    if preview_state is None:
        return jsonify({"success": False, "message": "Runner state is None"}), 404

    rgba_data = to_rgb(tf.constant(preview_state[None, ...]))[0].numpy()
    display_rgba_uint8 = np.uint8(np.clip(rgba_data, 0, 1) * 255)
    h, w, _ = display_rgba_uint8.shape
    final_rgba = np.dstack((display_rgba_uint8, np.full((h, w), 255, dtype=np.uint8)))

    return jsonify({"success": True, "height": h, "width": w, "pixels": final_rgba.flatten().tolist()})


@runner_bp.route('/set_runner_entropy', methods=['POST'])
def set_runner_entropy_route() -> Response:
    """Dynamically sets the entropy (noise) settings for the runner."""
    if not app_state.current_nca_runner:
        return jsonify({"success": False, "message": "Runner not initialized."}), 400
    try:
        data = request.json
        enable = bool(data.get('enable_entropy', DEFAULT_ENTROPY_ENABLED))
        strength = float(data.get('entropy_strength', DEFAULT_ENTROPY_STRENGTH))
        app_state.current_nca_runner.set_entropy_settings(enable, strength)
        return jsonify({"success": True, "message": "Runner entropy settings updated."})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error setting entropy: {e}"}), 500


@runner_bp.route('/runner_action', methods=['POST'])
def runner_action_route() -> Response:
    """Handles various actions for the runner, like reset or modification."""
    with app_state.run_thread_lock:
        if not app_state.current_nca_runner:
            return jsonify({"success": False, "message": "Runner not active."}), 400
        data = request.json
        action = data.get('action')
        if action == 'reset_runner':
            app_state.current_nca_runner.reset_state()
            message = "State reset to initial seed."
        elif action == 'modify_area':
            app_state.current_nca_runner.modify_area(
                data.get('tool_mode', 'erase'), float(data.get('norm_x')), float(data.get('norm_y')),
                float(data.get('brush_size_norm', 0.05)), int(data.get('canvas_render_width')),
                int(data.get('canvas_render_height')), data.get('draw_color_hex', '#000000')
            )
            message = f"Area {data.get('tool_mode', 'modified')}d."
        else:
            return jsonify({"success": False, "message": "Unknown action."}), 400
    return jsonify({"success": True, "message": message})