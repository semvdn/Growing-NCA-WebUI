# app_utils.py
"""
Utility functions for the Flask web application.

Contains helpers for generating image responses, managing threads,
and exporting models for TensorFlow.js.
"""
import io
import json
import time
import traceback
from typing import Optional

import numpy as np
import PIL.Image
import tensorflow as tf
from flask import send_file
from flask.wrappers import Response
from google.protobuf.json_format import MessageToDict
from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2

import app_state
from nca_globals import CHANNEL_N
from nca_model import CAModel
from nca_utils import np2pil, to_rgb


def get_preview_image_response(state_array_or_rgba: Optional[np.ndarray],
                               zoom_factor: int = 4,
                               default_width_px: int = 256,
                               default_height_px: int = 256) -> Response:
    """Converts a numpy array into a Flask response containing a PNG image."""
    img = None
    if state_array_or_rgba is None:
        img = PIL.Image.new('RGB', (default_width_px, default_height_px), color='grey')
    else:
        np_array = np.asarray(state_array_or_rgba)
        if np_array.ndim == 3 and np_array.shape[-1] == CHANNEL_N:
            rgb_array = to_rgb(tf.constant(np_array[None, ...]))[0].numpy()
            img = np2pil(rgb_array, zoom_factor=zoom_factor)
        elif np_array.ndim == 3 and np_array.shape[-1] == 4:
            alpha = np.clip(np_array[..., 3:], 0, 1)
            rgb_array = 1.0 - alpha + np_array[..., :3]
            img = np2pil(rgb_array, zoom_factor=zoom_factor)
        else:
            tf.print(f"get_preview_image_response: Unexpected array shape {getattr(np_array, 'shape', 'Not an array')}")
            img = PIL.Image.new('RGB', (default_width_px, default_height_px), color='lightgrey')

    img_io = io.BytesIO()
    if img.width == 0 or img.height == 0:
        img = PIL.Image.new('RGB', (zoom_factor, zoom_factor), color='pink')
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', as_attachment=False, download_name=f'preview_{time.time()}.png')


def ensure_trainer_stopped() -> None:
    """Signals the training thread to stop and waits for it to exit."""
    if app_state.training_thread and app_state.training_thread.is_alive():
        tf.print("Signalling training thread to stop...")
        app_state.stop_training_event.set()
        app_state.training_thread.join(timeout=7)
        if app_state.training_thread.is_alive():
            tf.print("Warning: Training thread did not stop in time.")
        app_state.training_thread = None
    app_state.stop_training_event.clear()


def ensure_runner_stopped() -> None:
    """Signals the runner thread to stop and waits for it to exit."""
    if app_state.running_thread and app_state.running_thread.is_alive():
        tf.print("Signalling running thread to stop...")
        app_state.stop_running_event.set()
        app_state.running_thread.join(timeout=7)
        if app_state.running_thread.is_alive():
            tf.print("Warning: Running thread did not stop in time.")
        app_state.running_thread = None
    app_state.stop_running_event.clear()


def export_model_for_tfjs(ca_model: CAModel, save_path: str) -> None:
    """Exports a CA model to a TensorFlow.js compatible JSON format."""
    concrete_func = ca_model.call.get_concrete_function(
        x=tf.TensorSpec([None, None, None, CHANNEL_N]),
        fire_rate=tf.constant(0.5),
        angle=tf.constant(0.0),
        step_size=tf.constant(1.0),
        enable_entropy=tf.constant(False),
        entropy_strength=tf.constant(0.0)
    )
    constant_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = constant_func.graph.as_graph_def()
    graph_json = MessageToDict(graph_def)
    graph_json['versions'] = dict(producer='1.14', minConsumer='1.14')
    model_json = {
        'format': 'graph-model',
        'modelTopology': graph_json,
        'weightsManifest': [],
    }
    with open(save_path, 'w') as f:
        json.dump(model_json, f)