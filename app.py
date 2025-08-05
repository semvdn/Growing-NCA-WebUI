# app.py
"""
Main entry point for the Neural Cellular Automata Flask application.

This script creates the Flask app instance, registers the route blueprints
from other modules, and starts the development server.

To run the application:
    python app.py
"""
import os
import tensorflow as tf
from flask import Flask, render_template

# Import the blueprints for different route groups
from app_routes_trainer import trainer_bp
from app_routes_runner import runner_bp

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure folders
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# Register the route blueprints with the Flask app
app.register_blueprint(trainer_bp)
app.register_blueprint(runner_bp)

# --- Main Route ---
@app.route('/')
def index_route() -> str:
    """Serves the main HTML page of the application."""
    return render_template('index.html')

# --- Main Execution Block ---
if __name__ == '__main__':
    tf.print(f"TensorFlow Version: {tf.__version__}")
    tf.print(f"Is TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu_dev in physical_devices:
                tf.config.experimental.set_memory_growth(gpu_dev, True)
            tf.print(f"Found GPUs: {physical_devices}, memory growth enabled.")
        except RuntimeError as e:
            tf.print(f"GPU Memory Growth Error: {e}")
    else:
        tf.print("No GPUs found. Running on CPU.")

    # use_reloader=False is important to prevent background threads from
    # being initialized multiple times in debug mode.
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)