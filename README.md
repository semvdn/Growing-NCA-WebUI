# Neural Cellular Automata Web UI

A web-based application for interactively training and running Neural Cellular Automata (NCA) models, allowing users to define target patterns, configure training parameters, and visualize real-time evolution.

## Features

*   **Interactive Target Definition**: Draw custom target patterns directly on a canvas or upload image files (PNG, JPG).
*   **Flexible Training Modes**: Support for "Growing", "Persistent", and "Regenerating" NCA experiments.
*   **Configurable Training Parameters**: Adjust fire rate, batch size, pool size, learning rate, and apply entropy (noise) during training.
*   **Model Management**: Save trained models as checkpoints and load existing models to resume training or for running.
*   **Real-time Visualization**: Live preview of training progress and NCA evolution.
*   **Interactive Runner Mode**: Explore trained NCA models with tools to erase or draw on the live simulation, and control simulation speed (FPS).
*   **Capture Tools**: Take screenshots and record videos of the NCA evolution.

## Installation

### Prerequisites

*   Python 3.x
*   `pip` (Python package installer)

### Setup Steps

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/Growing-NCA-WebUI.git
    cd Growing-NCA-WebUI
    ```
    (Note: Replace `https://github.com/your-repo/Growing-NCA-WebUI.git` with the actual repository URL if available.)

2.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    This will install necessary libraries like `numpy`, `flask`, `pillow`, `requests`, and `tensorflow`.

3.  **TensorFlow Installation (Optional - for GPU support)**:
    For GPU acceleration, ensure you have a compatible NVIDIA GPU and CUDA Toolkit installed. Then, install the GPU version of TensorFlow:
    ```bash
    pip install tensorflow[and-cuda] # For TensorFlow 2.10+
    # Or for older versions: pip install tensorflow-gpu
    ```
    If you do not have a compatible GPU, the CPU version of TensorFlow will be used automatically.

## Usage

### Starting the Application

To start the Flask web application, run:

```bash
python app.py
```

The application will typically run on `http://127.0.0.1:5000/` (or `http://localhost:5000/`). Open this URL in your web browser.

### Web Interface Overview

The web interface is divided into two main tabs: **Training** and **Run NCA**.

#### Training Tab

This tab allows you to define a target pattern and train an NCA model to reproduce it.

1.  **Define Target Pattern**:
    *   **Draw on Canvas**: Use the drawing tools (color picker, brush size, opacity, eraser) to create your own target pattern. Click "Confirm Drawing as Target" and provide a name.
    *   **Upload Image File**: Click "Choose File" to select a PNG or JPG image from your computer, then click "Load Image File for Trainer".

2.  **Configure Training**:
    *   **Experiment Type**: Choose between "Growing", "Persistent", or "Regenerating" NCA behaviors.
    *   **Parameters**: Adjust `Fire Rate`, `Batch Size`, `Pool Size`, and `Learning Rate`.
    *   **Entropy (Noise)**: Optionally enable and adjust `Entropy Strength` to introduce noise during training, which can help with robustness.

3.  **Control Training**:
    *   Click "Initialize Trainer with Target" after defining your target and configuring parameters.
    *   Click "Start Training" to begin the training process.
    *   Click "Stop Training" to pause the training. The model state is preserved.
    *   Click "Save Trained Model" to save the current model weights and metadata as a checkpoint in the `models/` directory.

4.  **Load Existing Model to Continue Training**:
    *   Upload a previously saved `.weights.h5` model file using "Choose File" and click "Load Trainer Model". This will restore the model and its training state, allowing you to continue training.

5.  **Capture Tools**:
    *   "Take Screenshot": Capture the current state of the training preview.
    *   "Start Recording" / "Stop Recording": Record the training evolution as a video.
    *   "White Background": Toggle a white background for captures.

#### Run NCA Tab

This tab allows you to load a trained NCA model and interact with its live simulation.

1.  **Load Model**:
    *   **Upload Model File**: Upload a `.weights.h5` model file from your computer.
    *   **Load Current Training Model**: Load the model currently being trained (or the last saved state of the trainer) directly into the runner.
    *   If no file is chosen for "Load from File", the server will attempt to load the most recently saved model.

2.  **Control Runner**:
    *   "Start Running Loop": Begin the NCA simulation.
    *   "Stop Running Loop": Pause the simulation.
    *   "Reset Runner State to Seed": Reset the simulation to its initial seed state.

3.  **Interaction Tools**:
    *   Select "Erase" or "Draw" mode.
    *   Choose a `Draw color` (for draw mode).
    *   Adjust `Brush Size`.
    *   Click and drag on the "NCA Runner" canvas to modify the live simulation.

4.  **Speed Control**:
    *   Adjust the "Target FPS" slider to control the simulation speed.

5.  **Entropy (Noise)**:
    *   Optionally enable and adjust `Entropy Strength` to introduce noise during the live simulation.

6.  **Capture Tools**:
    *   "Take Screenshot": Capture the current state of the runner preview.
    *   "Start Recording" / "Stop Recording": Record the runner evolution as a video.
    *   "White Background": Toggle a white background for captures.

## Project Structure

*   `app.py`: The main Flask application, handling all web routes, global state, and threading for training and running loops.
*   `nca_globals.py`: Defines global constants and configuration parameters used throughout the application.
*   `nca_model.py`: Contains the TensorFlow Keras `CAModel` class, defining the neural network architecture for the NCA.
*   `nca_trainer.py`: Implements the core training logic, including the training loop, loss calculation, and saving of best models.
*   `nca_runner.py`: Manages the NCA simulation loop for the "Run NCA" tab, handling state updates, history, and user interactions.
*   `nca_utils.py`: Provides utility functions such as image loading, state manipulation, and sample pooling.
*   `requirements.txt`: Lists all Python dependencies required to run the application.
*   `templates/`: Directory containing HTML templates, primarily `index.html` for the main web interface.
*   `static/`: Contains static web assets, including `js/script.js` for frontend logic and styling.
*   `models/`: A directory where trained NCA models (weights and metadata) are saved and loaded.
*   `uploads/`: A temporary directory for storing user-uploaded target images.
*   `TODO.md`: A markdown file outlining planned features and improvements for the project.

## Dependencies

The project relies on the following Python libraries, listed in `requirements.txt`:

*   `numpy`: For numerical operations, especially with array manipulation.
*   `flask`: The web framework used to build the application.
*   `pillow`: For image processing tasks (e.g., loading and saving images).
*   `requests`: (Potentially used for internal API calls or external resources, though not explicitly visible in core logic provided).
*   `tensorflow`: The deep learning framework powering the Neural Cellular Automata model.

## Future Enhancements

Based on the `TODO.md` file, here are some planned future enhancements:

*   Further modularize `script.js` and `app.py` for improved maintainability.
*   Add network visualization and the option to inspect individual cells in the "Run NCA" tab.
*   Explore advanced features like animations and 3D NCA simulations.
*   Allow for custom neural network architectures to be defined and used.

## License

This project is open-source and available under the MIT License.