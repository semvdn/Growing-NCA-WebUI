# app_state.py
"""
Manages the global state for the Neural CA web application.

This module centralizes all shared state variables and threading primitives
to avoid circular dependencies and make state management explicit.
"""
import threading
from typing import Optional, Tuple

import numpy as np
import PIL.Image
from nca_runner import NCARunner
from nca_trainer import NCATrainer
from nca_globals import DEFAULT_RUNNER_SLEEP_DURATION

# --- Trainer State ---
current_nca_trainer: Optional[NCATrainer] = None
trainer_target_image_rgba: Optional[np.ndarray] = None
trainer_target_source_kind: Optional[str] = None
trainer_actual_target_shape: Optional[Tuple[int, ...]] = None
trainer_target_image_name: str = "unknown_image"
trainer_target_image_loaded_or_drawn: str = "unknown"
original_drawn_image_pil: Optional[PIL.Image.Image] = None
current_training_run_id: Optional[str] = None
current_training_run_dir: Optional[str] = None

# --- Runner State ---
current_nca_runner: Optional[NCARunner] = None
runner_sleep_duration: float = DEFAULT_RUNNER_SLEEP_DURATION

# --- Threading Control ---
training_thread: Optional[threading.Thread] = None
stop_training_event = threading.Event()
train_thread_lock = threading.Lock()

running_thread: Optional[threading.Thread] = None
stop_running_event = threading.Event()
run_thread_lock = threading.Lock()