# nca_globals.py
"""Global constants for the NCA application."""

CHANNEL_N = 16
TARGET_SIZE = 40 # This is the internal size of the target after processing
TARGET_PADDING = 16
DEFAULT_FIRE_RATE = 0.5
DEFAULT_BATCH_SIZE = 8
DEFAULT_POOL_SIZE = 1024
HISTORY_MAX_SIZE = 200
DEFAULT_RUNNER_SLEEP_DURATION = 0.05
RUNNER_SLEEP_DURATION_TRAINING_ACTIVE = 0.2 # Slower FPS when training is active (e.g., 5 FPS)

# New for drawn patterns (client-side canvas size for drawing target)
# This should match the display size of the drawing canvas on the trainer tab.
# The actual TARGET_SIZE (e.g., 40) is what it gets resized to internally.
DRAW_CANVAS_DISPLAY_SIZE = 256 # pixels (e.g., 64x64 NCA grid zoomed 4x)