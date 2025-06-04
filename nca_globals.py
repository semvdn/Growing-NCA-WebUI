# nca_globals.py
"""Global constants for the NCA application."""

CHANNEL_N = 16  # Number of CA state channels
TARGET_SIZE = 40 # Default target image size
TARGET_PADDING = 16 # Padding around the target image
DEFAULT_FIRE_RATE = 0.5
DEFAULT_BATCH_SIZE = 8
DEFAULT_POOL_SIZE = 1024
HISTORY_MAX_SIZE = 200 # Max history steps for runner to prevent memory issues