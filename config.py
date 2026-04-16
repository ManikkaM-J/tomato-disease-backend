"""
config.py
---------
Central configuration for the
Multi-Label Tomato Leaf Disease Detection API.

All paths, constants, and thresholds defined here.
Never hardcode these values in other files.
"""

import os

# ── Base directory ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Paths ─────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH    = os.path.join(BASE_DIR, "model", "tomato_leaf_disease_model.h5")

# ── Image settings ────────────────────────────────────────────────────────────
IMAGE_SIZE     = (224, 224)
IMAGE_CHANNELS = 3

# ── Upload limits ─────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024   # 10 MB

# ── Disease classes ───────────────────────────────────────────────────────────
# ORDER MUST MATCH the folder order used during training (alphabetical).
# flow_from_directory assigns:
#   Bacterial_Spot   -> index 0
#   Healthy          -> index 1
#   Late_Blight      -> index 2
#   Yellow_Leaf_Curl -> index 3
CLASS_LABELS = [
    "Bacterial Spot",    # index 0
    "Healthy",           # index 1
    "Late Blight",       # index 2
    "Yellow Leaf Curl",  # index 3
]

# ── Multi-label detection threshold ──────────────────────────────────────────
# A disease is DETECTED if its sigmoid output >= DISEASE_THRESHOLD.
# 0.70 = 70% confidence required (strict mode).
# Increase to 0.80 for more strict detection.
# Decrease to 0.50 for more sensitive detection.
DISEASE_THRESHOLD = 0.70

# ── Flask debug flag ──────────────────────────────────────────────────────────
DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
