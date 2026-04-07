"""
Application configuration constants.
All paths are absolute (derived from this file's location),
so the app works regardless of the working directory.
"""
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----- File paths -----
HAND_LANDMARKER_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
MODEL_PATH           = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH          = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENCODER_PATH         = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

# ----- Camera -----
CAMERA_INDEX = 0  # 0 = default webcam; change to 1, 2 … for another camera

# ----- MediaPipe hand detection -----
DETECTION_CONFIDENCE = 0.7   # minimum confidence to detect a hand (0.0 – 1.0)
TRACKING_CONFIDENCE  = 0.7   # minimum confidence to keep tracking a hand (0.0 – 1.0)

# ----- Gesture prediction -----
PREDICTION_THRESHOLD = 0.88  # minimum model confidence to return a gesture

# ----- Action control -----
GESTURE_COOLDOWN = 0.4   # seconds between repeated same-gesture actions

# Control mode:
#   "scroll"   – gestures scroll the page (good for background mode)
#   "keyboard" – gestures send arrow keys (good for presentations)
CONTROL_MODE  = "scroll"
SCROLL_AMOUNT = 10  # scroll units per gesture (only used in "scroll" mode)

# ----- Optional settings override -----
# If settings.json exists in the project root, its values override the defaults above.
_settings_path = os.path.join(BASE_DIR, "settings.json")
if os.path.exists(_settings_path):
    try:
        with open(_settings_path) as f:
            s = json.load(f)
        CAMERA_INDEX         = int(s.get("camera_index",         CAMERA_INDEX))
        DETECTION_CONFIDENCE = float(s.get("detection_confidence",  DETECTION_CONFIDENCE))
        TRACKING_CONFIDENCE  = float(s.get("tracking_confidence",   TRACKING_CONFIDENCE))
        PREDICTION_THRESHOLD = float(s.get("prediction_threshold",  PREDICTION_THRESHOLD))
        GESTURE_COOLDOWN     = float(s.get("gesture_cooldown",      GESTURE_COOLDOWN))
        CONTROL_MODE         = str(s.get("control_mode",            CONTROL_MODE))
        SCROLL_AMOUNT        = int(s.get("scroll_amount",           SCROLL_AMOUNT))
    except Exception:
        pass
