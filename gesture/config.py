"""
Application configuration constants.
All paths are absolute (derived from this file's location),
so the app works regardless of the working directory.
"""
import os

# Root directory of the project (one level above this file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----- File paths -----
HAND_LANDMARKER_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
MODEL_PATH           = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH          = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENCODER_PATH         = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

# ----- Camera -----
CAMERA_INDEX = 0  # 0 = default (built-in) webcam; change to 1, 2 … for another camera

# ----- MediaPipe hand detection thresholds -----
DETECTION_CONFIDENCE = 0.7   # minimum confidence to detect a hand (0.0 – 1.0)
TRACKING_CONFIDENCE  = 0.7   # minimum confidence to keep tracking a hand (0.0 – 1.0)

# ----- Gesture prediction -----
# The model returns a gesture name only when its probability is >= this threshold.
# Lower value = more sensitive but more false positives.
PREDICTION_THRESHOLD = 0.75

# ----- Action control -----
# Minimum delay (in seconds) between two consecutive actions for the same gesture.
# Prevents the same key from firing repeatedly while the hand is held still.
GESTURE_COOLDOWN = 0.4

# Control mode:
#   "scroll"   – gestures scroll the page with the mouse (good for background mode)
#   "keyboard" – gestures send arrow keys (good for presentations / video players)
CONTROL_MODE = "scroll"

# Number of scroll units per gesture (only used when CONTROL_MODE = "scroll")
SCROLL_AMOUNT = 10

# ----- Settings override (written by gui/wizard.py) -----
# If settings.json exists in the project root, its values override the defaults above.
import json as _json
_settings_file = os.path.join(BASE_DIR, "settings.json")
if os.path.exists(_settings_file):
    try:
        with open(_settings_file) as _f:
            _s = _json.load(_f)
        CAMERA_INDEX          = int(_s.get("camera_index",         CAMERA_INDEX))
        DETECTION_CONFIDENCE  = float(_s.get("detection_confidence",  DETECTION_CONFIDENCE))
        TRACKING_CONFIDENCE   = float(_s.get("tracking_confidence",   TRACKING_CONFIDENCE))
        PREDICTION_THRESHOLD  = float(_s.get("prediction_threshold",  PREDICTION_THRESHOLD))
        GESTURE_COOLDOWN      = float(_s.get("gesture_cooldown",      GESTURE_COOLDOWN))
        CONTROL_MODE          = str(_s.get("control_mode",            CONTROL_MODE))
        SCROLL_AMOUNT         = int(_s.get("scroll_amount",           SCROLL_AMOUNT))
        del _s
    except Exception:  # nosec B110 — intentional: malformed settings.json must not crash the app
        pass
del _settings_file, _json
