"""
Konfigurační konstanty aplikace.
Všechny cesty jsou absolutní (odvozené od umístění tohoto souboru),
takže aplikace funguje bez ohledu na pracovní adresář.
"""
import os

# Kořenový adresář projektu (o jednu úroveň výše než tento soubor)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----- Cesty k souborům -----
HAND_LANDMARKER_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
MODEL_PATH           = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH          = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENCODER_PATH         = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

# ----- Kamera -----
CAMERA_INDEX = 0  # 0 = výchozí (vestavěná) webkamera; změň na 1, 2 … pro jinou

# ----- MediaPipe -----
DETECTION_CONFIDENCE = 0.7   # minimální spolehlivost detekce ruky (0.0–1.0)
TRACKING_CONFIDENCE  = 0.7   # minimální spolehlivost sledování ruky (0.0–1.0)

# ----- Predikce -----
# Model vrátí název gesta pouze pokud je jeho pravděpodobnost >= tento práh.
# Nižší hodnota = citlivější, ale více falešně pozitivních výsledků.
PREDICTION_THRESHOLD = 0.75

# ----- Ovládání -----
# Minimální prodleva (v sekundách) mezi dvěma po sobě jdoucími akcemi
# stejného gesta. Brání opakovanému spouštění jedné klávesy při klidném držení ruky.
GESTURE_COOLDOWN = 1.0

# ----- Settings override (written by setup_gui.py) -----
# If settings.json exists in the project root, its values take precedence.
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
        del _s
    except Exception:  # nosec B110 — intentional: malformed settings.json must not crash the app
        pass
del _settings_file, _json
