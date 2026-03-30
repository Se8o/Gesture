"""
Modul pro rozpoznávání gest ruky v reálném čase.

Spojuje dvě vrstvy:
  1. MediaPipe HandLandmarker – detekuje 21 kloubů ruky v každém snímku.
  2. Natrénovaný ML model (Random Forest) – z kloubů předpoví název gesta.
"""
import time

import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from . import config


class GestureRecognizer:
    """
    Inicializuje detektor ruky i ML model a nabídne jedinou metodu
    process(), která zpracuje snímek a vrátí predikci.
    """

    def __init__(self):
        # Načteme natrénovaný klasifikátor, scaler a enkodér labelů.
        # Pokud soubory neexistují (model nebyl natrénován), joblib vyhodí
        # FileNotFoundError – zachytíme ho v run.py a vypíšeme srozumitelnou chybu.
        self.model   = joblib.load(config.MODEL_PATH)
        self.scaler  = joblib.load(config.SCALER_PATH)
        self.encoder = joblib.load(config.ENCODER_PATH)

        # Nastavení MediaPipe HandLandmarker.
        # Používáme VIDEO mód, stejně jako při sběru dat v collect_data.py,
        # aby detektor viděl data ve stejném formátu jako při trénování.
        base_options = mp_python.BaseOptions(
            model_asset_path=config.HAND_LANDMARKER_PATH
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=config.DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.TRACKING_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
        )
        self._detector   = mp_vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()

    def process(self, rgb_frame: np.ndarray):
        """
        Zpracuje jeden snímek ve formátu RGB.

        Parametry
        ----------
        rgb_frame : np.ndarray
            Snímek z kamery převedený do RGB (výstup cv2.cvtColor(...BGR2RGB)).

        Vrátí
        ------
        gesture    : str | None   – název gesta, nebo None (ruka nenalezena / nízká conf.)
        confidence : float        – pravděpodobnost predikce (0.0–1.0)
        landmarks  : list | None  – 21 objektů s atributy x, y, z (pro kreslení)
        """
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result       = self._detector.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return None, 0.0, None

        landmarks = result.hand_landmarks[0]

        # Sestavíme vektor příznaků – stejné pořadí jako při sběru dat:
        # x0, y0, z0, x1, y1, z1, ..., x20, y20, z20  (63 hodnot)
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])

        # Škálování musí odpovídat tomu, jak byl model trénován.
        features_scaled = self.scaler.transform([features])

        # predict_proba vrátí pravděpodobnosti pro každou třídu.
        proba    = self.model.predict_proba(features_scaled)[0]
        max_prob = float(proba.max())

        # Pokud model není dostatečně přesvědčený, nevrátíme gesto.
        if max_prob < config.PREDICTION_THRESHOLD:
            return None, max_prob, landmarks

        pred_idx = int(proba.argmax())
        gesture  = self.encoder.inverse_transform([pred_idx])[0]
        return gesture, max_prob, landmarks

    def close(self):
        """Uvolní prostředky MediaPipe detektoru."""
        self._detector.close()
