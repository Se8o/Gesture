"""
Modul pro rozpoznávání gest ruky v reálném čase.

Spojuje dvě vrstvy:
  1. MediaPipe HandLandmarker – detekuje 21 kloubů ruky v každém snímku.
  2. Natrénovaný ML model (Random Forest) – z kloubů předpoví název gesta.

Hlasovací buffer:
  Posledních BUFFER_SIZE snímků se ukládá. Gesto se vrátí jakmile se
  stejné gesto objeví alespoň MIN_VOTES krát v bufferu – zachytí to
  i rychlá gesta, která by jinak prahem prošla jen v 1–2 snímcích.
"""
import time
from collections import deque, Counter

import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from . import config

# Interní práh pro zápis do bufferu – záměrně nižší než PREDICTION_THRESHOLD,
# aby se rychlá (trochu nejistá) gesta do hlasování vůbec dostala.
_BUFFER_THRESHOLD = 0.35


class GestureRecognizer:
    """
    Inicializuje detektor ruky i ML model a nabídne jedinou metodu
    process(), která zpracuje snímek a vrátí predikci.
    """

    BUFFER_SIZE = 5   # počet posledních snímků v hlasovacím okně
    MIN_VOTES   = 1   # kolik snímků musí souhlasit, aby se gesto vrátilo

    def __init__(self):
        self.model   = joblib.load(config.MODEL_PATH)
        self.scaler  = joblib.load(config.SCALER_PATH)
        self.encoder = joblib.load(config.ENCODER_PATH)

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

        # Hlasovací buffer: deque dvojic (gesture_nebo_None, confidence)
        self._buffer: deque = deque(maxlen=self.BUFFER_SIZE)

    def process(self, rgb_frame: np.ndarray):
        """
        Zpracuje jeden snímek ve formátu RGB.

        Vrátí
        ------
        gesture    : str | None   – název gesta, nebo None
        confidence : float        – pravděpodobnost nejjistějšího kandidáta
        landmarks  : list | None  – 21 kloubů pro kreslení
        """
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result       = self._detector.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            self._buffer.append((None, 0.0))
            return None, 0.0, None

        landmarks = result.hand_landmarks[0]

        # Vektor příznaků: x0,y0,z0 … x20,y20,z20 (63 hodnot)
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])

        features_scaled = self.scaler.transform([features])
        proba    = self.model.predict_proba(features_scaled)[0]
        max_prob = float(proba.max())

        # Do bufferu zapíšeme gesto pokud přesáhlo interní (nižší) práh.
        # Tím se do hlasování dostanou i rychlá, trochu nejistá gesta.
        if max_prob >= _BUFFER_THRESHOLD:
            pred_idx = int(proba.argmax())
            raw_gesture = self.encoder.inverse_transform([pred_idx])[0]
        else:
            raw_gesture = None

        self._buffer.append((raw_gesture, max_prob))

        # Hlasování: spočítej výskyty každého gesta v bufferu
        votes = Counter(g for g, _ in self._buffer if g is not None)
        if not votes:
            return None, max_prob, landmarks

        best_gesture, count = votes.most_common(1)[0]

        # Vrátíme gesto pokud dostalo dost hlasů
        if count >= self.MIN_VOTES:
            # Confidence = nejvyšší hodnota tohoto gesta v bufferu
            best_conf = max(
                c for g, c in self._buffer if g == best_gesture
            )
            return best_gesture, best_conf, landmarks

        return None, max_prob, landmarks

    def close(self):
        """Uvolní prostředky MediaPipe detektoru."""
        self._detector.close()
