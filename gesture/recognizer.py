"""
Real-time hand gesture recognition.

MediaPipe HandLandmarker detects 21 hand joint coordinates per frame.
A trained Random Forest model predicts the gesture name from those coordinates.
"""
import time

import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from . import config


class GestureRecognizer:
    """Loads the hand detector and ML model, exposes a single process() method."""

    def __init__(self):
        self.model   = joblib.load(config.MODEL_PATH)
        self.scaler  = joblib.load(config.SCALER_PATH)
        self.encoder = joblib.load(config.ENCODER_PATH)

        base_options = mp_python.BaseOptions(model_asset_path=config.HAND_LANDMARKER_PATH)
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
        Process one RGB frame and return a gesture prediction.

        Returns
        -------
        gesture    : str | None  – gesture name, or None if nothing recognised
        confidence : float       – probability of the predicted gesture
        landmarks  : list | None – 21 landmark objects for drawing (or None)
        """
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result       = self._detector.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return None, 0.0, None

        landmarks = result.hand_landmarks[0]

        # Build feature vector: x, y, z for each of the 21 joints (63 values)
        features        = [v for lm in landmarks for v in (lm.x, lm.y, lm.z)]
        features_scaled = self.scaler.transform([features])
        proba           = self.model.predict_proba(features_scaled)[0]
        confidence      = float(proba.max())

        if confidence < config.PREDICTION_THRESHOLD:
            return None, confidence, landmarks

        gesture = self.encoder.inverse_transform([int(proba.argmax())])[0]
        return gesture, confidence, landmarks

    def close(self):
        """Release MediaPipe detector resources."""
        self._detector.close()
