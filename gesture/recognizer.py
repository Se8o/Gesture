"""
Real-time hand gesture recognition module.

Two-layer architecture:
  1. MediaPipe HandLandmarker – detects 21 hand joint coordinates per frame.
  2. Trained ML model (Random Forest) – predicts the gesture name from those coordinates.

Voting buffer:
  The last BUFFER_SIZE frames are kept in a sliding window. A gesture is
  returned only when it appears at least MIN_VOTES times in the buffer.
  This smooths out brief mis-predictions and catches fast gestures that
  would otherwise only appear in 1–2 frames.
"""
import time
from collections import deque, Counter

import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from . import config

# Internal confidence threshold for writing to the buffer.
# Intentionally lower than PREDICTION_THRESHOLD so that quick,
# slightly uncertain gestures can still participate in voting.
_BUFFER_THRESHOLD = 0.35


class GestureRecognizer:
    """
    Loads the hand detector and the ML model, then exposes a single
    process() method that takes one RGB frame and returns a prediction.
    """

    BUFFER_SIZE = 5   # number of recent frames kept in the voting window
    MIN_VOTES   = 1   # minimum votes in the buffer to return a gesture

    def __init__(self):
        # Load the three trained model artefacts produced by ml/train.py
        self.model   = joblib.load(config.MODEL_PATH)
        self.scaler  = joblib.load(config.SCALER_PATH)
        self.encoder = joblib.load(config.ENCODER_PATH)

        # Configure MediaPipe HandLandmarker in VIDEO mode (stateful tracking)
        base_options = mp_python.BaseOptions(
            model_asset_path=config.HAND_LANDMARKER_PATH
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,                                          # track one hand only
            min_hand_detection_confidence=config.DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.TRACKING_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
        )
        self._detector   = mp_vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()   # reference point for VIDEO mode timestamps

        # Voting buffer: deque of (gesture_name_or_None, confidence) pairs
        self._buffer: deque = deque(maxlen=self.BUFFER_SIZE)

    def process(self, rgb_frame: np.ndarray):
        """
        Process one RGB frame and return a gesture prediction.

        Returns
        -------
        gesture    : str | None  – gesture name, or None if nothing recognised
        confidence : float       – probability of the most likely gesture
        landmarks  : list | None – 21 landmark objects for drawing (or None)
        """
        # Wrap the frame in a MediaPipe Image object with a monotonically increasing timestamp
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result       = self._detector.detect_for_video(mp_image, timestamp_ms)

        # No hand found in this frame
        if not result.hand_landmarks:
            self._buffer.append((None, 0.0))
            return None, 0.0, None

        landmarks = result.hand_landmarks[0]

        # Build the feature vector: x, y, z for each of the 21 joints (63 values total)
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])

        # Scale the features using the same StandardScaler fitted during training
        features_scaled = self.scaler.transform([features])
        # Get the probability distribution over all gesture classes
        proba    = self.model.predict_proba(features_scaled)[0]
        max_prob = float(proba.max())

        # Write to the voting buffer only if above the internal (lower) threshold
        if max_prob >= _BUFFER_THRESHOLD:
            pred_idx    = int(proba.argmax())
            raw_gesture = self.encoder.inverse_transform([pred_idx])[0]
        else:
            raw_gesture = None

        self._buffer.append((raw_gesture, max_prob))

        # Count how many times each gesture appears in the recent buffer
        votes = Counter(g for g, _ in self._buffer if g is not None)
        if not votes:
            return None, max_prob, landmarks

        best_gesture, count = votes.most_common(1)[0]

        # Only return a gesture if it has enough votes in the buffer
        if count >= self.MIN_VOTES:
            # Use the highest confidence seen for this gesture in the buffer
            best_conf = max(
                c for g, c in self._buffer if g == best_gesture
            )
            # Final check: must also exceed the external prediction threshold
            if best_conf >= config.PREDICTION_THRESHOLD:
                return best_gesture, best_conf, landmarks

        return None, max_prob, landmarks

    def close(self):
        """Release MediaPipe detector resources."""
        self._detector.close()
