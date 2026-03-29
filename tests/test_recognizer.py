"""
Tests for src/recognizer.py — gesture prediction logic.

MediaPipe and joblib are mocked so no real model files or camera are needed.
"""
import importlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import make_landmark, make_landmarks_list


# ── Fixture: fully-mocked GestureRecognizer ───────────────────────────────────

@pytest.fixture
def recognizer(mock_ml_models):
    """
    Build a GestureRecognizer where every external dependency is mocked:
      - joblib.load      → returns fake model / scaler / encoder
      - MediaPipe        → HandLandmarker.create_from_options returns a mock detector
    """
    model, scaler, encoder = mock_ml_models

    # MediaPipe mocks
    mock_detector = MagicMock()
    mock_mp       = MagicMock()
    mock_mp_python = MagicMock()
    mock_mp_vision = MagicMock()
    mock_mp_vision.HandLandmarker.create_from_options.return_value = mock_detector
    mock_mp_vision.RunningMode.VIDEO = "VIDEO"

    with (
        patch("src.recognizer.joblib.load", side_effect=[model, scaler, encoder]),
        patch("src.recognizer.mp",          mock_mp),
        patch("src.recognizer.mp_python",   mock_mp_python),
        patch("src.recognizer.mp_vision",   mock_mp_vision),
    ):
        from src.recognizer import GestureRecognizer
        importlib.reload(importlib.import_module("src.recognizer"))
        rec = GestureRecognizer.__new__(GestureRecognizer)
        rec.model    = model
        rec.scaler   = scaler
        rec.encoder  = encoder
        rec._detector   = mock_detector
        rec._start_time = 0.0
        rec._mp         = mock_mp
        yield rec, mock_detector, model, scaler, encoder


# ── No hand detected ──────────────────────────────────────────────────────────

class TestNoHand:
    def test_returns_none_when_no_landmarks(self, recognizer):
        rec, detector, *_ = recognizer
        result = MagicMock()
        result.hand_landmarks = []
        detector.detect_for_video.return_value = result

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("src.recognizer.mp.Image"), \
             patch("src.recognizer.time.time", return_value=1.0):
            gesture, conf, landmarks = rec.process(frame)

        assert gesture    is None
        assert conf       == 0.0
        assert landmarks  is None


# ── Hand detected, low confidence ─────────────────────────────────────────────

class TestLowConfidence:
    def test_returns_none_gesture_when_below_threshold(self, recognizer):
        rec, detector, model, scaler, encoder = recognizer

        # All classes equally probable → max = 0.25, below default threshold 0.75
        model.predict_proba.return_value = np.array([[0.25, 0.25, 0.25, 0.25]])

        hand_landmarks = make_landmarks_list()
        result = MagicMock()
        result.hand_landmarks = [hand_landmarks]
        detector.detect_for_video.return_value = result

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("src.recognizer.mp.Image"), \
             patch("src.recognizer.time.time", return_value=1.0):
            gesture, conf, landmarks = rec.process(frame)

        assert gesture is None
        assert conf    == pytest.approx(0.25)
        assert landmarks is not None   # landmarks still returned for drawing

    def test_confidence_value_matches_max_proba(self, recognizer):
        rec, detector, model, *_ = recognizer
        model.predict_proba.return_value = np.array([[0.1, 0.6, 0.2, 0.1]])

        result = MagicMock()
        result.hand_landmarks = [make_landmarks_list()]
        detector.detect_for_video.return_value = result

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("src.recognizer.mp.Image"), \
             patch("src.recognizer.time.time", return_value=1.0):
            _, conf, _ = rec.process(frame)

        assert conf == pytest.approx(0.6)


# ── Hand detected, high confidence ────────────────────────────────────────────

class TestHighConfidence:
    def test_returns_gesture_name_above_threshold(self, recognizer):
        rec, detector, model, scaler, encoder = recognizer

        # Class 0 has probability 0.92 — well above threshold
        model.predict_proba.return_value = np.array([[0.92, 0.03, 0.03, 0.02]])
        encoder.inverse_transform.return_value = np.array(["posun nahoru"])

        result = MagicMock()
        result.hand_landmarks = [make_landmarks_list()]
        detector.detect_for_video.return_value = result

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("src.recognizer.mp.Image"), \
             patch("src.recognizer.time.time", return_value=1.0):
            gesture, conf, landmarks = rec.process(frame)

        assert gesture   == "posun nahoru"
        assert conf      == pytest.approx(0.92)
        assert landmarks is not None

    def test_encoder_called_with_argmax_index(self, recognizer):
        rec, detector, model, scaler, encoder = recognizer

        proba = np.array([[0.05, 0.05, 0.85, 0.05]])  # argmax = 2
        model.predict_proba.return_value = proba
        encoder.inverse_transform.return_value = np.array(["posun doprava"])

        result = MagicMock()
        result.hand_landmarks = [make_landmarks_list()]
        detector.detect_for_video.return_value = result

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("src.recognizer.mp.Image"), \
             patch("src.recognizer.time.time", return_value=1.0):
            rec.process(frame)

        encoder.inverse_transform.assert_called_once_with([2])


# ── Feature extraction ────────────────────────────────────────────────────────

class TestFeatureExtraction:
    def test_scaler_receives_63_features(self, recognizer):
        rec, detector, model, scaler, encoder = recognizer
        model.predict_proba.return_value = np.array([[0.92, 0.03, 0.03, 0.02]])
        encoder.inverse_transform.return_value = np.array(["posun nahoru"])

        lms = make_landmarks_list(21)
        result = MagicMock()
        result.hand_landmarks = [lms]
        detector.detect_for_video.return_value = result

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("src.recognizer.mp.Image"), \
             patch("src.recognizer.time.time", return_value=1.0):
            rec.process(frame)

        called_features = scaler.transform.call_args[0][0]
        assert len(called_features[0]) == 63   # 21 × 3


# ── close() ───────────────────────────────────────────────────────────────────

class TestClose:
    def test_close_calls_detector_close(self, recognizer):
        rec, detector, *_ = recognizer
        rec.close()
        detector.close.assert_called_once()
