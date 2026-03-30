"""
Tests for src/app.py — drawing helpers and run() startup behaviour.

cv2.imshow and camera I/O are mocked so no display or webcam is needed.
"""
import sys
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from tests.conftest import make_landmark, make_landmarks_list


# ── Drawing helpers ────────────────────────────────────────────────────────────

class TestDrawHand:
    """_draw_hand should draw lines and circles without raising."""

    @pytest.fixture
    def frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def landmarks(self):
        return make_landmarks_list(21)

    def test_does_not_raise(self, frame, landmarks):
        from gesture.app import _draw_hand
        _draw_hand(frame, landmarks, h=480, w=640)   # must not raise

    def test_frame_is_modified(self, frame, landmarks):
        """At least some pixels should be set after drawing."""
        from gesture.app import _draw_hand
        original_sum = frame.sum()
        _draw_hand(frame, landmarks, h=480, w=640)
        assert frame.sum() != original_sum

    def test_accepts_various_resolutions(self, landmarks):
        from gesture.app import _draw_hand
        for h, w in [(240, 320), (720, 1280), (1080, 1920)]:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            _draw_hand(frame, landmarks, h=h, w=w)   # must not raise


class TestDrawUI:
    """_draw_ui should render without raising for all state combinations."""

    @pytest.fixture
    def frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_no_gesture(self, frame):
        from gesture.app import _draw_ui
        _draw_ui(frame, gesture=None, confidence=0.0, triggered=False)

    def test_gesture_detected_not_triggered(self, frame):
        from gesture.app import _draw_ui
        _draw_ui(frame, gesture="posun nahoru", confidence=0.82, triggered=False)

    def test_gesture_triggered(self, frame):
        from gesture.app import _draw_ui
        _draw_ui(frame, gesture="posun dolu", confidence=0.91, triggered=True)

    def test_all_four_gestures(self, frame):
        from gesture.app import _draw_ui
        for g in ("posun nahoru", "posun dolu", "posun doprava", "posun doleva"):
            f = np.zeros((480, 640, 3), dtype=np.uint8)
            _draw_ui(f, gesture=g, confidence=0.88, triggered=True)

    def test_frame_modified_with_gesture(self, frame):
        from gesture.app import _draw_ui
        _draw_ui(frame, gesture="posun nahoru", confidence=0.9, triggered=False)
        assert frame.sum() > 0


# ── run() startup: missing model files ────────────────────────────────────────

class TestRunStartup:
    def test_exits_if_model_missing(self, tmp_path, monkeypatch):
        """run() must exit(1) when model files don't exist."""
        import gesture.config as cfg
        monkeypatch.setattr(cfg, "MODEL_PATH",  str(tmp_path / "model.pkl"))
        monkeypatch.setattr(cfg, "SCALER_PATH", str(tmp_path / "scaler.pkl"))
        monkeypatch.setattr(cfg, "ENCODER_PATH",str(tmp_path / "label_encoder.pkl"))

        with pytest.raises(SystemExit) as exc_info:
            from gesture.app import run
            run()
        assert exc_info.value.code == 1

    def test_exits_if_camera_fails(self, tmp_path, monkeypatch):
        """run() must exit(1) when the camera cannot be opened."""
        import gesture.config as cfg
        # Create fake (empty) model files so the file-existence check passes
        for name in ("model.pkl", "scaler.pkl", "label_encoder.pkl"):
            (tmp_path / name).touch()
        monkeypatch.setattr(cfg, "MODEL_PATH",  str(tmp_path / "model.pkl"))
        monkeypatch.setattr(cfg, "SCALER_PATH", str(tmp_path / "scaler.pkl"))
        monkeypatch.setattr(cfg, "ENCODER_PATH",str(tmp_path / "label_encoder.pkl"))

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False   # camera unavailable

        mock_recognizer = MagicMock()

        with (
            patch("gesture.app.GestureRecognizer", return_value=mock_recognizer),
            patch("gesture.app.cv2.VideoCapture",  return_value=mock_cap),
            pytest.raises(SystemExit) as exc_info,
        ):
            from gesture.app import run
            run()

        assert exc_info.value.code == 1


# ── HAND_CONNECTIONS constant ─────────────────────────────────────────────────

class TestHandConnections:
    def test_all_indices_in_range(self):
        from gesture.app import HAND_CONNECTIONS
        for a, b in HAND_CONNECTIONS:
            assert 0 <= a <= 20
            assert 0 <= b <= 20

    def test_no_self_loops(self):
        from gesture.app import HAND_CONNECTIONS
        for a, b in HAND_CONNECTIONS:
            assert a != b
