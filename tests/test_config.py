"""
Tests for src/config.py — default values, path constants, and the
settings.json override mechanism.
"""
import importlib
import json
from pathlib import Path

import pytest

import gesture.config as cfg


# ── Default values ─────────────────────────────────────────────────────────────

class TestDefaults:
    def test_camera_index_is_non_negative_int(self):
        assert isinstance(cfg.CAMERA_INDEX, int)
        assert cfg.CAMERA_INDEX >= 0

    def test_detection_confidence_in_range(self):
        assert 0.0 < cfg.DETECTION_CONFIDENCE <= 1.0

    def test_tracking_confidence_in_range(self):
        assert 0.0 < cfg.TRACKING_CONFIDENCE <= 1.0

    def test_prediction_threshold_in_range(self):
        assert 0.0 < cfg.PREDICTION_THRESHOLD <= 1.0

    def test_gesture_cooldown_positive(self):
        assert cfg.GESTURE_COOLDOWN > 0


# ── Path constants ─────────────────────────────────────────────────────────────

class TestPaths:
    def test_base_dir_exists(self):
        assert Path(cfg.BASE_DIR).is_dir()

    def test_hand_landmarker_path_defined(self):
        assert isinstance(cfg.HAND_LANDMARKER_PATH, str)
        assert cfg.HAND_LANDMARKER_PATH.endswith(".task")

    def test_model_path_in_models_dir(self):
        assert "models" in cfg.MODEL_PATH
        assert cfg.MODEL_PATH.endswith(".pkl")

    def test_scaler_path_in_models_dir(self):
        assert "models" in cfg.SCALER_PATH
        assert cfg.SCALER_PATH.endswith(".pkl")

    def test_encoder_path_in_models_dir(self):
        assert "models" in cfg.ENCODER_PATH
        assert cfg.ENCODER_PATH.endswith(".pkl")

    def test_hand_landmarker_file_exists(self):
        """The pre-trained MediaPipe model must be committed to the repo."""
        assert Path(cfg.HAND_LANDMARKER_PATH).is_file(), (
            "hand_landmarker.task is missing from models/ — it must be present in the repo."
        )


# ── settings.json override ─────────────────────────────────────────────────────

class TestSettingsOverride:
    """
    The override block in config.py reads settings.json from BASE_DIR at
    import time.  We write the file, reload the module, then clean up.
    """

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        """Remove any settings.json we create and reload config to defaults."""
        settings_path = Path(cfg.BASE_DIR) / "settings.json"
        original = settings_path.read_text() if settings_path.exists() else None
        yield settings_path
        # Restore original state
        if original is not None:
            settings_path.write_text(original)
        elif settings_path.exists():
            settings_path.unlink()
        importlib.reload(cfg)

    def test_camera_index_overridden(self, _cleanup):
        settings_path = _cleanup
        settings_path.write_text(json.dumps({"camera_index": 2}))
        importlib.reload(cfg)
        assert cfg.CAMERA_INDEX == 2

    def test_cooldown_overridden(self, _cleanup):
        settings_path = _cleanup
        settings_path.write_text(json.dumps({"gesture_cooldown": 2.5}))
        importlib.reload(cfg)
        assert cfg.GESTURE_COOLDOWN == pytest.approx(2.5)

    def test_all_keys_overridden(self, _cleanup):
        settings_path = _cleanup
        overrides = {
            "camera_index":         1,
            "detection_confidence": 0.85,
            "tracking_confidence":  0.80,
            "prediction_threshold": 0.90,
            "gesture_cooldown":     1.5,
        }
        settings_path.write_text(json.dumps(overrides))
        importlib.reload(cfg)
        assert cfg.CAMERA_INDEX          == 1
        assert cfg.DETECTION_CONFIDENCE  == pytest.approx(0.85)
        assert cfg.TRACKING_CONFIDENCE   == pytest.approx(0.80)
        assert cfg.PREDICTION_THRESHOLD  == pytest.approx(0.90)
        assert cfg.GESTURE_COOLDOWN      == pytest.approx(1.5)

    def test_malformed_json_falls_back_to_defaults(self, _cleanup):
        settings_path = _cleanup
        settings_path.write_text("{ not valid json }")
        importlib.reload(cfg)
        # Must not raise; values should stay at defaults
        assert isinstance(cfg.CAMERA_INDEX, int)
        assert 0.0 < cfg.DETECTION_CONFIDENCE <= 1.0
