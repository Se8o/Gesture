"""
Tests for setup_gui.py — settings persistence and model-ready checks.

No Tk window is created; only the pure-Python helper functions are tested.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import setup_gui


# ── load_settings / save_settings roundtrip ───────────────────────────────────

class TestSettingsPersistence:
    def test_load_returns_defaults_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(setup_gui, "SETTINGS_FILE", tmp_path / "settings.json")
        loaded = setup_gui.load_settings()
        assert loaded == setup_gui.DEFAULTS

    def test_save_creates_file(self, tmp_path, monkeypatch):
        path = tmp_path / "settings.json"
        monkeypatch.setattr(setup_gui, "SETTINGS_FILE", path)
        setup_gui.save_settings(setup_gui.DEFAULTS.copy())
        assert path.exists()

    def test_roundtrip_preserves_all_keys(self, tmp_path, monkeypatch):
        path = tmp_path / "settings.json"
        monkeypatch.setattr(setup_gui, "SETTINGS_FILE", path)

        original = {
            "camera_index":         2,
            "detection_confidence": 0.85,
            "tracking_confidence":  0.80,
            "prediction_threshold": 0.90,
            "gesture_cooldown":     2.5,
        }
        setup_gui.save_settings(original)
        loaded = setup_gui.load_settings()
        assert loaded == original

    def test_partial_settings_merged_with_defaults(self, tmp_path, monkeypatch):
        path = tmp_path / "settings.json"
        monkeypatch.setattr(setup_gui, "SETTINGS_FILE", path)
        path.write_text(json.dumps({"camera_index": 3}))

        loaded = setup_gui.load_settings()
        assert loaded["camera_index"] == 3
        # All other keys must still be present (filled from DEFAULTS)
        for key in setup_gui.DEFAULTS:
            assert key in loaded

    def test_malformed_json_returns_defaults(self, tmp_path, monkeypatch):
        path = tmp_path / "settings.json"
        monkeypatch.setattr(setup_gui, "SETTINGS_FILE", path)
        path.write_text("{ broken json !!!")

        loaded = setup_gui.load_settings()
        assert loaded == setup_gui.DEFAULTS

    def test_saved_file_is_valid_json(self, tmp_path, monkeypatch):
        path = tmp_path / "settings.json"
        monkeypatch.setattr(setup_gui, "SETTINGS_FILE", path)
        setup_gui.save_settings(setup_gui.DEFAULTS.copy())
        parsed = json.loads(path.read_text())
        assert isinstance(parsed, dict)


# ── models_ready() ────────────────────────────────────────────────────────────

class TestModelsReady:
    def test_false_when_no_model_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(setup_gui, "MODELS_DIR", tmp_path)
        assert setup_gui.models_ready() is False

    def test_false_when_only_some_files_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(setup_gui, "MODELS_DIR", tmp_path)
        (tmp_path / "model.pkl").touch()
        # scaler.pkl and label_encoder.pkl are still missing
        assert setup_gui.models_ready() is False

    def test_true_when_all_files_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(setup_gui, "MODELS_DIR", tmp_path)
        for name in setup_gui.REQUIRED_MODELS:
            (tmp_path / name).touch()
        assert setup_gui.models_ready() is True


# ── data_ready() ──────────────────────────────────────────────────────────────

class TestDataReady:
    def test_false_when_no_dataset(self, tmp_path, monkeypatch):
        monkeypatch.setattr(setup_gui, "DATA_FILE", tmp_path / "missing.csv")
        assert setup_gui.data_ready() is False

    def test_true_when_dataset_exists(self, tmp_path, monkeypatch):
        csv = tmp_path / "dataset.csv"
        csv.touch()
        monkeypatch.setattr(setup_gui, "DATA_FILE", csv)
        assert setup_gui.data_ready() is True


# ── DEFAULTS sanity ────────────────────────────────────────────────────────────

class TestDefaults:
    def test_all_required_keys_present(self):
        required = {
            "camera_index", "detection_confidence",
            "tracking_confidence", "prediction_threshold", "gesture_cooldown",
        }
        assert required.issubset(setup_gui.DEFAULTS.keys())

    def test_camera_index_is_int(self):
        assert isinstance(setup_gui.DEFAULTS["camera_index"], int)

    def test_confidence_values_in_range(self):
        for key in ("detection_confidence", "tracking_confidence", "prediction_threshold"):
            val = setup_gui.DEFAULTS[key]
            assert 0.0 < val <= 1.0, f"{key}={val} out of range"

    def test_cooldown_positive(self):
        assert setup_gui.DEFAULTS["gesture_cooldown"] > 0
