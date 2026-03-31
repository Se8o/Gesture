"""
Tests for ml/train.py — the model training pipeline.

Uses a small synthetic dataset so tests run in seconds, not minutes.
"""
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ml import train


# ── Helpers ────────────────────────────────────────────────────────────────────

def run_train(data_path: Path, model_dir: Path):
    """
    Run train.train() with DATA_PATH and MODEL_DIR redirected to temp dirs.
    """
    with (
        patch.object(train, "DATA_PATH", str(data_path)),
        patch.object(train, "MODEL_DIR", str(model_dir)),
    ):
        train.train()


# ── Happy path ────────────────────────────────────────────────────────────────

class TestTrainPipeline:
    def test_creates_model_pkl(self, synthetic_csv, tmp_path):
        run_train(synthetic_csv, tmp_path)
        assert (tmp_path / "model.pkl").exists()

    def test_creates_scaler_pkl(self, synthetic_csv, tmp_path):
        run_train(synthetic_csv, tmp_path)
        assert (tmp_path / "scaler.pkl").exists()

    def test_creates_label_encoder_pkl(self, synthetic_csv, tmp_path):
        run_train(synthetic_csv, tmp_path)
        assert (tmp_path / "label_encoder.pkl").exists()

    def test_model_is_random_forest(self, synthetic_csv, tmp_path):
        run_train(synthetic_csv, tmp_path)
        model = joblib.load(tmp_path / "model.pkl")
        assert isinstance(model, RandomForestClassifier)

    def test_scaler_is_standard_scaler(self, synthetic_csv, tmp_path):
        run_train(synthetic_csv, tmp_path)
        scaler = joblib.load(tmp_path / "scaler.pkl")
        assert isinstance(scaler, StandardScaler)

    def test_encoder_is_label_encoder(self, synthetic_csv, tmp_path):
        run_train(synthetic_csv, tmp_path)
        encoder = joblib.load(tmp_path / "label_encoder.pkl")
        assert isinstance(encoder, LabelEncoder)

    def test_encoder_knows_all_five_gestures(self, synthetic_csv, tmp_path):
        run_train(synthetic_csv, tmp_path)
        encoder = joblib.load(tmp_path / "label_encoder.pkl")
        expected = {"posun nahoru", "posun dolu", "posun doprava", "posun doleva", "pauza"}
        assert set(encoder.classes_) == expected

    def test_model_achieves_decent_accuracy(self, synthetic_csv, tmp_path):
        """
        Synthetic data has clearly separated class means — a trained RF
        should achieve > 70 % accuracy even on 200 samples.
        """
        run_train(synthetic_csv, tmp_path)
        model   = joblib.load(tmp_path / "model.pkl")
        scaler  = joblib.load(tmp_path / "scaler.pkl")
        encoder = joblib.load(tmp_path / "label_encoder.pkl")

        df     = pd.read_csv(synthetic_csv)
        X      = df.iloc[:, 1:].values
        y      = encoder.transform(df.iloc[:, 0].values)
        X_sc   = scaler.transform(X)
        preds  = model.predict(X_sc)
        acc    = (preds == y).mean()
        assert acc > 0.70, f"Accuracy too low: {acc:.2%}"

    def test_scaler_fit_only_on_train_not_full_dataset(self, synthetic_csv, tmp_path):
        """The scaler must be fit on training data only."""
        run_train(synthetic_csv, tmp_path)
        scaler = joblib.load(tmp_path / "scaler.pkl")
        sample = np.random.rand(1, 63)
        transformed = scaler.transform(sample)
        assert transformed.shape == (1, 63)

    def test_model_predicts_for_single_sample(self, synthetic_csv, tmp_path):
        run_train(synthetic_csv, tmp_path)
        model  = joblib.load(tmp_path / "model.pkl")
        scaler = joblib.load(tmp_path / "scaler.pkl")
        sample = np.random.rand(1, 63)
        proba  = model.predict_proba(scaler.transform(sample))
        assert proba.shape == (1, 5)
        assert abs(proba.sum() - 1.0) < 1e-6


# ── Error handling ─────────────────────────────────────────────────────────────

class TestTrainErrors:
    def test_exits_when_dataset_missing(self, tmp_path):
        missing = tmp_path / "no_such_file.csv"
        with pytest.raises(SystemExit) as exc_info:
            run_train(missing, tmp_path)
        assert exc_info.value.code == 1

    def test_handles_csv_with_nan_rows(self, tmp_path):
        """NaN rows should be silently dropped, not cause a crash."""
        df = pd.read_csv(
            Path(__file__).parent.parent / "data" / "dataset.csv",
            nrows=50,
        )
        df.iloc[0, 3] = float("nan")
        csv_path = tmp_path / "with_nan.csv"
        df.to_csv(csv_path, index=False)

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        run_train(csv_path, model_dir)
        assert (model_dir / "model.pkl").exists()
