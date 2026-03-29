"""
Shared pytest fixtures for the Gesture Remote Controller test suite.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Synthetic dataset ──────────────────────────────────────────────────────────

GESTURE_CLASSES = ["posun nahoru", "posun dolu", "posun doprava", "posun doleva"]
N_PER_CLASS = 40          # 160 rows total — fast but big enough for sklearn
N_FEATURES  = 63          # 21 landmarks × 3 coords (x, y, z)


def _make_df() -> pd.DataFrame:
    """Create a reproducible synthetic gesture dataset."""
    rng = np.random.default_rng(42)
    rows = []
    for i, gesture in enumerate(GESTURE_CLASSES):
        # Give each class a distinct mean so the model can actually learn
        coords = rng.normal(loc=float(i), scale=0.3, size=(N_PER_CLASS, N_FEATURES))
        labels = np.full(N_PER_CLASS, gesture)
        rows.append(np.column_stack([labels, coords]))

    cols = ["label"] + [
        f"{axis}{idx}"
        for idx in range(21)
        for axis in ("x", "y", "z")
    ]
    df = pd.DataFrame(np.vstack(rows), columns=cols)
    # Cast feature columns to float
    df[cols[1:]] = df[cols[1:]].astype(float)
    return df


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return _make_df()


@pytest.fixture
def synthetic_csv(tmp_path, synthetic_df) -> Path:
    """Write the synthetic dataset to a temp CSV and return its path."""
    path = tmp_path / "dataset.csv"
    synthetic_df.to_csv(path, index=False)
    return path


# ── Mock keyboard ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_keyboard():
    """Patch pynput's keyboard controller so no real key-presses happen."""
    with patch("src.controller.KeyboardController") as cls_mock:
        instance = MagicMock()
        cls_mock.return_value = instance
        yield instance


# ── Mock ML artefacts ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_ml_models():
    """
    Return (model, scaler, encoder) mocks that mimic the sklearn interface
    used by GestureRecognizer.
    """
    model   = MagicMock()
    scaler  = MagicMock()
    encoder = MagicMock()

    # scaler.transform returns the input unchanged (already scaled)
    scaler.transform.side_effect = lambda x: np.array(x)

    # model.predict_proba returns equal probabilities across 4 classes by default
    proba = np.array([[0.25, 0.25, 0.25, 0.25]])
    model.predict_proba.return_value = proba

    # encoder.inverse_transform returns a gesture name
    encoder.inverse_transform.return_value = np.array(["posun nahoru"])

    return model, scaler, encoder


# ── Mock MediaPipe landmark ────────────────────────────────────────────────────

def make_landmark(x=0.5, y=0.5, z=0.0):
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    return lm


def make_landmarks_list(n=21):
    """21 fake hand-landmark objects."""
    return [make_landmark(x=i * 0.04, y=i * 0.03) for i in range(n)]
