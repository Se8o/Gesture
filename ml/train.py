#!/usr/bin/env python3
"""
Train the gesture recognition ML model from collected data.

Reads:   data/dataset.csv
Writes:  models/model.pkl          – trained Random Forest classifier
         models/scaler.pkl         – fitted StandardScaler
         models/label_encoder.pkl  – fitted LabelEncoder

Usage (from the project root):
    python ml/train.py
"""
import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Project root is two levels up from this file (ml/train.py → project root)
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "dataset.csv")
MODEL_DIR = os.path.join(ROOT, "models")


def train() -> None:
    # ------------------------------------------------------------------ #
    # 1. Load the dataset
    # ------------------------------------------------------------------ #
    if not os.path.isfile(DATA_PATH):
        print(f"ERROR: Dataset not found: {DATA_PATH}")
        sys.exit(1)

    print(f"Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # ------------------------------------------------------------------ #
    # 2. Clean the data — drop rows where MediaPipe failed to detect a hand
    # ------------------------------------------------------------------ #
    before  = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    removed = before - len(df)
    print(f"Records after cleaning: {len(df)}  (removed NaN rows: {removed})")

    # ------------------------------------------------------------------ #
    # 3. Split into features (X) and target labels (y)
    # ------------------------------------------------------------------ #
    X = df.iloc[:, 1:].values   # 63 numeric columns — hand joint coordinates
    y = df.iloc[:, 0].values    # gesture name (text label)

    # Print class distribution so we can spot imbalances
    classes, counts = zip(*sorted(
        {cls: (y == cls).sum() for cls in set(y)}.items()
    ))
    print(f"\nClass distribution ({len(classes)} gestures, {X.shape[1]} features):")
    for cls, cnt in zip(classes, counts):
        print(f"  {cls}: {cnt} samples")

    # ------------------------------------------------------------------ #
    # 4. Encode text labels to integers
    #    LabelEncoder assigns a number to each class alphabetically.
    # ------------------------------------------------------------------ #
    encoder = LabelEncoder()
    y_enc   = encoder.fit_transform(y)

    # ------------------------------------------------------------------ #
    # 5. Split into training (80 %) and test (20 %) sets
    #    stratify=y_enc ensures both sets have the same class proportions.
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # ------------------------------------------------------------------ #
    # 6. Scale features — StandardScaler (mean=0, std=1)
    #    IMPORTANT: fit only on training data to avoid data leakage.
    # ------------------------------------------------------------------ #
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit + transform training set
    X_test  = scaler.transform(X_test)        # transform only (no fit) on test set

    # ------------------------------------------------------------------ #
    # 7. Train the Random Forest classifier
    #    100 decision trees, each voting on the gesture class.
    # ------------------------------------------------------------------ #
    print("\nTraining RandomForestClassifier (100 trees) ...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,        # use all available CPU cores
    )
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # 8. Evaluate on the held-out test set
    # ------------------------------------------------------------------ #
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\nTest set accuracy: {acc:.4f}  ({acc * 100:.1f} %)")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # ------------------------------------------------------------------ #
    # 9. Save the model and helper objects to disk
    #    All three files are needed by the application at runtime.
    # ------------------------------------------------------------------ #
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,   os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler,  os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print(f"\nFiles saved to: {MODEL_DIR}/")
    print("  model.pkl, scaler.pkl, label_encoder.pkl")
    print("\nModel is ready. Run the app with:  python run.py")


if __name__ == "__main__":
    train()
