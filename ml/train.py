#!/usr/bin/env python3
"""
Trénování ML modelu z nasbíraných dat.

Čte:      data/dataset.csv
Zapisuje: models/model.pkl
          models/scaler.pkl
          models/label_encoder.pkl

Spuštění (z kořenového adresáře projektu):
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

# Kořenový adresář projektu (o dvě úrovně výše: ml/ → Gesture/)
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "dataset.csv")
MODEL_DIR = os.path.join(ROOT, "models")


def train() -> None:
    # ------------------------------------------------------------------ #
    # 1. Načtení dat
    # ------------------------------------------------------------------ #
    if not os.path.isfile(DATA_PATH):
        print(f"CHYBA: Dataset nebyl nalezen: {DATA_PATH}")
        sys.exit(1)

    print(f"Nacitam dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # ------------------------------------------------------------------ #
    # 2. Čištění dat
    # ------------------------------------------------------------------ #
    before = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    removed = before - len(df)
    print(f"Zaznamu po cisteni: {len(df)}  (odebrano NaN: {removed})")

    # ------------------------------------------------------------------ #
    # 3. Rozdělení na příznaky (X) a cílovou proměnnou (y)
    # ------------------------------------------------------------------ #
    X = df.iloc[:, 1:].values   # 63 numerických sloupců (souřadnice kloubů)
    y = df.iloc[:, 0].values    # textové názvy gest

    classes, counts = zip(*sorted(
        {cls: (y == cls).sum() for cls in set(y)}.items()
    ))
    print(f"\nRozdělení tříd ({len(classes)} gestura, {X.shape[1]} příznaků):")
    for cls, cnt in zip(classes, counts):
        print(f"  {cls}: {cnt} záznamu")

    # ------------------------------------------------------------------ #
    # 4. Kódování textových labelů na čísla
    # ------------------------------------------------------------------ #
    encoder = LabelEncoder()
    y_enc   = encoder.fit_transform(y)

    # ------------------------------------------------------------------ #
    # 5. Rozdělení na trénovací a testovací sadu (80 / 20)
    # stratify zajistí, že obě sady mají stejné rozložení tříd
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # ------------------------------------------------------------------ #
    # 6. Škálování – StandardScaler (průměr=0, odchylka=1)
    # fit POUZE na trénovacích datech!
    # ------------------------------------------------------------------ #
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)   # pouze transform, NE fit

    # ------------------------------------------------------------------ #
    # 7. Trénování modelu – Random Forest Classifier
    # ------------------------------------------------------------------ #
    print("\nTrenuji RandomForestClassifier (100 stromu) ...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,        # využije všechna dostupná CPU jádra
    )
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # 8. Vyhodnocení na testovací sadě
    # ------------------------------------------------------------------ #
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\nPresnost na testovacich datech: {acc:.4f}  ({acc * 100:.1f} %)")
    print("\nKlasifikacni zprava:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # ------------------------------------------------------------------ #
    # 9. Uložení modelu a pomocných objektů
    # ------------------------------------------------------------------ #
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,   os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler,  os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print(f"\nSoubory ulozeny do: {MODEL_DIR}/")
    print("  model.pkl, scaler.pkl, label_encoder.pkl")
    print("\nModel je pripraveny. Spust aplikaci prikazem:  python run.py")


if __name__ == "__main__":
    train()
