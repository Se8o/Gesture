#!/usr/bin/env python3
"""
Vstupní bod aplikace Gesture Remote Controller.

Spuštění – normální režim (okno s kamerou):
    python run.py

Spuštění – na pozadí (ikona v liště, žádné okno):
    python run.py --background

Před prvním spuštěním je nutné natrénovat model:
    python train.py
"""
import sys
import os
import argparse

# Přidáme kořenový adresář projektu do sys.path, aby fungovaly importy z balíčku src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gesture.app import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gesture Remote Controller – ovládání PC gesty ruky"
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Spustí aplikaci na pozadí bez okna kamery (ovládání přes ikonu v liště)",
    )
    args = parser.parse_args()
    run(background=args.background)
