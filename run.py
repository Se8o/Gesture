#!/usr/bin/env python3
"""
Vstupní bod aplikace Gesture Remote Controller.

Spuštění:
    python run.py

Před prvním spuštěním je nutné natrénovat model:
    python train.py
"""
import sys
import os

# Přidáme kořenový adresář projektu do sys.path, aby fungovaly importy z balíčku src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.app import run

if __name__ == "__main__":
    run()
