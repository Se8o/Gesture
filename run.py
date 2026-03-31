#!/usr/bin/env python3
"""
Entry point for Gesture Remote Controller.

Normal mode (camera window):
    python run.py

Background mode (system tray, no window):
    python run.py --background

The model must be trained before the first run:
    python ml/train.py
"""
import sys
import os
import argparse

# Add the project root to sys.path so imports from the gesture/ package work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gesture.app import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gesture Remote Controller — control your PC with hand gestures"
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Run in background mode without a camera window (controlled via system tray icon)",
    )
    args = parser.parse_args()
    run(background=args.background)
