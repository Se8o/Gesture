"""
Translates a recognised gesture name into an operating system action.

Gesture mapping:
  posun nahoru  → scroll up   /  ↑ arrow key  (depends on CONTROL_MODE)
  posun dolu    → scroll down /  ↓ arrow key
  posun doprava → next browser tab  (Ctrl+Tab)
  posun doleva  → previous browser tab  (Ctrl+Shift+Tab)
  pauza         → spacebar  (play / pause)

Tab switching and pause use a streak mechanism: the gesture must appear in
HOTKEY_MIN_STREAK consecutive frames before the action fires, preventing
accidental activation mid-movement.
"""
import time
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Controller as MouseController


class GestureController:
    """Receives a gesture name and executes the corresponding action."""

    GESTURE_TO_SCROLL = {
        "posun nahoru": (0,  1),
        "posun dolu":   (0, -1),
    }
    GESTURE_TO_KEY = {
        "posun nahoru": Key.up,
        "posun dolu":   Key.down,
    }
    GESTURE_TO_HOTKEY = {
        "posun doprava": ([Key.ctrl],            Key.tab),   # Ctrl+Tab
        "posun doleva":  ([Key.ctrl, Key.shift], Key.tab),   # Ctrl+Shift+Tab
    }
    GESTURE_TO_SIMPLE_KEY = {
        "pauza": Key.space,
    }

    HOTKEY_MIN_CONFIDENCE = 0.85
    HOTKEY_MIN_STREAK     = 4

    def __init__(self, cooldown: float, mode: str = "keyboard", scroll_amount: int = 5):
        self._keyboard      = KeyboardController()
        self._mouse         = MouseController()
        self._cooldown      = cooldown
        self._mode          = mode
        self._scroll_amount = scroll_amount
        self._last_time     = 0.0
        self._last_gesture  = None
        self._streak_gesture = None
        self._streak_count   = 0

    def _check_streak(self, gesture: str, confidence: float) -> bool:
        """Increment streak counter; return True when threshold is reached."""
        if confidence >= self.HOTKEY_MIN_CONFIDENCE and gesture == self._streak_gesture:
            self._streak_count += 1
        else:
            self._streak_gesture = gesture
            self._streak_count   = 1 if confidence >= self.HOTKEY_MIN_CONFIDENCE else 0

        if self._streak_count >= self.HOTKEY_MIN_STREAK:
            self._streak_count   = 0
            self._streak_gesture = None
            return True
        return False

    def _reset_streak(self):
        self._streak_gesture = None
        self._streak_count   = 0

    def execute(self, gesture: str, confidence: float = 1.0) -> bool:
        """
        Execute the action for the given gesture if cooldown has elapsed.

        Returns True if an action was performed, False otherwise.
        """
        now = time.time()
        if gesture == self._last_gesture and now - self._last_time < self._cooldown:
            return False

        if gesture in self.GESTURE_TO_SIMPLE_KEY:
            if not self._check_streak(gesture, confidence):
                return False
            self._keyboard.tap(self.GESTURE_TO_SIMPLE_KEY[gesture])

        elif gesture in self.GESTURE_TO_HOTKEY:
            if not self._check_streak(gesture, confidence):
                return False
            modifiers, key = self.GESTURE_TO_HOTKEY[gesture]
            for mod in modifiers:
                self._keyboard.press(mod)
            self._keyboard.tap(key)
            for mod in reversed(modifiers):
                self._keyboard.release(mod)

        elif self._mode == "scroll":
            self._reset_streak()
            direction = self.GESTURE_TO_SCROLL.get(gesture)
            if direction is None:
                return False
            dx, dy = direction
            self._mouse.scroll(dx * self._scroll_amount, dy * self._scroll_amount)

        else:
            self._reset_streak()
            key = self.GESTURE_TO_KEY.get(gesture)
            if key is None:
                return False
            self._keyboard.tap(key)

        self._last_time    = now
        self._last_gesture = gesture
        return True
