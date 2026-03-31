"""
Translates a recognised gesture name into an operating system action.

Gesture mapping:
  posun nahoru  → scroll up   /  ↑ arrow key  (depends on control mode)
  posun dolu    → scroll down /  ↓ arrow key  (depends on control mode)
  posun doprava → switch to right browser tab  (Ctrl+Tab)
  posun doleva  → switch to left browser tab   (Ctrl+Shift+Tab)
  pauza         → spacebar  (play / pause video)

Tab switching and pause use a streak mechanism:
  The gesture must appear in HOTKEY_MIN_STREAK consecutive frames before
  the action fires. This prevents accidental activation when the hand
  is moving between scroll gestures.
"""
import time
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Controller as MouseController


class GestureController:
    """Receives a gesture name and executes the corresponding action."""

    # Gesture → arrow key mapping (used in keyboard mode)
    GESTURE_TO_KEY = {
        "posun nahoru": Key.up,
        "posun dolu":   Key.down,
    }

    # Gesture → (dx, dy) scroll direction
    # Positive dy = scroll up, negative dy = scroll down
    GESTURE_TO_SCROLL = {
        "posun nahoru": (0,  1),
        "posun dolu":   (0, -1),
    }

    # Gesture → keyboard shortcut: (list of modifier keys, main key)
    GESTURE_TO_HOTKEY = {
        "posun doprava": ([Key.ctrl],            Key.tab),   # Ctrl+Tab
        "posun doleva":  ([Key.ctrl, Key.shift], Key.tab),   # Ctrl+Shift+Tab
    }

    # Gesture → single key press (works in both control modes)
    GESTURE_TO_SIMPLE_KEY = {
        "pauza": Key.space,
    }

    HOTKEY_MIN_CONFIDENCE = 0.70  # minimum model confidence to count toward a streak
    HOTKEY_MIN_STREAK     = 4     # number of consecutive frames required to fire the action

    def __init__(self, cooldown: float, mode: str = "keyboard", scroll_amount: int = 5):
        self._keyboard      = KeyboardController()
        self._mouse         = MouseController()
        self._cooldown      = cooldown       # seconds between repeated same-gesture actions
        self._mode          = mode           # "scroll" or "keyboard"
        self._scroll_amount = scroll_amount  # scroll units per gesture
        self._last_time     = 0.0            # timestamp of the last executed action
        self._last_gesture  = None           # name of the last executed gesture

        # Streak tracking for hotkey/pause gestures
        self._hotkey_candidate = None   # gesture currently building up a streak
        self._hotkey_streak    = 0      # how many consecutive frames it has appeared

    def execute(self, gesture: str, confidence: float = 1.0) -> bool:
        """
        Execute the action for the given gesture if the cooldown has elapsed.

        Returns True  if an action was performed.
        Returns False if the gesture is on cooldown, streak is not met, or
                      confidence is too low.
        """
        now = time.time()

        # Block the same gesture if it was just executed within the cooldown window
        if gesture == self._last_gesture and now - self._last_time < self._cooldown:
            return False

        # ── Simple key gestures (pause / spacebar) ────────────────────────────
        if gesture in self.GESTURE_TO_SIMPLE_KEY:
            # Increment streak if this is the same gesture with enough confidence
            if confidence >= self.HOTKEY_MIN_CONFIDENCE and gesture == self._hotkey_candidate:
                self._hotkey_streak += 1
            else:
                # New gesture or low confidence — reset streak counter
                self._hotkey_candidate = gesture
                self._hotkey_streak    = 1 if confidence >= self.HOTKEY_MIN_CONFIDENCE else 0

            # Not enough consecutive frames yet — wait
            if self._hotkey_streak < self.HOTKEY_MIN_STREAK:
                return False

            # Streak reached — press the key and reset
            self._hotkey_streak    = 0
            self._hotkey_candidate = None
            self._keyboard.tap(self.GESTURE_TO_SIMPLE_KEY[gesture])

        # ── Tab switching gestures (doprava / doleva) ─────────────────────────
        elif gesture in self.GESTURE_TO_HOTKEY:
            if confidence >= self.HOTKEY_MIN_CONFIDENCE and gesture == self._hotkey_candidate:
                self._hotkey_streak += 1
            else:
                # New gesture or low confidence — reset streak
                self._hotkey_candidate = gesture
                self._hotkey_streak    = 1 if confidence >= self.HOTKEY_MIN_CONFIDENCE else 0

            if self._hotkey_streak < self.HOTKEY_MIN_STREAK:
                return False

            # Streak reached — send the keyboard shortcut and reset
            self._hotkey_streak    = 0
            self._hotkey_candidate = None
            modifiers, key = self.GESTURE_TO_HOTKEY[gesture]
            for mod in modifiers:
                self._keyboard.press(mod)      # press each modifier key
            self._keyboard.tap(key)            # tap the main key
            for mod in reversed(modifiers):
                self._keyboard.release(mod)    # release modifiers in reverse order

        # ── Scroll mode (nahoru / dolu) ───────────────────────────────────────
        elif self._mode == "scroll":
            self._hotkey_candidate = None
            self._hotkey_streak    = 0
            direction = self.GESTURE_TO_SCROLL.get(gesture)
            if direction is None:
                return False
            dx, dy = direction
            self._mouse.scroll(dx * self._scroll_amount, dy * self._scroll_amount)

        # ── Keyboard mode (nahoru / dolu) ─────────────────────────────────────
        else:
            self._hotkey_candidate = None
            self._hotkey_streak    = 0
            key = self.GESTURE_TO_KEY.get(gesture)
            if key is None:
                return False
            self._keyboard.tap(key)

        # Record the time and gesture name of this successful action
        self._last_time    = now
        self._last_gesture = gesture
        return True
