"""
Tests for gesture/controller.py — gesture-to-action mapping, streak, and cooldown logic.
"""
import time
from unittest.mock import call, patch

import pytest
from pynput.keyboard import Key

from gesture.controller import GestureController


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_controller(cooldown=0.0, mode="keyboard"):
    """Create a GestureController with a mocked keyboard and mouse."""
    with patch("gesture.controller.KeyboardController") as kb_mock, \
         patch("gesture.controller.MouseController") as ms_mock:
        ctrl = GestureController(cooldown=cooldown, mode=mode)
        ctrl._keyboard = kb_mock.return_value
        ctrl._mouse    = ms_mock.return_value
        return ctrl


def fire_streak(ctrl, gesture, n=None, confidence=1.0):
    """Call execute() enough times to satisfy the streak requirement."""
    if n is None:
        n = GestureController.HOTKEY_MIN_STREAK
    result = False
    for _ in range(n):
        result = ctrl.execute(gesture, confidence=confidence)
    return result


# ── Gesture categories ────────────────────────────────────────────────────────

class TestGestureCategories:
    """Verify gestures are in the correct internal maps."""

    def test_scroll_gestures_in_scroll_map(self):
        assert "posun nahoru" in GestureController.GESTURE_TO_SCROLL
        assert "posun dolu"   in GestureController.GESTURE_TO_SCROLL

    def test_keyboard_gestures_in_key_map(self):
        assert "posun nahoru" in GestureController.GESTURE_TO_KEY
        assert "posun dolu"   in GestureController.GESTURE_TO_KEY

    def test_tab_gestures_in_hotkey_map(self):
        assert "posun doprava" in GestureController.GESTURE_TO_HOTKEY
        assert "posun doleva"  in GestureController.GESTURE_TO_HOTKEY

    def test_pauza_in_simple_key_map(self):
        assert "pauza" in GestureController.GESTURE_TO_SIMPLE_KEY

    def test_pauza_mapped_to_space(self):
        assert GestureController.GESTURE_TO_SIMPLE_KEY["pauza"] == Key.space

    def test_nahoru_mapped_to_up(self):
        assert GestureController.GESTURE_TO_KEY["posun nahoru"] == Key.up

    def test_dolu_mapped_to_down(self):
        assert GestureController.GESTURE_TO_KEY["posun dolu"] == Key.down


# ── Keyboard mode: scroll up/down ─────────────────────────────────────────────

class TestKeyboardMode:
    def test_nahoru_taps_up_key(self):
        ctrl = make_controller(mode="keyboard")
        assert ctrl.execute("posun nahoru") is True
        ctrl._keyboard.tap.assert_called_once_with(Key.up)

    def test_dolu_taps_down_key(self):
        ctrl = make_controller(mode="keyboard")
        assert ctrl.execute("posun dolu") is True
        ctrl._keyboard.tap.assert_called_once_with(Key.down)

    def test_unknown_gesture_returns_false(self):
        ctrl = make_controller(mode="keyboard")
        assert ctrl.execute("nezname gesto") is False
        ctrl._keyboard.tap.assert_not_called()

    def test_none_gesture_returns_false(self):
        ctrl = make_controller(mode="keyboard")
        assert ctrl.execute(None) is False
        ctrl._keyboard.tap.assert_not_called()


# ── Scroll mode ───────────────────────────────────────────────────────────────

class TestScrollMode:
    def test_nahoru_scrolls_up(self):
        ctrl = make_controller(mode="scroll")
        assert ctrl.execute("posun nahoru") is True
        ctrl._mouse.scroll.assert_called_once()
        _, dy = ctrl._mouse.scroll.call_args[0]
        assert dy > 0

    def test_dolu_scrolls_down(self):
        ctrl = make_controller(mode="scroll")
        assert ctrl.execute("posun dolu") is True
        ctrl._mouse.scroll.assert_called_once()
        _, dy = ctrl._mouse.scroll.call_args[0]
        assert dy < 0


# ── Hotkey gestures (doprava / doleva) ────────────────────────────────────────

class TestHotkeyGestures:
    def test_doprava_needs_full_streak(self):
        ctrl = make_controller()
        streak = GestureController.HOTKEY_MIN_STREAK
        for i in range(streak - 1):
            assert ctrl.execute("posun doprava", confidence=1.0) is False
        assert ctrl.execute("posun doprava", confidence=1.0) is True

    def test_doleva_needs_full_streak(self):
        ctrl = make_controller()
        streak = GestureController.HOTKEY_MIN_STREAK
        for i in range(streak - 1):
            assert ctrl.execute("posun doleva", confidence=1.0) is False
        assert ctrl.execute("posun doleva", confidence=1.0) is True

    def test_doprava_sends_ctrl_tab(self):
        ctrl = make_controller()
        fire_streak(ctrl, "posun doprava")
        ctrl._keyboard.press.assert_any_call(Key.ctrl)
        ctrl._keyboard.tap.assert_called_with(Key.tab)

    def test_doleva_sends_ctrl_shift_tab(self):
        ctrl = make_controller()
        fire_streak(ctrl, "posun doleva")
        ctrl._keyboard.press.assert_any_call(Key.ctrl)
        ctrl._keyboard.press.assert_any_call(Key.shift)
        ctrl._keyboard.tap.assert_called_with(Key.tab)

    def test_low_confidence_resets_streak(self):
        ctrl = make_controller()
        ctrl.execute("posun doprava", confidence=1.0)   # streak = 1
        ctrl.execute("posun doprava", confidence=0.1)   # below threshold → reset
        # Need full streak again — previous progress lost
        streak = GestureController.HOTKEY_MIN_STREAK
        for _ in range(streak - 1):
            ctrl.execute("posun doprava", confidence=1.0)
        assert ctrl.execute("posun doprava", confidence=1.0) is True

    def test_switching_gesture_resets_streak(self):
        ctrl = make_controller()
        ctrl.execute("posun doprava", confidence=1.0)  # streak = 1
        ctrl.execute("posun doleva",  confidence=1.0)  # different gesture → reset
        # doprava streak is gone — needs full streak again
        for _ in range(GestureController.HOTKEY_MIN_STREAK - 1):
            ctrl.execute("posun doprava", confidence=1.0)
        assert ctrl.execute("posun doprava", confidence=1.0) is True


# ── Pauza (simple key with streak) ───────────────────────────────────────────

class TestPauza:
    def test_pauza_needs_full_streak(self):
        ctrl = make_controller()
        streak = GestureController.HOTKEY_MIN_STREAK
        for _ in range(streak - 1):
            assert ctrl.execute("pauza", confidence=1.0) is False
        assert ctrl.execute("pauza", confidence=1.0) is True

    def test_pauza_taps_space(self):
        ctrl = make_controller()
        fire_streak(ctrl, "pauza")
        ctrl._keyboard.tap.assert_called_with(Key.space)


# ── Cooldown logic ─────────────────────────────────────────────────────────────

class TestCooldown:
    def test_same_gesture_blocked_within_cooldown(self):
        ctrl = make_controller(cooldown=5.0, mode="keyboard")
        assert ctrl.execute("posun nahoru") is True
        assert ctrl.execute("posun nahoru") is False

    def test_same_gesture_allowed_after_cooldown(self, monkeypatch):
        ctrl = make_controller(cooldown=1.0, mode="keyboard")
        ctrl.execute("posun nahoru")
        monkeypatch.setattr("gesture.controller.time.time",
                            lambda: ctrl._last_time + 1.1)
        assert ctrl.execute("posun nahoru") is True

    def test_different_gesture_not_blocked_by_cooldown(self):
        ctrl = make_controller(cooldown=5.0, mode="keyboard")
        assert ctrl.execute("posun nahoru") is True
        assert ctrl.execute("posun dolu")   is True

    def test_last_gesture_updated_after_execute(self):
        ctrl = make_controller(mode="keyboard")
        ctrl.execute("posun nahoru")
        assert ctrl._last_gesture == "posun nahoru"

    def test_last_time_updated_after_execute(self):
        ctrl = make_controller(mode="keyboard")
        before = time.time()
        ctrl.execute("posun nahoru")
        assert ctrl._last_time >= before

    def test_zero_cooldown_always_allows_same_gesture(self):
        ctrl = make_controller(cooldown=0.0, mode="keyboard")
        for _ in range(5):
            assert ctrl.execute("posun nahoru") is True
