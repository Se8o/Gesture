"""
Tests for src/controller.py — gesture-to-key mapping and cooldown logic.
"""
import time
from unittest.mock import call, patch

import pytest
from pynput.keyboard import Key

from src.controller import GestureController


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_controller(cooldown=1.0):
    """Create a GestureController with a mocked keyboard."""
    with patch("src.controller.KeyboardController") as cls_mock:
        ctrl = GestureController(cooldown=cooldown)
        ctrl._keyboard = cls_mock.return_value   # expose mock for assertions
        return ctrl


# ── Gesture mapping ────────────────────────────────────────────────────────────

class TestGestureMapping:
    """Every supported gesture must map to the correct arrow key."""

    EXPECTED = {
        "posun nahoru":  Key.up,
        "posun dolu":    Key.down,
        "posun doprava": Key.right,
        "posun doleva":  Key.left,
    }

    def test_all_gestures_present_in_map(self):
        assert set(GestureController.GESTURE_TO_KEY.keys()) == set(self.EXPECTED.keys())

    @pytest.mark.parametrize("gesture,expected_key", EXPECTED.items())
    def test_correct_key_pressed(self, gesture, expected_key):
        ctrl = make_controller()
        result = ctrl.execute(gesture)
        assert result is True
        ctrl._keyboard.tap.assert_called_once_with(expected_key)

    def test_unknown_gesture_returns_false(self):
        ctrl = make_controller()
        result = ctrl.execute("nezname gesto")
        assert result is False
        ctrl._keyboard.tap.assert_not_called()

    def test_none_gesture_returns_false(self):
        ctrl = make_controller()
        result = ctrl.execute(None)
        assert result is False
        ctrl._keyboard.tap.assert_not_called()


# ── Cooldown logic ─────────────────────────────────────────────────────────────

class TestCooldown:
    def test_same_gesture_blocked_within_cooldown(self):
        ctrl = make_controller(cooldown=5.0)
        assert ctrl.execute("posun nahoru") is True   # first press → OK
        assert ctrl.execute("posun nahoru") is False  # too soon → blocked

    def test_same_gesture_allowed_after_cooldown(self, monkeypatch):
        ctrl = make_controller(cooldown=1.0)
        ctrl.execute("posun nahoru")

        # Fast-forward time past the cooldown
        monkeypatch.setattr("src.controller.time.time",
                            lambda: ctrl._last_time + 1.1)

        assert ctrl.execute("posun nahoru") is True

    def test_different_gesture_not_blocked_by_cooldown(self):
        """Switching gestures must bypass the same-gesture cooldown."""
        ctrl = make_controller(cooldown=5.0)
        assert ctrl.execute("posun nahoru") is True
        # Immediately try a *different* gesture — should succeed
        assert ctrl.execute("posun dolu")   is True

    def test_key_only_tapped_when_not_in_cooldown(self):
        ctrl = make_controller(cooldown=5.0)
        ctrl.execute("posun doprava")
        ctrl.execute("posun doprava")  # blocked
        ctrl._keyboard.tap.assert_called_once()  # only one actual tap

    def test_last_gesture_updated_after_execute(self):
        ctrl = make_controller()
        ctrl.execute("posun doleva")
        assert ctrl._last_gesture == "posun doleva"

    def test_last_time_updated_after_execute(self):
        ctrl = make_controller()
        before = time.time()
        ctrl.execute("posun doleva")
        assert ctrl._last_time >= before

    def test_zero_cooldown_always_allows_same_gesture(self):
        ctrl = make_controller(cooldown=0.0)
        for _ in range(5):
            assert ctrl.execute("posun nahoru") is True
