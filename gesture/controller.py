"""
Modul pro převod rozpoznaného gesta na akci operačního systému.

Podporuje dva režimy:
  "keyboard" – odesílá šipkové klávesy (prezentace, videopřehrávač, ...)
  "scroll"   – odesílá události rolování myší (web, dokumenty, nahrazuje trackpad)

Mapování gest:
  posun nahoru  → ↑ klávesa  /  scroll nahoru
  posun dolu    → ↓ klávesa  /  scroll dolů
  posun doprava → → klávesa  /  scroll doprava
  posun doleva  ← ← klávesa  /  scroll doleva
"""
import time
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Controller as MouseController


class GestureController:
    """Přijímá název gesta a provede odpovídající akci (klávesa nebo scroll)."""

    GESTURE_TO_KEY = {
        "posun nahoru":  Key.up,
        "posun dolu":    Key.down,
        "posun doprava": Key.right,
        "posun doleva":  Key.left,
    }

    # (dx, dy) – kladné dy = scroll nahoru, záporné dy = scroll dolů
    GESTURE_TO_SCROLL = {
        "posun nahoru":  (0,  1),
        "posun dolu":    (0, -1),
        "posun doprava": (1,  0),
        "posun doleva":  (-1, 0),
    }

    def __init__(self, cooldown: float, mode: str = "keyboard", scroll_amount: int = 5):
        """
        Parametry
        ---------
        cooldown      : minimální prodleva (s) mezi dvěma akcemi stejného gesta
        mode          : "keyboard" nebo "scroll"
        scroll_amount : počet jednotek scrollu na jedno gesto
        """
        self._keyboard      = KeyboardController()
        self._mouse         = MouseController()
        self._cooldown      = cooldown
        self._mode          = mode
        self._scroll_amount = scroll_amount
        self._last_time     = 0.0
        self._last_gesture  = None

    def execute(self, gesture: str) -> bool:
        """
        Provede akci pro dané gesto, pokud uplynula dostatečná prodleva.

        Vrátí True  pokud byla akce provedena.
        Vrátí False pokud je aktivní cooldown, nebo gesto není rozpoznáno.
        """
        now = time.time()

        if gesture == self._last_gesture and now - self._last_time < self._cooldown:
            return False

        if self._mode == "scroll":
            smer = self.GESTURE_TO_SCROLL.get(gesture)
            if smer is None:
                return False
            dx, dy = smer
            self._mouse.scroll(dx * self._scroll_amount, dy * self._scroll_amount)
        else:
            key = self.GESTURE_TO_KEY.get(gesture)
            if key is None:
                return False
            self._keyboard.tap(key)

        self._last_time    = now
        self._last_gesture = gesture
        return True
