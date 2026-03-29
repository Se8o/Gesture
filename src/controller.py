"""
Modul pro převod rozpoznaného gesta na akci operačního systému.

Každé gesto je namapováno na šipkovou klávesu, která je universálně
použitelná pro ovládání prezentací, videí, webového prohlížeče i
mediálních přehrávačů.

Mapování:
  posun nahoru  → šipka Nahoru   (hlasitost ↑, předchozí snímek, scroll nahoru)
  posun dolu    → šipka Dolů     (hlasitost ↓, další snímek, scroll dolů)
  posun doprava → šipka Doprava  (přetočení vpřed, další snímek prezentace)
  posun doleva  → šipka Doleva   (přetočení zpět, předchozí snímek)
"""
import time
from pynput.keyboard import Key, Controller as KeyboardController


class GestureController:
    """Přijímá název gesta a odesílá odpovídající stisk klávesy."""

    GESTURE_TO_KEY = {
        "posun nahoru":  Key.up,
        "posun dolu":    Key.down,
        "posun doprava": Key.right,
        "posun doleva":  Key.left,
    }

    def __init__(self, cooldown: float):
        """
        Parametry
        ---------
        cooldown : float
            Minimální prodleva v sekundách mezi dvěma akcemi stejného gesta.
            Různá gesta lze spustit ihned po sobě bez čekání.
        """
        self._keyboard      = KeyboardController()
        self._cooldown      = cooldown
        self._last_time     = 0.0
        self._last_gesture  = None

    def execute(self, gesture: str) -> bool:
        """
        Provede akci pro dané gesto, pokud uplynula dostatečná prodleva.

        Vrátí True  pokud byla klávesa stisknuta.
        Vrátí False pokud je ještě aktivní cooldown (akce se neprovede).
        """
        now = time.time()

        # Stejné gesto smí opakovat až po uplynutí cooldownu.
        # Jiné gesto může spustit okamžitě (bez čekání).
        if gesture == self._last_gesture and now - self._last_time < self._cooldown:
            return False

        key = self.GESTURE_TO_KEY.get(gesture)
        if key is None:
            return False

        # tap() = press + release (jeden stisk klávesy)
        self._keyboard.tap(key)
        self._last_time    = now
        self._last_gesture = gesture
        return True
