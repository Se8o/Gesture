"""
Modul pro převod rozpoznaného gesta na akci operačního systému.

Mapování gest:
  posun nahoru  → scroll nahoru  /  ↑ klávesa (keyboard režim)
  posun dolu    → scroll dolů    /  ↓ klávesa (keyboard režim)
  posun doprava → přepnutí na pravý tab prohlížeče (Ctrl+Tab)
  posun doleva  → přepnutí na levý tab prohlížeče (Ctrl+Shift+Tab)
  pauza         → mezerník (spuštění / pozastavení videa)

Přepínání tabů a pauza používají streak mechanismus – gesto musí být detekováno
HOTKEY_MIN_STREAK po sobě jdoucích snímků, než se akce spustí.
To zabrání náhodné aktivaci při pohybu ruky při scrollování.
"""
import time
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Controller as MouseController


class GestureController:
    """Přijímá název gesta a provede odpovídající akci."""

    GESTURE_TO_KEY = {
        "posun nahoru": Key.up,
        "posun dolu":   Key.down,
    }

    # (dx, dy) – kladné dy = scroll nahoru, záporné dy = scroll dolů
    GESTURE_TO_SCROLL = {
        "posun nahoru": (0,  1),
        "posun dolu":   (0, -1),
    }

    # Klávesové zkratky pro přepínání tabů: (seznam modifikátorů, hlavní klávesa)
    GESTURE_TO_HOTKEY = {
        "posun doprava": ([Key.ctrl],            Key.tab),
        "posun doleva":  ([Key.ctrl, Key.shift], Key.tab),
    }

    # Gesta mapovaná na jednoduchou klávesu (fungují v obou režimech)
    GESTURE_TO_SIMPLE_KEY = {
        "pauza": Key.space,
    }

    HOTKEY_MIN_CONFIDENCE = 0.70  # minimální jistota modelu pro tab/pauza gesto
    HOTKEY_MIN_STREAK     = 4     # počet po sobě jdoucích snímků nutných ke spuštění

    def __init__(self, cooldown: float, mode: str = "keyboard", scroll_amount: int = 5):
        self._keyboard      = KeyboardController()
        self._mouse         = MouseController()
        self._cooldown      = cooldown
        self._mode          = mode
        self._scroll_amount = scroll_amount
        self._last_time     = 0.0
        self._last_gesture  = None

        # Streak counter pro tab gesta
        self._hotkey_candidate = None
        self._hotkey_streak    = 0

    def execute(self, gesture: str, confidence: float = 1.0) -> bool:
        """
        Provede akci pro dané gesto, pokud uplynula dostatečná prodleva.

        Vrátí True  pokud byla akce provedena.
        Vrátí False pokud je aktivní cooldown, streak nestačí, nebo nízká jistota.
        """
        now = time.time()

        if gesture == self._last_gesture and now - self._last_time < self._cooldown:
            return False

        # ── Jednoduché klávesy (pauza atd.) ──────────────────────────────────
        if gesture in self.GESTURE_TO_SIMPLE_KEY:
            if confidence >= self.HOTKEY_MIN_CONFIDENCE and gesture == self._hotkey_candidate:
                self._hotkey_streak += 1
            else:
                self._hotkey_candidate = gesture
                self._hotkey_streak    = 1 if confidence >= self.HOTKEY_MIN_CONFIDENCE else 0

            if self._hotkey_streak < self.HOTKEY_MIN_STREAK:
                return False

            self._hotkey_streak    = 0
            self._hotkey_candidate = None
            self._keyboard.tap(self.GESTURE_TO_SIMPLE_KEY[gesture])

        # ── Tab přepínání (doleva / doprava) ──────────────────────────────────
        elif gesture in self.GESTURE_TO_HOTKEY:
            if confidence >= self.HOTKEY_MIN_CONFIDENCE and gesture == self._hotkey_candidate:
                self._hotkey_streak += 1
            else:
                # Nové gesto nebo nedostatečná jistota – reset streaku
                self._hotkey_candidate = gesture
                self._hotkey_streak    = 1 if confidence >= self.HOTKEY_MIN_CONFIDENCE else 0

            if self._hotkey_streak < self.HOTKEY_MIN_STREAK:
                return False

            # Streak dosažen – spusť zkratku a resetuj
            self._hotkey_streak    = 0
            self._hotkey_candidate = None
            modifiers, key = self.GESTURE_TO_HOTKEY[gesture]
            for mod in modifiers:
                self._keyboard.press(mod)
            self._keyboard.tap(key)
            for mod in reversed(modifiers):
                self._keyboard.release(mod)

        # ── Scroll nahoru / dolů ──────────────────────────────────────────────
        elif self._mode == "scroll":
            self._hotkey_candidate = None
            self._hotkey_streak    = 0
            smer = self.GESTURE_TO_SCROLL.get(gesture)
            if smer is None:
                return False
            dx, dy = smer
            self._mouse.scroll(dx * self._scroll_amount, dy * self._scroll_amount)

        # ── Klávesový režim nahoru / dolů ─────────────────────────────────────
        else:
            self._hotkey_candidate = None
            self._hotkey_streak    = 0
            key = self.GESTURE_TO_KEY.get(gesture)
            if key is None:
                return False
            self._keyboard.tap(key)

        self._last_time    = now
        self._last_gesture = gesture
        return True
