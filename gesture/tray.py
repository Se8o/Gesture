"""
Systémová ikona v liště pro Gesture Remote Controller.

Zobrazí ikonu v menu liště (macOS) nebo v systémové liště (Windows / Linux)
a umožní aplikaci běžet na pozadí bez viditelného okna.

Vyžaduje balíčky: pystray, Pillow
"""
import threading

import pystray
from PIL import Image, ImageDraw


def _vytvor_ikonu() -> Image.Image:
    """Nakreslí jednoduchou ikonu ruky (64 × 64 px) pro systémovou lištu."""
    vel = 64
    img = Image.new("RGBA", (vel, vel), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    modra = (60, 120, 220, 255)

    # Dlaň
    draw.rounded_rectangle([10, 26, 54, 58], radius=9, fill=modra)

    # Pět prstů
    for x in [10, 18, 26, 34, 42]:
        draw.rounded_rectangle([x, 6, x + 7, 32], radius=3, fill=modra)

    return img


class SystemovyTray:
    """
    Spravuje ikonu aplikace v systémové liště.

    Použití
    -------
        stop = threading.Event()
        tray = SystemovyTray(stop)
        tray.spustit()   # blokující – volej z hlavního vlákna
    """

    def __init__(self, stop_event: threading.Event):
        self._stop = stop_event
        self._icon = None

    def _ukoncit(self, icon, item):
        """Callback pro položku menu 'Ukončit'."""
        self._stop.set()
        icon.stop()

    def spustit(self) -> None:
        """Zobrazí ikonu v liště a blokuje hlavní vlákno do zvolení Ukončit."""
        obr = _vytvor_ikonu()
        menu = pystray.Menu(
            pystray.MenuItem("Gesture Controller", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Ukončit", self._ukoncit),
        )
        self._icon = pystray.Icon("gesture_controller", obr, "Gesture Controller", menu)
        self._icon.run()
