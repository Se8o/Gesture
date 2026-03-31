"""
System tray icon for Gesture Remote Controller.

Displays an icon in the menu bar (macOS) or system tray (Windows / Linux)
so the application can run in the background without a visible window.

Requires: pystray, Pillow
"""
import threading

import pystray
from PIL import Image, ImageDraw


def _create_icon() -> Image.Image:
    """Draw a simple hand icon (64 × 64 px) for the system tray."""
    size = 64
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    blue = (60, 120, 220, 255)

    # Palm
    draw.rounded_rectangle([10, 26, 54, 58], radius=9, fill=blue)

    # Five fingers
    for x in [10, 18, 26, 34, 42]:
        draw.rounded_rectangle([x, 6, x + 7, 32], radius=3, fill=blue)

    return img


class SystemovyTray:
    """
    Manages the application's system tray icon.

    Usage
    -----
        stop = threading.Event()
        tray = SystemovyTray(stop)
        tray.spustit()   # blocking — call from the main thread
    """

    def __init__(self, stop_event: threading.Event):
        self._stop = stop_event   # shared event to signal the camera thread to stop
        self._icon = None

    def _quit(self, icon, item):
        """Callback for the 'Quit' menu item."""
        self._stop.set()   # tell the camera thread to stop
        icon.stop()        # stop the tray icon loop

    def spustit(self) -> None:
        """Show the tray icon and block the main thread until the user clicks Quit."""
        image = _create_icon()
        menu  = pystray.Menu(
            pystray.MenuItem("Gesture Controller", None, enabled=False),  # title (disabled)
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit),
        )
        self._icon = pystray.Icon("gesture_controller", image, "Gesture Controller", menu)
        self._icon.run()   # blocks until icon.stop() is called
