#!/usr/bin/env python3
"""
Odstraní automatické spouštění Gesture Remote Controller.

Spuštění:
    python uninstall.py
"""
import os
import platform
import subprocess
import sys

APP_ID  = "gesture-controller"
APP_KEY = "GestureController"


def _macos() -> None:
    plist_path = os.path.expanduser(f"~/Library/LaunchAgents/com.{APP_ID}.plist")
    if os.path.exists(plist_path):
        subprocess.run(["launchctl", "unload", plist_path], check=False)
        os.remove(plist_path)
        print(f"LaunchAgent odstraněn: {plist_path}")
    else:
        print("LaunchAgent nebyl nalezen.")


def _windows() -> None:
    import winreg  # noqa: F401

    klic_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        with winreg.OpenKey(                    # type: ignore[attr-defined]
            winreg.HKEY_CURRENT_USER,           # type: ignore[attr-defined]
            klic_path, 0,
            winreg.KEY_SET_VALUE,               # type: ignore[attr-defined]
        ) as klic:
            winreg.DeleteValue(klic, APP_KEY)   # type: ignore[attr-defined]
        print(f"Záznam {APP_KEY} odstraněn z registru.")
    except FileNotFoundError:
        print("Záznam v registru nebyl nalezen.")


def _linux() -> None:
    desktop_path = os.path.expanduser(f"~/.config/autostart/{APP_ID}.desktop")
    if os.path.exists(desktop_path):
        os.remove(desktop_path)
        print(f"Autostart soubor odstraněn: {desktop_path}")
    else:
        print("Autostart soubor nebyl nalezen.")


def main() -> None:
    system = platform.system()
    if system == "Darwin":
        _macos()
    elif system == "Windows":
        _windows()
    elif system == "Linux":
        _linux()
    else:
        print(f"Nepodporovaný operační systém: {system}")
        sys.exit(1)
    print("Odinstalace dokončena.")


if __name__ == "__main__":
    main()
