#!/usr/bin/env python3
"""
Instalátor automatického spouštění Gesture Remote Controller.

Registruje aplikaci tak, aby se spustila automaticky při přihlášení uživatele.
Aplikace poté běží na pozadí jako ikona v systémové liště.

Spuštění:
    python install.py

Odinstalace:
    python uninstall.py
"""
import os
import platform
import subprocess
import sys

# Cesta k Pythonu a skriptu run.py v tomto projektu
PYTHON  = sys.executable
RUN_PY  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "run.py"))
APP_ID  = "gesture-controller"
APP_KEY = "GestureController"


def _macos() -> None:
    """Vytvoří LaunchAgent plist – spustí aplikaci při přihlášení na macOS."""
    plist_dir  = os.path.expanduser("~/Library/LaunchAgents")
    log_path   = os.path.expanduser(f"~/Library/Logs/{APP_ID}.log")
    plist_path = os.path.join(plist_dir, f"com.{APP_ID}.plist")

    os.makedirs(plist_dir, exist_ok=True)

    obsah = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.{APP_ID}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{PYTHON}</string>
        <string>{RUN_PY}</string>
        <string>--background</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>{log_path}</string>
    <key>StandardErrorPath</key>
    <string>{log_path}</string>
</dict>
</plist>
"""
    with open(plist_path, "w") as f:
        f.write(obsah)

    # Načteme agent okamžitě (bez nutnosti odhlásit se)
    subprocess.run(["launchctl", "load", plist_path], check=False)
    print(f"LaunchAgent vytvořen: {plist_path}")
    print(f"Log: {log_path}")


def _windows() -> None:
    """Přidá záznam do registru – spustí aplikaci při přihlášení na Windows."""
    import winreg  # noqa: F401 – dostupný pouze na Windows

    klic_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    prikaz    = f'"{PYTHON}" "{RUN_PY}" --background'

    with winreg.OpenKey(                        # type: ignore[attr-defined]
        winreg.HKEY_CURRENT_USER,               # type: ignore[attr-defined]
        klic_path, 0,
        winreg.KEY_SET_VALUE,                   # type: ignore[attr-defined]
    ) as klic:
        winreg.SetValueEx(klic, APP_KEY, 0,     # type: ignore[attr-defined]
                          winreg.REG_SZ, prikaz)  # type: ignore[attr-defined]

    print(f"Autostart registrován v registru (HKCU\\...\\Run\\{APP_KEY}).")


def _linux() -> None:
    """Vytvoří .desktop soubor v ~/.config/autostart – spustí aplikaci při přihlášení."""
    autostart_dir  = os.path.expanduser("~/.config/autostart")
    desktop_path   = os.path.join(autostart_dir, f"{APP_ID}.desktop")

    os.makedirs(autostart_dir, exist_ok=True)

    obsah = f"""[Desktop Entry]
Type=Application
Name=Gesture Controller
Comment=Ovládání počítače gesty ruky
Exec={PYTHON} {RUN_PY} --background
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
"""
    with open(desktop_path, "w") as f:
        f.write(obsah)

    os.chmod(desktop_path, 0o755)
    print(f"Autostart soubor vytvořen: {desktop_path}")


def main() -> None:
    system = platform.system()
    print(f"Instaluji autostart pro: {system}")
    print(f"  Python : {PYTHON}")
    print(f"  Skript : {RUN_PY}")
    print()

    if system == "Darwin":
        _macos()
    elif system == "Windows":
        _windows()
    elif system == "Linux":
        _linux()
    else:
        print(f"Nepodporovaný operační systém: {system}")
        sys.exit(1)

    print()
    print("Hotovo! Aplikace se spustí automaticky při příštím přihlášení.")
    print("Pro okamžité spuštění na pozadí spusť:")
    print("    python run.py --background")


if __name__ == "__main__":
    main()
