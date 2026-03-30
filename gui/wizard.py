#!/usr/bin/env python3
"""
Gesture Remote Controller — Průvodce nastavením (Setup Wizard)

Spuštění:
    python gui/wizard.py
"""
import tkinter as tk
from tkinter import ttk, messagebox
import json
import subprocess
import sys
import threading
from pathlib import Path

# ── Cesty ─────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent          # Gesture/
SETTINGS_FILE  = ROOT / "settings.json"
MODELS_DIR     = ROOT / "models"
DATA_FILE      = ROOT / "data" / "dataset.csv"
TRAIN_SCRIPT   = ROOT / "ml" / "train.py"
RUN_SCRIPT     = ROOT / "run.py"
COLLECT_SCRIPT = ROOT / "ml" / "collect.py"

REQUIRED_MODELS = ["model.pkl", "scaler.pkl", "label_encoder.pkl"]

DEFAULTS: dict = {
    "camera_index":         0,
    "detection_confidence": 0.70,
    "tracking_confidence":  0.70,
    "prediction_threshold": 0.75,
    "gesture_cooldown":     1.0,
}

# ── Barevná paleta ─────────────────────────────────────────────────────────────
BG      = "#f1f5f9"
SURFACE = "#ffffff"
SURF2   = "#f8fafc"
ACCENT  = "#4f46e5"
ACC_H   = "#3730a3"
SUCCESS = "#059669"
ERROR   = "#dc2626"
WARN    = "#d97706"
TEXT    = "#0f172a"
TEXT2   = "#475569"
TEXT3   = "#94a3b8"
BORDER  = "#e2e8f0"
BORDER2 = "#cbd5e1"
LOG_BG  = "#0f172a"
LOG_FG  = "#94a3b8"

FONT        = "Helvetica Neue"
FONT_MONO   = "Menlo"


# ── Pomocné funkce ─────────────────────────────────────────────────────────────

def load_settings() -> dict:
    try:
        return {**DEFAULTS, **json.loads(SETTINGS_FILE.read_text())}
    except Exception:
        return DEFAULTS.copy()


def save_settings(s: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(s, indent=2))


def models_ready() -> bool:
    return all((MODELS_DIR / m).exists() for m in REQUIRED_MODELS)


def data_ready() -> bool:
    return DATA_FILE.exists()


# ── Hlavní GUI ─────────────────────────────────────────────────────────────────

class SetupGUI:
    WIDTH = 500

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Gesture Remote Controller")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self._settings = load_settings()
        self._vars: dict[str, tk.Variable] = {}
        self._status_var = tk.StringVar()

        self._build()
        self._center()
        self._refresh_status()

    # ── Pomocné metody okna ────────────────────────────────────────────────────

    def _center(self) -> None:
        self.root.update_idletasks()
        w = self.root.winfo_reqwidth()
        h = self.root.winfo_reqheight()
        x = (self.root.winfo_screenwidth()  - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"+{x}+{y}")

    # ── Továrny widgetů ────────────────────────────────────────────────────────

    def _card(self, parent: tk.Widget, title: str, icon: str = "") -> tk.Frame:
        """Bílá karta s nadpisem. Vrátí vnitřní frame."""
        outer = tk.Frame(parent, bg=BG)
        outer.pack(fill="x", padx=16, pady=(0, 10))

        border = tk.Frame(outer, bg=BORDER2)
        border.pack(fill="x")
        inner = tk.Frame(border, bg=SURFACE)
        inner.pack(fill="x", padx=1, pady=1)

        head = tk.Frame(inner, bg=SURF2)
        head.pack(fill="x")
        label = f"  {icon}  {title}" if icon else f"  {title}"
        tk.Label(head, text=label,
                 font=(FONT, 11, "bold"),
                 bg=SURF2, fg=TEXT, pady=9, anchor="w").pack(fill="x", padx=6)

        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x")

        body = tk.Frame(inner, bg=SURFACE)
        body.pack(fill="x", padx=16, pady=12)
        return body

    def _flat_btn(self, parent: tk.Widget, text: str, cmd,
                  bg=SURFACE, fg=TEXT2) -> tk.Button:
        return tk.Button(
            parent, text=text, command=cmd,
            font=(FONT, 10), bg=bg, fg=fg,
            activebackground=BORDER, activeforeground=TEXT,
            relief="flat", pady=8, cursor="hand2", bd=0,
            highlightbackground=BORDER2, highlightthickness=1,
        )

    def _slider_row(self, parent: tk.Widget, label: str, key: str,
                    from_: float, to: float, step: float,
                    fmt: str = "{:.2f}") -> tk.DoubleVar:
        """Slider s popiskem a živým výpisem hodnoty."""
        frame = tk.Frame(parent, bg=SURFACE)
        frame.pack(fill="x", pady=(0, 10))

        var = tk.DoubleVar(value=self._settings.get(key, DEFAULTS[key]))
        self._vars[key] = var

        top = tk.Frame(frame, bg=SURFACE)
        top.pack(fill="x")
        tk.Label(top, text=label, font=(FONT, 10),
                 bg=SURFACE, fg=TEXT2).pack(side="left")
        val_lbl = tk.Label(top, text=fmt.format(var.get()),
                           font=(FONT, 10, "bold"),
                           bg=SURFACE, fg=ACCENT)
        val_lbl.pack(side="right")

        slider = ttk.Scale(frame, variable=var, from_=from_, to=to,
                           orient="horizontal",
                           command=lambda v: val_lbl.config(text=fmt.format(float(v))))
        slider.pack(fill="x", pady=(3, 0))

        def _snap(_event=None):
            snapped = round(var.get() / step) * step
            snapped = max(from_, min(to, snapped))
            var.set(round(snapped, 10))
            val_lbl.config(text=fmt.format(snapped))

        slider.bind("<ButtonRelease-1>", _snap)
        return var

    # ── Sestavení GUI ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        root = self.root

        # Záhlaví
        hdr = tk.Frame(root, bg=ACCENT)
        hdr.pack(fill="x")

        tk.Label(hdr, text="✋  Gesture Remote Controller",
                 font=(FONT, 18, "bold"),
                 bg=ACCENT, fg="white", pady=18).pack()

        self._status_badge = tk.Label(
            hdr, textvariable=self._status_var,
            font=(FONT, 10), bg=ACCENT, fg="#c7d2fe", pady=6,
        )
        self._status_badge.pack()
        tk.Frame(hdr, bg=ACCENT, height=12).pack()

        tk.Frame(root, bg=BG, height=12).pack()

        # Kamera
        cam_body = self._card(root, "Camera", "📷")

        cam_var = tk.IntVar(value=int(self._settings.get("camera_index", 0)))
        self._vars["camera_index"] = cam_var

        options = [("0 — Built-in", 0), ("1 — USB #1", 1),
                   ("2 — USB #2",   2), ("3 — USB #3", 3)]
        grid = tk.Frame(cam_body, bg=SURFACE)
        grid.pack(fill="x")
        for i, (lbl, val) in enumerate(options):
            tk.Radiobutton(
                grid, text=lbl, variable=cam_var, value=val,
                font=(FONT, 10), bg=SURFACE, fg=TEXT2,
                selectcolor=SURFACE, activebackground=SURFACE,
                highlightthickness=0,
            ).grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 28), pady=2)

        # Parametry detekce
        det_body = self._card(root, "Detection Parameters", "🎯")

        self._slider_row(det_body, "Hand Detection Confidence",
                         "detection_confidence", 0.10, 1.0, 0.05)
        self._slider_row(det_body, "Hand Tracking Confidence",
                         "tracking_confidence",  0.10, 1.0, 0.05)
        self._slider_row(det_body, "Gesture Prediction Threshold",
                         "prediction_threshold", 0.10, 1.0, 0.05)
        self._slider_row(det_body, "Gesture Cooldown (seconds)",
                         "gesture_cooldown",     0.1,  3.0, 0.1, fmt="{:.1f} s")

        # Stav modelu
        self._model_body = self._card(root, "Model Status", "🤖")

        self._model_rows = tk.Frame(self._model_body, bg=SURFACE)
        self._model_rows.pack(fill="x")
        self._rebuild_model_rows()

        btns = tk.Frame(self._model_body, bg=SURFACE)
        btns.pack(fill="x", pady=(10, 0))

        self._train_btn = tk.Button(
            btns, text="Train Model",
            command=self._train,
            font=(FONT, 10, "bold"),
            bg=ACCENT, fg="white",
            activebackground=ACC_H, activeforeground="white",
            relief="flat", padx=18, pady=7, cursor="hand2", bd=0,
        )
        self._train_btn.pack(side="left")

        hint_text = "⚠  dataset.csv missing" if not data_ready() else "✓  dataset.csv found"
        hint_color = WARN if not data_ready() else SUCCESS
        tk.Label(btns, text=hint_text,
                 font=(FONT, 9), bg=SURFACE, fg=hint_color).pack(side="left", padx=10)

        # Log výstupu
        log_body = self._card(root, "Output Log", "📋")

        self._log = tk.Text(
            log_body, height=6,
            font=(FONT_MONO, 9),
            bg=LOG_BG, fg=LOG_FG,
            insertbackground=LOG_FG,
            relief="flat", bd=0,
            padx=8, pady=6,
            state="disabled",
        )
        self._log.pack(fill="x")
        self._log.tag_config("ok",   foreground="#34d399")
        self._log.tag_config("err",  foreground="#f87171")
        self._log.tag_config("info", foreground="#60a5fa")
        self._log.tag_config("cmd",  foreground="#fbbf24")

        # Zápatí
        foot = tk.Frame(root, bg=BG)
        foot.pack(fill="x", padx=16, pady=(4, 20))

        self._launch_btn = tk.Button(
            foot, text="▶   Launch App",
            command=self._launch,
            font=(FONT, 13, "bold"),
            bg=SUCCESS, fg="white",
            activebackground="#047857", activeforeground="white",
            relief="flat", pady=13, cursor="hand2", bd=0,
        )
        self._launch_btn.pack(fill="x", pady=(0, 8))

        row2 = tk.Frame(foot, bg=BG)
        row2.pack(fill="x")
        self._flat_btn(row2, "Save Settings",
                       self._save_explicit).pack(side="left", fill="x",
                                                 expand=True, padx=(0, 4))
        self._flat_btn(row2, "Collect Training Data",
                       self._collect).pack(side="left", fill="x",
                                           expand=True, padx=(4, 0))

    # ── Stav modelu ────────────────────────────────────────────────────────────

    def _rebuild_model_rows(self) -> None:
        for w in self._model_rows.winfo_children():
            w.destroy()
        for name in REQUIRED_MODELS:
            exists = (MODELS_DIR / name).exists()
            icon, color = ("✓", SUCCESS) if exists else ("✗", ERROR)
            row = tk.Frame(self._model_rows, bg=SURFACE)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{icon}  {name}",
                     font=(FONT_MONO, 10), bg=SURFACE, fg=color).pack(side="left")

    def _refresh_status(self) -> None:
        if models_ready():
            self._status_var.set("● All systems ready — you can launch the app")
            self._status_badge.config(fg="#a5f3c7")
        else:
            self._status_var.set("⚠  Model not trained — click Train Model first")
            self._status_badge.config(fg="#fde68a")

    # ── Pomocné metody logu ────────────────────────────────────────────────────

    def _log_write(self, msg: str, tag: str = "") -> None:
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n", tag)
        self._log.see("end")
        self._log.configure(state="disabled")

    # ── Akce ──────────────────────────────────────────────────────────────────

    def _collect_settings(self) -> dict:
        s = {}
        for key, var in self._vars.items():
            val = var.get()
            s[key] = int(val) if key == "camera_index" else round(float(val), 4)
        return s

    def _save(self) -> None:
        self._settings = self._collect_settings()
        save_settings(self._settings)

    def _save_explicit(self) -> None:
        self._save()
        self._log_write("Settings saved to settings.json", "ok")

    def _train(self) -> None:
        if not data_ready():
            messagebox.showerror(
                "No Training Data",
                f"Dataset not found:\n{DATA_FILE}\n\nUse 'Collect Training Data' first.",
            )
            return

        self._train_btn.config(state="disabled", text="Training…")
        self._log_write("Starting model training…", "cmd")

        def _run() -> None:
            try:
                proc = subprocess.Popen(
                    [sys.executable, str(TRAIN_SCRIPT)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=str(ROOT),
                )
                for line in proc.stdout:
                    stripped = line.rstrip()
                    if stripped:
                        self.root.after(0, self._log_write, stripped, "info")
                proc.wait()
                if proc.returncode == 0:
                    self.root.after(0, self._log_write, "Training complete!", "ok")
                    self.root.after(0, self._rebuild_model_rows)
                    self.root.after(0, self._refresh_status)
                else:
                    self.root.after(0, self._log_write,
                                    "Training failed — see output above.", "err")
            except Exception as exc:
                self.root.after(0, self._log_write, f"Error: {exc}", "err")
            finally:
                self.root.after(0, lambda: self._train_btn.config(
                    state="normal", text="Train Model"))

        threading.Thread(target=_run, daemon=True).start()

    def _launch(self) -> None:
        self._save()

        if not models_ready():
            if not messagebox.askyesno(
                "Model Missing",
                "Trained model files are missing.\n"
                "The app will fail to start without them.\n\n"
                "Launch anyway?",
            ):
                return

        self._log_write("Launching Gesture Remote Controller…", "cmd")
        subprocess.Popen([sys.executable, str(RUN_SCRIPT)], cwd=str(ROOT))
        self.root.after(800, self.root.destroy)

    def _collect(self) -> None:
        import tkinter.simpledialog as sd
        gesture = sd.askstring(
            "Collect Training Data",
            "Enter gesture name:\n(e.g.  up  down  left  right)",
            parent=self.root,
        )
        if not gesture:
            return
        gesture = gesture.strip().lower()
        if not gesture:
            return
        self._log_write(f"Opening data collection for '{gesture}'…", "cmd")
        subprocess.Popen(
            [sys.executable, str(COLLECT_SCRIPT), gesture],
            cwd=str(ROOT),
        )

    # ── Spuštění ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    SetupGUI().run()
