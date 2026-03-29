"""
Hlavní smyčka aplikace Gesture Remote Controller.

Propojuje tři moduly:
  - config      – konfigurační konstanty
  - recognizer  – detekce ruky + predikce gesta
  - controller  – překlad gesta na stisk klávesy

Ovládání:
  Q ... ukončit aplikaci
"""
import sys
import cv2
import numpy as np

from . import config
from .recognizer import GestureRecognizer
from .controller import GestureController


# Spojnice kloubů ruky pro kreslení kostry (stejné jako v collect_data.py)
HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),    # palec
    (0, 5),  (5, 6),  (6, 7),  (7, 8),    # ukazovák
    (0, 9),  (9, 10), (10, 11),(11, 12),  # prostředník
    (0, 13),(13, 14),(14, 15),(15, 16),   # prsteník
    (0, 17),(17, 18),(18, 19),(19, 20),   # malík
    (5, 9),  (9, 13),(13, 17),            # příčné spojení dlaně
]


def _draw_hand(frame: np.ndarray, landmarks, h: int, w: int) -> None:
    """Nakreslí bílou kostru ruky a zelené tečky na klouby."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for px, py in pts:
        cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)


def _draw_ui(frame: np.ndarray, gesture, confidence: float, triggered: bool) -> None:
    """
    Vykreslí informační panel v horní části okna:
      - název detekovaného gesta + jeho pravděpodobnost
      - zelená barva = akce právě spuštěna; žlutá = gesto detekováno, cooldown
    """
    h, w = frame.shape[:2]

    # Poloprůhledný černý pruh jako pozadí textu
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if gesture:
        color = (0, 255, 80) if triggered else (0, 200, 255)
        cv2.putText(frame, f"{gesture}  {confidence:.0%}",
                    (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    else:
        cv2.putText(frame, "Zadne gesto",
                    (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (110, 110, 110), 2)

    cv2.putText(frame, "Q = konec",
                (w - 130, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)


def run() -> None:
    """Spustí hlavní smyčku aplikace. Blokující volání – vrátí se po stisku Q."""

    # Ověříme, že modely existují, dříve než začneme otevírat kameru
    import os
    for path, label in [
        (config.MODEL_PATH,  "model.pkl"),
        (config.SCALER_PATH, "scaler.pkl"),
        (config.ENCODER_PATH,"label_encoder.pkl"),
    ]:
        if not os.path.isfile(path):
            print(f"CHYBA: Soubor '{label}' nebyl nalezen v {os.path.dirname(path)}/")
            print("Nejprve natrénuj model příkazem:  python train.py")
            sys.exit(1)

    recognizer = GestureRecognizer()
    controller = GestureController(cooldown=config.GESTURE_COOLDOWN)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        recognizer.close()
        print(f"CHYBA: Nelze otevřít kameru (index {config.CAMERA_INDEX}).")
        sys.exit(1)

    print("Gesture Remote Controller spusten. Stiskni 'Q' pro ukonceni.")

    gesture   = None
    confidence = 0.0
    triggered  = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("CHYBA: Nepodařilo se načíst snímek z kamery.")
                break

            # Zrcadlové překlopení – intuitivnější pro uživatele
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # Konverze BGR→RGB pro MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            gesture, confidence, landmarks = recognizer.process(rgb)

            if landmarks is not None:
                _draw_hand(frame, landmarks, h, w)

            triggered = controller.execute(gesture) if gesture else False

            _draw_ui(frame, gesture, confidence, triggered)
            cv2.imshow("Gesture Remote Controller", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Aplikace ukoncena.")
