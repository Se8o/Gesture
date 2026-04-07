import sys
import threading

import cv2
import numpy as np

from . import config
from .recognizer import GestureRecognizer
from .controller import GestureController


# Pairs of hand joint indices to draw the hand skeleton.
# MediaPipe numbers 21 joints (0 = wrist, 1–20 = finger joints).
HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),    # thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),    # index finger
    (0, 9),  (9, 10), (10, 11),(11, 12),  # middle finger
    (0, 13),(13, 14),(14, 15),(15, 16),   # ring finger
    (0, 17),(17, 18),(18, 19),(19, 20),   # pinky
    (5, 9),  (9, 13),(13, 17),            # palm cross-connections
]


def _draw_hand(frame: np.ndarray, landmarks, h: int, w: int) -> None:
    """Draw a white hand skeleton and green dots on each joint."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for px, py in pts:
        cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)


def _draw_ui(frame: np.ndarray, gesture, confidence: float, triggered: bool) -> None:
    """Render the HUD overlay at the top of the camera window."""
    h, w = frame.shape[:2]

    # Semi-transparent dark bar behind the text
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if gesture:
        # Green when the action fired, yellow when waiting for cooldown/streak
        color = (0, 255, 80) if triggered else (0, 200, 255)
        cv2.putText(frame, f"{gesture}  {confidence:.0%}",
                    (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    else:
        cv2.putText(frame, "No gesture",
                    (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (110, 110, 110), 2)

    cv2.putText(frame, "Q = quit",
                (w - 130, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)


def _camera_loop(stop_event: threading.Event, background: bool) -> None:
    """Camera capture and gesture recognition loop."""
    recognizer = GestureRecognizer()
    controller = GestureController(
        cooldown=config.GESTURE_COOLDOWN,
        mode=config.CONTROL_MODE,
        scroll_amount=config.SCROLL_AMOUNT,
    )

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        recognizer.close()
        print(f"ERROR: Cannot open camera (index {config.CAMERA_INDEX}).")
        if not background:
            sys.exit(1)
        stop_event.set()
        return

    if not background:
        print("Gesture Remote Controller started. Press 'Q' to quit.")

    gesture    = None
    confidence = 0.0
    triggered  = False

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            gesture, confidence, landmarks = recognizer.process(rgb)

            if landmarks is not None:
                _draw_hand(frame, landmarks, h, w)

            triggered = controller.execute(gesture, confidence) if gesture else False

            if not background:
                _draw_ui(frame, gesture, confidence, triggered)
                cv2.imshow("Gesture Remote Controller", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break
    finally:
        recognizer.close()
        cap.release()
        if not background:
            cv2.destroyAllWindows()
        print("Camera released.")


def run(background: bool = False) -> None:
    """Start the application."""
    import os

    for path, label in [
        (config.MODEL_PATH,   "model.pkl"),
        (config.SCALER_PATH,  "scaler.pkl"),
        (config.ENCODER_PATH, "label_encoder.pkl"),
    ]:
        if not os.path.isfile(path):
            print(f"ERROR: '{label}' not found in {os.path.dirname(path)}/")
            print("Train the model first:  python ml/train.py")
            sys.exit(1)

    stop_event = threading.Event()

    if background:
        thread = threading.Thread(
            target=_camera_loop,
            args=(stop_event, True),
            daemon=True,
        )
        thread.start()

        from .tray import SystemovyTray
        print("Gesture Controller running in background. Quit via tray icon.")
        tray = SystemovyTray(stop_event)
        tray.spustit()

        thread.join(timeout=3.0)
    else:
        _camera_loop(stop_event, False)
