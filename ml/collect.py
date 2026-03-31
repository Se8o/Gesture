"""
Hand gesture data collection script using the webcam.
Collected data is saved to a CSV file for later ML model training.

Note: uses the new MediaPipe Tasks API (version 0.10+),
      which replaced the older mp.solutions interface.

Usage (from the project root):
    python ml/collect.py "posun nahoru"
    python ml/collect.py "posun dolu"
    python ml/collect.py "posun doprava"
    python ml/collect.py "posun doleva"
    python ml/collect.py "pauza"

Controls:
  's' ... start / stop recording frames to CSV
  'q' ... quit the program
"""

import sys
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import csv
import time

# ============================================================
#  Command-line argument — gesture name to record
# ============================================================

# The gesture name can be passed as an argument: python ml/collect.py "posun nahoru"
# If no argument is given, the default value below is used.
if len(sys.argv) >= 2:
    GESTURE_NAME = sys.argv[1]
else:
    GESTURE_NAME = "posun dolu"   # default — change as needed

# ============================================================
#  File paths (absolute, derived from this file's location)
# ============================================================

# Project root is two levels above this script (ml/collect.py → project root)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the pre-trained MediaPipe hand detection model
MODEL_PATH = os.path.join(ROOT_DIR, "models", "hand_landmarker.task")

# Path to the output CSV file where collected samples are appended
CSV_FILE = os.path.join(ROOT_DIR, "data", "dataset.csv")

# ============================================================
#  Configuration constants
# ============================================================

CAMERA_INDEX               = 0    # 0 = built-in webcam
MIN_DETECTION_CONFIDENCE   = 0.7
MIN_TRACKING_CONFIDENCE    = 0.7

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ============================================================
#  Check that the MediaPipe model file exists
# ============================================================

if not os.path.isfile(MODEL_PATH):
    print("ERROR: Hand landmarker model not found: " + MODEL_PATH)
    print("Download it manually with:")
    print("  curl -L -o models/hand_landmarker.task \"" + MODEL_URL + "\"")
    sys.exit(1)

# Create the data/ folder if it does not exist yet
os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)

# ============================================================
#  Hand skeleton connections for drawing
# ============================================================

# Each pair (a, b) means: draw a line from joint a to joint b.
# MediaPipe numbers 21 hand joints (0 = wrist, 1-20 = finger joints).
HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),    # thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),    # index finger
    (0, 9),  (9, 10), (10, 11),(11, 12),  # middle finger
    (0, 13),(13, 14), (14, 15),(15, 16),  # ring finger
    (0, 17),(17, 18), (18, 19),(19, 20),  # pinky
    (5, 9),  (9, 13), (13, 17)            # palm cross-connections
]

# ============================================================
#  Initialise the MediaPipe hand detector (Tasks API)
# ============================================================

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

# VIDEO mode keeps state between frames for smoother tracking
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_hand_presence_confidence=MIN_TRACKING_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# ============================================================
#  Prepare the CSV output file
# ============================================================

# Header: label column + x,y,z for each of the 21 hand joints (64 columns total)
header = ["label"]
for i in range(21):
    header.append("x" + str(i))
    header.append("y" + str(i))
    header.append("z" + str(i))

file_exists = os.path.isfile(CSV_FILE)

# Open in append mode so data from previous sessions is preserved
csv_file   = open(CSV_FILE, mode="a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

# Write the header only if the file is new
if not file_exists:
    csv_writer.writerow(header)

# ============================================================
#  Open the webcam
# ============================================================

camera = cv2.VideoCapture(CAMERA_INDEX)
if not camera.isOpened():
    print("ERROR: Cannot open camera with index", CAMERA_INDEX)
    csv_file.close()
    detector.close()
    sys.exit(1)

# ============================================================
#  State variables
# ============================================================

recording  = False    # True when actively saving frames to CSV
saved      = 0        # total frames saved in this session
start_time = time.time()

# ============================================================
#  Main loop
# ============================================================

print("Program started.")
print("  Gesture to record:", GESTURE_NAME)
print("  Press 's' to start/stop recording.")
print("  Press 'q' to quit.")

while True:
    ok, frame = camera.read()
    if not ok:
        print("ERROR: Failed to read frame from camera.")
        break

    # Flip horizontally so the image acts as a mirror
    frame     = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Timestamp must be unique and monotonically increasing (VIDEO mode requirement)
    timestamp_ms = int((time.time() - start_time) * 1000)
    mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result       = detector.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        hand   = result.hand_landmarks[0]
        h, w, _ = frame.shape

        # Draw the white hand skeleton
        for start_idx, end_idx in HAND_CONNECTIONS:
            s = hand[start_idx]
            e = hand[end_idx]
            cv2.line(frame,
                     (int(s.x * w), int(s.y * h)),
                     (int(e.x * w), int(e.y * h)),
                     (255, 255, 255), 2)

        # Draw green dots on each joint
        for lm in hand:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)

        # Save joint coordinates to CSV when recording is active
        if recording:
            row = [GESTURE_NAME]
            for lm in hand:
                row.append(lm.x)
                row.append(lm.y)
                row.append(lm.z)
            csv_writer.writerow(row)
            csv_file.flush()   # flush immediately so data is not lost on crash
            saved += 1

    # Status text overlay
    if recording:
        status = "RECORDING: " + GESTURE_NAME + "  [frames: " + str(saved) + "]"
        color  = (0, 0, 255)    # red = recording in progress
    else:
        status = "PAUSED — press 'S' to start"
        color  = (0, 255, 0)    # green = waiting for input

    cv2.putText(frame, status, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, "'S' = start/stop  |  'Q' = quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Data collection — " + GESTURE_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Quitting...")
        break
    if key == ord("s"):
        recording = not recording
        if recording:
            print("Recording STARTED — gesture:", GESTURE_NAME)
        else:
            print("Recording PAUSED. Total frames saved:", saved)

# ============================================================
#  Clean up
# ============================================================

camera.release()
cv2.destroyAllWindows()
csv_file.close()
detector.close()

print("Program finished. Data saved to:", CSV_FILE)
print("Total frames recorded this session:", saved)
