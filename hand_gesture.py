"""Real-time hand gesture recognition from a webcam.

Uses MediaPipe's HandLandmarker to find 21 landmarks per hand, then classifies
the gesture from which fingers are extended.

Usage:
    python hand_gesture.py [--camera 0] [--hands 2]

Press 'q' or Esc to quit.
"""

import argparse
import math
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

# Landmark indices
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        print(f"Downloading hand landmark model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def fingers_up(lm) -> list[bool]:
    """Return [thumb, index, middle, ring, pinky] extended flags.

    A finger counts as extended when its tip is farther from the wrist than its
    PIP joint, which works regardless of hand rotation. The thumb is extended
    when its tip is farther from the pinky base than the thumb IP joint.
    """
    wrist = lm[WRIST]
    thumb = dist(lm[THUMB_TIP], lm[PINKY_MCP]) > dist(lm[THUMB_IP], lm[PINKY_MCP]) * 1.1
    others = [
        dist(lm[tip], wrist) > dist(lm[pip], wrist) * 1.1
        for tip, pip in ((INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                         (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP))
    ]
    return [thumb, *others]


def classify(lm) -> str:
    thumb, index, middle, ring, pinky = fingers_up(lm)
    hand_size = dist(lm[WRIST], lm[MIDDLE_MCP])

    # OK sign: thumb and index tips touching, other fingers up
    if dist(lm[THUMB_TIP], lm[INDEX_TIP]) < 0.35 * hand_size and middle and ring and pinky:
        return "OK"

    pattern = (thumb, index, middle, ring, pinky)
    if pattern == (True, False, False, False, False):
        # Thumb up vs down depends on whether the tip is above the knuckle (y grows downward)
        return "Thumbs Up" if lm[THUMB_TIP].y < lm[THUMB_MCP].y else "Thumbs Down"

    gestures = {
        (False, False, False, False, False): "Fist",
        (True, True, True, True, True): "Open Palm",
        (False, True, False, False, False): "Pointing",
        (False, True, True, False, False): "Peace",
        (True, True, False, False, True): "I Love You",
        (False, True, False, False, True): "Rock",
        (True, False, False, False, True): "Call Me",
        (True, True, False, False, False): "L",
        (False, True, True, True, False): "Three",
        (False, True, True, True, True): "Four",
    }
    return gestures.get(pattern, f"{sum(pattern)} finger(s)")


def draw_hand(frame, lm, label: str, handedness: str) -> None:
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)

    x0 = max(min(p[0] for p in pts) - 10, 0)
    y0 = max(min(p[1] for p in pts) - 10, 30)
    text = f"{handedness}: {label}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x0, y0 - th - 10), (x0 + tw + 10, y0), (0, 0, 0), -1)
    cv2.putText(frame, text, (x0 + 5, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam hand gesture recognition")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--hands", type=int, default=2, help="max number of hands")
    args = parser.parse_args()

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.hands,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    start = time.monotonic()
    prev = start
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(image, timestamp_ms)

            for lm, hd in zip(result.hand_landmarks, result.handedness):
                draw_hand(frame, lm, classify(lm), hd[0].category_name)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
