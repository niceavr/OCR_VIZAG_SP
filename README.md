# OCR_VIZAG_SP

## Hand gesture recognition

`hand_gesture.py` reads a webcam feed, tracks up to two hands with
[MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker),
and labels each hand with the gesture it is making.

### Requirements

- Python 3.10+
- A webcam

```
pip install mediapipe opencv-python
```

The hand landmark model (`hand_landmarker.task`, about 7 MB) downloads next to
the script on the first run.

### Usage

```
python hand_gesture.py                     # default webcam, up to 2 hands
python hand_gesture.py --camera 1 --hands 1
```

| Option     | Default | Description              |
|------------|---------|--------------------------|
| `--camera` | `0`     | Camera index to open     |
| `--hands`  | `2`     | Maximum hands to track   |

Press `q` or `Esc` in the video window to quit.

### Recognized gestures

| Gesture      | Fingers raised                         |
|--------------|----------------------------------------|
| Fist         | none                                   |
| Open Palm    | all five                               |
| Pointing     | index                                  |
| Peace        | index, middle                          |
| Three        | index, middle, ring                    |
| Four         | all except thumb                       |
| Thumbs Up    | thumb, pointing up                     |
| Thumbs Down  | thumb, pointing down                   |
| OK           | thumb and index tips touching, others up |
| L            | thumb, index                           |
| Rock         | index, pinky                           |
| Call Me      | thumb, pinky                           |
| I Love You   | thumb, index, pinky                    |

Any other combination is shown as a count, such as `2 finger(s)`.

### How it works

1. Each frame is mirrored and passed to MediaPipe's HandLandmarker, which
   returns 21 landmarks per hand.
2. `fingers_up()` decides which fingers are extended. A finger counts as
   extended when its tip is farther from the wrist than its middle joint, so
   this works at any hand angle.
3. `classify()` maps that pattern of raised fingers to a gesture name.

### Tuning

If gestures are misread, adjust the `1.1` extension threshold in
`fingers_up()` or the `0.35` touch distance for OK in `classify()`. To add a
gesture, add its finger pattern to the `gestures` dictionary in `classify()`.
