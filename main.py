import cv2
import os
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# --------------------
# Config
# --------------------
WIDTH, HEIGHT = 1280, 720
GESTURE_THRESHOLD = 300
FOLDER_PATH = "Presentation"

DRAW_THICKNESS = 12
POINTER_RADIUS = 12
SMOOTHING = 7  # bigger = smoother

# --------------------
# Load slides
# --------------------
slides = sorted(os.listdir(FOLDER_PATH), key=len)
slide_index = 0

# --------------------
# Camera
# --------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# --------------------
# MediaPipe HandLandmarker (Tasks API)
# --------------------
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# --------------------
# Drawing variables
# --------------------
prev_x, prev_y = 0, 0
canvas = None
annotations = [[]]
annotation_number = -1
annotation_start = False
button_pressed = False
counter = 0
DELAY = 30

# Small webcam overlay size
hs, ws = int(120 * 1), int(213 * 1)

# --------------------
# Helper function
# --------------------


def fingers_up(hand_landmarks):
    """Return which fingers are up: [thumb, index, middle, ring, pinky]"""
    if not hand_landmarks:
        return [0, 0, 0, 0, 0]
    lm = hand_landmarks

    fingers = []
    # Thumb
    if lm[4].x < lm[3].x:  # Right hand
        fingers.append(1)
    else:
        fingers.append(0)
    # Other fingers
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        fingers.append(1 if lm[tip].y < lm[pip].y else 0)
    return fingers


# --------------------
# Main loop
# --------------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load slide
    slide_path = os.path.join(FOLDER_PATH, slides[slide_index])
    slide = cv2.imread(slide_path)
    slide = cv2.resize(slide, (WIDTH, HEIGHT))
    if canvas is None:
        canvas = np.zeros_like(slide)

    # Draw gesture threshold line
    cv2.line(img, (0, GESTURE_THRESHOLD),
             (WIDTH, GESTURE_THRESHOLD), (0, 255, 0), 10)

    # Detect hands
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        lm_list = result.hand_landmarks[0]  # first hand
        fingers = fingers_up(lm_list)

        # Index finger tip (landmark 8)
        x = int(lm_list[8].x * WIDTH)
        y = int(lm_list[8].y * HEIGHT)

        # Smooth movement
        if prev_x == 0 and prev_y == 0:
            prev_x, prev_y = x, y
        curr_x = prev_x + (x - prev_x) // SMOOTHING
        curr_y = prev_y + (y - prev_y) // SMOOTHING
        prev_x, prev_y = curr_x, curr_y
        index_finger = (curr_x, curr_y)

        # Navigation gestures (above threshold)
        if lm_list[9].y * HEIGHT <= GESTURE_THRESHOLD:  # hand center
            # Left slide: Thumb only
            if fingers == [1, 0, 0, 0, 0] and slide_index > 0:
                slide_index -= 1
                annotations = [[]]
                annotation_number = -1
                annotation_start = False
                button_pressed = True
            # Right slide: Pinky only
            if fingers == [0, 0, 0, 0, 1] and slide_index < len(slides) - 1:
                slide_index += 1
                annotations = [[]]
                annotation_number = -1
                annotation_start = False
                button_pressed = True

        # Pointer mode (index + middle)
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(slide, index_finger, POINTER_RADIUS, (0, 0, 255), -1)

        # Drawing mode (index only)
        if fingers == [0, 1, 0, 0, 0]:
            if not annotation_start:
                annotation_start = True
                annotation_number += 1
                annotations.append([])
            annotations[annotation_number].append(index_finger)
            cv2.circle(slide, index_finger, POINTER_RADIUS, (0, 0, 255), -1)
        else:
            annotation_start = False

        # Erase mode (index + middle + ring)
        if fingers == [0, 1, 1, 1, 0] and annotations:
            annotations.pop(-1)
            annotation_number -= 1
            button_pressed = True

    else:
        prev_x, prev_y = 0, 0
        annotation_start = False

    # Button delay
    if button_pressed:
        counter += 1
        if counter > DELAY:
            counter = 0
            button_pressed = False

    # Draw annotations
    for ann in annotations:
        for j in range(1, len(ann)):
            cv2.line(slide, ann[j - 1], ann[j], (0, 0, 200), DRAW_THICKNESS)

    # Small webcam overlay
    img_small = cv2.resize(img, (ws, hs))
    h, w, _ = slide.shape
    slide[0:hs, w - ws: w] = img_small

    cv2.imshow("Slides", slide)
    cv2.imshow("Webcam", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
