import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Parameters
width, height = 1280, 720
gestureThreshold = 300
folderPath = "Presentation"

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector Setup
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Variables
delay = 30
buttonPressed = False
counter = 0
imgNumber = 0
annotations = [[]]
annotationNumber = -1
annotationStart = False
hs, ws = int(120 * 3), int(213 * 3)

# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
pathImages = [img for img in pathImages if not img.startswith('.')]
print(f"Presentation images: {pathImages}")


def fingersUp(hand_landmarks, handedness):
    """Count which fingers are up - returns list like [thumb, index, middle, ring, pinky]"""
    if not hand_landmarks or len(hand_landmarks) == 0:
        return [0, 0, 0, 0, 0]

    landmarks = hand_landmarks[0]
    fingers = []

    # Thumb
    # handedness is a list of lists, get the first classification
    is_right_hand = True
    try:
        if hasattr(handedness[0][0], 'category_name'):
            is_right_hand = handedness[0][0].category_name == "Right"
        elif hasattr(handedness[0][0], 'label'):
            is_right_hand = handedness[0][0].label == "Right"
    except:
        pass

    if is_right_hand:
        if landmarks[4].x < landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if landmarks[4].x > landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Other 4 fingers
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        if landmarks[tip].y < landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def draw_landmarks(img, hand_landmarks):
    """Draw hand landmarks on image"""
    if not hand_landmarks or len(hand_landmarks) == 0:
        return

    landmarks = hand_landmarks[0]
    h, w, _ = img.shape

    # Draw connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]

    for connection in connections:
        start_idx, end_idx = connection
        start_point = (int(landmarks[start_idx].x * w),
                       int(landmarks[start_idx].y * h))
        end_point = (int(landmarks[end_idx].x * w),
                     int(landmarks[end_idx].y * h))
        cv2.line(img, start_point, end_point, (0, 255, 0), 2)

    for landmark in landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)


print("\nGesture Controls:")
print("- Thumb up (above green line): Previous slide")
print("- Pinky up (above green line): Next slide")
print("- Index finger only: Draw on slide")
print("- Index + Middle fingers: Pointer (red circle)")
print("- Index + Middle + Ring fingers: Erase last drawing")
print("- Press 'q': Quit\n")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    if imgCurrent is None:
        print(f"Error loading image: {pathFullImage}")
        break

    # Convert to RGB and create MP Image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # Detect hands
    detection_result = detector.detect(mp_image)

    # Draw gesture threshold line
    cv2.line(img, (0, gestureThreshold),
             (width, gestureThreshold), (0, 255, 0), 10)

    if detection_result.hand_landmarks and not buttonPressed:
        hand_landmarks = detection_result.hand_landmarks
        handedness = detection_result.handedness

        # Draw hand landmarks
        draw_landmarks(img, hand_landmarks)

        # Get landmarks
        landmarks = hand_landmarks[0]

        # Get center of hand (palm base)
        cx = int(landmarks[9].x * width)
        cy = int(landmarks[9].y * height)

        # Get index finger tip position
        lm8_x = int(landmarks[8].x * width)
        lm8_y = int(landmarks[8].y * height)

        # Constrain values for easier drawing
        xVal = int(np.interp(lm8_x, [width // 2, width], [0, width]))
        yVal = int(np.interp(lm8_y, [150, height-150], [0, height]))
        indexFinger = (xVal, yVal)

        # Get which fingers are up
        fingers = fingersUp(hand_landmarks, handedness)

        # Navigation gestures (only above threshold line)
        if cy <= gestureThreshold:
            # Left - Thumb only
            if fingers == [1, 0, 0, 0, 0]:
                print("Left")
                buttonPressed = True
                if imgNumber > 0:
                    imgNumber -= 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False

            # Right - Pinky only
            if fingers == [0, 0, 0, 0, 1]:
                print("Right")
                buttonPressed = True
                if imgNumber < len(pathImages) - 1:
                    imgNumber += 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False

        # Pointer mode - Index and Middle fingers
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        # Drawing mode - Index finger only
        if fingers == [0, 1, 0, 0, 0]:
            if not annotationStart:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            annotations[annotationNumber].append(indexFinger)
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
        else:
            annotationStart = False

        # Erase - Index + Middle + Ring fingers
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                if len(annotations) > 0:
                    annotations.pop(-1)
                    annotationNumber -= 1
                    buttonPressed = True
    else:
        annotationStart = False

    # Button delay
    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    # Draw all annotations
    for i, annotation in enumerate(annotations):
        for j in range(len(annotation)):
            if j != 0:
                cv2.line(imgCurrent, annotation[j - 1],
                         annotation[j], (0, 0, 200), 12)

    # Add small webcam view to slide
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws: w] = imgSmall

    # Display slide number
    cv2.putText(imgCurrent, f"{imgNumber + 1}/{len(pathImages)}",
                (w - 150, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Slides", imgCurrent)
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nPresentation ended.")
