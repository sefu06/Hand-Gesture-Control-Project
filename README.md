# Hand Gesture Presentation Project

## Overview
This project allows you to **control a presentation using hand gestures** captured via a webcam. Users can navigate slides, draw annotations, use a pointer, and erase drawings—all without touching a mouse or keyboard.  

The system leverages **MediaPipe HandLandmarker**, **OpenCV**, and **NumPy** for real-time hand tracking and gesture-based control. It’s built in Python and designed for simplicity and smooth gesture interaction.

---

## Tech Stack

| Component | Purpose |
|-----------|---------|
| **Python 3.10** | Main programming language |
| **OpenCV** | Webcam access, image processing, drawing annotations, displaying slides |
| **NumPy** | Array operations for the canvas and coordinates |
| **MediaPipe Tasks API** | HandLandmarker model for detecting hand landmarks and gestures |

---

## Features

1. **Slide Navigation**
   - Move to the **next slide** by raising the **pinky finger** above the gesture threshold line.
   - Move to the **previous slide** by raising the **thumb** above the gesture threshold line.
   - Prevents accidental slide switching using a **button delay** mechanism.

2. **Pointer**
   - Shows a **red circle pointer** when **index + middle fingers** are raised.
   - Moves smoothly across the slide using **coordinate smoothing** to avoid jitter.

3. **Drawing**
   - Draw freehand on the slides using the **index finger only**.
   - Stores drawings as sequences of points in a list called `annotations`.
   - Smooths drawing movements for more natural curves.
   - Drawings continue even if finger detection flickers slightly (debounced).

4. **Erase**
   - Raise **index + middle + ring fingers** to erase the last drawn annotation.

5. **Webcam Overlay**
   - Small live webcam feed is shown on the top-right corner of the slide for reference.

---

## How It Works (Logic)

1. **Initialization**
   - Load all slides from the `Presentation` folder.
   - Initialize the webcam (`cv2.VideoCapture`) and set resolution.
   - Initialize the MediaPipe **HandLandmarker** model for single-hand detection.
   - Set up variables for drawing, annotations, pointer smoothing, and gesture detection.

2. **Main Loop**
   - Read each webcam frame and **flip horizontally** for mirror effect.
   - Convert frame to RGB for MediaPipe processing.
   - Load the current slide and initialize a **canvas** for drawing if needed.

3. **Hand Detection**
   - Pass the RGB frame to `HandLandmarker.detect()` to detect hands.
   - Retrieve the **21 hand landmarks** for the first hand.
   - Use landmarks to determine **finger states**:
     - `[thumb, index, middle, ring, pinky]` with `1` if raised, `0` if down.

4. **Gesture-Based Actions**
   - **Slide Navigation**: If the hand is above the `GESTURE_THRESHOLD` line:
     - Thumb → Previous slide
     - Pinky → Next slide
   - **Pointer Mode**: Index + middle → Red circle shows the finger tip location.
   - **Drawing Mode**: Index only → Append coordinates to `annotations` and draw line on canvas.
   - **Erase Mode**: Index + middle + ring → Remove last annotation.

5. **Smoothing**
   - Smooth finger movement using previous coordinates:
     ```python
     curr_x = prev_x + (x - prev_x) // SMOOTHING
     curr_y = prev_y + (y - prev_y) // SMOOTHING
     ```
   - Prevents the pointer or drawing from jumping abruptly.

6. **Drawing & Overlay**
   - Lines are drawn for each annotation in `annotations`.
   - Webcam feed resized and overlaid on the slide for real-time reference.
   - Slide and drawing combined using `cv2.addWeighted`.

7. **Controls**
   - Press `'q'` to quit.
   - Button delay ensures gestures don’t trigger multiple actions immediately.

## Demo
[![Watch the Demo](https://img.youtube.com/vi/2fPzL85btgw/0.jpg)](https://youtu.be/2fPzL85btgw)



