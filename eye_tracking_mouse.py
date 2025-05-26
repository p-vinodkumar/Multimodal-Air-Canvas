import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque

# Flags and smoothing
running = True
blink_counter = 0
double_blink_detected = False
eye_movement_history = deque(maxlen=5)
last_blink_time = time.time()

# Constants
BLINK_THRESHOLD = 0.2  # Adjust threshold for blink detection
CONSEC_BLINKS_REQUIRED = 2  # Number of blinks required for a click
TIME_WINDOW = 1  # Time within which double blink must occur

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.7)

screen_width, screen_height = pyautogui.size()

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_ear(landmarks, frame_w, frame_h):
    """Calculate Eye Aspect Ratio (EAR) to detect blinks."""
    p1 = (int(landmarks[362].x * frame_w), int(landmarks[362].y * frame_h))  # Right eye outer
    p2 = (int(landmarks[386].x * frame_w), int(landmarks[386].y * frame_h))  # Top eyelid
    p3 = (int(landmarks[374].x * frame_w), int(landmarks[374].y * frame_h))  # Bottom eyelid
    p4 = (int(landmarks[263].x * frame_w), int(landmarks[263].y * frame_h))  # Right eye inner

    ear = euclidean_distance(p2, p3) / (euclidean_distance(p1, p4) + 1e-6)
    return ear

def detect_eye_tracking():
    global running, blink_counter, double_blink_detected, last_blink_time

    cap = cv2.VideoCapture(0)

    while cap.isOpened() and running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        frame_h, frame_w, _ = frame.shape

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # ---------- Cursor Movement ----------
            # Use iris or eye corner center (more stable than single landmark)
            eye_x = (landmarks[474].x + landmarks[475].x + landmarks[476].x + landmarks[477].x) / 4
            eye_y = (landmarks[474].y + landmarks[475].y + landmarks[476].y + landmarks[477].y) / 4

            # Smooth cursor movement
            eye_movement_history.append((eye_x, eye_y))
            avg_x = np.mean([pt[0] for pt in eye_movement_history])
            avg_y = np.mean([pt[1] for pt in eye_movement_history])

            screen_x = int(avg_x * screen_width)
            screen_y = int(avg_y * screen_height)
            pyautogui.moveTo(screen_x, screen_y)

            # ---------- Blink Detection ----------
            ear = compute_ear(landmarks, frame_w, frame_h)

            if ear < BLINK_THRESHOLD:  # Blink detected
                if time.time() - last_blink_time < TIME_WINDOW:  # Check if blink happens within set time window
                    blink_counter += 1
                else:
                    blink_counter = 1  # Reset if too much time passes
                last_blink_time = time.time()

                if blink_counter >= CONSEC_BLINKS_REQUIRED:  # If two consecutive blinks detected
                    pyautogui.click()
                    print("Double Blink Detected: Click Performed!")
                    blink_counter = 0  # Reset counter after click

        cv2.imshow("Eye Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the function

if __name__ == "__main__":
    try:
        detect_eye_tracking()
    except KeyboardInterrupt:
        print("Program stopped by user")
        running = False
