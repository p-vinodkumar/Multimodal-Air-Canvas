import cv2
import mediapipe as mp
import pyautogui

running = False

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7)

screen_width, screen_height = pyautogui.size()

def detect_eye_tracking():
    global running
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get left eye landmarks (around landmarks 33, 133, 159)
                left_eye_x = face_landmarks.landmark[33].x * screen_width
                left_eye_y = face_landmarks.landmark[33].y * screen_height

                # Move cursor based on eye movement
                pyautogui.moveTo(int(left_eye_x), int(left_eye_y))

        cv2.imshow("Eye Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()