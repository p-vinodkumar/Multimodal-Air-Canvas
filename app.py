from flask import Flask, render_template, redirect, url_for
from threading import Thread
import hand_gesture_mouse
import eye_tracking_mouse  # Import eye tracking module

app = Flask(__name__)
detection_thread = None
mode = "hand"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start/<tracking_mode>")
def start_detection(tracking_mode):
    global detection_thread, mode
    mode = tracking_mode  # Set mode (hand/eye)
    
    if detection_thread is None or not detection_thread.is_alive():
        if mode == "hand":
            hand_gesture_mouse.running = True
            detection_thread = Thread(target=hand_gesture_mouse.detect_hand_gestures)
        elif mode == "eye":
            eye_tracking_mouse.running = True
            detection_thread = Thread(target=eye_tracking_mouse.detect_eye_tracking)
        detection_thread.start()
    
    return redirect(url_for("index"))

@app.route("/stop")
def stop_detection():
    hand_gesture_mouse.running = False
    eye_tracking_mouse.running = False
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)