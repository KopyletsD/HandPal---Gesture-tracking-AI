import csv
import argparse
import itertools
import threading
import subprocess
import win32api
import win32con
import win32gui
import time
from collections import Counter, deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import tkinter as tk
from pynput.mouse import Controller, Button

from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier


# Set of landmark indices that are drawn larger
LARGE_POINTS = {4, 8, 12, 16, 20}
CURSOR_CHANGE_COOLDOWN = 200
FRAME_SKIP = 5
last_cursor_change_time = 0
mouse = Controller()
CURSOR_PATH = ".\\cursor.cur"

def set_custom_cursor():
    hwnd = win32gui.GetForegroundWindow()
    hcursor = win32gui.LoadImage(0, CURSOR_PATH, win32con.IMAGE_CURSOR, 0, 0, win32con.LR_LOADFROMFILE)
    win32gui.SetCursor(hcursor)
    win32gui.PostMessage(hwnd, win32con.WM_SETCURSOR, hwnd, win32con.HTCLIENT)

def restore_default_cursor():
    win32gui.SetCursor(win32gui.LoadCursor(0, win32con.IDC_ARROW))  # Restore default arrow cursor

def update_cursor(hand_detected, prev_cursor_set):
    global last_cursor_change_time

    current_time = int(time.time() * 1000)  # Convert to milliseconds

    # Check cooldown
    if current_time - last_cursor_change_time < CURSOR_CHANGE_COOLDOWN:
        return  # Skip if cooldown is active

    if hand_detected and not prev_cursor_set:
        set_custom_cursor()
        prev_cursor_set = True
        last_cursor_change_time = current_time  # Reset cooldown
    elif not hand_detected and prev_cursor_set:
        restore_default_cursor()
        prev_cursor_set = False
        last_cursor_change_time = current_time  # Reset cooldown

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Camera device index or RTSP URL")
    parser.add_argument("--width", type=int, default=1920, help="Capture width")
    parser.add_argument("--height", type=int, default=1080, help="Capture height")
    parser.add_argument('--use_static_image_mode', action='store_true', 
                        help="Use static image mode (not recommended for real-time)")
    parser.add_argument("--min_detection_confidence", type=float, default=0.7, help="Min detection confidence")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5, help="Min tracking confidence")
    parser.add_argument("--rtsp_url", type=str, default="rtsp://sitai.duckdns.org:8554/webcam?tcp", 
                        help="RTSP URL for video stream")
    parser.add_argument("--use_opencv", action="store_true", default=False,
                        help="Use OpenCV for frame capture (True) or FFmpeg (False)")
    return parser.parse_args()


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [
        [min(int(lm.x * image_width), image_width - 1),
         min(int(lm.y * image_height), image_height - 1)]
        for lm in landmarks.landmark
    ]


def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0]
    # Convert landmarks to relative coordinates and flatten the list
    relative_landmarks = [(x - base_x, y - base_y) for x, y in landmark_list]
    flat_list = list(itertools.chain.from_iterable(relative_landmarks))
    max_value = max(map(abs, flat_list)) or 1  # avoid division by zero
    return [n / max_value for n in flat_list]


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    if not point_history:
        return []
    base_x, base_y = point_history[0]
    normalized_history = [((x - base_x) / image_width, (y - base_y) / image_height)
                          for x, y in point_history]
    return list(itertools.chain.from_iterable(normalized_history))


def logging_csv(number, mode, landmark_list, point_history_list, frame_count):
    if frame_count % FRAME_SKIP != 0:
        return  # Skip logging to reduce I/O overhead

    if mode == 1 and 0 <= number <= 9:
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            csv.writer(f).writerow([number, *landmark_list])
    elif mode == 2 and 0 <= number <= 9:
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            csv.writer(f).writerow([number, *point_history_list])


def draw_landmarks(image, landmark_list):
    for idx, (x, y) in enumerate(landmark_list):
        # Use a larger circle for certain landmark indices
        radius = 8 if idx in LARGE_POINTS else 5
        cv.circle(image, (x, y), radius, (255, 255, 255), -1)
        cv.circle(image, (x, y), radius, (0, 0, 0), 1)
    return image


def draw_info_text(image, handedness, hand_sign_text, finger_gesture_text):
    info_text = f"{handedness.classification[0].label}"
    if hand_sign_text:
        info_text += f" {hand_sign_text}"
    if finger_gesture_text:
        info_text = f"Finger Gesture: {info_text}"
    cv.putText(image, info_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, info_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image


def draw_point_history(image, point_history):
    for idx, point in enumerate(point_history):
        if point != [0, 0]:
            cv.circle(image, tuple(point), 1 + idx // 2, (152, 251, 152), 2)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    mode_strings = {1: "Logging Key Point", 2: "Logging Point History"}
    if mode in mode_strings:
        cv.putText(image, f"MODE: {mode_strings[mode]}", (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, f"NUM: {number}", (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


def move_mouse_to(hand_landmarks, screen_width, screen_height, alpha=0.6, threshold=5):
    normx = hand_landmarks.landmark[8].x
    normy = hand_landmarks.landmark[8].y
    target_x = int(normx * screen_width)
    target_y = int(normy * screen_height)
    current_x, current_y = mouse.position
    if abs(current_x - target_x) > threshold or abs(current_y - target_y) > threshold:
        new_x = int(current_x + alpha * (target_x - current_x))
        new_y = int(current_y + alpha * (target_y - current_y))
        mouse.position = (new_x, new_y)



def check_thumb_index_click(hand_landmarks):
    """Simula un click sinistro se il pollice e l'indice della mano sinistra si toccano."""
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    # Calcolo della distanza euclidea normalizzata
    distance = np.linalg.norm(np.array([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y]))

    return distance < 0.05  # Soglia per il click

def select_mode(key, mode):
    number = key - 48 if 48 <= key <= 57 else -1
    if key == ord('n'):
        mode = 0
    elif key == ord('k'):
        mode = 1
    elif key == ord('h'):
        mode = 2
    return number, mode


def detect_pointing_direction(hand_landmarks):
    # Use index finger landmarks: 5 is the base (MCP) and 8 is the fingertip.
    index_base = hand_landmarks.landmark[5]
    index_tip = hand_landmarks.landmark[8]
    v = np.array([
        index_tip.x - index_base.x,
        index_tip.y - index_base.y,
        index_tip.z - index_base.z
    ])
    norm = np.linalg.norm(v)
    if norm == 0:
        return "Direction Undetermined"
    # Define camera forward as [0, 0, -1] (since MediaPipe uses a right-handed system)
    cos_angle = -v[2] / norm  # negative because forward is negative Z
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    if angle < 70:
        return "Pointing forward"
    else:
        return "Pointing sideways"




class VideoStream:
    """
    Camera object that controls video streaming from the webcam in a separate thread.
    """
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.cap = cv.VideoCapture(src)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv.CAP_PROP_FPS, fps)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None  # Return safely instead of causing an error
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()

    

def main():
    args = get_args()

    rtsp_url = args.rtsp_url
    mjpeg_url = "udp://127.0.0.1:8554"
    # Initialize capture method based on the flag
    use_opencv = args.use_opencv
    stream = None
    process = None
    
    # Initialize the webcam or video stream
    if use_opencv:
        print("Using OpenCV VideoStream for capture...")
        stream = VideoStream(src=args.device, width=args.width, height=args.height, fps=30)
    else:
        print("Using FFmpeg for capture...")
        # Use a more portable way to locate ffmpeg
        import shutil
        ffmpeg_path = shutil.which('ffmpeg')  # Will find ffmpeg in PATH
        if not ffmpeg_path:
            ffmpeg_path = "..\ffmpeg.exe"  # Default to just the command name if not found
            
        ffmpeg_cmd = [
            ffmpeg_path, '-rtsp_transport', 'tcp', 
            '-analyzeduration', '10000000', '-probesize', '50000000',
            '-i', rtsp_url, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'
        ]
        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Load label files
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    fps_calc = CvFpsCalc(buffer_len=10)
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    mode = 0

    # Create mouse controller and determine screen dimensions once.
    root = tk.Tk()
    root.withdraw()  # Hide the Tkinter window
    screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()

    cap = cv.VideoCapture(mjpeg_url)
    prev_cursor_set = True
    hand_detected = False
    results_holder = [None]
    frame_count = 0
    while True:
        # Read frame based on selected capture method
        
        if use_opencv:
            ret, frame = stream.read()
            if not ret or frame is None:
                print("Error: Unable to read frame from OpenCV stream")
                break
        else:
            # Read frame from FFmpeg
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame from FFmpeg")
                break
            
        fps = fps_calc.get()
        key = cv.waitKey(1)
        if key == 27:  # ESC key to exit
            break
        number, mode = select_mode(key, mode)

        frame = cv.flip(frame, 1)  # Mirror the image for a more natural feel
        debug_image = frame.copy()

        # Process the image for hand detection
        def process_hand(image, hands, results_holder):
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results_holder[0] = hands.process(image_rgb)
        thread = threading.Thread(target=process_hand, args=(frame, hands, results_holder))
        thread.start()
        thread.join()  # Ensure the thread completes before accessing results

        results = results_holder[0]
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark = pre_process_landmark(landmark_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark)
                pointing_direction = detect_pointing_direction(hand_landmarks)
                label = handedness.classification[0].label
                if label == "Right":
                    move_mouse_to(hand_landmarks, screen_width, screen_height)
                    hand_detected = True 
                    set_custom_cursor() 
                    prev_cursor_set = True
                    

                elif label == "Left":
                    if check_thumb_index_click(hand_landmarks):
                        mouse.click(Button.left)
                        cv.putText(debug_image, "Click!", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Update point_history with the current frame's data
                # if pointing_direction == "Pointing forward":
                    # if hand_sign_id == 2:
                        # point_history.append(landmark_list[8])
                        # move_mouse_to(hand_landmarks, screen_width, screen_height)
                    # else:
                        # point_history.append([0, 0])
                # else:
                    # point_history.append([0, 0])

                # Compute the point history after appending the current frame
                pre_processed_point_history = pre_process_point_history(debug_image, list(point_history))
                frame_count += 1
                logging_csv(number, mode, pre_processed_landmark, pre_processed_point_history, frame_count)

                # Process point history for finger gesture classification
                finger_gesture_id = 0
                if len(point_history) == history_length:
                    finger_gesture_id = point_history_classifier(pre_processed_point_history)
                finger_gesture_history.append(finger_gesture_id)

                most_common_fg_id = Counter(finger_gesture_history).most_common(1)[0][0]

                # Draw the hand landmarks and informational text
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(debug_image, handedness,
                                           keypoint_classifier_labels[hand_sign_id],
                                           point_history_classifier_labels[most_common_fg_id])
                cv.putText(debug_image, pointing_direction, (10, 140),
                          cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv.LINE_AA)
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, list(point_history))
        debug_image = draw_info(debug_image, fps, mode, number)

        # Cambio del cursore
        update_cursor(hand_detected, prev_cursor_set)
        if hand_detected:
            if not prev_cursor_set:
                set_custom_cursor() 
                prev_cursor_set = True
        else:
            if prev_cursor_set:
                set_custom_cursor() 
                prev_cursor_set = False
        cv.imshow('Hand Gesture Recognition', debug_image)

    # Clean up based on the method used
    if use_opencv:
        if stream:
            stream.stop()
    else:
        if process:
            process.terminate()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
