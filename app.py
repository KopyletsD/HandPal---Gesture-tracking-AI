import csv
import argparse
import itertools
import threading
from collections import Counter, deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import tkinter as tk
import time
from pynput.mouse import Controller, Button
import subprocess
import webbrowser

from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier

# Set of landmark indices that are drawn larger
LARGE_POINTS = {4, 8, 12, 16, 20}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    # Lower resolution for faster processing
    parser.add_argument("--width", type=int, default=1920, help="Capture width")
    parser.add_argument("--height", type=int, default=1080, help="Capture height")
    parser.add_argument('--use_static_image_mode', action='store_true',
                        help="Use static image mode (not recommended for real-time)")
    parser.add_argument("--min_detection_confidence", type=float, default=0.7, help="Min detection confidence")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5, help="Min tracking confidence")
    return parser.parse_args()


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [
        [min(int(lm.x * image_width), image_width - 1),
         min(int(lm.y * image_height), image_height - 1)]
        for lm in landmarks.landmark
    ]

def create_menu(debug_image, menu_top_left, mouse_position, is_clicking):
    """
    Creates an improved menu with multiple application options and visual feedback.
    
    Args:
        debug_image: The image to draw the menu on
        menu_top_left: Top-left coordinates of the menu
        mouse_position: Current mouse position for hover detection
        is_clicking: Boolean indicating if the user is currently clicking
    
    Returns:
        The image with the menu drawn on it and a list of actions to execute
    """
    # Menu dimensions
    menu_width = 200
    menu_height = 300
    button_height = 50
    padding = 10
    
    # Menu background with semi-transparency
    overlay = debug_image.copy()
    menu_bottom_right = (menu_top_left[0] + menu_width, menu_top_left[1] + menu_height)
    cv.rectangle(overlay, menu_top_left, menu_bottom_right, (50, 50, 50), -1)
    cv.addWeighted(overlay, 0.7, debug_image, 0.3, 0, debug_image)
    
    # Menu border
    cv.rectangle(debug_image, menu_top_left, menu_bottom_right, (200, 200, 200), 2)
    
    # Menu title
    title_y = menu_top_left[1] + 30
    cv.putText(debug_image, "MENU", (menu_top_left[0] + 70, title_y), 
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)
    
    # Define menu buttons
    menu_items = [
        {"label": "Calculator", "icon": "🧮", "action": open_calculator, "color": (0, 120, 255)},
        {"label": "Browser", "icon": "🌐", "action": open_browser, "color": (0, 200, 0)},
        {"label": "Notepad", "icon": "📝", "action": open_notepad, "color": (255, 100, 0)},
        {"label": "Music", "icon": "🎵", "action": open_music_player, "color": (180, 0, 180)}
    ]
    
    # Draw menu items
    actions_to_execute = []
    
    for i, item in enumerate(menu_items):
        # Button position
        button_top = menu_top_left[1] + 50 + (i * (button_height + padding))
        button_bottom = button_top + button_height
        button_left = menu_top_left[0] + padding
        button_right = menu_top_left[0] + menu_width - padding
        
        # Check if mouse is hovering over this button
        is_hovering = (button_left <= mouse_position[0] <= button_right and 
                       button_top <= mouse_position[1] <= button_bottom)
        
        # Draw button with hover effect
        button_color = item["color"] if not is_hovering else tuple(min(c + 50, 255) for c in item["color"])
        cv.rectangle(debug_image, (button_left, button_top), (button_right, button_bottom), 
                     button_color, -1 if is_hovering else 2)
        
        # Icon and label
        icon_x = button_left + 20
        text_x = button_left + 50
        text_y = button_top + 32
        
        # Draw icon placeholder 
        cv.circle(debug_image, (icon_x, text_y - 10), 10, (255, 255, 255), -1)
        
        # Draw label
        cv.putText(debug_image, item["label"], (text_x, text_y), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        
        # Store action if clicking on this button
        if is_clicking and is_hovering:
            actions_to_execute.append(item["action"])
    
    return debug_image, actions_to_execute

def open_calculator():
    try:
        subprocess.Popen('calc.exe')  # For Windows
    except Exception as e:
        print(f"Error opening calculator: {e}")

def open_browser():
    try:
        webbrowser.open('https://www.google.com')
    except Exception as e:
        print(f"Error opening browser: {e}")

def open_notepad():
    try:
        subprocess.Popen('notepad.exe')  # For Windows
    except Exception as e:
        print(f"Error opening notepad: {e}")

def open_music_player():
    try:
        # For Windows, open the default music player
        subprocess.Popen('wmplayer.exe')
    except Exception as e:
        print(f"Error opening music player: {e}")

def draw_circle_on_right(image):
    # Get the image dimensions
    height, width, _ = image.shape

    # Set the circle's center at the right side of the image
    center = (width - 50, height // 2)  # 50px from the right edge, vertically centered

    # Set the radius and color of the circle
    radius = 30  # You can change this to your desired radius
    color = (0, 0, 255)  # Red color in BGR format (OpenCV uses BGR)

    # Draw the circle on the image
    cv.circle(image, center, radius, color, -1)  # -1 to fill the circle
    return image

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


def logging_csv(number, mode, landmark_list, point_history_list):
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


def move_mouse_to(hand_landmarks, screen_width, screen_height, mouse):
    normx = hand_landmarks.landmark[8].x
    normy = hand_landmarks.landmark[8].y
    mouse.position = (int(normx * screen_width), int(normy * screen_height))


def check_thumb_index_click(hand_landmarks):
    """
    Check if the thumb tip (landmark 4) is very close to the index finger base (landmark 5).
    If so, consider it a click.
    """
    thumb_tip = hand_landmarks.landmark[4]
    index_base = hand_landmarks.landmark[5]
    # Calculate Euclidean distance in normalized space (x, y)
    thumb_index_distance = np.linalg.norm(np.array([thumb_tip.x - index_base.x, thumb_tip.y - index_base.y]))
    # Adjust threshold as needed (here 0.05 is chosen arbitrarily)
    if thumb_index_distance < 0.05:
        return True
    return False


def select_mode(key, mode, menu_active):
    number = key - 48 if 48 <= key <= 57 else -1
    if key == ord('n'):
        mode = 0
    elif key == ord('k'):
        mode = 1
    elif key == ord('h'):
        mode = 2
    elif key == ord('m'):
        menu_active = not menu_active
    return number, mode, menu_active


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

def is_mouse_inside_circle(mouse_position, circle_center, radius):
    """
    Check if the mouse position is inside the circle.
    """
    distance = np.linalg.norm(np.array(mouse_position) - np.array(circle_center))
    return distance <= radius


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
            # Return a copy of the frame to avoid threading issues
            if self.ret:
                return self.ret, self.frame.copy()
            else:
                return False, None

    def stop(self):
        self.stopped = True
        self.cap.release()


def main():
    # Initialize variables
    menu_active = False
    alr_clicked = False
    
    args = get_args()

    # Initialize threaded video stream
    stream = VideoStream(src=args.device, width=args.width, height=args.height, fps=30)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,  # For real-time use, do not use static mode.
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
    mouse = Controller()
    root = tk.Tk()
    root.withdraw()  # Hide the Tkinter window
    screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()

    circle_center = (screen_width - 50, screen_height // 2)
    circle_radius = 30

    click_cooldown = 2.0  # 2 second cooldown between clicks
    last_click_time = 0.0  # Initialize last click time to 0
    menu_top_left = (20, 100)  # Position for the menu

    while True:
        fps = fps_calc.get()
        key = cv.waitKey(1)  # Reduced delay for higher FPS
        if key == 27:  # ESC key to exit
            break
        number, mode, menu_active = select_mode(key, mode, menu_active)

        ret, frame = stream.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)  # Mirror the image for a more natural feel.
        debug_image = frame.copy()

        # Process the image for hand detection.
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True

        is_clicking = False
        current_mouse_pos = mouse.position

        if results.multi_hand_landmarks:
            debug_image = draw_circle_on_right(debug_image)
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Calculate landmark coordinates
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                
                # Pre-process landmark coordinates for gesture recognition
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                # Append index finger tip coordinates to point history
                point_history.append(landmark_list[8])
                
                # Pre-process point history
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                # Point history classification if necessary
                if hand_sign_id == 2:  # Assuming 2 indicates a pointing gesture
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len >= 1:
                        finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                    else:
                        finger_gesture_id = 0
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(finger_gesture_history).most_common(1)[0][0]
                    finger_gesture_text = point_history_classifier_labels[most_common_fg_id]
                else:
                    finger_gesture_text = ""
                
                # Get hand gesture labels
                hand_sign_text = keypoint_classifier_labels[hand_sign_id]
                
                # Draw landmarks and info on the image
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(debug_image, handedness, hand_sign_text, finger_gesture_text)

                # Control the mouse based on hand position
                move_mouse_to(hand_landmarks, screen_width, screen_height, mouse)
                
                # Check for click gesture
                is_clicking = check_thumb_index_click(hand_landmarks)
                if is_clicking and time.time() - last_click_time > click_cooldown:
                    last_click_time = time.time()
                
                # Log data if in logging mode
                if mode in (1, 2):
                    logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)
                
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Draw the menu if active
        if menu_active:
            debug_image, actions = create_menu(debug_image, menu_top_left, current_mouse_pos, is_clicking)
            # Execute any actions triggered by menu interaction
            for action in actions:
                action()

        cv.imshow('Hand Gesture Recognition', debug_image)

    stream.stop()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()