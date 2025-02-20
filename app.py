import csv
import argparse
import itertools
import threading
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

def create_menu(debug_image, menu_top_left, menu_width, menu_height, dot_radius):
    # Draw rectangle (menu)
    cv.rectangle(debug_image, menu_top_left, 
                 (menu_top_left[0] + menu_width, menu_top_left[1] + menu_height), 
                 (255, 255, 255), 2)
    
    # Draw blue dots inside the menu
    num_dots = 5  # Number of dots
    space_between_dots = menu_height // (num_dots + 1)
    
    for i in range(num_dots):
        dot_center = (menu_top_left[0] + menu_width // 2, 
                      menu_top_left[1] + (i + 1) * space_between_dots)
        cv.circle(debug_image, dot_center, dot_radius, (255, 0, 0), -1)
    
    return debug_image


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
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()


def main():
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

    # Pre-calculate the restricted area (the "no-access" rectangle) coordinates.
    rect_top_left = (args.width // 4, args.height // 4)
    rect_bottom_right = (int(2 * args.width / 3.5), int(2 * args.height / 4))

    circle_center = (screen_width - 50, screen_height // 2)
    circle_radius = 30

    while True:
        fps = fps_calc.get()
        key = cv.waitKey(1)  # Reduced delay for higher FPS
        if key == 27:  # ESC key to exit
            break
        number, mode = select_mode(key, mode)

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

        if results.multi_hand_landmarks:
            debug_image = draw_circle_on_right(debug_image)
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark = pre_process_landmark(landmark_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark)
                pointing_direction = detect_pointing_direction(hand_landmarks)

                # --- Check for thumb-index touch to simulate mouse click ---
                if check_thumb_index_click(hand_landmarks):
                    move_mouse_to(hand_landmarks, screen_width, screen_height, mouse)
                    mouse.click(Button.left)

                # --- Process hand gestures and point history ---
                if pointing_direction == "Pointing forward":
                    if hand_sign_id == 2:
                        point_history.append(landmark_list[8])
                        move_mouse_to(hand_landmarks, screen_width, screen_height, mouse)
                    else:
                        point_history.append([0, 0])
                else:
                    point_history.append([0, 0])

                # Compute the point history after appending the current frame.
                pre_processed_point_history = pre_process_point_history(debug_image, list(point_history))
                logging_csv(number, mode, pre_processed_landmark, pre_processed_point_history)

                # Process point history for finger gesture classification.
                finger_gesture_id = 0
                if len(point_history) == history_length:
                    finger_gesture_id = point_history_classifier(pre_processed_point_history)
                finger_gesture_history.append(finger_gesture_id)
                menuActive = False
                menu_width = 300  # Width of the menu
                menu_height = 800  # Height of the menu
                dot_radius = 50
                menu_top_left = (screen_width - menu_width - 20, (screen_height - menu_height) // 2)
                circle_radius = 30  # Radius of the red circle
                circle_center = (screen_width - 50, screen_height // 2)  # Circle center near the right edge
                circle_clicked = False
               # Check if the mouse is inside the red circle and set the menu active
                if pointing_direction == "Pointing forward":
                   if menuActive == False:
                       if is_mouse_inside_circle(mouse.position, circle_center, circle_radius):
                           menuActive = True  # Activate the menu
                           # Do not draw the red circle anymore after it is clicked
                           circle_clicked = True  # Flag to track the click
                   if menuActive == True:
                       # Keep the menu visible
                       debug_image = create_menu(debug_image, menu_top_left, menu_width, menu_height, dot_radius)
                   
                   # If the circle hasn't been clicked, draw the red circle
                   if not circle_clicked:
                       debug_image = draw_circle_on_right(debug_image)
               
                       return debug_image
            #inserire qui la prossima modifica del menu



                most_common_fg_id = Counter(finger_gesture_history).most_common(1)[0][0]

                # Draw the hand landmarks and informational text.
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

        cv.imshow('Hand Gesture Recognition', debug_image)

    # Clean up
    stream.stop()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()