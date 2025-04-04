import csv
import argparse
import itertools
import threading
from collections import Counter, deque
import sys
import os

import cv2 as cv
import numpy as np
import mediapipe as mp
import tkinter as tk
import time
from pynput.mouse import Controller, Button
import subprocess
import webbrowser
from tkinter import Frame, Button as TkButton, Label, Toplevel, BOTH, X
from PIL import Image, ImageTk

from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier

# Set of landmark indices that are drawn larger
LARGE_POINTS = {4, 8, 12, 16, 20}

# CSV file path for applications
APP_CSV_PATH = 'applications.csv'

# Create default applications CSV file if not exists
def create_default_apps_csv():
    if not os.path.exists(APP_CSV_PATH):
        with open(APP_CSV_PATH, 'w', newline='', encoding='utf-8') as f:  # Specify utf-8 encoding
            writer = csv.writer(f)
            writer.writerow(['label', 'path', 'color', 'icon'])
            writer.writerow(['Calculator', 'C:\\Windows\\System32\\calc.exe', '#0078D7', '🧮'])
            writer.writerow(['Browser', 'https://www.google.com', '#00C800', '🌐'])
            writer.writerow(['Notepad', 'notepad.exe', '#FF6400', '📝'])

# Function to read applications from CSV
def read_applications_from_csv():
    create_default_apps_csv()  # Ensure CSV exists
    applications = []
    
    try:
        with open(APP_CSV_PATH, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                applications.append({
                    'label': row['label'],
                    'path': row['path'],
                    'color': row['color'] if 'color' in row else '#555555',
                    'icon': row['icon'] if 'icon' in row and row['icon'] else '📱'
                })
    except Exception as e:
        print(f"Error reading applications CSV: {e}")
        # Return some default applications in case of error
        applications = [
            {'label': 'Calculator', 'path': 'C:\\Windows\\System32\\calc.exe', 'color': '#0078D7', 'icon': '🧮'},
            {'label': 'Browser', 'path': 'https://www.google.com', 'color': '#00C800', 'icon': '🌐'},
            {'label': 'Notepad', 'path': 'notepad.exe', 'color': '#FF6400', 'icon': '📝'}
        ]
    
    return applications

# Function to launch application based on path
def launch_application(path):
    try:
        if path.startswith('http'):
            webbrowser.open(path)
        else:
            subprocess.Popen(path)
    except Exception as e:
        print(f"Error launching application {path}: {e}")

# Class for the floating menu
class FloatingMenu:
    def __init__(self, root=None):
        if root is None:
            self.root = tk.Tk()
            self.root.withdraw()  # Hide main window
        else:
            self.root = root
            
        self.window = Toplevel(self.root)
        self.window.title("Gesture Menu")
        self.window.attributes("-topmost", True)  # Always on top
        self.window.overrideredirect(True)  # Remove title bar
        self.window.geometry("280x400+50+50")  # Size and position
        self.window.configure(bg='#222831')  # Dark theme background
        
        # Load applications from CSV
        self.applications = read_applications_from_csv()
        
        # Create menu elements
        self.create_menu_elements()
        
        # Hide menu initially
        self.hide()
        
        # Menu state
        self.visible = False
        
        # For moving the window
        self.window.bind("<ButtonPress-1>", self.start_move)
        self.window.bind("<ButtonRelease-1>", self.stop_move)
        self.window.bind("<B1-Motion>", self.do_move)
        
    def create_menu_elements(self):
        # Title with modern design
        title_frame = Frame(self.window, bg='#222831', pady=15)
        title_frame.pack(fill=X)
        
        title_label = Label(title_frame, text="GESTURE CONTROL", font=("Helvetica", 14, "bold"), 
                           bg='#222831', fg='#EEEEEE')
        title_label.pack()
        
        subtitle = Label(title_frame, text="Select Application", font=("Helvetica", 10), 
                        bg='#222831', fg='#00ADB5')
        subtitle.pack(pady=(0, 10))
        
        # Container for buttons
        self.button_container = Frame(self.window, bg='#222831', padx=20)
        self.button_container.pack(fill=BOTH, expand=True)
        
        # Create application buttons dynamically from CSV
        self.buttons = []
        for app in self.applications:
            button_frame = Frame(self.button_container, bg='#222831', pady=8)
            button_frame.pack(fill=X)
            
            # Create button with app info
            button = TkButton(
                button_frame, 
                text=f"{app['icon']} {app['label']}", 
                bg=app['color'],
                fg="white",
                font=("Helvetica", 11),
                relief=tk.FLAT,
                borderwidth=0,
                padx=10,
                pady=8,
                width=20,
                command=lambda path=app['path']: launch_application(path)
            )
            button.pack(fill=X)
            self.buttons.append(button)
        
        # Close button at bottom
        bottom_frame = Frame(self.window, bg='#222831', pady=15)
        bottom_frame.pack(fill=X, side=tk.BOTTOM)
        
        close_button = TkButton(
            bottom_frame, 
            text="✖ Close Menu", 
            bg='#393E46',
            fg='#EEEEEE',
            font=("Helvetica", 10),
            relief=tk.FLAT,
            borderwidth=0,
            padx=10,
            pady=5,
            width=15,
            command=self.hide
        )
        close_button.pack(pady=5)
    
    def show(self):
        self.window.deiconify()
        self.visible = True
        
    def hide(self):
        self.window.withdraw()
        self.visible = False
        
    def toggle(self):
        if self.visible:
            self.hide()
        else:
            self.show()
    
    def start_move(self, event):
        self.x = event.x
        self.y = event.y
        
    def stop_move(self, event):
        self.x = None
        self.y = None
        
    def do_move(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.window.winfo_x() + deltax
        y = self.window.winfo_y() + deltay
        self.window.geometry(f"+{x}+{y}")

    def update(self):
        if hasattr(self, 'root'):
            self.root.update()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument('--use_static_image_mode', action='store_true',
                        help="Use static image mode (not recommended for real-time)")
    parser.add_argument("--min_detection_confidence", type=float, default=0.7, help="Min detection confidence")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5, help="Min tracking confidence")
    parser.add_argument("--run_in_background", action='store_true', 
                        help="Run recognition in background without showing the camera window")
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


def select_mode(key, mode, menu_active, floating_menu):
    number = key - 48 if 48 <= key <= 57 else -1
    if key == ord('n'):
        mode = 0
    elif key == ord('k'):
        mode = 1
    elif key == ord('h'):
        mode = 2
    elif key == ord('m'):
        # Toggle floating menu
        floating_menu.toggle()
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

def draw_circle_on_right(image):
    # Get the image dimensions
    height, width, _ = image.shape

    # Set the circle's center at the right side of the image
    center = (width - 50, height // 2)  # 50px from the right edge, vertically centered

    # Set the radius and color of the circle
    radius = 30  # You can change this to your desired radius
    # Draw pulsing circle effect
    color_intensity = 150 + int(50 * np.sin(time.time() * 5))  # Pulsing effect
    color = (0, color_intensity, 255)  # Blue color with pulsing intensity

    # Draw the circle on the image
    cv.circle(image, center, radius, color, -1)  # -1 to fill the circle
    
    # Add a small text label
    cv.putText(image, "Menu", (center[0]-20, center[1]+40), 
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    
    return image, center, radius

def perform_gesture_detection(image, circle_data, hands, keypoint_classifier, 
                            point_history_classifier, point_history, finger_gesture_history,
                            keypoint_classifier_labels, point_history_classifier_labels,
                            history_length, floating_menu):
    """
    Perform gesture detection and mouse control in a separate function.
    This function can be called in both the main app and background mode.
    """
    # Extract circle data if available
    circle_center, circle_radius = circle_data if circle_data else (None, None)
    
    # Initialize mouse controller
    mouse = Controller()
    
    # Convert image to RGB for MediaPipe
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    # Process image with MediaPipe Hands
    results = hands.process(image_rgb)
    
    # If no hands detected, clear point history
    if results.multi_hand_landmarks is None:
        point_history.appendleft([0, 0])
        return image, mouse.position, False
    
    # Variables for mouse control
    is_clicking = False
    mouse_position = mouse.position
    
    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        # Hand landmarks
        landmark_list = calc_landmark_list(image, hand_landmarks)
        
        # Normalization
        pre_processed_landmark_list = pre_process_landmark(landmark_list)
        pre_processed_point_history_list = pre_process_point_history(image, point_history)
        
        # Hand sign classification
        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
        if hand_sign_id == 2:  # Index finger extended
            point_history.appendleft(landmark_list[8])  # Index fingertip
        else:
            point_history.appendleft([0, 0])
        
        # Finger gesture classification
        finger_gesture_id = 0
        point_history_len = len(pre_processed_point_history_list)
        if point_history_len == (history_length * 2):
            finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
        finger_gesture_history.append(finger_gesture_id)
        most_common_fg_id = Counter(finger_gesture_history).most_common(1)[0][0]
        
        # MOUSE CONTROL AND MENU INTERACTION
        # Check if user is clicking
        is_clicking = check_thumb_index_click(hand_landmarks)
        
        # Move mouse with index finger
        move_mouse_to(hand_landmarks, 1920, 1080, mouse)  # Use screen resolution
        mouse_position = mouse.position
        
        # Execute click if detected
        if is_clicking:
            mouse.press(Button.left)
            mouse.release(Button.left)
        
        # Check if index finger is pointing towards circle (to activate menu)
        if circle_center and circle_radius and is_mouse_inside_circle(mouse_position, circle_center, circle_radius):
            floating_menu.show()
        
        # Draw hand landmarks and information on debug_image
        image = draw_landmarks(image, landmark_list)
        image = draw_point_history(image, point_history)
        image = draw_info_text(image, handedness, 
                              keypoint_classifier_labels[hand_sign_id],
                              point_history_classifier_labels[most_common_fg_id])
    
    return image, mouse_position, is_clicking

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
            return self.ret, self.frame.copy() if self.ret else None

    def stop(self):
        self.stopped = True

# Class to run gesture recognition in background
class GestureRecognitionThread(threading.Thread):
    def __init__(self, args):
        threading.Thread.__init__(self, daemon=True)
        self.args = args
        self.stopped = False
        
        # Create floating menu
        self.root = tk.Tk()
        self.root.withdraw()  # Hide main window
        self.floating_menu = FloatingMenu(self.root)
        
        # Initialize webcam
        self.video_stream = VideoStream(
            src=args.device,
            width=args.width,
            height=args.height,
            fps=30
        )
        
        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=args.use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        
        # Load classifiers
        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()
        
        # Read classifier labels
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
        with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            self.point_history_classifier_labels = [row[0] for row in csv.reader(f)]
        
        # Point history
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)
        
        # FPS calculator
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)
        
        # Mouse state
        self.mouse = Controller()
        self.mouse_position = (0, 0)
        self.is_clicking = False
        
        # Create mini-display to show camera input
        self.mini_display = Toplevel(self.root)
        self.mini_display.title("Camera Preview")
        self.mini_display.geometry("320x300+1600+50")  # Position in top right
        self.mini_display.attributes("-topmost", True)  # Always on top
        self.mini_display.configure(bg='#222831')  # Dark theme background
        self.mini_display.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handle closing
        
        # Create label to show camera feed
        self.camera_label = Label(self.mini_display, bg='#222831')
        self.camera_label.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Add button to show/hide menu
        self.menu_button = TkButton(self.mini_display, text="Toggle Menu", 
                                  bg="#00ADB5", fg="white", font=("Helvetica", 10, "bold"),
                                  relief=tk.FLAT, command=self.floating_menu.toggle)
        self.menu_button.pack(fill=X, padx=10, pady=5)
        
        # Add exit button
        self.exit_button = TkButton(self.mini_display, text="Exit Application", 
                                   bg="#D72000", fg="white", font=("Helvetica", 10),
                                   relief=tk.FLAT, command=self.stop)
        self.exit_button.pack(fill=X, padx=10, pady=5)
        
    def update_mini_display(self, image):
        """Update mini-display with current image"""
        # Resize image for mini-display
        small_image = cv.resize(image, (300, 200))
        # Convert from BGR to RGB
        small_image = cv.cvtColor(small_image, cv.COLOR_BGR2RGB)
        # Convert to format for tkinter
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(small_image))
        # Update label
        self.camera_label.config(image=img_tk)
        self.camera_label.image = img_tk  # Keep a reference
    
    def run(self):
        """Main loop for gesture recognition thread"""
        while not self.stopped:
            # Update tkinter UI
            self.root.update()
            
            # Read frame from camera
            ret, image = self.video_stream.read()
            if not ret:
                continue
                
            # Flip image horizontally
            image = cv.flip(image, 1)
            debug_image = image.copy()
            
            # Draw a circle on the right side of the image
            debug_image, circle_center, circle_radius = draw_circle_on_right(debug_image)
            circle_data = (circle_center, circle_radius)
            
            # Perform gesture detection
            debug_image, mouse_pos, is_click = perform_gesture_detection(
                debug_image, circle_data, self.hands, self.keypoint_classifier, 
                self.point_history_classifier, self.point_history, 
                self.finger_gesture_history, self.keypoint_classifier_labels,
                self.point_history_classifier_labels, self.history_length, 
                self.floating_menu
            )
            
            # Update FPS information
            fps = self.cvFpsCalc.get()
            debug_image = draw_info(debug_image, fps, 0, -1)
            
            # Update mini-display
            self.update_mini_display(debug_image)
            
            # Short pause to reduce CPU usage
            time.sleep(0.01)
        
        # Clean up when stopped
        self.cleanup()
    
    def stop(self):
        """Stop recognition thread and clean up resources"""
        self.stopped = True
        
    def on_closing(self):
        """Handle mini-display window closing"""
        self.stop()
        
    def cleanup(self):
        """Clean up resources"""
        self.video_stream.stop()
        self.mini_display.destroy()
        self.root.destroy()
        # Exit application
        os._exit(0)

def main():
    # Parse command line arguments
    args = get_args()
    
    # Create default applications CSV if it doesn't exist
    create_default_apps_csv()
    
    # If requested to run in background
    if args.run_in_background:
        # Start gesture recognition thread in background
        gesture_thread = GestureRecognitionThread(args)
        gesture_thread.start()
        # Use tkinter as main loop
        gesture_thread.root.mainloop()
        return
        
    # Initialize webcam with thread
    video_stream = VideoStream(
        src=args.device,
        width=args.width,
        height=args.height,
        fps=30
    )
    
    # Create floating menu
    root = tk.Tk()
    root.withdraw()  # Hide main window
    floating_menu = FloatingMenu(root)
    
    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    # Load classifiers
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Read classifier labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    # Initialize state variables
    mode = 0
    number = -1

    # Point history
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    # FPS calculator
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    
    # Mouse controller
    mouse = Controller()

    while True:
        # Update floating menu
        root.update()
        
        fps = cvFpsCalc.get()
        ret, image = video_stream.read()
        if not ret:
            break

        # Flip image horizontally to create a "mirror" effect
        image = cv.flip(image, 1)
        debug_image = image.copy()

        # Draw a circle on the right side of the image
        debug_image, circle_center, circle_radius = draw_circle_on_right(debug_image)
        circle_data = (circle_center, circle_radius)
        
        # Perform gesture detection
        debug_image, mouse_position, is_clicking = perform_gesture_detection(
            debug_image, circle_data, hands, keypoint_classifier, 
            point_history_classifier, point_history, finger_gesture_history,
            keypoint_classifier_labels, point_history_classifier_labels,
            history_length, floating_menu
        )

        # Draw information on FPS and mode
        debug_image = draw_info(debug_image, fps, mode, number)
        
        # User instructions
        cv.putText(debug_image, "Press 'M' to toggle menu, 'Q' to exit", (10, 150),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(debug_image, "Press 'B' to switch to background mode", (10, 180),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        # Show image
        cv.imshow('Hand Gesture Recognition', debug_image)

        # Handle keyboard input
        key = cv.waitKey(1)
        if key == 27 or key == ord('q'):  # ESC or 'q' to exit
            break
        if key == ord('b'):  # 'b' to switch to background mode
            # Switch to background mode
            video_stream.stop()
            cv.destroyAllWindows()
            
            # Start gesture recognition thread in background
            args.run_in_background = True
            gesture_thread = GestureRecognitionThread(args)
            gesture_thread.start()
            # Use tkinter as main loop
            gesture_thread.root.mainloop()
            return
            
        number, mode = select_mode(key, mode, False, floating_menu)

    # Clean up
    video_stream.stop()
    cv.destroyAllWindows()
    root.destroy()


if __name__ == '__main__':
    main()