import argparse
import itertools
import threading
import win32api
import win32con
import win32gui
import cv2 as cv
import numpy as np
import mediapipe as mp
import tkinter as tk
from pynput.mouse import Controller, Button
import win32gui
import win32con

CURSOR_PATH = "C:\\Users\\carminati.20133\\Desktop\\HandPal---Gesture-tracking-AI-click-function\\HandPal---Gesture-tracking-AI-click-function\\cursor.cur"


# Set of landmark indices that are drawn larger
LARGE_POINTS = {4, 8, 12, 16, 20}

mouse = Controller()

def set_custom_cursor():
    hwnd = win32gui.GetForegroundWindow()
    hcursor = win32gui.LoadImage(0, CURSOR_PATH, win32con.IMAGE_CURSOR, 0, 0, win32con.LR_LOADFROMFILE)
    win32gui.SetCursor(hcursor)
    win32gui.PostMessage(hwnd, win32con.WM_SETCURSOR, hwnd, win32con.HTCLIENT)





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
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

def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0]
    # Convert landmarks to relative coordinates and flatten the list
    relative_landmarks = [(x - base_x, y - base_y) for x, y in landmark_list]
    flat_list = list(itertools.chain.from_iterable(relative_landmarks))
    max_value = max(map(abs, flat_list)) or 1  # avoid division by zero
    return [n / max_value for n in flat_list]



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

def move_mouse_to(hand_landmarks, screen_width, screen_height):
    """Muove il mouse in base alla posizione dell'indice della mano destra."""
    normx = hand_landmarks.landmark[8].x
    normy = hand_landmarks.landmark[8].y
    mouse.position = (int(normx * screen_width), int(normy * screen_height))


def check_thumb_index_click(hand_landmarks):
    """Simula un click sinistro se il pollice e l'indice della mano sinistra si toccano."""
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    # Calcolo della distanza euclidea normalizzata
    distance = np.linalg.norm(np.array([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y]))

    return distance < 0.05  # Soglia per il click

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
            # Return a copy of the frame to avoid threading issues
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()
        
def change_cursor_on_index_pointing(hand_landmarks):
    """Cambia il cursore se l'indice punta verso lo schermo."""
    direction = detect_pointing_direction(hand_landmarks)
    if direction == "Pointing forward":
        # Cambia il colore e aumenta la dimensione del cursore
        set_custom_cursor()  # Puoi scegliere un cursore diverso o personalizzato
        # Puoi anche usare un cursore personalizzato se lo desideri, ad esempio:
        # set_cursor(win32con.IDC_HAND)
        # Per cambiare la dimensione, dipende dal sistema operativo
        # Alcuni sistemi permettono di usare "larger" cursori (ad esempio con `win32api`)
    else:
        # Ripristina il cursore normale
       set_custom_cursor() 

def main():
    args = get_args()

    # Inizializzazione della webcam
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # Inizializzazione MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    # Ottenere le dimensioni dello schermo
    root = tk.Tk()
    root.withdraw()
    screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()

    prev_cursor_set = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        debug_image = frame.copy()

        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        hand_detected = False  # Flag per controllare il cambio del cursore

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # "Left" o "Right"

                if label == "Right":
                    move_mouse_to(hand_landmarks, screen_width, screen_height)
                    hand_detected = True 
                    set_custom_cursor() 
                    prev_cursor_set = True
                    

                elif label == "Left":
                    if check_thumb_index_click(hand_landmarks):
                        mouse.click(Button.left)
                        cv.putText(debug_image, "Click!", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Disegna i punti della mano
                for idx, (x, y) in enumerate(calc_landmark_list(debug_image, hand_landmarks)):
                    radius = 8 if idx in LARGE_POINTS else 5
                    cv.circle(debug_image, (x, y), radius, (255, 255, 255), -1)
                    cv.circle(debug_image, (x, y), radius, (0, 0, 0), 1)

        # Cambio del cursore
        if hand_detected:
            if not prev_cursor_set:
                set_custom_cursor() 
                prev_cursor_set = True
        else:
            if prev_cursor_set:
                set_custom_cursor() 
                prev_cursor_set = False

        cv.imshow("Hand Tracking", debug_image)

        if cv.waitKey(1) & 0xFF == 27:  # ESC per uscire
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
