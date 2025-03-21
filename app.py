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
from tkinter import Frame, Button as TkButton, Label, Toplevel
from PIL import Image, ImageTk

from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier

# Set of landmark indices that are drawn larger
LARGE_POINTS = {4, 8, 12, 16, 20}

# Classe per il menu sempre visibile
class FloatingMenu:
    def __init__(self, root=None):
        if root is None:
            self.root = tk.Tk()
            self.root.withdraw()  # Nascondi la finestra principale
        else:
            self.root = root
            
        self.window = Toplevel(self.root)
        self.window.title("Gesture Menu")
        self.window.attributes("-topmost", True)  # Sempre in primo piano
        self.window.overrideredirect(True)  # Rimuovi la barra del titolo
        self.window.geometry("220x320+50+50")  # Dimensioni e posizione
        self.window.configure(bg='#333333')
        
        # Crea gli elementi del menu
        self.create_menu_elements()
        
        # Nascondi il menu all'inizio
        self.hide()
        
        # Stato del menu
        self.visible = False
        
        # Per spostare la finestra
        self.window.bind("<ButtonPress-1>", self.start_move)
        self.window.bind("<ButtonRelease-1>", self.stop_move)
        self.window.bind("<B1-Motion>", self.do_move)
        
    def create_menu_elements(self):
        # Titolo
        title_frame = Frame(self.window, bg='#333333')
        title_frame.pack(pady=10, fill=tk.X)
        
        title_label = Label(title_frame, text="GESTURE MENU", font=("Arial", 12, "bold"), 
                           bg='#333333', fg='white')
        title_label.pack()
        
        # Pulsanti
        self.buttons = []
        button_configs = [
            {"text": "Calculator", "bg": "#0078D7", "command": open_calculator},
            {"text": "Browser", "bg": "#00C800", "command": open_browser},
            {"text": "Notepad", "bg": "#FF6400", "command": open_notepad},
            {"text": "Music Player", "bg": "#B400B4", "command": open_music_player},
            {"text": "Close Menu", "bg": "#D72000", "command": self.hide}
        ]
        
        for config in button_configs:
            button_frame = Frame(self.window, bg='#333333')
            button_frame.pack(pady=5, padx=10, fill=tk.X)
            
            button = TkButton(button_frame, text=config["text"], bg=config["bg"], fg="white",
                           font=("Arial", 10), width=15, height=2, command=config["command"])
            button.pack(fill=tk.X)
            self.buttons.append(button)
    
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
    # Lower resolution for faster processing
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
    return image, center, radius

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
        # Toggle il menu flottante invece di quello OpenCV
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

def perform_gesture_detection(image, circle_data, hands, keypoint_classifier, 
                            point_history_classifier, point_history, finger_gesture_history,
                            keypoint_classifier_labels, point_history_classifier_labels,
                            history_length, floating_menu):
    """
    Esegui il rilevamento dei gesti e il controllo del mouse in una funzione separata.
    Questa funzione può essere chiamata sia nell'app principale che in modalità background.
    """
    # Estrai i dati del cerchio se disponibili
    circle_center, circle_radius = circle_data if circle_data else (None, None)
    
    # Inizializza il mouse controller
    mouse = Controller()
    
    # Converti l'immagine in RGB per MediaPipe
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    # Processa l'immagine con MediaPipe Hands
    results = hands.process(image_rgb)
    
    # Se non ci sono mani rilevate, svuota la storia dei punti
    if results.multi_hand_landmarks is None:
        point_history.appendleft([0, 0])
        return image, mouse.position, False
    
    # Variabili per il controllo del mouse
    is_clicking = False
    mouse_position = mouse.position
    
    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        # Punti di riferimento della mano
        landmark_list = calc_landmark_list(image, hand_landmarks)
        
        # Normalizzazione
        pre_processed_landmark_list = pre_process_landmark(landmark_list)
        pre_processed_point_history_list = pre_process_point_history(image, point_history)
        
        # Classificazione del gesto della mano
        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
        if hand_sign_id == 2:  # Indice esteso
            point_history.appendleft(landmark_list[8])  # Punta dell'indice
        else:
            point_history.appendleft([0, 0])
        
        # Classificazione del gesto delle dita
        finger_gesture_id = 0
        point_history_len = len(pre_processed_point_history_list)
        if point_history_len == (history_length * 2):
            finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
        finger_gesture_history.append(finger_gesture_id)
        most_common_fg_id = Counter(finger_gesture_history).most_common(1)[0][0]
        
        # CONTROLLO DEL MOUSE E INTERAZIONE MENU
        # Controlla se l'utente sta cliccando
        is_clicking = check_thumb_index_click(hand_landmarks)
        
        # Muovi il mouse con l'indice
        move_mouse_to(hand_landmarks, 1920, 1080, mouse)  # Usa risoluzione dello schermo
        mouse_position = mouse.position
        
        # Esegui click se è stato rilevato
        if is_clicking:
            mouse.press(Button.left)
            mouse.release(Button.left)
        
        # Controlla se l'indice sta puntando verso il cerchio (per attivare il menu)
        if circle_center and circle_radius and is_mouse_inside_circle(mouse_position, circle_center, circle_radius):
            floating_menu.show()
        
        # Disegna i punti della mano e le informazioni nel debug_image
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

# Classe per eseguire il riconoscimento gestuale in background
class GestureRecognitionThread(threading.Thread):
    def __init__(self, args):
        threading.Thread.__init__(self, daemon=True)
        self.args = args
        self.stopped = False
        
        # Crea il menu flottante
        self.root = tk.Tk()
        self.root.withdraw()  # Nascondi la finestra principale
        self.floating_menu = FloatingMenu(self.root)
        
        # Inizializza la webcam
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
        
        # Carica i classificatori
        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()
        
        # Leggi le etichette dei classificatori
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
        with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            self.point_history_classifier_labels = [row[0] for row in csv.reader(f)]
        
        # Storia dei punti
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)
        
        # Calcolatore FPS
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)
        
        # Stato del mouse
        self.mouse = Controller()
        self.mouse_position = (0, 0)
        self.is_clicking = False
        
        # Crea un mini-display per mostrare l'input della videocamera
        self.mini_display = Toplevel(self.root)
        self.mini_display.title("Camera Preview")
        self.mini_display.geometry("320x240+1600+50")  # Posiziona in alto a destra
        self.mini_display.attributes("-topmost", True)  # Sempre in primo piano
        self.mini_display.protocol("WM_DELETE_WINDOW", self.on_closing)  # Gestisci la chiusura
        
        # Crea un'etichetta per mostrare il feed della videocamera
        self.camera_label = Label(self.mini_display)
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Aggiungi un pulsante per mostrare/nascondere il menu
        self.menu_button = TkButton(self.mini_display, text="Menu On/Off", 
                                  bg="#555555", fg="white", command=self.floating_menu.toggle)
        self.menu_button.pack(fill=tk.X, pady=5)
        
        # Aggiungi un pulsante per uscire
        self.exit_button = TkButton(self.mini_display, text="Exit", 
                                   bg="#FF0000", fg="white", command=self.stop)
        self.exit_button.pack(fill=tk.X)
        
    def update_mini_display(self, image):
        """Aggiorna il mini-display con l'immagine corrente"""
        # Ridimensiona l'immagine per il mini-display
        small_image = cv.resize(image, (320, 240))
        # Converti da BGR a RGB
        small_image = cv.cvtColor(small_image, cv.COLOR_BGR2RGB)
        # Converti in formato per tkinter
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(small_image))
        # Aggiorna l'etichetta
        self.camera_label.config(image=img_tk)
        self.camera_label.image = img_tk  # Mantieni un riferimento
    
    def run(self):
        """Loop principale del thread di riconoscimento gestuale"""
        while not self.stopped:
            # Aggiorna tkinter UI
            self.root.update()
            
            # Leggi il frame dalla videocamera
            ret, image = self.video_stream.read()
            if not ret:
                continue
                
            # Ribalta l'immagine orizzontalmente
            image = cv.flip(image, 1)
            debug_image = image.copy()
            
            # Disegna un cerchio sulla destra dell'immagine
            debug_image, circle_center, circle_radius = draw_circle_on_right(debug_image)
            circle_data = (circle_center, circle_radius)
            
            # Esegui il rilevamento dei gesti
            debug_image, mouse_pos, is_click = perform_gesture_detection(
                debug_image, circle_data, self.hands, self.keypoint_classifier, 
                self.point_history_classifier, self.point_history, 
                self.finger_gesture_history, self.keypoint_classifier_labels,
                self.point_history_classifier_labels, self.history_length, 
                self.floating_menu
            )
            
            # Aggiorna le informazioni FPS
            fps = self.cvFpsCalc.get()
            debug_image = draw_info(debug_image, fps, 0, -1)
            
            # Aggiorna il mini-display
            self.update_mini_display(debug_image)
            
            # Pausa breve per ridurre l'utilizzo della CPU
            time.sleep(0.01)
        
        # Pulisci quando fermato
        self.cleanup()
    
    def stop(self):
        """Ferma il thread di riconoscimento e pulisci le risorse"""
        self.stopped = True
        
    def on_closing(self):
        """Gestisci la chiusura della finestra del mini-display"""
        self.stop()
        
    def cleanup(self):
        """Pulisci le risorse"""
        self.video_stream.stop()
        self.mini_display.destroy()
        self.root.destroy()
        # Esci dall'applicazione
        os._exit(0)

def main():
    # Analizza gli argomenti da riga di comando
    args = get_args()
    
    # Se richiesto di eseguire in background
    if args.run_in_background:
        # Avvia il thread di riconoscimento gestuale in background
        gesture_thread = GestureRecognitionThread(args)
        gesture_thread.start()
        # Usa tkinter come loop principale
        gesture_thread.root.mainloop()
        return
        
    # Inizializza la webcam con thread
    video_stream = VideoStream(
        src=args.device,
        width=args.width,
        height=args.height,
        fps=30
    )
    
    # Crea il menu flottante
    root = tk.Tk()
    root.withdraw()  # Nascondi la finestra principale
    floating_menu = FloatingMenu(root)
    
    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    # Carica i classificatori
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Leggi le etichette dei classificatori
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    # Inizializza le variabili di stato
    mode = 0
    number = -1

    # Storia dei punti
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    # Calcolatore FPS
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    
    # Controller del mouse
    mouse = Controller()

    while True:
        # Aggiorna il menu flottante
        root.update()
        
        fps = cvFpsCalc.get()
        ret, image = video_stream.read()
        if not ret:
            break

        # Ribalta l'immagine orizzontalmente per creare un effetto "specchio"
        image = cv.flip(image, 1)
        debug_image = image.copy()

        # Disegna un cerchio sulla destra dell'immagine
        debug_image, circle_center, circle_radius = draw_circle_on_right(debug_image)
        circle_data = (circle_center, circle_radius)
        
        # Esegui il rilevamento dei gesti
        debug_image, mouse_position, is_clicking = perform_gesture_detection(
            debug_image, circle_data, hands, keypoint_classifier, 
            point_history_classifier, point_history, finger_gesture_history,
            keypoint_classifier_labels, point_history_classifier_labels,
            history_length, floating_menu
        )

        # Disegna le informazioni su FPS e modalità
        debug_image = draw_info(debug_image, fps, mode, number)
        
        # Istruzioni per l'utente
        cv.putText(debug_image, "Press 'M' to toggle menu, 'Q' to exit", (10, 150),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(debug_image, "Press 'B' to switch to background mode", (10, 180),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        # Mostra l'immagine
        cv.imshow('Hand Gesture Recognition', debug_image)

        # Gestisci l'input da tastiera
        key = cv.waitKey(1)
        if key == 27 or key == ord('q'):  # ESC o 'q' per uscire
            break
        if key == ord('b'):  # 'b' per passare alla modalità background
            # Passa alla modalità background
            video_stream.stop()
            cv.destroyAllWindows()
            
            # Avvia il thread di riconoscimento gestuale in background
            args.run_in_background = True
            gesture_thread = GestureRecognitionThread(args)
            gesture_thread.start()
            # Usa tkinter come loop principale
            gesture_thread.root.mainloop()
            return
            
        number, mode = select_mode(key, mode, False, floating_menu)

    # Pulisci
    video_stream.stop()
    cv.destroyAllWindows()
    root.destroy()


if __name__ == '__main__':
    main()