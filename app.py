import argparse
import threading
import os
import sys
import time
import numpy as np
import cv2 as cv
import mediapipe as mp
import tkinter as tk
from pynput.mouse import Controller, Button
from collections import deque
import json
import logging
import ctypes  # For changing the Windows system cursor

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("handpal.log"), logging.StreamHandler()]
)
logger = logging.getLogger("HandPal")

# -----------------------------------------------------------------------------
# Windows Cursor Functions
# -----------------------------------------------------------------------------
def set_custom_cursor(cursor_path):
    """
    Cambia il cursore di Windows al cursore personalizzato specificato.
    Attenzione: Questa operazione impatta il sistema a livello globale.
    """
    try:
        user32 = ctypes.windll.user32
        # Carica il cursore personalizzato da file (assicurarsi che sia un file .cur)
        hCursor = user32.LoadCursorFromFileW(cursor_path)
        if hCursor == 0:
            raise Exception("Impossibile caricare il cursore personalizzato.")
        # OCR_NORMAL è l'ID del cursore di default (freccia)
        OCR_NORMAL = 32512
        if not user32.SetSystemCursor(hCursor, OCR_NORMAL):
            raise Exception("Impossibile impostare il cursore di sistema.")
        print("Cursore personalizzato impostato con successo.")
    except Exception as e:
        print(f"Errore nell'impostazione del cursore personalizzato: {e}")

def restore_default_cursor():
    """
    Ripristina i cursori di sistema predefiniti.
    """
    try:
        user32 = ctypes.windll.user32
        SPI_SETCURSORS = 0x57
        # Il flag 3 (SPIF_UPDATEINIFILE | SPIF_SENDCHANGE) aggiorna e notifica il sistema
        if not user32.SystemParametersInfoW(SPI_SETCURSORS, 0, None, 3):
            raise Exception("Impossibile ripristinare i cursori predefiniti.")
        print("Cursori predefiniti ripristinati.")
    except Exception as e:
        print(f"Errore nel ripristino dei cursori predefiniti: {e}")

# -----------------------------------------------------------------------------
# Config Class
# -----------------------------------------------------------------------------
class Config:
    """Classe per gestire la configurazione dell'applicazione."""
    DEFAULT_CONFIG = {
        "device": 0,
        "width": 1980, # Webcam capture width (can be high)
        "height": 1440, # Webcam capture height (can be high)
        "process_width": 640, # Lower resolution for processing
        "process_height": 360, # Lower resolution for processing
        "min_detection_confidence": 0.6,
        "min_tracking_confidence": 0.5,
        "use_static_image_mode": False,
        "smoothing_factor": 0.7,
        "click_cooldown": 0.5,
        "max_fps": 30,
        "gesture_sensitivity": 0.08,
        "inactivity_zone": 0.03,
        "flip_camera": True,
        "gesture_settings": {
            "scroll_sensitivity": 8,
            "drag_threshold": 0.15, # Placeholder, not currently used
            "double_click_time": 0.4
        },
        "calibration": {
            "enabled": True,
            "screen_margin": 0.15,
            "x_min": 0.2,
            "x_max": 0.8,
            "y_min": 0.2,
            "y_max": 0.8,
            "active": False # Internal state, not saved/loaded typically
        }
    }

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __init__(self, args=None):
        """Inizializza la configurazione combinando valori predefiniti e argomenti CLI."""
        self.config = self.DEFAULT_CONFIG.copy()
        config_path = os.path.expanduser("~/.handpal_config.json")

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                    # Deep update for nested dictionaries like 'calibration' and 'gesture_settings'
                    for key, value in saved_config.items():
                        if isinstance(value, dict) and key in self.config and isinstance(self.config[key], dict):
                            self.config[key].update(value)
                        else:
                            self.config[key] = value

                    # Ensure calibration values are valid after loading
                    calib = self.config["calibration"]
                    if abs(calib["x_max"] - calib["x_min"]) < 0.1:
                        logger.warning("Valori di calibrazione X troppo vicini, ripristino valori di default")
                        calib["x_min"] = self.DEFAULT_CONFIG["calibration"]["x_min"]
                        calib["x_max"] = self.DEFAULT_CONFIG["calibration"]["x_max"]

                    if abs(calib["y_max"] - calib["y_min"]) < 0.1:
                        logger.warning("Valori di calibrazione Y troppo vicini, ripristino valori di default")
                        calib["y_min"] = self.DEFAULT_CONFIG["calibration"]["y_min"]
                        calib["y_max"] = self.DEFAULT_CONFIG["calibration"]["y_max"]

                    logger.info("Configurazione caricata dal file")
            except Exception as e:
                logger.error(f"Errore nel caricamento della configurazione: {e}, usando default.")
                self.config = self.DEFAULT_CONFIG.copy() # Fallback to default on error

        # Override with CLI arguments if provided
        if args:
            for key, value in vars(args).items():
                # Handle nested args if needed in the future, for now only top-level
                if key in self.config and value is not None:
                    # Special handling for boolean flags like flip_camera
                    if isinstance(self.config[key], bool) and isinstance(value, bool):
                         self.config[key] = value
                    elif not isinstance(self.config[key], dict): # Avoid overwriting dicts directly
                        self.config[key] = value

        # Ensure calibration active state is false initially
        self.config["calibration"]["active"] = False

    def save(self):
        """Salva la configurazione in un file."""
        config_path = os.path.expanduser("~/.handpal_config.json")
        try:
            # Create a copy to save, excluding internal state like 'active'
            config_to_save = self.config.copy()
            if "calibration" in config_to_save and "active" in config_to_save["calibration"]:
                 del config_to_save["calibration"]["active"] # Don't save the active state

            with open(config_path, 'w') as f:
                json.dump(config_to_save, f, indent=2)
                logger.info("Configurazione salvata")
        except Exception as e:
            logger.error(f"Errore nel salvataggio della configurazione: {e}")

    def __getitem__(self, key):
        # Allow nested access like config['calibration']['x_min']
        if '.' in key:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        return self.config[key]

    def __setitem__(self, key, value):
         # Allow nested access like config['calibration.x_min'] = 0.1
        if '.' in key:
            keys = key.split('.')
            d = self.config
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value
        else:
            self.config[key] = value

# -----------------------------------------------------------------------------
# GestureRecognizer Class
# -----------------------------------------------------------------------------
class GestureRecognizer:
    """Classe specializzata per il riconoscimento dei gesti."""

    def __init__(self, config):
        self.config = config
        self.last_positions = {} # Store last known position for landmarks (e.g., index tip for scroll)
        self.gesture_state = {
            "drag_active": False, # Placeholder for future drag gesture
            "scroll_active": False,
            "last_click_time": 0,
            "last_click_position": (0, 0), # Store position for double click check
            "scroll_history": deque(maxlen=5), # History for scroll smoothing
            "last_gesture_name": None, # Last stable gesture detected
            "gesture_stable_count": 0,
            "active_gesture": None, # Currently performed action (click, scroll)
            "last_pose": "unknown", # Last detected hand pose
            "pose_stable_count": 0
        }
        self.POSE_STABILITY_THRESHOLD = 3 # Number of frames for a pose to be considered stable

    def _calculate_distance(self, p1, p2):
        """Calculates 2D Euclidean distance between two landmarks."""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def check_thumb_index_click(self, hand_landmarks, handedness):
        """Rileva il gesto di click (indice e pollice che si toccano), solo per la mano sinistra."""
        if handedness != "Left":
            return None
        # Ignore click if scrolling is active to prevent conflicts
        if self.gesture_state["scroll_active"]:
            return None

        thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP] # 4
        index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP] # 8

        distance = self._calculate_distance(thumb_tip, index_tip)
        current_time = time.time()
        click_detected = distance < self.config["gesture_sensitivity"]

        if click_detected:
            # Check cooldown first
            if (current_time - self.gesture_state["last_click_time"]) > self.config["click_cooldown"]:
                # Check for double click (time and position proximity)
                time_diff = current_time - self.gesture_state["last_click_time"]
                if time_diff < self.config["gesture_settings"]["double_click_time"]:
                    self.gesture_state["last_click_time"] = current_time
                    self.gesture_state["active_gesture"] = "click" # Keep it simple, action handles double
                    logger.debug("Double click gesture detected")
                    return "double_click"
                else:
                    self.gesture_state["last_click_time"] = current_time
                    self.gesture_state["active_gesture"] = "click"
                    logger.debug("Single click gesture detected")
                    return "click"
        else:
            if self.gesture_state["active_gesture"] == "click":
                 self.gesture_state["active_gesture"] = None

        return None

    def check_scroll_gesture(self, hand_landmarks, handedness):
        """Rileva il gesto di scorrimento (indice e medio estesi verticalmente), solo per la mano sinistra."""
        if handedness != "Left":
            return None
        if self.gesture_state["active_gesture"] == "click":
            return None

        index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        index_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]

        index_extended = index_tip.y < index_mcp.y
        middle_extended = middle_tip.y < middle_mcp.y
        fingers_close = abs(index_tip.x - middle_tip.x) < 0.08
        ring_folded = ring_tip.y > index_mcp.y
        pinky_folded = pinky_tip.y > middle_mcp.y

        is_scroll_pose = index_extended and middle_extended and fingers_close and ring_folded and pinky_folded

        scroll_delta = None
        index_tip_id = 8

        if is_scroll_pose:
            if not self.gesture_state["scroll_active"]:
                self.gesture_state["scroll_active"] = True
                self.gesture_state["scroll_history"].clear()
                self.gesture_state["active_gesture"] = "scroll"
                self.last_positions[index_tip_id] = (index_tip.x, index_tip.y)
                logger.debug("Scroll gesture started")

            if index_tip_id in self.last_positions:
                prev_y = self.last_positions[index_tip_id][1]
                curr_y = index_tip.y
                delta_y = (curr_y - prev_y) * self.config["gesture_settings"]["scroll_sensitivity"] * 10

                if abs(delta_y) > 0.01:
                    self.gesture_state["scroll_history"].append(delta_y)

                if len(self.gesture_state["scroll_history"]) > 0:
                    smooth_delta = sum(self.gesture_state["scroll_history"]) / len(self.gesture_state["scroll_history"])
                    if abs(smooth_delta) > 0.05:
                         scroll_delta = smooth_delta
                         self.last_positions[index_tip_id] = (index_tip.x, index_tip.y)

            if self.gesture_state["scroll_active"]:
                 self.last_positions[index_tip_id] = (index_tip.x, index_tip.y)
        else:
            if self.gesture_state["scroll_active"]:
                self.gesture_state["scroll_active"] = False
                self.gesture_state["scroll_history"].clear()
                if index_tip_id in self.last_positions:
                    del self.last_positions[index_tip_id]
                if self.gesture_state["active_gesture"] == "scroll":
                    self.gesture_state["active_gesture"] = None
                logger.debug("Scroll gesture ended")

        return scroll_delta

    def detect_hand_pose(self, hand_landmarks, handedness):
        """Analizza la posa complessiva della mano. Returns a string like 'pointing', 'open_palm', etc."""
        if handedness == "Right":
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
            if index_tip.y < index_mcp.y:
                return "pointing"
            else:
                return "unknown"

        if handedness == "Left":
            landmarks = hand_landmarks.landmark
            wrist = landmarks[mp.solutions.hands.HandLandmark.WRIST]
            thumb_tip = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = landmarks[mp.solutions.hands.HandLandmark.PINKY_TIP]

            thumb_mcp = landmarks[mp.solutions.hands.HandLandmark.THUMB_MCP]
            index_mcp = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
            middle_mcp = landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
            ring_mcp = landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_MCP]
            pinky_mcp = landmarks[mp.solutions.hands.HandLandmark.PINKY_MCP]

            thumb_extended = abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x)
            index_extended = index_tip.y < index_mcp.y
            middle_extended = middle_tip.y < middle_mcp.y
            ring_extended = ring_tip.y < ring_mcp.y
            pinky_extended = pinky_tip.y < pinky_mcp.y

            current_pose = "unknown"
            num_extended = sum([index_extended, middle_extended, ring_extended, pinky_extended])

            if num_extended == 0 and not thumb_extended:
                 current_pose = "closed_fist"
            elif index_extended and num_extended == 1:
                 current_pose = "pointing"
            elif index_extended and middle_extended and num_extended == 2:
                 current_pose = "two_fingers"
            elif num_extended >= 4:
                 current_pose = "open_palm"

            if current_pose == self.gesture_state["last_pose"]:
                self.gesture_state["pose_stable_count"] += 1
            else:
                self.gesture_state["last_pose"] = current_pose
                self.gesture_state["pose_stable_count"] = 0

            if self.gesture_state["pose_stable_count"] >= self.POSE_STABILITY_THRESHOLD:
                return current_pose
            else:
                return current_pose

        return "unknown"

# -----------------------------------------------------------------------------
# MotionSmoother Class
# -----------------------------------------------------------------------------
class MotionSmoother:
    """Classe per smoothing del movimento del mouse con filtro di media mobile."""

    def __init__(self, config, history_size=5):
        self.config = config
        self.history_size = history_size
        self.position_history = deque(maxlen=history_size)
        self.last_smoothed_position = None
        self.inactivity_zone_size = config["inactivity_zone"]
        self.smoothing_factor = config["smoothing_factor"]

    def update(self, target_x, target_y):
        """Aggiorna la cronologia delle posizioni e calcola la nuova posizione smussata."""
        current_target = (target_x, target_y)

        if not self.position_history:
            self.position_history.append(current_target)
            self.last_smoothed_position = current_target
            return current_target

        prev_target = self.position_history[-1]
        screen_w = self.config.get("screen_width", 1920)
        screen_h = self.config.get("screen_height", 1080)
        dx = abs(target_x - prev_target[0]) / screen_w
        dy = abs(target_y - prev_target[1]) / screen_h

        if dx < self.inactivity_zone_size and dy < self.inactivity_zone_size and self.last_smoothed_position:
             return self.last_smoothed_position

        self.position_history.append(current_target)

        alpha = self.smoothing_factor
        if self.last_smoothed_position:
            smooth_x = int(alpha * target_x + (1 - alpha) * self.last_smoothed_position[0])
            smooth_y = int(alpha * target_y + (1 - alpha) * self.last_smoothed_position[1])
        else:
            smooth_x, smooth_y = target_x, target_y

        self.last_smoothed_position = (smooth_x, smooth_y)
        return smooth_x, smooth_y

    def reset(self):
        """Resets the smoother's history."""
        self.position_history.clear()
        self.last_smoothed_position = None

# -----------------------------------------------------------------------------
# HandPal Class
# -----------------------------------------------------------------------------
class HandPal:
    """Classe principale che gestisce l'applicazione HandPal."""

    def __init__(self, config):
        self.config = config
        self.mouse = Controller()
        self.gesture_recognizer = GestureRecognizer(config)
        try:
            root = tk.Tk()
            root.withdraw()
            self.screen_size = (root.winfo_screenwidth(), root.winfo_screenheight())
            root.destroy()
        except tk.TclError:
             logger.warning("Could not initialize Tkinter to get screen size. Using defaults (1920x1080).")
             self.screen_size = (1920, 1080)

        self.config["screen_width"] = self.screen_size[0]
        self.config["screen_height"] = self.screen_size[1]

        self.motion_smoother = MotionSmoother(config)
        self.running = False
        self.cap = None
        self.mp_hands = mp.solutions.hands
        self.hands = None
        self.last_cursor_pos = None
        self.last_right_hand_pos = None
        self.fps_stats = deque(maxlen=30)

        self.calibration_active = False
        self.calibration_points = []
        self.calibration_step = 0
        self.calibration_corners = [
            "in alto a sinistra dello SCHERMO",
            "in alto a destra dello SCHERMO",
            "in basso a destra dello SCHERMO",
            "in basso a sinistra dello SCHERMO"
        ]

        self.debug_mode = False
        self.tracking_enabled = True

        self.debug_values = {
            "fps": 0.0,
            "last_action_time": time.time(),
            "cursor_history": deque(maxlen=20),
            "screen_mapping": {"raw": (0, 0), "mapped": (0, 0), "smoothed": (0,0)},
            "left_hand_pose": "unknown",
            "right_hand_pose": "unknown",
            "active_gesture": "None"
        }

    def start(self):
        """Avvia l'applicazione."""
        logger.info("Avvio HandPal...")
        try:
            self.cap = cv.VideoCapture(self.config["device"])
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.config["width"])
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.config["height"])
            actual_width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
            logger.info(f"Webcam richiesta {self.config['width']}x{self.config['height']}, ottenuta {actual_width}x{actual_height}")

            if not self.cap.isOpened():
                logger.error(f"Impossibile aprire la webcam (dispositivo {self.config['device']}).")
                return False

            self.hands = self.mp_hands.Hands(
                static_image_mode=self.config["use_static_image_mode"],
                max_num_hands=2,
                min_detection_confidence=self.config["min_detection_confidence"],
                min_tracking_confidence=self.config["min_tracking_confidence"]
            )

            self.running = True
            self.thread = threading.Thread(target=self.main_loop, daemon=True)
            self.thread.start()
            logger.info("HandPal avviato con successo!")
            return True

        except Exception as e:
            logger.exception(f"Errore durante l'avvio: {e}")
            self.stop()
            return False

    def stop(self):
        """Ferma l'applicazione."""
        if not self.running:
            return
        logger.info("Fermando HandPal...")
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
             self.thread.join(timeout=0.5)

        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.debug("Webcam rilasciata.")
        if self.hands is not None:
            self.hands.close()
            self.hands = None
            logger.debug("MediaPipe Hands chiuso.")
        cv.destroyAllWindows()
        logger.info("HandPal fermato.")

    def map_to_screen(self, x, y):
        """Mappa le coordinate normalizzate della mano (0-1) allo schermo."""
        try:
            if self.debug_mode:
                self.debug_values["screen_mapping"]["raw"] = (x, y)

            x_calib = x
            y_calib = y

            if self.config["calibration"]["enabled"]:
                x_min = self.config["calibration"]["x_min"]
                x_max = self.config["calibration"]["x_max"]
                y_min = self.config["calibration"]["y_min"]
                y_max = self.config["calibration"]["y_max"]

                x_range = x_max - x_min
                y_range = y_max - y_min

                if x_range < 0.01 or y_range < 0.01:
                     if self.debug_mode: logger.warning("Calibration range too small, ignoring.")
                else:
                    x_calib = max(0.0, min(1.0, (x - x_min) / x_range))
                    y_calib = max(0.0, min(1.0, (y - y_min) / y_range))

            margin = self.config["calibration"]["screen_margin"]
            x_expanded = x_calib * (1.0 + 2.0 * margin) - margin
            y_expanded = y_calib * (1.0 + 2.0 * margin) - margin

            screen_x = int(x_expanded * self.screen_size[0])
            screen_y = int(y_expanded * self.screen_size[1])

            screen_x = max(0, min(screen_x, self.screen_size[0] - 1))
            screen_y = max(0, min(screen_y, self.screen_size[1] - 1))

            if self.debug_mode:
                self.debug_values["screen_mapping"]["mapped"] = (screen_x, screen_y)

            return screen_x, screen_y

        except Exception as e:
            logger.error(f"Errore durante la mappatura coordinate: {e}")
            screen_x = int(x * self.screen_size[0])
            screen_y = int(y * self.screen_size[1])
            return max(0, min(screen_x, self.screen_size[0] - 1)), max(0, min(screen_y, self.screen_size[1] - 1))

    def process_frame(self, frame):
        """Processa un singolo frame della webcam, ottimizzato per due mani."""
        process_start_time = time.perf_counter()
        process_frame = cv.resize(frame, (self.config["process_width"], self.config["process_height"]))
        rgb_frame = cv.cvtColor(process_frame, cv.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        results = self.hands.process(rgb_frame)

        rgb_frame.flags.writeable = True
        display_frame = process_frame.copy()
        frame_height, frame_width = display_frame.shape[:2]

        left_hand_landmarks = None
        right_hand_landmarks = None
        left_handedness = None
        right_handedness = None

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness_enum = results.multi_handedness[idx].classification[0]
                handedness_label = handedness_enum.label
                if handedness_label == "Right":
                    right_hand_landmarks = hand_landmarks
                    right_handedness = handedness_label
                elif handedness_label == "Left":
                    left_hand_landmarks = hand_landmarks
                    left_handedness = handedness_label

            self.draw_landmarks(display_frame, results.multi_hand_landmarks)

        self.last_right_hand_pos = None
        if right_hand_landmarks:
            index_tip = right_hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            right_hand_pos = (index_tip.x, index_tip.y)
            self.last_right_hand_pos = right_hand_pos

            if self.calibration_active:
                tip_x_disp = int(right_hand_pos[0] * frame_width)
                tip_y_disp = int(right_hand_pos[1] * frame_height)
                cv.circle(display_frame, (tip_x_disp, tip_y_disp), 10, (0, 255, 0), -1)
            elif self.tracking_enabled:
                try:
                    screen_x, screen_y = self.map_to_screen(right_hand_pos[0], right_hand_pos[1])
                    smooth_x, smooth_y = self.motion_smoother.update(screen_x, screen_y)
                    if self.last_cursor_pos != (smooth_x, smooth_y):
                        self.mouse.position = (smooth_x, smooth_y)
                        self.last_cursor_pos = (smooth_x, smooth_y)
                        if self.debug_mode:
                            self.debug_values["screen_mapping"]["smoothed"] = (smooth_x, smooth_y)
                            self.debug_values["cursor_history"].append((smooth_x, smooth_y))
                except Exception as e:
                    logger.error(f"Errore nell'aggiornamento del cursore: {e}")
                    self.motion_smoother.reset()

            if self.debug_mode:
                 self.debug_values["right_hand_pose"] = self.gesture_recognizer.detect_hand_pose(right_hand_landmarks, "Right")
        else:
            self.motion_smoother.reset()

        if left_hand_landmarks and not self.calibration_active:
            scroll_amount = self.gesture_recognizer.check_scroll_gesture(left_hand_landmarks, "Left")

            if scroll_amount is not None:
                scroll_clicks = int(scroll_amount * -1)
                if scroll_clicks != 0:
                    self.mouse.scroll(0, scroll_clicks)
                    self.debug_values["last_action_time"] = time.time()
                    self.debug_values["active_gesture"] = f"Scroll ({scroll_clicks})"
            else:
                click_gesture = self.gesture_recognizer.check_thumb_index_click(left_hand_landmarks, "Left")
                if click_gesture == "click":
                    self.mouse.click(Button.left)
                    self.debug_values["last_action_time"] = time.time()
                    self.debug_values["active_gesture"] = "Click"
                    logger.info("Click sinistro")
                elif click_gesture == "double_click":
                    self.mouse.click(Button.left, 2)
                    self.debug_values["last_action_time"] = time.time()
                    self.debug_values["active_gesture"] = "Double Click"
                    logger.info("Doppio click")

            if self.debug_mode:
                 self.debug_values["left_hand_pose"] = self.gesture_recognizer.detect_hand_pose(left_hand_landmarks, "Left")
                 if scroll_amount is None and click_gesture is None:
                     current_active = self.gesture_recognizer.gesture_state["active_gesture"]
                     self.debug_values["active_gesture"] = str(current_active) if current_active else "None"

        self.draw_overlays(display_frame)

        process_end_time = time.perf_counter()
        frame_proc_time = process_end_time - process_start_time
        self.fps_stats.append(frame_proc_time)

        return display_frame

    def draw_landmarks(self, frame, multi_hand_landmarks):
        """Disegna i landmark delle mani sul frame."""
        if not multi_hand_landmarks:
            return
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        for hand_landmarks in multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    def draw_overlays(self, frame):
        """Disegna testi informativi, debug e istruzioni di calibrazione."""
        h, w = frame.shape[:2]
        overlay_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale_small = 0.4
        font_scale_medium = 0.5
        line_type = 1

        if self.calibration_active:
            overlay = frame.copy()
            cv.rectangle(overlay, (0, 0), (w, 100), bg_color, -1)
            cv.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            cv.putText(frame, f"CALIBRAZIONE ({self.calibration_step+1}/4): Muovi mano DESTRA",
                       (10, 20), font, font_scale_medium, overlay_color, line_type)
            cv.putText(frame, f"{self.calibration_corners[self.calibration_step]}",
                       (10, 40), font, font_scale_medium, overlay_color, line_type)
            cv.putText(frame, "Premi SPAZIO per confermare",
                       (10, 60), font, font_scale_medium, overlay_color, line_type)
            cv.putText(frame, "(ESC per annullare)",
                       (10, 80), font, font_scale_medium, overlay_color, line_type)

            corner_radius = 10
            inactive_color = (180, 180, 180)
            active_color = (0, 0, 255)
            cv.circle(frame, (corner_radius, corner_radius), corner_radius, active_color if self.calibration_step == 0 else inactive_color, -1)
            cv.circle(frame, (w - corner_radius, corner_radius), corner_radius, active_color if self.calibration_step == 1 else inactive_color, -1)
            cv.circle(frame, (w - corner_radius, h - corner_radius), corner_radius, active_color if self.calibration_step == 2 else inactive_color, -1)
            cv.circle(frame, (corner_radius, h - corner_radius), corner_radius, active_color if self.calibration_step == 3 else inactive_color, -1)

        else:
            cv.putText(frame, "C:Calibra | Q:Esci | D:Debug", (10, h - 10),
                      font, font_scale_small, overlay_color, line_type)

            if self.fps_stats:
                avg_duration = sum(self.fps_stats) / len(self.fps_stats)
                fps = 1.0 / avg_duration if avg_duration > 0 else 0
                self.debug_values["fps"] = fps
                cv.putText(frame, f"FPS: {fps:.1f}", (w - 70, 20),
                          font, font_scale_medium, overlay_color, line_type)

            if self.debug_mode:
                debug_h = 150
                overlay = frame.copy()
                cv.rectangle(overlay, (0, h - debug_h), (w, h), bg_color, -1)
                cv.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                y_pos = h - debug_h + 15

                debug_text1 = f"L-Pose: {self.debug_values['left_hand_pose']} | R-Pose: {self.debug_values['right_hand_pose']} | Gest: {self.debug_values['active_gesture']}"
                cv.putText(frame, debug_text1, (10, y_pos), font, font_scale_small, overlay_color, line_type)
                y_pos += 20

                raw = self.debug_values['screen_mapping']['raw']
                mapped = self.debug_values['screen_mapping']['mapped']
                smoothed = self.debug_values['screen_mapping']['smoothed']
                debug_text2 = f"Raw:({raw[0]:.2f},{raw[1]:.2f}) Map:({mapped[0]},{mapped[1]}) Smooth:({smoothed[0]},{smoothed[1]})"
                cv.putText(frame, debug_text2, (10, y_pos), font, font_scale_small, overlay_color, line_type)
                y_pos += 20

                calib = self.config['calibration']
                debug_text3 = f"Calib: X[{calib['x_min']:.2f}-{calib['x_max']:.2f}] Y[{calib['y_min']:.2f}-{calib['y_max']:.2f}] En:{calib['enabled']}"
                cv.putText(frame, debug_text3, (10, y_pos), font, font_scale_small, overlay_color, line_type)
                y_pos += 20

                gest_sens = self.config['gesture_sensitivity']
                cooldown = self.config['click_cooldown']
                smooth_f = self.config['smoothing_factor']
                scroll_sens = self.config['gesture_settings']['scroll_sensitivity']
                debug_text4 = f"Sens:{gest_sens:.2f} Cool:{cooldown:.1f} Smooth:{smooth_f:.1f} Scroll:{scroll_sens}"
                cv.putText(frame, debug_text4, (10, y_pos), font, font_scale_small, overlay_color, line_type)
                y_pos += 20

                time_since_action = time.time() - self.debug_values["last_action_time"]
                debug_text5 = f"Last Action: {time_since_action:.1f}s ago"
                cv.putText(frame, debug_text5, (10, y_pos), font, font_scale_small, overlay_color, line_type)
                y_pos += 20

                if len(self.debug_values["cursor_history"]) > 1:
                    trace_color = (0, 255, 255)
                    for i in range(1, len(self.debug_values["cursor_history"])):
                        p1_screen = self.debug_values["cursor_history"][i-1]
                        p2_screen = self.debug_values["cursor_history"][i]
                        p1_frame = (int(p1_screen[0] * w / self.screen_size[0]), int(p1_screen[1] * h / self.screen_size[1]))
                        p2_frame = (int(p2_screen[0] * w / self.screen_size[0]), int(p2_screen[1] * h / self.screen_size[1]))
                        cv.line(frame, p1_frame, p2_frame, trace_color, 1)

    def main_loop(self):
        """Loop principale dell'applicazione."""
        frame_time_target = 1.0 / self.config["max_fps"]
        last_frame_time = time.perf_counter()

        while self.running:
            loop_start_time = time.perf_counter()

            if self.cap is None or not self.cap.isOpened():
                logger.error("Webcam non disponibile nel loop principale.")
                time.sleep(0.5)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("Frame non valido ricevuto dalla webcam.")
                time.sleep(0.1)
                continue

            if self.config["flip_camera"]:
                frame = cv.flip(frame, 1)

            try:
                processed_frame = self.process_frame(frame)
            except Exception as e:
                 logger.exception(f"Errore irreversibile durante process_frame: {e}")
                 processed_frame = frame
                 cv.putText(processed_frame, "ERROR PROCESSING FRAME", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv.imshow("HandPal", processed_frame)

            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                logger.info("Comando 'q' ricevuto, fermando...")
                self.stop()
            elif key == ord('c') and not self.calibration_active:
                self.start_calibration()
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                logger.info(f"Modalità debug: {'attivata' if self.debug_mode else 'disattivata'}")
            elif key == 27 and self.calibration_active:  # ESC
                logger.info("Calibrazione annullata dall'utente.")
                self.cancel_calibration()
            elif key == 32 and self.calibration_active:  # Spacebar
                self.process_calibration_step()

            loop_end_time = time.perf_counter()
            elapsed_time = loop_end_time - loop_start_time
            sleep_time = frame_time_target - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            last_frame_time = loop_end_time

        logger.debug("Main loop terminato.")
        cv.destroyAllWindows()

    def start_calibration(self):
        """Avvia il processo di calibrazione."""
        if self.calibration_active:
            logger.warning("Tentativo di avviare la calibrazione mentre è già attiva.")
            return

        self.calibration_active = True
        self.calibration_points = []
        self.calibration_step = 0
        self.tracking_enabled = False
        self.motion_smoother.reset()
        logger.info("Calibrazione avviata. Segui le istruzioni sullo schermo.")
        print("\n--- CALIBRAZIONE ---")
        print("Usa la mano DESTRA. Posiziona l'indice negli angoli dello SCHERMO.")
        print("Premi SPAZIO per confermare ogni angolo. Premi ESC per annullare.")
        print(f"Inizia con: {self.calibration_corners[0]}\n")

    def cancel_calibration(self):
        """Annulla la calibrazione in corso."""
        self.calibration_active = False
        self.calibration_points = []
        self.calibration_step = 0
        self.tracking_enabled = True
        logger.info("Calibrazione annullata.")
        print("\nCalibrazione annullata.\n")

    def process_calibration_step(self):
        """Processa un singolo passo della calibrazione (quando SPAZIO viene premuto)."""
        if not self.calibration_active:
            return

        if self.last_right_hand_pos is None:
            logger.warning("Impossibile registrare punto di calibrazione: mano destra non rilevata.")
            print("Mano destra non rilevata. Assicurati che sia visibile e riprova.")
            return

        current_pos = self.last_right_hand_pos
        self.calibration_points.append(current_pos)
        logger.info(f"Punto di calibrazione {self.calibration_step + 1}/4 registrato: ({current_pos[0]:.3f}, {current_pos[1]:.3f})")

        self.calibration_step += 1
        if self.calibration_step >= 4:
            self.complete_calibration()
        else:
            print(f"Punto {self.calibration_step} registrato. Ora posiziona la mano {self.calibration_corners[self.calibration_step]} e premi SPAZIO.")

    def complete_calibration(self):
        """Completa la calibrazione usando i punti registrati."""
        logger.info("Completamento calibrazione...")
        if len(self.calibration_points) != 4:
            logger.error(f"Calibrazione fallita: numero errato di punti ({len(self.calibration_points)}).")
            print("\nErrore: Numero errato di punti registrati. Calibrazione fallita.\n")
            self.cancel_calibration()
            return

        try:
            x_values = [p[0] for p in self.calibration_points]
            y_values = [p[1] for p in self.calibration_points]

            x_min_calib = min(x_values)
            x_max_calib = max(x_values)
            y_min_calib = min(y_values)
            y_max_calib = max(y_values)

            min_range = 0.05
            if (x_max_calib - x_min_calib < min_range) or (y_max_calib - y_min_calib < min_range):
                 logger.warning(f"Range di calibrazione troppo piccolo: X={x_max_calib-x_min_calib:.3f}, Y={y_max_calib-y_min_calib:.3f}. Calibrazione potrebbe essere imprecisa.")
                 print("Attenzione: L'area di calibrazione rilevata è molto piccola.")

            self.config["calibration"]["x_min"] = x_min_calib
            self.config["calibration"]["x_max"] = x_max_calib
            self.config["calibration"]["y_min"] = y_min_calib
            self.config["calibration"]["y_max"] = y_max_calib
            self.config["calibration"]["enabled"] = True

            self.config.save()

            logger.info(f"Calibrazione completata e salvata. Valori: X[{x_min_calib:.3f}-{x_max_calib:.3f}], Y[{y_min_calib:.3f}-{y_max_calib:.3f}]")
            print("\nCalibrazione completata e salvata con successo!\n")

        except Exception as e:
            logger.exception(f"Errore durante il completamento della calibrazione: {e}")
            print("\nErrore durante il salvataggio della calibrazione. Modifiche non salvate.\n")
        finally:
            self.calibration_active = False
            self.calibration_points = []
            self.calibration_step = 0
            self.tracking_enabled = True

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
def parse_arguments():
    """Analizza gli argomenti della linea di comando."""
    parser = argparse.ArgumentParser(description="HandPal - Controllo del mouse con i gesti delle mani")

    # Device and Resolution
    parser.add_argument('--device', type=int, help='ID del dispositivo webcam (es. 0, 1)')
    parser.add_argument('--width', type=int, help='Larghezza desiderata acquisizione webcam')
    parser.add_argument('--height', type=int, help='Altezza desiderata acquisizione webcam')

    # Features and Modes
    parser.add_argument('--debug', action='store_true', help='Abilita modalità debug con overlay informativo')
    parser.add_argument('--no-flip', dest='flip_camera', action='store_false', default=None, help='Disabilita ribaltamento orizzontale immagine webcam')
    parser.add_argument('--calibrate', action='store_true', help='Avvia la calibrazione all\'avvio dell\'applicazione')

    # Configuration Management
    parser.add_argument('--reset-config', action='store_true', help='Ripristina la configurazione ai valori predefiniti ed esce')
    parser.add_argument('--config-file', type=str, default=os.path.expanduser("~/.handpal_config.json"), help='Percorso file di configurazione')

    # Custom Cursor Option
    parser.add_argument('--cursor-file', type=str, default="red_cursor.cur", help="Percorso del file cursore personalizzato (.cur) da utilizzare")

    return parser.parse_args()

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    """Funzione principale dell'applicazione."""
    args = parse_arguments()

    # Handle config reset first
    if args.reset_config:
        config_path = args.config_file
        try:
            if os.path.exists(config_path):
                os.remove(config_path)
                print(f"Configurazione rimossa ({config_path}). Verranno usati i valori predefiniti al prossimo avvio.")
            else:
                print("Nessun file di configurazione trovato da resettare.")
            return 0
        except Exception as e:
            print(f"Errore durante la rimozione del file di configurazione: {e}")
            return 1

    # If a custom cursor file is provided, set the custom cursor
    if args.cursor_file:
        set_custom_cursor(args.cursor_file)

    config = Config(args)
    app = HandPal(config)

    if args.debug:
        app.debug_mode = True
        logger.setLevel(logging.DEBUG)
        logger.debug("Modalità debug attivata da argomento CLI.")

    if app.start():
        if args.calibrate:
            app.start_calibration()

        try:
            while app.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Interruzione da tastiera ricevuta (Ctrl+C).")
        finally:
            app.stop()
            logger.info("Applicazione terminata.")
            # Restore the default Windows cursors before exiting
            if args.cursor_file:
                restore_default_cursor()
            return 0
    else:
        logger.error("Avvio dell'applicazione fallito.")
        if args.cursor_file:
            restore_default_cursor()
        return 1

if __name__ == "__main__":
    sys.exit(main())
