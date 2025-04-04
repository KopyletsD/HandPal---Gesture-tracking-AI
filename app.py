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

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("handpal.log"), logging.StreamHandler()]
)
logger = logging.getLogger("HandPal")

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
                # pos_diff = self._calculate_distance(index_tip, self.gesture_state["last_click_position"]) # Using index tip as reference

                # Double click check: within time limit
                # Removed position check for simplicity, time is usually enough
                if time_diff < self.config["gesture_settings"]["double_click_time"]:
                    # Update time to prevent immediate re-triggering as single click
                    self.gesture_state["last_click_time"] = current_time
                    self.gesture_state["active_gesture"] = "click" # Keep it simple, action handles double
                    logger.debug("Double click gesture detected")
                    return "double_click"
                else:
                    # Single click
                    self.gesture_state["last_click_time"] = current_time
                    # Store position for potential double click next time
                    # self.gesture_state["last_click_position"] = index_tip # Store landmark object directly? No, store coords.
                    # self.gesture_state["last_click_position"] = (index_tip.x, index_tip.y) # Store coords
                    self.gesture_state["active_gesture"] = "click"
                    logger.debug("Single click gesture detected")
                    return "click"
        else:
            # Reset active gesture if it was 'click' and no longer clicking
            if self.gesture_state["active_gesture"] == "click":
                 self.gesture_state["active_gesture"] = None

        return None


    def check_scroll_gesture(self, hand_landmarks, handedness):
        """Rileva il gesto di scorrimento (indice e medio estesi verticalmente), solo per la mano sinistra."""
        if handedness != "Left":
            return None
        # Ignore scroll if clicking is active
        if self.gesture_state["active_gesture"] == "click":
            return None

        # Landmarks
        index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP] # 8
        middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP] # 12
        index_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP] # 5
        middle_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP] # 9
        ring_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP] # 16
        pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP] # 20

        # Conditions for scroll gesture:
        # 1. Index and Middle fingers extended (tip higher than base)
        index_extended = index_tip.y < index_mcp.y
        middle_extended = middle_tip.y < middle_mcp.y
        # 2. Index and Middle fingers relatively close together horizontally
        fingers_close = abs(index_tip.x - middle_tip.x) < 0.08 # Adjust threshold as needed
        # 3. Ring and Pinky fingers folded (tip lower than index/middle base)
        ring_folded = ring_tip.y > index_mcp.y
        pinky_folded = pinky_tip.y > middle_mcp.y # Check against its own base or index? Index seems more robust.

        is_scroll_pose = index_extended and middle_extended and fingers_close and ring_folded and pinky_folded

        scroll_delta = None
        index_tip_id = 8 # Use landmark index as key for last_positions

        if is_scroll_pose:
            if not self.gesture_state["scroll_active"]:
                # Start of scroll gesture
                self.gesture_state["scroll_active"] = True
                self.gesture_state["scroll_history"].clear()
                self.gesture_state["active_gesture"] = "scroll"
                self.last_positions[index_tip_id] = (index_tip.x, index_tip.y) # Initialize position
                logger.debug("Scroll gesture started")

            # Calculate vertical movement based on index finger tip
            if index_tip_id in self.last_positions:
                prev_y = self.last_positions[index_tip_id][1]
                curr_y = index_tip.y
                # Delta calculation: Positive delta means finger moved down (scroll down typically)
                # Negative delta means finger moved up (scroll up typically)
                # Multiply by sensitivity; screen height isn't directly relevant here, it's relative motion
                delta_y = (curr_y - prev_y) * self.config["gesture_settings"]["scroll_sensitivity"] * 10 # Added multiplier for more sensitivity

                # Add to history only if movement is significant enough to avoid noise
                if abs(delta_y) > 0.01: # Threshold for significant movement
                    self.gesture_state["scroll_history"].append(delta_y)

                # Calculate smoothed delta if history is populated
                if len(self.gesture_state["scroll_history"]) > 0:
                    smooth_delta = sum(self.gesture_state["scroll_history"]) / len(self.gesture_state["scroll_history"])

                    # Only return a value if the smoothed delta is significant
                    if abs(smooth_delta) > 0.05: # Threshold for smoothed output
                         scroll_delta = smooth_delta # Return the smoothed value
                         # Update last position only when a significant movement is processed
                         self.last_positions[index_tip_id] = (index_tip.x, index_tip.y)

            # Always update the reference position for the *next* frame's calculation if scroll is active
            # This prevents large jumps if there was a pause in significant movement
            if self.gesture_state["scroll_active"]:
                 self.last_positions[index_tip_id] = (index_tip.x, index_tip.y)

        else:
            # Gesture ended or not active
            if self.gesture_state["scroll_active"]:
                self.gesture_state["scroll_active"] = False
                self.gesture_state["scroll_history"].clear()
                if index_tip_id in self.last_positions:
                    del self.last_positions[index_tip_id] # Clear last position when scroll ends
                if self.gesture_state["active_gesture"] == "scroll":
                    self.gesture_state["active_gesture"] = None
                logger.debug("Scroll gesture ended")

        return scroll_delta # Returns None if no significant scroll, otherwise the delta


    def detect_hand_pose(self, hand_landmarks, handedness):
        """Analizza la posa complessiva della mano. Returns a string like 'pointing', 'open_palm', etc."""
        # Right hand is only used for pointing check (cursor control)
        if handedness == "Right":
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
            # Basic check: is index finger extended?
            if index_tip.y < index_mcp.y:
                return "pointing"
            else:
                return "unknown" # Or maybe 'fist' if other fingers are also down? Keep it simple.

        # Left hand: More detailed pose recognition for gestures
        if handedness == "Left":
            # Get key landmarks
            landmarks = hand_landmarks.landmark
            wrist = landmarks[mp.solutions.hands.HandLandmark.WRIST]
            thumb_tip = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = landmarks[mp.solutions.hands.HandLandmark.PINKY_TIP]

            # Get finger bases (MCP joints)
            thumb_mcp = landmarks[mp.solutions.hands.HandLandmark.THUMB_MCP] # Not typically used for extension check
            index_mcp = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
            middle_mcp = landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
            ring_mcp = landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_MCP]
            pinky_mcp = landmarks[mp.solutions.hands.HandLandmark.PINKY_MCP]

            # Check finger extensions (tip higher than base)
            # Thumb extension is tricky (horizontal movement), check relative to wrist or MCP
            # Simple check: thumb tip further from wrist horizontally than its base (MCP)
            thumb_extended = abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x)
            index_extended = index_tip.y < index_mcp.y
            middle_extended = middle_tip.y < middle_mcp.y
            ring_extended = ring_tip.y < ring_mcp.y
            pinky_extended = pinky_tip.y < pinky_mcp.y

            # Determine pose based on extended fingers
            current_pose = "unknown"
            num_extended = sum([index_extended, middle_extended, ring_extended, pinky_extended])

            if num_extended == 0 and not thumb_extended: # Check thumb too? Maybe just fingers.
                 current_pose = "closed_fist"
            elif index_extended and num_extended == 1:
                 current_pose = "pointing"
            elif index_extended and middle_extended and num_extended == 2:
                 # Could be 'peace' or 'scroll' - scroll check is separate and more specific
                 # Let's call it 'two_fingers' generically here
                 current_pose = "two_fingers" # Was 'peace'
            elif num_extended >= 4: # Allow for slight errors, 4 or 5 extended
                 current_pose = "open_palm"
            # Add more poses if needed (e.g., 'thumb_up')

            # Stabilize pose detection
            if current_pose == self.gesture_state["last_pose"]:
                self.gesture_state["pose_stable_count"] += 1
            else:
                self.gesture_state["last_pose"] = current_pose
                self.gesture_state["pose_stable_count"] = 0

            # Return stabilized pose if stable enough
            if self.gesture_state["pose_stable_count"] >= self.POSE_STABILITY_THRESHOLD:
                return current_pose
            else:
                # Return previous stable pose during transition to avoid flickering
                # Or return current unstable pose? Let's return current for responsiveness.
                return current_pose # Return the currently detected pose even if not stable yet

        return "unknown" # Default


class MotionSmoother:
    """Classe per smoothing del movimento del mouse con filtro di media mobile."""

    def __init__(self, config, history_size=5): # Reduced history size slightly
        self.config = config
        self.history_size = history_size
        self.position_history = deque(maxlen=history_size)
        self.last_smoothed_position = None # Store the last *output* position
        self.inactivity_zone_size = config["inactivity_zone"] # Relative to screen size
        self.smoothing_factor = config["smoothing_factor"] # Exponential moving average factor

    def update(self, target_x, target_y):
        """Aggiorna la cronologia delle posizioni e calcola la nuova posizione smussata."""
        current_target = (target_x, target_y)

        # If history is empty, initialize
        if not self.position_history:
            self.position_history.append(current_target)
            self.last_smoothed_position = current_target
            return current_target

        # Get the previous target position from history for inactivity check
        prev_target = self.position_history[-1]

        # Calculate raw movement delta relative to screen size
        # Use screen dimensions from config or dynamically? Assume config is updated.
        # Getting screen size dynamically every time might be slow. Get it once in HandPal.
        screen_w, screen_h = self.config["width"], self.config["height"] # Use process size? No, screen size.
        # Need access to actual screen size here. Pass it in or get from config?
        # Let's assume HandPal updates config with screen size if needed, or pass it.
        # For now, use placeholder or assume HandPal class provides it.
        # Let's modify HandPal to pass screen_size to the smoother or update config.
        # --> Modification needed in HandPal.__init__ and potentially map_to_screen

        # Let's assume self.config has screen_width, screen_height (HandPal should set this)
        screen_w = self.config.get("screen_width", 1920) # Default fallback
        screen_h = self.config.get("screen_height", 1080)

        dx = abs(target_x - prev_target[0]) / screen_w
        dy = abs(target_y - prev_target[1]) / screen_h

        # Apply inactivity zone based on the *raw* target movement
        if dx < self.inactivity_zone_size and dy < self.inactivity_zone_size and self.last_smoothed_position:
             # If movement is small, return the last *smoothed* position to prevent drift
             return self.last_smoothed_position

        # Add current target to history *after* inactivity check
        self.position_history.append(current_target)

        # Calculate smoothed position using Exponential Moving Average (EMA)
        # Simple EMA: smooth = alpha * current + (1 - alpha) * last_smooth
        # Alpha is the smoothing_factor
        alpha = self.smoothing_factor
        if self.last_smoothed_position:
            smooth_x = int(alpha * target_x + (1 - alpha) * self.last_smoothed_position[0])
            smooth_y = int(alpha * target_y + (1 - alpha) * self.last_smoothed_position[1])
        else:
            # First time after inactivity or start
            smooth_x, smooth_y = target_x, target_y

        # Update the last smoothed position
        self.last_smoothed_position = (smooth_x, smooth_y)

        return smooth_x, smooth_y

    def reset(self):
        """Resets the smoother's history."""
        self.position_history.clear()
        self.last_smoothed_position = None


class HandPal:
    """Classe principale che gestisce l'applicazione HandPal."""

    def __init__(self, config):
        self.config = config
        self.mouse = Controller()
        self.gesture_recognizer = GestureRecognizer(config)
        # Get screen size once for mapping and smoothing
        try:
            root = tk.Tk()
            root.withdraw() # Hide the main window
            self.screen_size = (root.winfo_screenwidth(), root.winfo_screenheight())
            root.destroy()
        except tk.TclError:
             logger.warning("Could not initialize Tkinter to get screen size. Using defaults (1920x1080).")
             self.screen_size = (1920, 1080)

        # Update config with actual screen size for smoother
        self.config["screen_width"] = self.screen_size[0]
        self.config["screen_height"] = self.screen_size[1]

        self.motion_smoother = MotionSmoother(config) # Now config has screen size
        self.running = False
        self.cap = None
        self.mp_hands = mp.solutions.hands
        self.hands = None
        self.last_cursor_pos = None
        self.last_right_hand_pos = None # Store normalized coords (0-1)
        self.fps_stats = deque(maxlen=30) # For calculating rolling FPS

        # Calibration state
        self.calibration_active = False
        self.calibration_points = [] # Stores normalized coords (0-1) during calibration
        self.calibration_step = 0
        self.calibration_corners = [
            "in alto a sinistra dello SCHERMO",
            "in alto a destra dello SCHERMO",
            "in basso a destra dello SCHERMO",
            "in basso a sinistra dello SCHERMO"
        ]

        # App state flags
        self.debug_mode = False
        self.tracking_enabled = True # Can be used to pause tracking

        # Debug info dictionary
        self.debug_values = {
            "fps": 0.0,
            "last_action_time": time.time(),
            "cursor_history": deque(maxlen=20), # Longer history for drawing trace
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
            # Try setting desired capture resolution, but don't fail if it doesn't work
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.config["width"])
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.config["height"])
            # Verify resolution
            actual_width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
            logger.info(f"Webcam richiesta {self.config['width']}x{self.config['height']}, ottenuta {actual_width}x{actual_height}")


            if not self.cap.isOpened():
                logger.error(f"Impossibile aprire la webcam (dispositivo {self.config['device']}).")
                return False

            # Initialize MediaPipe Hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=self.config["use_static_image_mode"],
                max_num_hands=2, # Crucial: Detect up to two hands
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
            self.stop() # Ensure cleanup if start fails
            return False

    def stop(self):
        """Ferma l'applicazione."""
        if not self.running:
            return
        logger.info("Fermando HandPal...")
        self.running = False
        # Wait briefly for the thread to finish processing the current frame
        if hasattr(self, 'thread') and self.thread.is_alive():
             self.thread.join(timeout=0.5) # Wait max 0.5 seconds

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
            # Store raw coordinates for debugging
            if self.debug_mode:
                self.debug_values["screen_mapping"]["raw"] = (x, y)

            x_calib = x
            y_calib = y

            # Apply calibration if enabled and valid
            if self.config["calibration"]["enabled"]:
                x_min = self.config["calibration"]["x_min"]
                x_max = self.config["calibration"]["x_max"]
                y_min = self.config["calibration"]["y_min"]
                y_max = self.config["calibration"]["y_max"]

                # Prevent division by zero or near-zero
                x_range = x_max - x_min
                y_range = y_max - y_min

                if x_range < 0.01 or y_range < 0.01:
                     # Fallback to no calibration if range is too small
                     if self.debug_mode: logger.warning("Calibration range too small, ignoring.")
                     pass # Use original x, y
                else:
                    # Normalize within the calibrated range, clamping to 0-1
                    x_calib = max(0.0, min(1.0, (x - x_min) / x_range))
                    y_calib = max(0.0, min(1.0, (y - y_min) / y_range))

            # Apply screen margin expansion
            margin = self.config["calibration"]["screen_margin"]
            # Expand the 0-1 range to (0-margin) to (1+margin) before scaling
            x_expanded = x_calib * (1.0 + 2.0 * margin) - margin
            y_expanded = y_calib * (1.0 + 2.0 * margin) - margin

            # Scale to screen size
            screen_x = int(x_expanded * self.screen_size[0])
            screen_y = int(y_expanded * self.screen_size[1])

            # Clamp final coordinates to screen boundaries
            screen_x = max(0, min(screen_x, self.screen_size[0] - 1))
            screen_y = max(0, min(screen_y, self.screen_size[1] - 1))

            # Store mapped coordinates for debugging
            if self.debug_mode:
                self.debug_values["screen_mapping"]["mapped"] = (screen_x, screen_y)

            return screen_x, screen_y

        except Exception as e:
            logger.error(f"Errore durante la mappatura coordinate: {e}")
            # Fallback to simple direct mapping
            screen_x = int(x * self.screen_size[0])
            screen_y = int(y * self.screen_size[1])
            return max(0, min(screen_x, self.screen_size[0] - 1)), max(0, min(screen_y, self.screen_size[1] - 1))


    # =========================================================================
    # REFACTORED process_frame
    # =========================================================================
    def process_frame(self, frame):
        """Processa un singolo frame della webcam, ottimizzato per due mani."""
        # 1. Pre-processing
        process_start_time = time.perf_counter()
        # Resize for faster processing
        process_frame = cv.resize(frame, (self.config["process_width"], self.config["process_height"]))
        # Convert BGR to RGB
        rgb_frame = cv.cvtColor(process_frame, cv.COLOR_BGR2RGB)
        # Improve performance by marking frame as not writeable
        rgb_frame.flags.writeable = False

        # 2. MediaPipe Hand Detection
        results = self.hands.process(rgb_frame)

        # 3. Post-processing and Drawing Setup
        rgb_frame.flags.writeable = True # Make writeable again if needed (though we draw on display_frame)
        display_frame = process_frame.copy() # Draw on the smaller processed frame
        frame_height, frame_width = display_frame.shape[:2]

        # 4. Identify Left/Right Hands (if detected)
        left_hand_landmarks = None
        right_hand_landmarks = None
        left_handedness = None
        right_handedness = None

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness_enum = results.multi_handedness[idx].classification[0]
                handedness_label = handedness_enum.label # "Left" or "Right"
                # score = handedness_enum.score # Confidence score

                if handedness_label == "Right":
                    right_hand_landmarks = hand_landmarks
                    right_handedness = handedness_label
                elif handedness_label == "Left":
                    left_hand_landmarks = hand_landmarks
                    left_handedness = handedness_label

            # Draw landmarks for all detected hands
            self.draw_landmarks(display_frame, results.multi_hand_landmarks)

        # 5. Process Right Hand (Cursor Control / Calibration)
        self.last_right_hand_pos = None # Reset last known position each frame
        if right_hand_landmarks:
            # Get index finger tip position (normalized 0-1)
            index_tip = right_hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            right_hand_pos = (index_tip.x, index_tip.y)
            self.last_right_hand_pos = right_hand_pos # Store for calibration use

            # --- Calibration Logic ---
            if self.calibration_active:
                # Draw feedback circle on the display frame
                tip_x_disp = int(right_hand_pos[0] * frame_width)
                tip_y_disp = int(right_hand_pos[1] * frame_height)
                cv.circle(display_frame, (tip_x_disp, tip_y_disp), 10, (0, 255, 0), -1)
                # No cursor movement during calibration

            # --- Cursor Control Logic ---
            elif self.tracking_enabled:
                try:
                    # Map normalized coords to screen coords
                    screen_x, screen_y = self.map_to_screen(right_hand_pos[0], right_hand_pos[1])

                    # Apply smoothing
                    smooth_x, smooth_y = self.motion_smoother.update(screen_x, screen_y)

                    # Update mouse position if changed significantly (handled by smoother's inactivity zone)
                    if self.last_cursor_pos != (smooth_x, smooth_y):
                        self.mouse.position = (smooth_x, smooth_y)
                        self.last_cursor_pos = (smooth_x, smooth_y)
                        if self.debug_mode:
                            self.debug_values["screen_mapping"]["smoothed"] = (smooth_x, smooth_y)
                            self.debug_values["cursor_history"].append((smooth_x, smooth_y))

                except Exception as e:
                    logger.error(f"Errore nell'aggiornamento del cursore: {e}")
                    self.motion_smoother.reset() # Reset smoother on error

            # --- Right Hand Pose (for debug/info) ---
            if self.debug_mode:
                 self.debug_values["right_hand_pose"] = self.gesture_recognizer.detect_hand_pose(right_hand_landmarks, "Right")

        else:
            # No right hand detected, reset smoother if it was active
            self.motion_smoother.reset()


        # 6. Process Left Hand (Gestures)
        if left_hand_landmarks and not self.calibration_active:
            # --- Gesture Recognition ---
            # Check for scroll first (usually mutually exclusive with click)
            scroll_amount = self.gesture_recognizer.check_scroll_gesture(left_hand_landmarks, "Left")

            if scroll_amount is not None:
                # Perform scroll action
                # Convert relative scroll amount to scroll wheel 'clicks'
                # Adjust the divisor for sensitivity
                scroll_clicks = int(scroll_amount * -1) # Invert direction if needed, pynput positive is down
                if scroll_clicks != 0:
                    self.mouse.scroll(0, scroll_clicks)
                    self.debug_values["last_action_time"] = time.time()
                    self.debug_values["active_gesture"] = f"Scroll ({scroll_clicks})"
                    # logger.debug(f"Scroll: {scroll_clicks}") # Reduce log spam
            else:
                # If not scrolling, check for click
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
                # else: # No specific action gesture detected
                #      if self.debug_values["active_gesture"] not in ["None", "Scroll (...)"]: # Avoid clearing scroll status immediately
                #          self.debug_values["active_gesture"] = "None"


            # --- Left Hand Pose (for debug/info) ---
            if self.debug_mode:
                 self.debug_values["left_hand_pose"] = self.gesture_recognizer.detect_hand_pose(left_hand_landmarks, "Left")
                 # Update active gesture based on recognizer state if no action happened this frame
                 if scroll_amount is None and click_gesture is None:
                     current_active = self.gesture_recognizer.gesture_state["active_gesture"]
                     self.debug_values["active_gesture"] = str(current_active) if current_active else "None"


        # 7. Draw Overlays (Info, Debug, Calibration)
        self.draw_overlays(display_frame)

        # 8. Calculate Processing Time & Update FPS Stats
        process_end_time = time.perf_counter()
        frame_proc_time = process_end_time - process_start_time
        self.fps_stats.append(frame_proc_time) # Store duration

        return display_frame
    # =========================================================================
    # END REFACTORED process_frame
    # =========================================================================


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

        # --- Calibration Instructions ---
        if self.calibration_active:
            # Semi-transparent background
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

            # Draw target corners on frame
            corner_radius = 10
            inactive_color = (180, 180, 180)
            active_color = (0, 0, 255)
            cv.circle(frame, (corner_radius, corner_radius), corner_radius, active_color if self.calibration_step == 0 else inactive_color, -1)
            cv.circle(frame, (w - corner_radius, corner_radius), corner_radius, active_color if self.calibration_step == 1 else inactive_color, -1)
            cv.circle(frame, (w - corner_radius, h - corner_radius), corner_radius, active_color if self.calibration_step == 2 else inactive_color, -1)
            cv.circle(frame, (corner_radius, h - corner_radius), corner_radius, active_color if self.calibration_step == 3 else inactive_color, -1)

        # --- General Info / Debug Mode ---
        else:
            # Basic instructions at the bottom
            cv.putText(frame, "C:Calibra | Q:Esci | D:Debug", (10, h - 10),
                      font, font_scale_small, overlay_color, line_type)

            # FPS Display
            if self.fps_stats:
                # Calculate FPS based on average duration
                avg_duration = sum(self.fps_stats) / len(self.fps_stats)
                fps = 1.0 / avg_duration if avg_duration > 0 else 0
                self.debug_values["fps"] = fps
                cv.putText(frame, f"FPS: {fps:.1f}", (w - 70, 20),
                          font, font_scale_medium, overlay_color, line_type)

            # Debug Information Panel
            if self.debug_mode:
                # Semi-transparent background for debug info
                debug_h = 150 # Height of the debug panel
                overlay = frame.copy()
                cv.rectangle(overlay, (0, h - debug_h), (w, h), bg_color, -1)
                cv.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                y_pos = h - debug_h + 15 # Starting y position for text

                # Row 1: Poses & Active Gesture
                debug_text1 = f"L-Pose: {self.debug_values['left_hand_pose']} | R-Pose: {self.debug_values['right_hand_pose']} | Gest: {self.debug_values['active_gesture']}"
                cv.putText(frame, debug_text1, (10, y_pos), font, font_scale_small, overlay_color, line_type)
                y_pos += 20

                # Row 2: Mapping Info
                raw = self.debug_values['screen_mapping']['raw']
                mapped = self.debug_values['screen_mapping']['mapped']
                smoothed = self.debug_values['screen_mapping']['smoothed']
                debug_text2 = f"Raw:({raw[0]:.2f},{raw[1]:.2f}) Map:({mapped[0]},{mapped[1]}) Smooth:({smoothed[0]},{smoothed[1]})"
                cv.putText(frame, debug_text2, (10, y_pos), font, font_scale_small, overlay_color, line_type)
                y_pos += 20

                # Row 3: Calibration Values
                calib = self.config['calibration']
                debug_text3 = f"Calib: X[{calib['x_min']:.2f}-{calib['x_max']:.2f}] Y[{calib['y_min']:.2f}-{calib['y_max']:.2f}] En:{calib['enabled']}"
                cv.putText(frame, debug_text3, (10, y_pos), font, font_scale_small, overlay_color, line_type)
                y_pos += 20

                # Row 4: Config Values (Sensitivity, Cooldown, Smoothing)
                gest_sens = self.config['gesture_sensitivity']
                cooldown = self.config['click_cooldown']
                smooth_f = self.config['smoothing_factor']
                scroll_sens = self.config['gesture_settings']['scroll_sensitivity']
                debug_text4 = f"Sens:{gest_sens:.2f} Cool:{cooldown:.1f} Smooth:{smooth_f:.1f} Scroll:{scroll_sens}"
                cv.putText(frame, debug_text4, (10, y_pos), font, font_scale_small, overlay_color, line_type)
                y_pos += 20

                # Row 5: Timings
                time_since_action = time.time() - self.debug_values["last_action_time"]
                debug_text5 = f"Last Action: {time_since_action:.1f}s ago"
                cv.putText(frame, debug_text5, (10, y_pos), font, font_scale_small, overlay_color, line_type)
                y_pos += 20

                # Draw cursor trace (if history exists)
                if len(self.debug_values["cursor_history"]) > 1:
                    trace_color = (0, 255, 255) # Cyan
                    for i in range(1, len(self.debug_values["cursor_history"])):
                        # Convert screen coords back to display frame coords for drawing
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

            # --- Frame Acquisition ---
            if self.cap is None or not self.cap.isOpened():
                logger.error("Webcam non disponibile nel loop principale.")
                time.sleep(0.5)
                continue # Try again

            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("Frame non valido ricevuto dalla webcam.")
                time.sleep(0.1)
                continue

            # --- Frame Flipping ---
            if self.config["flip_camera"]:
                frame = cv.flip(frame, 1)

            # --- Frame Processing ---
            try:
                processed_frame = self.process_frame(frame)
            except Exception as e:
                 logger.exception(f"Errore irreversibile durante process_frame: {e}")
                 # Decide whether to stop or just log and continue
                 # self.stop() # Option: stop on critical error
                 processed_frame = frame # Show raw frame on error
                 # Add error message to frame?
                 cv.putText(processed_frame, "ERROR PROCESSING FRAME", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


            # --- Display Frame ---
            cv.imshow("HandPal", processed_frame)

            # --- Handle Keyboard Input ---
            key = cv.waitKey(1) & 0xFF # Wait minimal time

            if key == ord('q'):
                logger.info("Comando 'q' ricevuto, fermando...")
                self.stop()
            elif key == ord('c') and not self.calibration_active:
                self.start_calibration()
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                logger.info(f"Modalità debug: {'attivata' if self.debug_mode else 'disattivata'}")
                # Clear debug history on toggle? Optional.
                # self.debug_values["cursor_history"].clear()
            elif key == 27 and self.calibration_active:  # ESC
                logger.info("Calibrazione annullata dall'utente.")
                self.cancel_calibration()
            elif key == 32 and self.calibration_active:  # Spacebar
                self.process_calibration_step()
            # Add other keys if needed (e.g., pause tracking)
            # elif key == ord('p'):
            #     self.tracking_enabled = not self.tracking_enabled
            #     logger.info(f"Tracking {'abilitato' if self.tracking_enabled else 'disabilitato'}")


            # --- FPS Limiting ---
            loop_end_time = time.perf_counter()
            elapsed_time = loop_end_time - loop_start_time
            sleep_time = frame_time_target - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Update last frame time for next iteration's potential use (though not currently used)
            last_frame_time = loop_end_time # Use perf_counter for consistency

        logger.debug("Main loop terminato.")
        cv.destroyAllWindows() # Ensure window is closed when loop exits


    def start_calibration(self):
        """Avvia il processo di calibrazione."""
        if self.calibration_active:
            logger.warning("Tentativo di avviare la calibrazione mentre è già attiva.")
            return

        self.calibration_active = True
        # self.config["calibration"]["active"] = True # This is internal state, not config
        self.calibration_points = []
        self.calibration_step = 0
        self.tracking_enabled = False # Disable cursor movement during calibration
        self.motion_smoother.reset() # Reset smoother state
        logger.info("Calibrazione avviata. Segui le istruzioni sullo schermo.")
        print("\n--- CALIBRAZIONE ---")
        print("Usa la mano DESTRA. Posiziona l'indice negli angoli dello SCHERMO.")
        print("Premi SPAZIO per confermare ogni angolo. Premi ESC per annullare.")
        print(f"Inizia con: {self.calibration_corners[0]}\n")

    def cancel_calibration(self):
        """Annulla la calibrazione in corso."""
        self.calibration_active = False
        # self.config["calibration"]["active"] = False
        self.calibration_points = []
        self.calibration_step = 0
        self.tracking_enabled = True # Re-enable tracking
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

        # Store the *normalized* coordinates of the right hand's index finger
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
            self.cancel_calibration() # Reset state
            return

        try:
            # Calculate min/max from the recorded normalized points
            x_values = [p[0] for p in self.calibration_points]
            y_values = [p[1] for p in self.calibration_points]

            x_min_calib = min(x_values)
            x_max_calib = max(x_values)
            y_min_calib = min(y_values)
            y_max_calib = max(y_values)

            # Basic validation: ensure min < max and range is not too small
            min_range = 0.05 # Minimum acceptable range
            if (x_max_calib - x_min_calib < min_range) or (y_max_calib - y_min_calib < min_range):
                 logger.warning(f"Range di calibrazione troppo piccolo: X={x_max_calib-x_min_calib:.3f}, Y={y_max_calib-y_min_calib:.3f}. Calibrazione potrebbe essere imprecisa.")
                 # Optionally, add padding or reject calibration
                 # Example padding:
                 # x_min_calib = max(0.0, x_min_calib - 0.02)
                 # x_max_calib = min(1.0, x_max_calib + 0.02)
                 # y_min_calib = max(0.0, y_min_calib - 0.02)
                 # y_max_calib = min(1.0, y_max_calib + 0.02)
                 print("Attenzione: L'area di calibrazione rilevata è molto piccola.")


            # Update configuration with new values
            self.config["calibration"]["x_min"] = x_min_calib
            self.config["calibration"]["x_max"] = x_max_calib
            self.config["calibration"]["y_min"] = y_min_calib
            self.config["calibration"]["y_max"] = y_max_calib
            self.config["calibration"]["enabled"] = True # Ensure calibration is marked as enabled

            # Save the updated configuration
            self.config.save()

            logger.info(f"Calibrazione completata e salvata. Valori: X[{x_min_calib:.3f}-{x_max_calib:.3f}], Y[{y_min_calib:.3f}-{y_max_calib:.3f}]")
            print("\nCalibrazione completata e salvata con successo!\n")

        except Exception as e:
            logger.exception(f"Errore durante il completamento della calibrazione: {e}")
            print("\nErrore durante il salvataggio della calibrazione. Modifiche non salvate.\n")
            # Optionally revert to default calibration values? Or just keep the old ones.
            # Reverting might be safer if completion failed badly.
            # self.config['calibration'].update(self.config.DEFAULT_CONFIG['calibration'])

        finally:
            # Reset calibration state regardless of success/failure
            self.calibration_active = False
            # self.config["calibration"]["active"] = False
            self.calibration_points = []
            self.calibration_step = 0
            self.tracking_enabled = True # Re-enable tracking


def parse_arguments():
    """Analizza gli argomenti della linea di comando."""
    parser = argparse.ArgumentParser(description="HandPal - Controllo del mouse con i gesti delle mani")

    # Device and Resolution
    parser.add_argument('--device', type=int, help='ID del dispositivo webcam (es. 0, 1)')
    parser.add_argument('--width', type=int, help='Larghezza desiderata acquisizione webcam')
    parser.add_argument('--height', type=int, help='Altezza desiderata acquisizione webcam')

    # Features and Modes
    parser.add_argument('--debug', action='store_true', help='Abilita modalità debug con overlay informativo')
    parser.add_argument('--no-flip', dest='flip_camera', action='store_false', default=None, help='Disabilita ribaltamento orizzontale immagine webcam') # Default is handled by Config class
    parser.add_argument('--calibrate', action='store_true', help='Avvia la calibrazione all\'avvio dell\'applicazione')

    # Configuration Management
    parser.add_argument('--reset-config', action='store_true', help='Ripristina la configurazione ai valori predefiniti ed esce')
    parser.add_argument('--config-file', type=str, default=os.path.expanduser("~/.handpal_config.json"), help='Percorso file di configurazione')


    return parser.parse_args()


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
            return 0 # Exit after reset
        except Exception as e:
            print(f"Errore durante la rimozione del file di configurazione: {e}")
            return 1


    # Initialize configuration (loads from file, then overrides with args)
    config = Config(args)

    # Create and configure the main application object
    app = HandPal(config)

    # Set debug mode from args if specified
    if args.debug:
        app.debug_mode = True
        logger.setLevel(logging.DEBUG) # Lower log level if debug arg is passed
        logger.debug("Modalità debug attivata da argomento CLI.")


    # Start the application
    if app.start():
        # Start calibration immediately if requested
        if args.calibrate:
            # Need a small delay to ensure the window is up? Maybe not.
            # time.sleep(0.5)
            app.start_calibration()

        # Keep the main thread alive while the app runs in its own thread
        try:
            while app.running:
                time.sleep(0.1) # Main thread doesn't need to do much
        except KeyboardInterrupt:
            logger.info("Interruzione da tastiera ricevuta (Ctrl+C).")
        finally:
            app.stop() # Ensure clean shutdown
            logger.info("Applicazione terminata.")
            return 0
    else:
        logger.error("Avvio dell'applicazione fallito.")
        return 1


if __name__ == "__main__":
    sys.exit(main())