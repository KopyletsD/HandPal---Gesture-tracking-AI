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
import queue # For thread-safe communication

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
    # Check if running on Windows
    if os.name != 'nt':
        logger.warning("Impostazione cursore personalizzato supportata solo su Windows.")
        return False

    try:
        # Check if file exists before trying to load
        if not os.path.exists(cursor_path):
            raise FileNotFoundError(f"File cursore non trovato: {cursor_path}")
        user32 = ctypes.windll.user32
        hCursor = user32.LoadCursorFromFileW(cursor_path)
        if hCursor == 0:
            # Get more error info if possible
            error_code = ctypes.get_last_error()
            raise Exception(f"Impossibile caricare il cursore personalizzato (Errore {error_code}). Assicurarsi che sia un file .cur valido.")
        OCR_NORMAL = 32512 # Standard Arrow cursor ID
        if not user32.SetSystemCursor(hCursor, OCR_NORMAL):
            error_code = ctypes.get_last_error()
            # Note: Setting system cursors might require specific privileges.
            raise Exception(f"Impossibile impostare il cursore di sistema (Errore {error_code}). Potrebbero mancare i permessi.")
        logger.info(f"Cursore personalizzato '{os.path.basename(cursor_path)}' impostato con successo.")
        return True # Indicate success
    except FileNotFoundError as e:
        logger.error(f"Errore nell'impostazione del cursore personalizzato: {e}")
    except Exception as e:
        logger.error(f"Errore nell'impostazione del cursore personalizzato: {e}")
    return False # Indicate failure

def restore_default_cursor():
    """
    Ripristina i cursori di sistema predefiniti (solo Windows).
    """
    # Check if running on Windows
    if os.name != 'nt':
        logger.debug("Ripristino cursore predefinito saltato (non su Windows).")
        return

    try:
        user32 = ctypes.windll.user32
        SPI_SETCURSORS = 0x57
        # The last parameter '3' means SPIF_UPDATEINIFILE | SPIF_SENDCHANGE
        if not user32.SystemParametersInfoW(SPI_SETCURSORS, 0, None, 3):
            error_code = ctypes.get_last_error()
            raise Exception(f"Impossibile ripristinare i cursori predefiniti (Errore {error_code}).")
        logger.info("Cursori predefiniti ripristinati.")
    except Exception as e:
        logger.error(f"Errore nel ripristino dei cursori predefiniti: {e}")

# -----------------------------------------------------------------------------
# Config Class
# -----------------------------------------------------------------------------
class Config:
    """Classe per gestire la configurazione dell'applicazione."""
    DEFAULT_CONFIG = {
        # Webcam & Processing
        "device": 0,
        "width": 1280, # Default webcam width
        "height": 720, # Default webcam height
        "process_width": 640, # Lower resolution for processing
        "process_height": 360, # Lower resolution for processing
        "flip_camera": True,

        # OpenCV Display Window
        "display_width": None, # Use None to default to frame size
        "display_height": None,

        # MediaPipe
        "min_detection_confidence": 0.6,
        "min_tracking_confidence": 0.5,
        "use_static_image_mode": False,

        # Control & Smoothing
        "smoothing_factor": 0.7, # Exponential Moving Average factor (higher = smoother, more lag)
        "inactivity_zone": 0.015, # Normalized screen size threshold to stop smoothing

        # Gestures
        "click_cooldown": 0.4, # Seconds between clicks
        "gesture_sensitivity": 0.07, # Distance threshold for click (normalized)
        "gesture_settings": {
            "scroll_sensitivity": 12, # Multiplier for scroll speed
            "drag_threshold": 0.15, # Placeholder, not implemented
            "double_click_time": 0.35 # Max time between clicks for double click
        },

        # Calibration
        "calibration": {
            "enabled": True, # Whether to use calibration values
            "screen_margin": 0.1, # Expand mapping beyond calibrated area
            "x_min": 0.15, # Default calibrated min X (normalized)
            "x_max": 0.85, # Default calibrated max X
            "y_min": 0.15, # Default calibrated min Y
            "y_max": 0.85, # Default calibrated max Y
            "active": False # Internal state, do not save
        },

        # Performance
        "max_fps": 60, # Target FPS for the main loop (display)

        # Appearance
        "custom_cursor_path": "red_cursor.cur" # Path to custom cursor file
    }

    def _deep_update(self, target_dict, source_dict):
        """Recursively update nested dictionaries."""
        for key, value in source_dict.items():
            if isinstance(value, dict) and key in target_dict and isinstance(target_dict[key], dict):
                self._deep_update(target_dict[key], value)
            else:
                target_dict[key] = value

    def __init__(self, args=None):
        """Inizializza la configurazione combinando valori predefiniti e argomenti CLI."""
        # Start with a deep copy of defaults
        self.config = json.loads(json.dumps(self.DEFAULT_CONFIG))

        config_path = os.path.expanduser("~/.handpal_config.json")

        # Load from file if it exists
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                    # Deep update the current config with saved values
                    self._deep_update(self.config, saved_config)
                    logger.info(f"Configurazione caricata da {config_path}")
            except json.JSONDecodeError as e:
                logger.error(f"Errore nel parsing del file di configurazione {config_path}: {e}. Usando default/CLI.")
            except Exception as e:
                logger.error(f"Errore nel caricamento della configurazione da {config_path}: {e}, usando default/CLI.")

        # Override with CLI arguments if provided
        if args:
            cli_args = vars(args)
            for key, value in cli_args.items():
                # Only update if CLI arg is provided (not None)
                if value is not None:
                    # Handle special cases like --no-flip
                    if key == 'flip_camera' and value is False:
                         self.config['flip_camera'] = False
                    # Handle nested keys if needed (e.g., future --calibration.enabled=False)
                    elif '.' in key:
                        keys = key.split('.')
                        d = self.config
                        try:
                            for k in keys[:-1]:
                                d = d[k]
                            if keys[-1] in d:
                                target_type = type(d[keys[-1]])
                                d[keys[-1]] = target_type(value) # Update existing nested key
                        except (KeyError, TypeError, ValueError):
                             logger.warning(f"Impossibile impostare l'argomento CLI nidificato '{key}' (valore '{value}')")
                    # Handle top-level keys
                    elif key in self.config:
                        # Try to convert CLI arg type to config type if possible and not a dict
                        if not isinstance(self.config[key], dict):
                            try:
                                target_type = type(self.config[key])
                                # Handle boolean explicitly if default is None from argparse
                                if target_type == bool and isinstance(value, bool):
                                    self.config[key] = value
                                elif target_type != bool: # Avoid converting non-bools to bool
                                    self.config[key] = target_type(value)
                                else: # Keep the value as is if types mismatch significantly
                                     self.config[key] = value
                            except (ValueError, TypeError):
                                 logger.warning(f"Impossibile convertire l'argomento CLI '{key}' ({value}) al tipo {target_type}. Usando valore fornito.")
                                 self.config[key] = value
                        else:
                            # Don't overwrite entire dicts from simple CLI args
                            logger.warning(f"Ignorando l'argomento CLI '{key}' perché corrisponde a una sezione dizionario nella configurazione.")
                    # Special handling for args not directly in config (e.g. --cursor-file)
                    elif key == 'cursor_file':
                         self.config['custom_cursor_path'] = value


        # Ensure calibration active state is false initially and validate values
        self.config["calibration"]["active"] = False
        self._validate_calibration()
        self._validate_display_dims()
        logger.debug(f"Configurazione finale inizializzata: {self.config}")


    def _validate_calibration(self):
        """Checks if calibration values are valid and resets if not."""
        calib = self.config["calibration"]
        default_calib = self.DEFAULT_CONFIG["calibration"]
        reset_x = False
        reset_y = False

        # Check types and basic range
        if not all(isinstance(calib[k], (int, float)) for k in ["x_min", "x_max", "y_min", "y_max"]):
             logger.warning("Valori di calibrazione non numerici, ripristino default.")
             reset_x = reset_y = True
        elif not (0 <= calib["x_min"] < calib["x_max"] <= 1):
            logger.warning(f"Valori di calibrazione X non validi ({calib['x_min']:.2f}, {calib['x_max']:.2f}), ripristino default X.")
            reset_x = True
        elif abs(calib["x_max"] - calib["x_min"]) < 0.05: # Minimum range check
            logger.warning(f"Range di calibrazione X troppo piccolo ({calib['x_max'] - calib['x_min']:.2f}), ripristino default X.")
            reset_x = True

        if not reset_y and not (0 <= calib["y_min"] < calib["y_max"] <= 1):
            logger.warning(f"Valori di calibrazione Y non validi ({calib['y_min']:.2f}, {calib['y_max']:.2f}), ripristino default Y.")
            reset_y = True
        elif not reset_y and abs(calib["y_max"] - calib["y_min"]) < 0.05: # Minimum range check
            logger.warning(f"Range di calibrazione Y troppo piccolo ({calib['y_max'] - calib['y_min']:.2f}), ripristino default Y.")
            reset_y = True

        if reset_x:
            calib["x_min"] = default_calib["x_min"]
            calib["x_max"] = default_calib["x_max"]
        if reset_y:
            calib["y_min"] = default_calib["y_min"]
            calib["y_max"] = default_calib["y_max"]

    def _validate_display_dims(self):
        """Checks if display dimensions are valid integers or None."""
        w = self.config.get("display_width")
        h = self.config.get("display_height")
        valid_w = (w is None) or (isinstance(w, int) and w > 0)
        valid_h = (h is None) or (isinstance(h, int) and h > 0)
        if not valid_w:
             logger.warning(f"Valore display_width ('{w}') non valido, verrà ignorato.")
             self.config["display_width"] = None
        if not valid_h:
             logger.warning(f"Valore display_height ('{h}') non valido, verrà ignorato.")
             self.config["display_height"] = None

    def get(self, key, default=None):
        """Helper to access config values, including nested keys like 'calibration.x_min'."""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            # logger.warning(f"Chiave di configurazione '{key}' non trovata, usando default: {default}")
            return default

    def set(self, key, value):
        """Helper to set config values, including nested keys."""
        keys = key.split('.')
        d = self.config
        try:
            for k in keys[:-1]:
                # Create nested dicts if they don't exist? Risky. Assume they exist.
                if k not in d or not isinstance(d[k], dict):
                     logger.error(f"Impossibile impostare la chiave '{key}': percorso intermedio '{k}' non è un dizionario.")
                     return False
                d = d[k]
            d[keys[-1]] = value
            # Re-validate specific sections if necessary
            if key.startswith("calibration."): self._validate_calibration()
            if key.startswith("display_"): self._validate_display_dims()
            return True
        except (KeyError, TypeError):
             logger.error(f"Impossibile impostare la chiave di configurazione '{key}': percorso non valido.")
             return False

    def save(self):
        """Salva la configurazione corrente in un file JSON."""
        config_path = os.path.expanduser("~/.handpal_config.json")
        try:
            # Create a deep copy to save, excluding internal state like 'active'
            config_to_save = json.loads(json.dumps(self.config)) # Deep copy
            if "calibration" in config_to_save and "active" in config_to_save["calibration"]:
                 del config_to_save["calibration"]["active"] # Don't save the active state

            with open(config_path, 'w') as f:
                json.dump(config_to_save, f, indent=2)
                logger.info(f"Configurazione salvata in {config_path}")
        except Exception as e:
            logger.error(f"Errore nel salvataggio della configurazione: {e}")

    # Allow dictionary-like access, e.g., config['device'] or config['calibration.x_min']
    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)


# -----------------------------------------------------------------------------
# GestureRecognizer Class
# -----------------------------------------------------------------------------
class GestureRecognizer:
    """Classe specializzata per il riconoscimento dei gesti."""

    def __init__(self, config):
        self.config = config
        self.last_positions = {} # Store last known position for landmarks (e.g., index tip for scroll)
        self.gesture_state = {
            "drag_active": False,
            "scroll_active": False,
            "last_click_time": 0,
            "last_click_button": None,
            "scroll_history": deque(maxlen=5), # For smoothing scroll delta
            "active_gesture": None, # ('click', 'scroll', etc.)
            "last_pose": {"Left": "unknown", "Right": "unknown"}, # Track pose per hand
            "pose_stable_count": {"Left": 0, "Right": 0} # Frames the pose has been stable
        }
        self.POSE_STABILITY_THRESHOLD = 3 # Number of consecutive frames for a pose to be stable

    def _calculate_distance(self, p1, p2):
        """Calculates 2D Euclidean distance between two landmarks (normalized coords)."""
        if p1 is None or p2 is None or not hasattr(p1, 'x') or not hasattr(p2, 'x'):
            return float('inf')
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def check_thumb_index_click(self, hand_landmarks, handedness):
        """Rileva il gesto di click (indice e pollice che si toccano), solo per la mano SINISTRA."""
        if handedness != "Left" or hand_landmarks is None:
            return None
        # Ignore click if scrolling is active to prevent conflicts
        if self.gesture_state["scroll_active"]:
            return None

        try:
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP] # ID 4
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP] # ID 8
        except (IndexError, TypeError):
             # logger.warning("Landmark indice/pollice non trovato per il click.")
             return None

        distance = self._calculate_distance(thumb_tip, index_tip)
        current_time = time.time()
        click_sensitivity = self.config.get("gesture_sensitivity", 0.07)
        click_detected = distance < click_sensitivity

        gesture = None
        if click_detected:
            # Only register a new click if cooldown has passed
            cooldown = self.config.get("click_cooldown", 0.4)
            if (current_time - self.gesture_state["last_click_time"]) > cooldown:
                # Check for double click (within double_click_time)
                double_click_time_limit = self.config.get("gesture_settings.double_click_time", 0.35)
                time_diff_since_last = current_time - self.gesture_state["last_click_time"]

                if time_diff_since_last < double_click_time_limit and self.gesture_state["last_click_button"] == Button.left:
                    gesture = "double_click"
                    logger.debug(f"Double click gesture detected (dist: {distance:.3f})")
                else:
                    gesture = "click"
                    logger.debug(f"Single click gesture detected (dist: {distance:.3f})")

                # Update state ONLY if a click/double_click is confirmed
                self.gesture_state["last_click_time"] = current_time
                self.gesture_state["last_click_button"] = Button.left # Assuming left click for now
                self.gesture_state["active_gesture"] = "click" # Mark as active click/double_click

        elif self.gesture_state["active_gesture"] == "click":
             # Reset active gesture state if fingers move apart after a click was active
             self.gesture_state["active_gesture"] = None

        return gesture


    def check_scroll_gesture(self, hand_landmarks, handedness):
        """
        Rileva il gesto di scorrimento: mano SINISTRA, indice e medio estesi verticalmente,
        vicini tra loro, altre dita piegate.
        """
        if handedness != "Left" or hand_landmarks is None:
            return None
        # Ignore scroll if clicking is active
        if self.gesture_state["active_gesture"] == "click":
            return None

        try:
            lm = hand_landmarks.landmark
            index_tip = lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP] # 8
            middle_tip = lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP] # 12
            index_mcp = lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP] # 5
            middle_mcp = lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP] # 9
            ring_tip = lm[mp.solutions.hands.HandLandmark.RING_FINGER_TIP] # 16
            pinky_tip = lm[mp.solutions.hands.HandLandmark.PINKY_TIP] # 20
        except (IndexError, TypeError):
            # logger.warning("Landmark mancanti per il rilevamento dello scroll.")
            return None

        # --- Check pose conditions ---
        # 1. Index/Middle extended vertically (tip Y significantly smaller than MCP Y)
        y_extension_threshold = 0.05 # How much higher tip needs to be than MCP
        index_extended = index_tip.y < index_mcp.y - y_extension_threshold
        middle_extended = middle_tip.y < middle_mcp.y - y_extension_threshold

        # 2. Index/Middle close horizontally
        x_closeness_threshold = 0.08 # Max horizontal distance between tips
        fingers_close = abs(index_tip.x - middle_tip.x) < x_closeness_threshold

        # 3. Ring/Pinky folded (tip Y larger than relevant MCP Y)
        y_fold_threshold = 0.01 # Tolerance for folding
        ring_folded = ring_tip.y > middle_mcp.y + y_fold_threshold # Compare to middle MCP for stability
        pinky_folded = pinky_tip.y > middle_mcp.y + y_fold_threshold

        is_scroll_pose = index_extended and middle_extended and fingers_close and ring_folded and pinky_folded

        # --- Calculate scroll delta ---
        scroll_delta = None
        index_tip_id = 8 # Use landmark ID as a unique key for position tracking

        if is_scroll_pose:
            # Start or continue scroll state
            if not self.gesture_state["scroll_active"]:
                self.gesture_state["scroll_active"] = True
                self.gesture_state["scroll_history"].clear()
                self.gesture_state["active_gesture"] = "scroll"
                # Initialize position when scroll starts
                self.last_positions[index_tip_id] = index_tip.y
                logger.debug("Scroll gesture started")

            # Calculate delta only if scroll is active and we have a previous position
            if index_tip_id in self.last_positions:
                prev_y = self.last_positions[index_tip_id]
                curr_y = index_tip.y
                # Raw delta: Positive means finger moved down screen (scroll down)
                raw_delta_y = curr_y - prev_y

                # Add to history only if movement is significant enough (avoid noise)
                movement_threshold = 0.002 # Small normalized movement needed
                if abs(raw_delta_y) > movement_threshold:
                    # Scale delta by sensitivity *before* smoothing
                    scaled_delta = raw_delta_y * self.config.get("gesture_settings.scroll_sensitivity", 10)
                    self.gesture_state["scroll_history"].append(scaled_delta)
                    # Update last position only when significant movement is detected & added
                    self.last_positions[index_tip_id] = curr_y

                # Calculate smoothed delta if history is populated
                if len(self.gesture_state["scroll_history"]) > 0:
                    smooth_delta = sum(self.gesture_state["scroll_history"]) / len(self.gesture_state["scroll_history"])
                    # Apply scroll only if smoothed delta is significant (avoids tiny scrolls)
                    apply_threshold = 0.1 # Threshold after scaling/smoothing
                    if abs(smooth_delta) > apply_threshold:
                         scroll_delta = smooth_delta # This value will be used by HandPal
                         # No need to update last_positions here, it's updated when movement added to history

            # If scroll just started or no movement, ensure last_position is current
            elif index_tip_id not in self.last_positions and self.gesture_state["scroll_active"]:
                 self.last_positions[index_tip_id] = index_tip.y


        else:
            # If pose is lost, deactivate scroll state
            if self.gesture_state["scroll_active"]:
                self.gesture_state["scroll_active"] = False
                self.gesture_state["scroll_history"].clear()
                if index_tip_id in self.last_positions:
                    del self.last_positions[index_tip_id]
                if self.gesture_state["active_gesture"] == "scroll":
                    self.gesture_state["active_gesture"] = None
                logger.debug("Scroll gesture ended")

        # Return the smoothed, scaled delta (or None if no scroll action)
        return scroll_delta

    def detect_hand_pose(self, hand_landmarks, handedness):
        """Analizza la posa complessiva della mano (semplificato). Returns a string like 'pointing', 'open_palm', etc."""
        if hand_landmarks is None or handedness not in ["Left", "Right"]:
            return "unknown"

        try:
            lm = hand_landmarks.landmark
            wrist = lm[mp.solutions.hands.HandLandmark.WRIST]
            thumb_tip = lm[mp.solutions.hands.HandLandmark.THUMB_TIP]
            index_tip = lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = lm[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = lm[mp.solutions.hands.HandLandmark.PINKY_TIP]

            index_mcp = lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
            middle_mcp = lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
            ring_mcp = lm[mp.solutions.hands.HandLandmark.RING_FINGER_MCP]
            pinky_mcp = lm[mp.solutions.hands.HandLandmark.PINKY_MCP]
        except (IndexError, TypeError):
             # logger.warning(f"Landmark mancanti per il rilevamento della posa ({handedness}).")
             return "unknown"

        # Check finger extensions relative to MCP joints (vertical check, smaller Y is higher)
        y_extension_threshold = 0.03 # How much higher tip needs to be than MCP
        thumb_extended = thumb_tip.y < wrist.y - y_extension_threshold # Compare thumb to wrist
        index_extended = index_tip.y < index_mcp.y - y_extension_threshold
        middle_extended = middle_tip.y < middle_mcp.y - y_extension_threshold
        ring_extended = ring_tip.y < ring_mcp.y - y_extension_threshold
        pinky_extended = pinky_tip.y < pinky_mcp.y - y_extension_threshold

        num_extended = sum([index_extended, middle_extended, ring_extended, pinky_extended])

        # --- Determine Pose ---
        current_pose = "unknown"
        # Check specific poses first
        if handedness == "Left" and self.gesture_state["scroll_active"]:
            current_pose = "scroll_pose" # Prioritize active scroll gesture
        elif index_extended and num_extended == 1:
             current_pose = "pointing"
        elif index_extended and middle_extended and num_extended == 2:
             current_pose = "two_fingers" # Peace sign / Victory
        elif num_extended >= 4:
             current_pose = "open_palm"
        elif num_extended == 0 and not thumb_extended: # Check thumb folded too
             current_pose = "closed_fist"
        # Add more specific poses here if needed (e.g., thumbs up)

        # --- Pose Stability Check ---
        last_pose = self.gesture_state["last_pose"].get(handedness, "unknown")
        stable_count = self.gesture_state["pose_stable_count"].get(handedness, 0)

        if current_pose == last_pose and current_pose != "unknown":
            stable_count += 1
        else:
            stable_count = 0 # Reset count if pose changed or is unknown

        self.gesture_state["last_pose"][handedness] = current_pose
        self.gesture_state["pose_stable_count"][handedness] = stable_count

        # Return the pose only if it's stable or if it's a special active gesture
        if stable_count >= self.POSE_STABILITY_THRESHOLD or current_pose == "scroll_pose":
            return current_pose
        else:
            # Return the previous stable pose if current one isn't stable yet?
            # Or return current unstable? Let's return current for responsiveness in debug.
            # Could return f"unstable_{current_pose}"
            return current_pose # Return current pose even if unstable

# -----------------------------------------------------------------------------
# MotionSmoother Class (Exponential Moving Average)
# -----------------------------------------------------------------------------
class MotionSmoother:
    """Applica smoothing al movimento usando Exponential Moving Average (EMA)."""

    def __init__(self, config):
        self.config = config
        self.last_smoothed_position = None
        self._update_alpha() # Calculate initial alpha
        # Inactivity zone based on normalized screen distance (squared for efficiency)
        self.inactivity_zone_size_sq = self.config.get("inactivity_zone", 0.015)**2

    def _update_alpha(self):
        """Calculate EMA alpha based on smoothing_factor."""
        # Clamp factor between 0.01 (heavy smoothing) and 0.99 (minimal smoothing)
        smoothing_factor = max(0.01, min(0.99, self.config.get("smoothing_factor", 0.7)))
        # Alpha = 1 - smoothing_factor (higher factor -> lower alpha -> more smoothing)
        self.alpha = 1.0 - smoothing_factor
        logger.debug(f"MotionSmoother alpha updated to: {self.alpha:.3f} (factor: {smoothing_factor:.2f})")

    def update(self, target_x, target_y, screen_width, screen_height):
        """Aggiorna e calcola la nuova posizione smussata (pixel coords)."""
        current_target_pixels = (target_x, target_y)

        if self.last_smoothed_position is None:
            self.last_smoothed_position = current_target_pixels
            return current_target_pixels # Return immediately on first update

        # --- Inactivity Check ---
        # Calculate normalized distance between current target and last smoothed position
        last_smoothed_norm_x = self.last_smoothed_position[0] / screen_width
        last_smoothed_norm_y = self.last_smoothed_position[1] / screen_height
        target_norm_x = target_x / screen_width
        target_norm_y = target_y / screen_height

        dx_norm = target_norm_x - last_smoothed_norm_x
        dy_norm = target_norm_y - last_smoothed_norm_y
        dist_sq_norm = dx_norm**2 + dy_norm**2

        # If target hasn't moved significantly from the last *smoothed* point (normalized check)
        if dist_sq_norm < self.inactivity_zone_size_sq:
             # Return the previous smoothed position to prevent micro-jitters
             return self.last_smoothed_position

        # --- Apply EMA Filter (on pixel coordinates) ---
        # Ensure alpha is up-to-date in case config changed
        self._update_alpha()

        smooth_x = int(self.alpha * target_x + (1 - self.alpha) * self.last_smoothed_position[0])
        smooth_y = int(self.alpha * target_y + (1 - self.alpha) * self.last_smoothed_position[1])

        self.last_smoothed_position = (smooth_x, smooth_y)
        return smooth_x, smooth_y

    def reset(self):
        """Resets the smoother's state."""
        self.last_smoothed_position = None
        logger.debug("MotionSmoother reset.")

# -----------------------------------------------------------------------------
# Detection Thread
# -----------------------------------------------------------------------------
class DetectionThread(threading.Thread):
    """Thread dedicato alla cattura e all'elaborazione MediaPipe."""
    def __init__(self, config, cap, hands_instance, data_queue, stop_event):
        super().__init__(daemon=True, name="DetectionThread")
        self.config = config
        self.cap = cap
        self.hands = hands_instance
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.process_width = config.get("process_width", 640)
        self.process_height = config.get("process_height", 360)
        self.flip_camera = config.get("flip_camera", True)
        logger.info(f"DetectionThread initializzato (process res: {self.process_width}x{self.process_height}, flip: {self.flip_camera})")

    def run(self):
        logger.info("DetectionThread avviato")
        frame_count = 0
        start_time = time.perf_counter()

        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                logger.error("Webcam non disponibile nel DetectionThread.")
                time.sleep(0.5) # Wait before retrying
                continue

            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning("Frame non valido ricevuto dalla webcam nel DetectionThread.")
                    time.sleep(0.05) # Short pause if frame read fails
                    continue

                # --- Frame Preparation ---
                if self.flip_camera:
                    frame = cv.flip(frame, 1) # Flip horizontally

                # Create a copy for processing (resized)
                process_frame = cv.resize(frame, (self.process_width, self.process_height), interpolation=cv.INTER_LINEAR)
                # Convert the processing frame to RGB for MediaPipe
                rgb_frame = cv.cvtColor(process_frame, cv.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False # Performance optimization

                # --- Perform Detection ---
                results = self.hands.process(rgb_frame)
                rgb_frame.flags.writeable = True # Re-enable writeable if needed later

                # --- Output ---
                # Put the *original* (flipped) frame and detection results onto the queue.
                # The main thread will handle drawing on this original frame.
                output_data = (frame, results)

                # Use try_put with a timeout or check queue size to avoid blocking indefinitely
                # if the main thread is slow. This strategy drops older frames if necessary.
                try:
                    self.data_queue.put(output_data, block=False) # Non-blocking put
                except queue.Full:
                    # Queue is full, meaning main thread is lagging. Discard oldest and put newest.
                    try:
                        self.data_queue.get_nowait() # Discard the oldest item
                        self.data_queue.put_nowait(output_data) # Put the latest item
                        # logger.debug("Detection queue full, dropped oldest frame.")
                    except queue.Empty:
                        pass # Should not happen if Full was raised, but handle anyway
                    except queue.Full:
                        # logger.warning("Detection queue still full after trying to discard, skipping frame.")
                        pass # Skip if still full after trying to clear

                frame_count += 1

            except Exception as e:
                logger.exception(f"Errore nel loop del DetectionThread: {e}")
                time.sleep(0.1) # Prevent rapid error loops

        # --- Thread Cleanup ---
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        logger.info(f"DetectionThread terminato. Elaborati {frame_count} frames in {elapsed:.2f}s (Avg FPS: {avg_fps:.1f})")
        # No need to close hands or release cap here, main thread's stop() handles it


# -----------------------------------------------------------------------------
# HandPal Class (Main Application Logic)
# -----------------------------------------------------------------------------
class HandPal:
    """Classe principale che gestisce l'applicazione HandPal."""

    def __init__(self, config):
        self.config = config
        self.mouse = Controller()
        self.gesture_recognizer = GestureRecognizer(config)

        # Get screen size (needed for mapping and smoother)
        try:
            root = tk.Tk()
            root.withdraw() # Hide the main window
            self.screen_size = (root.winfo_screenwidth(), root.winfo_screenheight())
            root.destroy()
            logger.info(f"Dimensioni schermo rilevate: {self.screen_size[0]}x{self.screen_size[1]}")
        except tk.TclError as e:
             logger.warning(f"Impossibile inizializzare Tkinter per ottenere dimensioni schermo ({e}). Usando default (1920x1080).")
             self.screen_size = (1920, 1080)

        self.motion_smoother = MotionSmoother(config)

        # Threading and State
        self.running = False
        self.stop_event = threading.Event()
        self.detection_thread = None
        self.data_queue = queue.Queue(maxsize=2) # Queue for (original_frame, results)

        # Resources
        self.cap = None
        self.mp_hands = mp.solutions.hands
        self.hands_instance = None # MediaPipe Hands object
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Tracking & State Variables
        self.last_cursor_pos = None
        self.last_right_hand_pos_norm = None # Store normalized (0-1) position of right hand tip
        self.tracking_enabled = True # Controls if mouse movement is active
        self.debug_mode = False

        # Calibration State
        self.calibration_active = False
        self.calibration_points = [] # Stores normalized points collected
        self.calibration_step = 0
        self.calibration_corners = [ # User-friendly names for corners
            "in alto a sinistra",
            "in alto a destra",
            "in basso a destra",
            "in basso a sinistra"
        ]

        # Debug Info Dictionary
        self.debug_values = {
            "main_fps": 0.0,
            "detection_fps": 0.0, # Note: Hard to measure accurately here, best estimate from thread log
            "last_action_time": time.time(), # Timestamp of last gesture action
            "cursor_history": deque(maxlen=50), # Store recent screen coords for trace
            "screen_mapping": {"raw_norm": (0, 0), "calib_norm": (0,0), "mapped_px": (0, 0), "smoothed_px": (0,0)},
            "left_hand_pose": "unknown",
            "right_hand_pose": "unknown",
            "active_gesture": "None",
            "queue_size": 0
        }
        self.fps_stats = deque(maxlen=60) # Track main loop frame durations for FPS calculation

    def start(self):
        """Avvia l'applicazione, la webcam e il thread di rilevamento."""
        if self.running:
            logger.warning("HandPal è già in esecuzione.")
            return True

        logger.info("Avvio HandPal...")
        self.stop_event.clear() # Reset stop event in case of restart

        try:
            # --- Initialize Webcam ---
            self.cap = cv.VideoCapture(self.config["device"], cv.CAP_DSHOW) # Use DirectShow backend if available
            if not self.cap.isOpened():
                logger.warning(f"Impossibile aprire webcam {self.config['device']} con DShow, ritento senza backend.")
                self.cap = cv.VideoCapture(self.config["device"])

            if not self.cap.isOpened():
                logger.error(f"Impossibile aprire la webcam (dispositivo {self.config['device']}). Assicurarsi che non sia usata da altre applicazioni.")
                return False

            # Set desired properties (best effort)
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.config["width"])
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.config["height"])
            self.cap.set(cv.CAP_PROP_FPS, self.config.get("max_fps", 60)) # Request desired FPS
             # Reduce buffer size for lower latency (if supported)
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

            actual_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv.CAP_PROP_FPS)
            logger.info(f"Webcam richiesta {self.config['width']}x{self.config['height']}, ottenuta {actual_width}x{actual_height} @ {actual_fps:.1f} FPS (Buffer: {self.cap.get(cv.CAP_PROP_BUFFERSIZE)})")
            # Update config with actual dimensions if different? Optional.
            # self.config['width'] = actual_width
            # self.config['height'] = actual_height

            # --- Initialize MediaPipe Hands ---
            logger.debug("Inizializzazione MediaPipe Hands...")
            self.hands_instance = self.mp_hands.Hands(
                static_image_mode=self.config["use_static_image_mode"],
                max_num_hands=2,
                min_detection_confidence=self.config["min_detection_confidence"],
                min_tracking_confidence=self.config["min_tracking_confidence"]
            )
            logger.debug("MediaPipe Hands inizializzato.")


            # --- Start Detection Thread ---
            if not self.hands_instance:
                 raise Exception("MediaPipe Hands non inizializzato correttamente.")

            logger.debug("Avvio del DetectionThread...")
            self.detection_thread = DetectionThread(
                self.config, self.cap, self.hands_instance, self.data_queue, self.stop_event
            )
            self.detection_thread.start()
            logger.debug("DetectionThread avviato.")


            self.running = True
            logger.info("HandPal avviato con successo! Il loop principale inizierà.")
            return True

        except Exception as e:
            logger.exception(f"Errore catastrofico durante l'avvio: {e}")
            self.stop() # Ensure cleanup if start fails partway through
            return False

    def stop(self):
        """Ferma l'applicazione, i thread e rilascia le risorse."""
        if not self.running and not self.stop_event.is_set():
             logger.debug("Stop chiamato ma HandPal non era in esecuzione. Assicurando che stop_event sia impostato.")
             self.stop_event.set()
             # Ensure windows closed if stop called after main loop but before full exit
             cv.destroyAllWindows()
             return

        if not self.running:
            logger.debug("Stop chiamato ma HandPal è già fermo o in fase di arresto.")
            return

        logger.info("Fermando HandPal...")
        self.running = False # Signal main loop to stop
        self.stop_event.set() # Signal detection thread to stop

        # Wait for detection thread to finish
        if self.detection_thread and self.detection_thread.is_alive():
            logger.debug("In attesa della terminazione del DetectionThread...")
            self.detection_thread.join(timeout=2.0) # Wait max 2 seconds
            if self.detection_thread.is_alive():
                 logger.warning("DetectionThread non si è fermato entro il timeout.")
            else:
                 logger.debug("DetectionThread terminato.")

        # Release MediaPipe Hands instance
        # Note: Newer MediaPipe versions might not have/need an explicit close().
        # Relying on thread exit and GC might be sufficient.
        if hasattr(self.hands_instance, 'close') and callable(self.hands_instance.close):
             try:
                 logger.debug("Chiusura istanza MediaPipe Hands...")
                 self.hands_instance.close()
                 logger.debug("Istanza MediaPipe Hands chiusa.")
             except Exception as e:
                 logger.error(f"Errore durante la chiusura di MediaPipe Hands: {e}")
        self.hands_instance = None # Allow GC

        # Release camera resource
        if self.cap and self.cap.isOpened():
            logger.debug("Rilascio webcam...")
            self.cap.release()
            logger.debug("Webcam rilasciata.")
            self.cap = None

        # Close OpenCV windows
        cv.destroyAllWindows()
        logger.info("Finestre OpenCV chiuse.")
        logger.info("HandPal fermato.")

    def map_to_screen(self, x_norm, y_norm):
        """Mappa le coordinate normalizzate della mano (0-1 da frame) allo schermo (pixel)."""
        screen_w, screen_h = self.screen_size

        if self.debug_mode:
            self.debug_values["screen_mapping"]["raw_norm"] = (f"{x_norm:.3f}", f"{y_norm:.3f}")

        x_calib = x_norm
        y_calib = y_norm

        # Apply calibration if enabled
        if self.config.get("calibration.enabled", True):
            x_min = self.config.get("calibration.x_min", 0.15)
            x_max = self.config.get("calibration.x_max", 0.85)
            y_min = self.config.get("calibration.y_min", 0.15)
            y_max = self.config.get("calibration.y_max", 0.85)

            x_range = x_max - x_min
            y_range = y_max - y_min

            # Avoid division by zero or tiny range
            if x_range < 0.01 or y_range < 0.01:
                 if self.debug_mode: logger.warning("Range di calibrazione troppo piccolo, calibrazione ignorata per questo frame.")
                 # Fall through to use uncalibrated x_norm, y_norm
            else:
                # Clamp input coordinates to calibration bounds *before* normalizing within the range
                x_clamped = max(x_min, min(x_norm, x_max))
                y_clamped = max(y_min, min(y_norm, y_max))
                # Normalize within the calibrated range to 0-1
                x_calib = (x_clamped - x_min) / x_range
                y_calib = (y_clamped - y_min) / y_range

        if self.debug_mode:
            self.debug_values["screen_mapping"]["calib_norm"] = (f"{x_calib:.3f}", f"{y_calib:.3f}")


        # Apply screen margin expansion (expand the 0-1 calibrated range)
        margin = self.config.get("calibration.screen_margin", 0.1)
        # Scale the 0-1 range to cover 1.0 + 2*margin, then shift left by margin
        x_expanded = x_calib * (1.0 + 2.0 * margin) - margin
        y_expanded = y_calib * (1.0 + 2.0 * margin) - margin

        # Map expanded normalized coordinates to screen pixel coordinates
        screen_x = int(x_expanded * screen_w)
        screen_y = int(y_expanded * screen_h)

        # Clamp final coordinates to stay within screen boundaries
        screen_x = max(0, min(screen_x, screen_w - 1))
        screen_y = max(0, min(screen_y, screen_h - 1))

        if self.debug_mode:
            self.debug_values["screen_mapping"]["mapped_px"] = (screen_x, screen_y)

        return screen_x, screen_y

    def process_results(self, results):
        """Processa i risultati di MediaPipe per controllare il mouse e rilevare gesti."""
        left_hand_landmarks = None
        right_hand_landmarks = None
        left_handedness_label = "Unknown"
        right_handedness_label = "Unknown"
        self.last_right_hand_pos_norm = None # Reset before processing new results

        # --- Assign Landmarks to Left/Right Hands ---
        if results and results.multi_hand_landmarks and results.multi_handedness:
            if len(results.multi_hand_landmarks) != len(results.multi_handedness):
                 logger.warning("Mismatch tra numero di mani rilevate e info handedness.")
            else:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    try:
                        handedness_info = results.multi_handedness[i].classification[0]
                        label = handedness_info.label # Should be 'Left' or 'Right'
                        # score = handedness_info.score # Could use score threshold if needed
                    except (IndexError, AttributeError) as e:
                        logger.warning(f"Errore nell'accesso a handedness per la mano {i}: {e}")
                        continue # Skip this hand if handedness is broken

                    if label == "Right":
                        right_hand_landmarks = hand_landmarks
                        right_handedness_label = label
                    elif label == "Left":
                        left_hand_landmarks = hand_landmarks
                        left_handedness_label = label

        # --- Right Hand: Cursor Movement ---
        if right_hand_landmarks:
            try:
                # Use Index finger tip (landmark 8) for cursor control
                index_tip = right_hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                # Store normalized position (relative to processed frame dimensions)
                # These coords are used for calibration and mapping
                self.last_right_hand_pos_norm = (index_tip.x, index_tip.y)

                # Only move mouse if tracking is enabled and not calibrating
                if self.tracking_enabled and not self.calibration_active:
                    # Map normalized coords to screen pixels
                    target_x, target_y = self.map_to_screen(index_tip.x, index_tip.y)

                    # Apply smoothing
                    smooth_x, smooth_y = self.motion_smoother.update(target_x, target_y, self.screen_size[0], self.screen_size[1])

                    # Move mouse only if the smoothed position actually changed
                    if self.last_cursor_pos is None or smooth_x != self.last_cursor_pos[0] or smooth_y != self.last_cursor_pos[1]:
                        self.mouse.position = (smooth_x, smooth_y)
                        self.last_cursor_pos = (smooth_x, smooth_y)
                        if self.debug_mode:
                             self.debug_values["screen_mapping"]["smoothed_px"] = (smooth_x, smooth_y)
                             # Add to history *after* smoothing and potential inactivity stop
                             self.debug_values["cursor_history"].append(self.last_cursor_pos)

            except IndexError:
                 logger.warning("Landmark indice destro (8) non trovato per controllo cursore.")
                 self.motion_smoother.reset() # Reset smoother if tracking point lost
            except Exception as e:
                logger.error(f"Errore nell'aggiornamento del cursore: {e}")
                self.motion_smoother.reset()
        else:
            # No right hand detected, reset smoother
            self.motion_smoother.reset()
            # Clear right hand position if lost
            self.last_right_hand_pos_norm = None

        # --- Left Hand: Gestures (Click, Scroll) ---
        action_performed = False # Flag to track if a gesture action happened this frame
        if left_hand_landmarks and not self.calibration_active:
            # Check for scroll first (often involves a distinct pose)
            scroll_amount_smoothed = self.gesture_recognizer.check_scroll_gesture(left_hand_landmarks, "Left")

            if scroll_amount_smoothed is not None:
                # Convert scroll delta to integer scroll 'clicks' for pynput
                # Recognizer delta: positive = finger down = scroll down (negative value for pynput)
                scroll_clicks = int(scroll_amount_smoothed * -1) # Invert sign and convert to int
                if scroll_clicks != 0:
                    try:
                        self.mouse.scroll(0, scroll_clicks) # dx=0, dy=scroll_clicks
                        if self.debug_mode:
                            self.debug_values["last_action_time"] = time.time()
                            self.debug_values["active_gesture"] = f"Scroll ({scroll_clicks})"
                        action_performed = True
                        logger.debug(f"Scroll eseguito: {scroll_clicks} units")
                    except Exception as e:
                         logger.error(f"Errore durante l'invio dello scroll: {e}")

            # If not scrolling, check for click gesture
            if not action_performed:
                click_gesture = self.gesture_recognizer.check_thumb_index_click(left_hand_landmarks, "Left")
                if click_gesture == "click":
                    try:
                        self.mouse.click(Button.left, 1) # Single click
                        logger.info("Click sinistro eseguito")
                        if self.debug_mode:
                            self.debug_values["last_action_time"] = time.time()
                            self.debug_values["active_gesture"] = "Click"
                        action_performed = True
                    except Exception as e:
                         logger.error(f"Errore durante l'invio del click: {e}")
                elif click_gesture == "double_click":
                    try:
                        self.mouse.click(Button.left, 2) # Double click
                        logger.info("Doppio click sinistro eseguito")
                        if self.debug_mode:
                            self.debug_values["last_action_time"] = time.time()
                            self.debug_values["active_gesture"] = "Double Click"
                        action_performed = True
                    except Exception as e:
                         logger.error(f"Errore durante l'invio del doppio click: {e}")

        # --- Update Debug Info (Poses, Active Gesture if no action) ---
        if self.debug_mode:
             # Update poses regardless of actions
             self.debug_values["left_hand_pose"] = self.gesture_recognizer.detect_hand_pose(left_hand_landmarks, "Left")
             self.debug_values["right_hand_pose"] = self.gesture_recognizer.detect_hand_pose(right_hand_landmarks, "Right")

             # If no specific action was triggered, reflect the general gesture state
             if not action_performed:
                 current_internal_gesture = self.gesture_recognizer.gesture_state.get("active_gesture")
                 self.debug_values["active_gesture"] = str(current_internal_gesture) if current_internal_gesture else "None"


    def draw_landmarks(self, frame, multi_hand_landmarks):
        """Disegna i landmark delle mani e le connessioni sul frame."""
        if not multi_hand_landmarks:
            return frame # Return original frame if no hands

        annotated_frame = frame # Draw directly on the frame passed

        for hand_landmarks in multi_hand_landmarks:
            # Use default styles for landmarks and connections
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=hand_landmarks,
                connections=self.mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style())

        return annotated_frame # Return frame with drawings

    def draw_overlays(self, frame, results):
        """Disegna testi informativi, debug, istruzioni di calibrazione e landmark."""
        h, w = frame.shape[:2]
        overlay_color = (255, 255, 255) # White
        bg_color = (0, 0, 0) # Black
        accent_color = (0, 255, 255) # Cyan/Yellow
        error_color = (0, 0, 255) # Red
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale_small = 0.4
        font_scale_medium = 0.5
        line_type = 1
        thickness_normal = 1
        thickness_bold = 2

        # Draw landmarks first (if any)
        if results and results.multi_hand_landmarks:
            frame = self.draw_landmarks(frame, results.multi_hand_landmarks)

        # --- Calibration Overlay ---
        if self.calibration_active:
            # Semi-transparent background for instructions at the top
            overlay_bg = frame.copy()
            cv.rectangle(overlay_bg, (0, 0), (w, 100), bg_color, -1)
            alpha = 0.7
            frame = cv.addWeighted(overlay_bg, alpha, frame, 1 - alpha, 0)

            # Instructions Text
            cv.putText(frame, f"CALIBRAZIONE ({self.calibration_step+1}/4): Muovi mano DESTRA",
                       (10, 25), font, font_scale_medium, overlay_color, thickness_normal, line_type)
            cv.putText(frame, f"Punta indice verso angolo {self.calibration_corners[self.calibration_step]}",
                       (10, 50), font, font_scale_medium, overlay_color, thickness_normal, line_type)
            cv.putText(frame, "Premi SPAZIO per confermare",
                       (10, 75), font, font_scale_medium, accent_color, thickness_bold, line_type)
            cv.putText(frame, "(Premi ESC per annullare)",
                       (w - 150, 20), font, font_scale_small, overlay_color, thickness_normal, line_type)


            # Draw target corner indicators on the frame
            corner_radius = 15
            inactive_color = (100, 100, 100) # Grey
            active_color = error_color # Red
            fill = -1 # Filled circle

            corners_px = [
                (corner_radius, corner_radius),               # Top-Left
                (w - corner_radius, corner_radius),          # Top-Right
                (w - corner_radius, h - corner_radius),      # Bottom-Right
                (corner_radius, h - corner_radius)           # Bottom-Left
            ]
            for i, corner_px in enumerate(corners_px):
                 color = active_color if self.calibration_step == i else inactive_color
                 cv.circle(frame, corner_px, corner_radius, color, fill)
                 cv.circle(frame, corner_px, corner_radius, overlay_color, 1) # White border


            # Draw green circle representing current hand position during calibration
            if self.last_right_hand_pos_norm:
                tip_x_frame = int(self.last_right_hand_pos_norm[0] * w)
                tip_y_frame = int(self.last_right_hand_pos_norm[1] * h)
                cv.circle(frame, (tip_x_frame, tip_y_frame), 10, (0, 255, 0), -1) # Bright Green filled circle


        # --- Normal Operation / Debug Overlay ---
        else:
            # Basic instructions at the bottom
            info_text = "C: Calibra | D: Debug | Q: Esci"
            cv.putText(frame, info_text, (10, h - 10),
                      font, font_scale_small, overlay_color, line_type)

            # FPS Display Top Right
            fps = self.debug_values["main_fps"]
            cv.putText(frame, f"FPS: {fps:.1f}", (w - 80, 20),
                      font, font_scale_medium, overlay_color, line_type)

            # --- Debug Mode Panel (Bottom) ---
            if self.debug_mode:
                debug_panel_height = 110 # Adjust height based on content
                # Semi-transparent background for debug info
                overlay_bg = frame.copy()
                cv.rectangle(overlay_bg, (0, h - debug_panel_height), (w, h), bg_color, -1)
                alpha = 0.7
                frame = cv.addWeighted(overlay_bg, alpha, frame, 1 - alpha, 0)

                # Starting Y position for text inside the panel
                y_pos = h - debug_panel_height + 15

                # Line 1: Poses & Gesture State
                debug_text1 = f"L:{self.debug_values['left_hand_pose']} R:{self.debug_values['right_hand_pose']} Gest:{self.debug_values['active_gesture']}"
                cv.putText(frame, debug_text1, (10, y_pos), font, font_scale_small, overlay_color, line_type)
                y_pos += 18

                # Line 2: Screen Mapping Coordinates (Raw Norm -> Calib Norm -> Mapped Px -> Smoothed Px)
                map_info = self.debug_values['screen_mapping']
                debug_text2 = f"Map: Raw{map_info['raw_norm']} Cal{map_info['calib_norm']} -> MapPx{map_info['mapped_px']} -> SmoothPx{map_info['smoothed_px']}"
                cv.putText(frame, debug_text2, (10, y_pos), font, font_scale_small, overlay_color, line_type)
                y_pos += 18

                # Line 3: Calibration Settings
                calib = self.config['calibration']
                calib_en = calib.get('enabled', False)
                calib_text = f"Calib X[{calib.get('x_min',0):.2f}-{calib.get('x_max',1):.2f}] Y[{calib.get('y_min',0):.2f}-{calib.get('y_max',1):.2f}] En:{calib_en}"
                cv.putText(frame, calib_text, (10, y_pos), font, font_scale_small, overlay_color, line_type)
                y_pos += 18

                # Line 4: Timings & Queue Size
                time_since_action = time.time() - self.debug_values["last_action_time"]
                q_size = self.debug_values["queue_size"]
                timing_text = f"LastAct:{time_since_action:.1f}s ago | QSize:{q_size}"
                cv.putText(frame, timing_text, (10, y_pos), font, font_scale_small, overlay_color, line_type)
                y_pos += 18

                # Line 5: Key Parameters (Sensitivity, Cooldown, Smoothing, Scroll Sens)
                g_sens = self.config.get('gesture_sensitivity', 0.0)
                c_cool = self.config.get('click_cooldown', 0.0)
                s_fact = self.config.get('smoothing_factor', 0.0)
                sc_sens = self.config.get('gesture_settings.scroll_sensitivity', 0)
                params_text = f"Param: G-Sens:{g_sens:.2f} C-Cool:{c_cool:.1f} SmoothF:{s_fact:.1f} ScrSens:{sc_sens}"
                cv.putText(frame, params_text, (10, y_pos), font, font_scale_small, overlay_color, line_type)
                # y_pos += 18 # Increment if adding more lines

                # Draw cursor trace (on the frame, not just overlay)
                if len(self.debug_values["cursor_history"]) > 1:
                    trace_color = accent_color # Use accent color for trace
                    # History contains screen pixel coordinates
                    points_screen = np.array(list(self.debug_values["cursor_history"]), dtype=np.int32)

                    # Need to scale screen coords back to frame coords for drawing
                    # Avoid division by zero if screen size is somehow invalid
                    if self.screen_size[0] > 0 and self.screen_size[1] > 0:
                        points_frame = points_screen.copy()
                        points_frame[:, 0] = points_frame[:, 0] * w // self.screen_size[0]
                        points_frame[:, 1] = points_frame[:, 1] * h // self.screen_size[1]
                        # Clamp points to frame boundaries just in case
                        points_frame[:, 0] = np.clip(points_frame[:, 0], 0, w - 1)
                        points_frame[:, 1] = np.clip(points_frame[:, 1], 0, h - 1)

                        cv.polylines(frame, [points_frame], isClosed=False, color=trace_color, thickness=1)
                    else:
                         logger.warning("Dimensione schermo non valida per il disegno della traccia cursore.")


        return frame # Return the frame with all overlays drawn


    def main_loop(self):
        """Loop principale dell'applicazione: gestisce UI, input, consuma dati dal thread."""
        logger.info("Main loop avviato.")
        target_frame_time = 1.0 / self.config.get("max_fps", 60) # Target time per frame (for potential sleeps)
        last_loop_time = time.perf_counter()

        # Create the display window beforehand using WINDOW_NORMAL (allows resizing)
        window_name = "HandPal"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        logger.debug(f"Finestra OpenCV '{window_name}' creata.")
        # Set initial size based on config if specified? Might be overridden by first frame.
        initial_w = self.config.get('display_width')
        initial_h = self.config.get('display_height')
        if initial_w and initial_h:
            try:
                cv.resizeWindow(window_name, initial_w, initial_h)
                logger.info(f"Dimensione finestra impostata a {initial_w}x{initial_h}")
            except Exception as e:
                logger.warning(f"Impossibile impostare dimensione iniziale finestra: {e}")


        while self.running:
            loop_start_time = time.perf_counter()

            # --- 1. Get Data from Detection Thread ---
            display_frame = None
            results = None
            try:
                # Get data with a timeout to prevent blocking indefinitely if detection thread hangs
                original_frame, results = self.data_queue.get(block=True, timeout=0.5) # Wait up to 0.5s
                display_frame = original_frame.copy() # Work on a copy for drawing/display
                if self.debug_mode:
                    self.debug_values["queue_size"] = self.data_queue.qsize()

            except queue.Empty:
                # No data received within timeout. Detection thread might be slow, stalled, or stopped.
                # Continue loop to handle input, maybe display the last frame or a message?
                logger.debug("Nessun dato dalla coda di rilevamento nel timeout.")
                # If we have no frame at all, we can't really do much. Wait briefly.
                if display_frame is None:
                    time.sleep(0.01)
                    continue # Skip rest of loop if no frame ever received

            except Exception as e:
                 logger.exception(f"Errore nel recupero dati dalla coda: {e}")
                 # Maybe stop the app? For now, just log and try to continue.
                 time.sleep(0.1)
                 continue

            # --- 2. Process Data and Control Mouse (only if data received) ---
            if display_frame is not None and results is not None:
                try:
                    self.process_results(results)
                except Exception as e:
                    logger.exception(f"Errore irreversibile durante process_results: {e}")
                    # Draw error message directly on frame if possible
                    cv.putText(display_frame, "ERROR PROCESSING RESULTS", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # --- 3. Draw Overlays (only if frame available) ---
            if display_frame is not None:
                try:
                    # Draw landmarks, debug info, calibration instructions etc.
                    display_frame = self.draw_overlays(display_frame, results) # Pass results for landmark drawing
                except Exception as e:
                    logger.exception(f"Errore durante il disegno degli overlay: {e}")
                    # Draw error message if possible
                    cv.putText(display_frame, "ERROR DRAWING OVERLAYS", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                # --- 4. Resize Frame for Display (if specified in config) ---
                display_w = self.config.get('display_width')
                display_h = self.config.get('display_height')
                final_display_frame = display_frame # Default to the drawn frame

                # Only resize if BOTH width and height are valid positive integers
                if display_w and display_h: # Already validated in Config to be int > 0 or None
                    try:
                        # Check if frame size already matches target to avoid unnecessary resize
                        current_h, current_w = display_frame.shape[:2]
                        if current_w != display_w or current_h != display_h:
                            final_display_frame = cv.resize(display_frame, (display_w, display_h), interpolation=cv.INTER_LINEAR)
                    except Exception as e:
                        logger.error(f"Errore durante il ridimensionamento del frame a {display_w}x{display_h}: {e}")
                        # Fallback to original frame if resize fails
                        final_display_frame = display_frame

                # --- 5. Display Frame ---
                cv.imshow(window_name, final_display_frame)

            # --- 6. Handle Keyboard Input ---
            key = cv.waitKey(1) & 0xFF # Use waitKey(1) for max responsiveness

            if key == ord('q'):
                logger.info("Comando 'q' ricevuto, fermando...")
                self.stop()
                break # Exit loop immediately

            elif key == ord('c'):
                if not self.calibration_active:
                    self.start_calibration()
                else:
                     logger.warning("Comando 'c' ignorato: calibrazione già attiva.")

            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                # Reset history and other debug values when toggling? Optional.
                if self.debug_mode:
                    self.debug_values["cursor_history"].clear()
                logger.info(f"Modalità debug: {'attivata' if self.debug_mode else 'disattivata'}")

            elif key == 27: # ESC key
                if self.calibration_active:
                    logger.info("Calibrazione annullata dall'utente (ESC).")
                    self.cancel_calibration()
                # else: # Optional: ESC could also quit the app if not calibrating
                #     logger.info("Comando ESC ricevuto, fermando...")
                #     self.stop()
                #     break

            elif key == 32: # Spacebar key
                if self.calibration_active:
                    self.process_calibration_step()
                # else: # Optional: Spacebar could toggle tracking?
                #      self.tracking_enabled = not self.tracking_enabled
                #      logger.info(f"Tracking mouse {'abilitato' if self.tracking_enabled else 'disabilitato'}")


            # --- 7. FPS Calculation & Optional Delay ---
            loop_end_time = time.perf_counter()
            elapsed_time = loop_end_time - loop_start_time
            self.fps_stats.append(elapsed_time) # Store duration of this loop iteration

            # Calculate average FPS over the deque window
            if self.fps_stats:
                avg_duration = sum(self.fps_stats) / len(self.fps_stats)
                current_fps = 1.0 / avg_duration if avg_duration > 0 else 0
                self.debug_values["main_fps"] = current_fps

            # Optional: Sleep to maintain target FPS for the main loop
            # This can reduce CPU usage if processing/display is faster than target FPS
            # sleep_time = target_frame_time - elapsed_time
            # if sleep_time > 0:
            #     time.sleep(sleep_time)

            last_loop_time = loop_end_time
        # --- End of main loop ---

        logger.info("Main loop terminato.")
        # Ensure windows are closed after loop exits (stop() should also do this)
        cv.destroyAllWindows()

    # --- Calibration Methods ---
    def start_calibration(self):
        """Avvia il processo di calibrazione guidata."""
        if self.calibration_active:
            logger.warning("Tentativo di avviare la calibrazione mentre è già attiva.")
            return

        logger.info("Avvio calibrazione...")
        self.calibration_active = True
        self.config["calibration"]["active"] = True # Update internal state in config
        self.calibration_points = [] # Reset collected points
        self.calibration_step = 0    # Start at the first corner
        self.tracking_enabled = False # Disable mouse movement during calibration
        self.motion_smoother.reset() # Reset smoother state
        print("\n--- AVVIO CALIBRAZIONE ---")
        print("Usa la mano DESTRA. Muovi la punta del dito indice verso gli angoli indicati.")
        print("Assicurati che la mano sia ben visibile.")
        print("Premi SPAZIO per registrare la posizione per ogni angolo.")
        print("Premi ESC per annullare in qualsiasi momento.")
        print(f"\nPASSO 1: Punta verso l'angolo {self.calibration_corners[self.calibration_step]} e premi SPAZIO.\n")

    def cancel_calibration(self):
        """Annulla la calibrazione in corso e ripristina lo stato precedente."""
        if not self.calibration_active:
            return
        logger.info("Calibrazione annullata.")
        self.calibration_active = False
        self.config["calibration"]["active"] = False
        self.calibration_points = []
        self.calibration_step = 0
        self.tracking_enabled = True # Re-enable mouse movement
        print("\n--- CALIBRAZIONE ANNULLATA ---")
        # Do not restore previous calibration values, just stop the process.

    def process_calibration_step(self):
        """Registra il punto di calibrazione corrente (quando SPAZIO viene premuto)."""
        if not self.calibration_active:
            return

        if self.last_right_hand_pos_norm is None:
            logger.warning("Impossibile registrare punto di calibrazione: mano destra non rilevata.")
            print("(!) ATTENZIONE: Mano destra non rilevata. Assicurati che sia visibile e riprova.")
            # Play a sound? Flash the screen?
            return # Don't proceed to next step

        # Record the *normalized* coordinates from the last frame
        current_pos_norm = self.last_right_hand_pos_norm
        self.calibration_points.append(current_pos_norm)
        corner_name = self.calibration_corners[self.calibration_step]
        logger.info(f"Punto di calibrazione {self.calibration_step + 1}/4 ({corner_name}) registrato: ({current_pos_norm[0]:.3f}, {current_pos_norm[1]:.3f})")
        print(f"-> Punto {self.calibration_step + 1}/4 ({corner_name}) registrato.")

        # Move to the next step
        self.calibration_step += 1

        if self.calibration_step >= 4:
            # All points collected, complete the calibration
            self.complete_calibration()
        else:
            # Prompt for the next corner
            next_corner_name = self.calibration_corners[self.calibration_step]
            print(f"\nPASSO {self.calibration_step + 1}: Punta verso l'angolo {next_corner_name} e premi SPAZIO.\n")

    def complete_calibration(self):
        """Calcola i limiti, salva la configurazione e termina la calibrazione."""
        logger.info("Completamento calibrazione...")
        if len(self.calibration_points) != 4:
            logger.error(f"Calibrazione fallita: numero errato di punti ({len(self.calibration_points)}). Annullamento.")
            print("\n(!) ERRORE: Numero errato di punti registrati ({len(self.calibration_points)} invece di 4). Calibrazione fallita.\n")
            self.cancel_calibration() # Cancel if wrong number of points
            return

        try:
            # Extract X and Y coordinates from the collected (normalized) points
            x_values = [p[0] for p in self.calibration_points]
            y_values = [p[1] for p in self.calibration_points]

            # Determine min/max from the collected points
            # Add/subtract a small epsilon to ensure min < max even if user points exactly at the edge?
            # No, simple min/max should be fine.
            x_min_calib = min(x_values)
            x_max_calib = max(x_values)
            y_min_calib = min(y_values)
            y_max_calib = max(y_values)

            # Basic validation of the calculated range
            min_required_range = 0.05 # Minimum acceptable range (e.g., 5% of frame width/height)
            valid_range = True
            if (x_max_calib - x_min_calib < min_required_range):
                 logger.warning(f"Range di calibrazione X rilevato molto piccolo: {x_max_calib-x_min_calib:.3f}. La calibrazione potrebbe essere imprecisa.")
                 print("\n(!) ATTENZIONE: L'area X di calibrazione rilevata è molto piccola. Potrebbe essere imprecisa.")
                 valid_range = False # Or maybe just warn? Let's proceed but warn.
            if (y_max_calib - y_min_calib < min_required_range):
                 logger.warning(f"Range di calibrazione Y rilevato molto piccolo: {y_max_calib-y_min_calib:.3f}. La calibrazione potrebbe essere imprecisa.")
                 print("\n(!) ATTENZIONE: L'area Y di calibrazione rilevata è molto piccola. Potrebbe essere imprecisa.")
                 valid_range = False

            # Update configuration with new calibration boundaries
            self.config.set("calibration.x_min", x_min_calib)
            self.config.set("calibration.x_max", x_max_calib)
            self.config.set("calibration.y_min", y_min_calib)
            self.config.set("calibration.y_max", y_max_calib)
            self.config.set("calibration.enabled", True) # Ensure calibration is marked as enabled

            # Save the updated configuration to the file
            self.config.save()

            logger.info(f"Calibrazione completata e salvata. Valori: X[{x_min_calib:.3f}-{x_max_calib:.3f}], Y[{y_min_calib:.3f}-{y_max_calib:.3f}]")
            print("\n--- CALIBRAZIONE COMPLETATA E SALVATA! ---")
            print("Il movimento del mouse è ora attivo con la nuova calibrazione.")

        except Exception as e:
            logger.exception(f"Errore durante il completamento o salvataggio della calibrazione: {e}")
            print("\n(!) ERRORE durante il salvataggio della calibrazione. Le modifiche potrebbero NON essere state salvate.")
            # Should we disable calibration if save failed? Maybe safer.
            self.config.set("calibration.enabled", False)
        finally:
            # Reset calibration state variables regardless of success/failure
            self.calibration_active = False
            self.config["calibration"]["active"] = False # Set internal state back
            self.calibration_points = []
            self.calibration_step = 0
            self.tracking_enabled = True # Re-enable tracking


# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
def parse_arguments():
    """Analizza gli argomenti della linea di comando."""
    parser = argparse.ArgumentParser(
        description="HandPal - Controllo del mouse con i gesti delle mani usando MediaPipe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )

    # --- Device and Resolution ---
    cfg_defaults = Config.DEFAULT_CONFIG # Get defaults for help message
    parser.add_argument('--device', type=int, default=None, # Default handled by Config class
                        help=f'ID del dispositivo webcam (es. 0, 1). Default: {cfg_defaults["device"]}')
    parser.add_argument('--width', type=int, default=None,
                        help=f'Larghezza desiderata acquisizione webcam. Default: {cfg_defaults["width"]}')
    parser.add_argument('--height', type=int, default=None,
                        help=f'Altezza desiderata acquisizione webcam. Default: {cfg_defaults["height"]}')
    parser.add_argument('--display-width', type=int, default=1280,
                        help='Larghezza desiderata (pixel) della finestra di anteprima OpenCV. Default: dimensione frame originale.')
    parser.add_argument('--display-height', type=int, default=720,
                        help='Altezza desiderata (pixel) della finestra di anteprima OpenCV. Default: dimensione frame originale.')

    # --- Features and Modes ---
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Abilita modalità debug con overlay informativo e logging dettagliato.')
    # Use store_true for flip, then handle logic based on --no-flip presence
    parser.add_argument('--flip-camera', action='store_true', default=None, # Default handled by Config
                         help=f'Ribalta orizzontalmente l\'immagine della webcam (utile per effetto specchio). Default: {cfg_defaults["flip_camera"]}')
    parser.add_argument('--no-flip', dest='flip_camera', action='store_false',
                         help='Disabilita il ribaltamento orizzontale dell\'immagine webcam.')
    parser.add_argument('--calibrate', action='store_true', default=False,
                        help='Avvia la calibrazione guidata all\'avvio dell\'applicazione.')

    # --- Configuration Management ---
    parser.add_argument('--reset-config', action='store_true', default=False,
                        help='Rimuove il file di configurazione esistente (~/.handpal_config.json) ed esce. Al prossimo avvio verranno usati i valori predefiniti.')
    parser.add_argument('--config-file', type=str, default=os.path.expanduser("~/.handpal_config.json"),
                        help='Percorso completo del file di configurazione JSON.')

    # --- Appearance ---
    parser.add_argument('--cursor-file', type=str, default="red_cursor.cur", # Default handled by Config
                        help=f'Percorso del file cursore personalizzato (.cur) da utilizzare (solo Windows). Default: {cfg_defaults["custom_cursor_path"]}')

    args = parser.parse_args()

    # Handle flip_camera logic: args.flip_camera will be False if --no-flip is used,
    # True if --flip-camera is used, None otherwise. Config class handles None correctly.
    if args.flip_camera is False:
         logger.debug("Argomento --no-flip rilevato.")
    elif args.flip_camera is True:
         logger.debug("Argomento --flip-camera rilevato.")


    return args

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    """Funzione principale dell'applicazione."""
    args = parse_arguments()
    custom_cursor_was_set = False # Flag to track if we need to restore the cursor

    # --- Setup Logging Level based on Debug Flag ---
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Logging impostato a livello DEBUG.")
    else:
        logger.setLevel(logging.INFO)
        for handler in logger.handlers:
             # Keep StreamHandler at INFO, but FileHandler could be DEBUG?
             if isinstance(handler, logging.FileHandler):
                 handler.setLevel(logging.INFO) # Or DEBUG if you want detailed file logs always
             else:
                 handler.setLevel(logging.INFO)
        logger.info("Logging impostato a livello INFO.")


    # --- Handle Config Reset ---
    if args.reset_config:
        config_path = args.config_file
        try:
            if os.path.exists(config_path):
                os.remove(config_path)
                logger.info(f"File di configurazione rimosso: {config_path}")
                print(f"Configurazione rimossa ({config_path}). Verranno usati i valori predefiniti al prossimo avvio.")
            else:
                logger.info("Nessun file di configurazione trovato da resettare.")
                print("Nessun file di configurazione trovato da resettare.")
            # Should we also restore default cursor here? Might be unexpected.
            # Let's only reset the config file.
            return 0 # Exit successfully after reset
        except Exception as e:
            logger.error(f"Errore durante la rimozione del file di configurazione: {e}")
            print(f"Errore durante la rimozione del file di configurazione: {e}")
            return 1 # Indicate failure

    # --- Initialize Configuration ---
    # Config loads defaults, then file, then applies CLI args
    config = Config(args)

    # --- Custom Cursor Handling (Windows Only) ---
    if os.name == 'nt':
        # Use path from config (which might have been overridden by CLI's --cursor-file)
        custom_cursor_path_config = config.get('custom_cursor_path')
        if custom_cursor_path_config:
            # Check if it's just a filename (assume relative to script) or a full path
            if not os.path.isabs(custom_cursor_path_config) and not os.path.exists(custom_cursor_path_config):
                 script_dir = os.path.dirname(os.path.abspath(__file__))
                 potential_path = os.path.join(script_dir, custom_cursor_path_config)
                 if os.path.exists(potential_path):
                     resolved_cursor_path = potential_path
                     logger.debug(f"Percorso cursore relativo risolto in: {resolved_cursor_path}")
                 else:
                      resolved_cursor_path = None
                      logger.warning(f"File cursore '{custom_cursor_path_config}' non trovato nel percorso corrente o nella directory dello script.")
            else:
                 # Path is absolute or exists in current dir
                 resolved_cursor_path = custom_cursor_path_config

            if resolved_cursor_path and os.path.exists(resolved_cursor_path):
                logger.info(f"Tentativo di impostare il cursore personalizzato: {resolved_cursor_path}")
                if set_custom_cursor(resolved_cursor_path):
                    custom_cursor_was_set = True # Mark that we need to restore it later
            elif custom_cursor_path_config: # Path was given but not found/resolved
                 logger.error(f"File cursore specificato '{custom_cursor_path_config}' non trovato o non accessibile.")
        else:
             logger.debug("Nessun percorso cursore personalizzato specificato nella configurazione.")

    # --- Initialize Main Application ---
    app = None
    try:
        app = HandPal(config)

        # Set debug mode in app instance if CLI flag was set
        if args.debug:
            app.debug_mode = True
            logger.debug("Modalità debug attivata nell'istanza HandPal.")

        # --- Start Application ---
        logger.info("Tentativo di avviare HandPal...")
        if app.start():
            # If calibrate flag is set, start calibration after a short delay
            if args.calibrate:
                logger.info("Flag --calibrate rilevato, avvio calibrazione tra poco...")
                # Need a slight delay to ensure the window is fully initialized?
                time.sleep(0.75) # Increased delay slightly
                app.start_calibration()

            # Run the main loop (blocks until 'q' is pressed or stop() is called)
            app.main_loop()
            # Main loop finished normally (e.g., user pressed 'q')

        else:
            # app.start() failed
            logger.error("Avvio dell'applicazione HandPal fallito. Controllare log precedenti.")
            print("ERRORE: Avvio dell'applicazione fallito. Controllare il file handpal.log per dettagli.")
            return 1 # Indicate failure

    except KeyboardInterrupt:
        logger.info("Interruzione da tastiera ricevuta (Ctrl+C). Fermando HandPal...")
        # The finally block will handle stopping the app
    except Exception as e:
         # Catch any unexpected errors in the main execution flow
         logger.exception(f"Errore non gestito nel blocco main: {e}")
         print(f"ERRORE non gestito: {e}. Controllare handpal.log.")
         return 1 # Indicate failure
    finally:
        # --- Cleanup ---
        logger.info("Esecuzione blocco finally per cleanup...")
        if app and app.running:
            logger.info("HandPal ancora in esecuzione, chiamando stop()...")
            app.stop()
        elif app:
             logger.debug("App esiste ma non è in esecuzione (potrebbe essersi fermata correttamente o fallita prima). Assicurando cleanup.")
             app.stop() # Call stop anyway to ensure resources are released if partially started

        # Restore default cursor only if we successfully set a custom one
        if custom_cursor_was_set:
            logger.info("Ripristino cursore di sistema predefinito...")
            restore_default_cursor()

        logger.info("Applicazione HandPal terminata.")
        print("\nHandPal terminato.")
        # Add a small delay to allow logs to flush completely?
        time.sleep(0.1)

    return 0 # Indicate success

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Optional: Add script directory to Python path if needed, though usually not necessary
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # if script_dir not in sys.path:
    #     sys.path.insert(0, script_dir)

    # Call the main function and exit with its return code
    sys.exit(main())