import argparse
import threading
import os
import sys
import time
import numpy as np
import cv2 as cv
import mediapipe as mp
import tkinter as tk
from tkinter import Toplevel, Frame, Label, Button as TkButton, BOTH, X
from pynput.mouse import Controller, Button
from collections import deque
import json
import logging
import ctypes
import queue
import csv
import subprocess
import webbrowser
try:
    from PIL import Image, ImageDraw, ImageFont
    from PIL import __version__ as PIL_VERSION_STR # Get Pillow version string
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    PIL_VERSION_STR = "0.0.0" # Default if PIL not found, or handle as None
    # This print statement is a simple notification.
    # If you have a logger configured for TutorialManager, you could use that.
    print("Warning: Pillow library (PIL) not found. Emojis in tutorial might not render correctly.")

# It's good practice to parse the version string into a tuple for easier comparison
# Do this once after import if PIL is available
PIL_VERSION_TUPLE = (0,0,0)
if PIL_AVAILABLE:
    try:
        PIL_VERSION_TUPLE = tuple(map(int, PIL_VERSION_STR.split('.')))
    except ValueError:
        print(f"Warning: Could not parse Pillow version string: {PIL_VERSION_STR}")
        PIL_VERSION_TUPLE = (0,0,0) # Fallback if parsing fails


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("handpal.log"), logging.StreamHandler()]
)
logger = logging.getLogger("HandPal")

# Variabile globale per l'istanza di HandPal (per il tutorial)
handpal_instance = None

# -----------------------------------------------------------------------------
# Windows Cursor Functions
# -----------------------------------------------------------------------------
def set_custom_cursor(cursor_path):
    if os.name != 'nt': return False
    try:
        if not os.path.exists(cursor_path): raise FileNotFoundError(f"Cursor file not found: {cursor_path}")
        user32 = ctypes.windll.user32; hCursor = user32.LoadCursorFromFileW(cursor_path)
        if hCursor == 0: raise Exception(f"Failed to load cursor (Error {ctypes.get_last_error()}).")
        OCR_NORMAL = 32512
        if not user32.SetSystemCursor(hCursor, OCR_NORMAL): raise Exception(f"Failed to set system cursor (Error {ctypes.get_last_error()}).")
        logger.info(f"Custom cursor '{os.path.basename(cursor_path)}' set.")
        return True
    except Exception as e: logger.error(f"Error setting custom cursor: {e}"); return False

def restore_default_cursor():
    if os.name != 'nt': return
    try:
        user32 = ctypes.windll.user32; SPI_SETCURSORS = 0x57
        if not user32.SystemParametersInfoW(SPI_SETCURSORS, 0, None, 3): raise Exception(f"Failed to restore default cursors (Error {ctypes.get_last_error()}).")
        logger.info("Default system cursors restored.")
    except Exception as e: logger.error(f"Error restoring default cursors: {e}")

# -----------------------------------------------------------------------------
# Config Class
# -----------------------------------------------------------------------------
class Config:
    DEFAULT_CONFIG = {
        "device": 0,
        "width": 1920,
        "height": 1440,
        "process_width": 1920,
        "process_height": 1440,
        "flip_camera": True,
        "display_width": None,
        "display_height": None,
        "min_detection_confidence": 0.6,
        "min_tracking_confidence": 0.5,
        "use_static_image_mode": False,
        "smoothing_factor": 0.7,
        "inactivity_zone": 0.015,
        "click_cooldown": 0.4,
        "gesture_sensitivity": 0.02,
        "gesture_settings": {"scroll_sensitivity": 4, "double_click_time": 0.35},
        "calibration": {
            "enabled": True,
            "screen_margin": 0.1,
            "x_min": 0.15,
            "x_max": 0.85,
            "y_min": 0.15,
            "y_max": 0.85,
            "active": False
        },
        "max_fps": 60,
        "custom_cursor_path": "red_cursor.cur"
    }

    CONFIG_FILENAME = os.path.expanduser("~/.handpal_config.json")

    def _deep_update(self, target, source):
        for k, v in source.items():
            if isinstance(v, dict) and k in target and isinstance(target[k], dict): self._deep_update(target[k], v)
            else: target[k] = v

    def __init__(self, args=None):
        self.config = json.loads(json.dumps(self.DEFAULT_CONFIG)) # Deep copy
        if os.path.exists(self.CONFIG_FILENAME):
            try:
                with open(self.CONFIG_FILENAME, 'r') as f: self._deep_update(self.config, json.load(f))
                logger.info(f"Config loaded from {self.CONFIG_FILENAME}")
            except Exception as e: logger.error(f"Error loading config: {e}")
        if args: self._apply_cli_args(args)
        self.config["calibration"]["active"] = False
        self._validate_calibration(); self._validate_display_dims()
        logger.debug(f"Final config: {self.config}")

    def _apply_cli_args(self, args):
        cli_args = vars(args)
        for key, value in cli_args.items():
            if value is not None:
                if key == 'flip_camera' and value is False: self.config['flip_camera'] = False
                elif key == 'cursor_file': self.config['custom_cursor_path'] = value
                elif '.' in key: logger.warning(f"Nested CLI arg '{key}' ignored (not supported).")
                elif key in self.config:
                    if not isinstance(self.config[key], dict):
                        try: self.config[key] = type(self.config[key])(value)
                        except Exception: self.config[key] = value
                    else: logger.warning(f"Ignoring CLI arg '{key}' (dict in config).")

    def _validate_calibration(self):
        calib = self.config["calibration"]; default = self.DEFAULT_CONFIG["calibration"]; rx = ry = False
        nums = ["x_min", "x_max", "y_min", "y_max"]
        if not all(isinstance(calib.get(k), (int, float)) for k in nums): rx=ry=True; logger.warning("Non-numeric calibration values, reset.")
        if not (0<=calib.get("x_min",-1)<calib.get("x_max",-1)<=1 and abs(calib.get("x_max",0)-calib.get("x_min",0))>=0.05): rx=True; logger.warning("Invalid X calibration, reset.")
        if not (0<=calib.get("y_min",-1)<calib.get("y_max",-1)<=1 and abs(calib.get("y_max",0)-calib.get("y_min",0))>=0.05): ry=True; logger.warning("Invalid Y calibration, reset.")
        if rx: calib["x_min"], calib["x_max"] = default["x_min"], default["x_max"]
        if ry: calib["y_min"], calib["y_max"] = default["y_min"], default["y_max"]

    def _validate_display_dims(self):
        w, h = self.get("display_width"), self.get("display_height")
        if not ((w is None) or (isinstance(w, int) and w > 0)): self.set("display_width", None); logger.warning("Invalid display_width.")
        if not ((h is None) or (isinstance(h, int) and h > 0)): self.set("display_height", None); logger.warning("Invalid display_height.")

    def get(self, key, default=None):
        try: val = self.config; keys = key.split('.'); [val := val[k] for k in keys]; return val
        except Exception: return default

    def set(self, key, value):
        try: d = self.config; keys = key.split('.'); [d := d[k] for k in keys[:-1]]; d[keys[-1]] = value
        except Exception: logger.error(f"Failed to set config key '{key}'."); return False
        if key.startswith("calibration."): self._validate_calibration()
        if key.startswith("display_"): self._validate_display_dims()
        return True

    def save(self):
        try:
            save_cfg = json.loads(json.dumps(self.config)); save_cfg.get("calibration", {}).pop("active", None)
            with open(self.CONFIG_FILENAME, 'w') as f: json.dump(save_cfg, f, indent=2)
            logger.info(f"Config saved to {self.CONFIG_FILENAME}")
        except Exception as e: logger.error(f"Error saving config: {e}")

    def __getitem__(self, key): return self.get(key)
    def __setitem__(self, key, value): self.set(key, value)

# -----------------------------------------------------------------------------
# GestureRecognizer Class
# -----------------------------------------------------------------------------
class GestureRecognizer:
    def __init__(self, config):
        self.config = config
        self.last_positions = {}
        self.gesture_state = {
            "scroll_active": False,
            "last_click_time": 0,
            "last_click_button": None,
            "scroll_history": deque(maxlen=5),
            "active_gesture": None,
            "last_pose": {"Left": "U", "Right": "U"},
            "pose_stable_count": {"Left": 0, "Right": 0},
            "fist_drag_active": False
        }
        self.POSE_STABILITY_THRESHOLD = 3

    def _dist(self, p1, p2):
        if None in [p1, p2] or not all(hasattr(p, 'x') for p in [p1,p2]): return float('inf')
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def check_thumb_index_click(self, hand_lm, handedness):
        if handedness != "Left" or hand_lm is None or self.gesture_state["scroll_active"] or self.gesture_state["fist_drag_active"]:
            return None
        try:
            thumb, index = hand_lm.landmark[4], hand_lm.landmark[8]
        except (IndexError, TypeError):
            return None
        dist = self._dist(thumb, index)
        now = time.time()
        thresh = self.config["gesture_sensitivity"]
        gesture = None
        
        if dist < thresh: 
            cooldown = self.config["click_cooldown"]
            if (now - self.gesture_state["last_click_time"]) > cooldown:
                double_click_t = self.config["gesture_settings"]["double_click_time"]
                if (now - self.gesture_state["last_click_time"]) < double_click_t and \
                   self.gesture_state["last_click_button"] == Button.left:
                    gesture = "double_click"
                    self.gesture_state["active_gesture"] = "double_click"
                else:
                    gesture = "click"
                    self.gesture_state["active_gesture"] = "click"
                self.gesture_state["last_click_time"] = now
                self.gesture_state["last_click_button"] = Button.left
        elif self.gesture_state["active_gesture"] in ["click", "double_click"]:
            self.gesture_state["active_gesture"] = None
        
        return gesture

    def check_fist_drag(self, hand_lm, handedness):
        if handedness != "Left" or hand_lm is None or self.gesture_state["scroll_active"]:
            return None
            
        raw_pose = self.detect_raw_pose(hand_lm)
        is_fist = raw_pose == "Fist"
        
        if self.gesture_state["fist_drag_active"]:
            try:
                lm=hand_lm.landmark
                it,mt,rt,pt = lm[8],lm[12],lm[16],lm[20]
                im,mm,rm,pm = lm[5],lm[9],lm[13],lm[17]
                y_ext = 0.03
                i_ext = it.y < im.y - y_ext
                m_ext = mt.y < mm.y - y_ext
                r_ext = rt.y < rm.y - y_ext 
                p_ext = pt.y < pm.y - y_ext
                num_ext = sum([i_ext,m_ext,r_ext,p_ext])
                still_dragging = num_ext <= 1
            except (IndexError, TypeError):
                still_dragging = False
            
            if still_dragging:
                return "drag_continue"
            else:
                self.gesture_state["fist_drag_active"] = False
                self.gesture_state["active_gesture"] = None
                return "drag_end"
        
        if is_fist and not self.gesture_state["fist_drag_active"]:
            self.gesture_state["fist_drag_active"] = True
            self.gesture_state["active_gesture"] = "drag"
            return "drag_start"
        
        return None

    def detect_raw_pose(self, hand_lm):
        if hand_lm is None: return "U"
        try: 
            lm = hand_lm.landmark
            it,mt,rt,pt = lm[8],lm[12],lm[16],lm[20]
            im,mm,rm,pm = lm[5],lm[9],lm[13],lm[17]
            y_ext = 0.03
            i_ext = it.y < im.y - y_ext
            m_ext = mt.y < mm.y - y_ext
            r_ext = rt.y < rm.y - y_ext 
            p_ext = pt.y < pm.y - y_ext
            num_ext = sum([i_ext,m_ext,r_ext,p_ext])
            
            if num_ext == 0: return "Fist"
            if num_ext >= 4: return "Open"
            if i_ext and num_ext == 1: return "Point"
            if i_ext and m_ext and num_ext == 2: return "Two"
            return "Other"
        except (IndexError, TypeError):
            return "U"

    def check_scroll_gesture(self, hand_landmarks, handedness):
         if handedness != "Left": return None
         if self.gesture_state["active_gesture"] in ["click", "double_click"] or self.gesture_state["fist_drag_active"]:
             return None
             
         index_tip = hand_landmarks.landmark[8]
         middle_tip = hand_landmarks.landmark[12]
         index_mcp = hand_landmarks.landmark[5]
         middle_mcp = hand_landmarks.landmark[9]
         ring_tip = hand_landmarks.landmark[16]
         pinky_tip = hand_landmarks.landmark[20]
         
         index_extended = index_tip.y < index_mcp.y
         middle_extended = middle_tip.y < middle_mcp.y
         fingers_close = abs(index_tip.x - middle_tip.x) < 0.08
         ring_pinky_folded = (ring_tip.y > index_mcp.y) and (pinky_tip.y > index_mcp.y)
         
         if index_extended and middle_extended and fingers_close and ring_pinky_folded:
             if not self.gesture_state["scroll_active"]:
                 self.gesture_state["scroll_active"] = True
                 self.gesture_state["scroll_history"].clear()
                 self.gesture_state["active_gesture"] = "Scroll"
             
             if 8 in self.last_positions:
                 prev_y = self.last_positions[8][1]
                 curr_y = index_tip.y
                 delta_y = (curr_y - prev_y) * self.config["gesture_settings"]["scroll_sensitivity"]
                 
                 if abs(delta_y) > 0.0005:
                     self.gesture_state["scroll_history"].append(delta_y)
                 
                 if len(self.gesture_state["scroll_history"]) > 0:
                     smooth_delta = sum(self.gesture_state["scroll_history"]) / len(self.gesture_state["scroll_history"])
                     if abs(smooth_delta) > 0.001:
                         self.last_positions[8] = (index_tip.x, index_tip.y)
                         return smooth_delta * 100
             
             self.last_positions[8] = (index_tip.x, index_tip.y)
         else:
             if self.gesture_state["scroll_active"]:
                 self.gesture_state["scroll_active"] = False
                 self.gesture_state["scroll_history"].clear()
                 if self.gesture_state["active_gesture"] == "Scroll":
                     self.gesture_state["active_gesture"] = None
         return None

    def detect_hand_pose(self, hand_lm, handedness):
        if hand_lm is None or handedness not in ["Left", "Right"]: return "U"
        try:
            lm=hand_lm.landmark
            w,tt,it,mt,rt,pt=lm[0],lm[4],lm[8],lm[12],lm[16],lm[20]
            im,mm,rm,pm=lm[5],lm[9],lm[13],lm[17]
        except (IndexError, TypeError): return "U"

        y_ext = 0.03
        i_ext=it.y<im.y-y_ext
        m_ext=mt.y<mm.y-y_ext
        r_ext=rt.y<rm.y-y_ext
        p_ext=pt.y<pm.y-y_ext
        num_ext = sum([i_ext,m_ext,r_ext,p_ext])
        pose = "U"

        if handedness=="Left" and self.gesture_state["scroll_active"]:
            pose="Scroll"
        elif num_ext == 0:
            pose="Fist"
        elif i_ext and num_ext == 1:
            pose="Point"
        elif i_ext and m_ext and num_ext == 2:
            pose="Two"
        elif num_ext >= 4:
            pose="Open"

        last_known_pose=self.gesture_state["last_pose"].get(handedness,"U")
        stable_count=self.gesture_state["pose_stable_count"].get(handedness,0)

        if pose == last_known_pose and pose != "U":
            stable_count += 1
        else:
            stable_count = 0
        
        self.gesture_state["last_pose"][handedness] = pose
        self.gesture_state["pose_stable_count"][handedness] = stable_count

        if pose == "Fist" and stable_count >= 1:
            return pose
        
        if pose == "Scroll" or (pose != "U" and stable_count >= self.POSE_STABILITY_THRESHOLD) :
            return pose
        else:
            return last_known_pose + "?"

# -----------------------------------------------------------------------------
# MotionSmoother Class
# -----------------------------------------------------------------------------
class MotionSmoother:
    def __init__(self, config):
        self.config = config; self.last_smooth_pos = None; self._update_alpha()
        self.inactive_zone_sq = self.config["inactivity_zone"]**2

    def _update_alpha(self):
        factor = max(0.01, min(0.99, self.config["smoothing_factor"]))
        self.alpha = 1.0 - factor

    def update(self, target_x, target_y, screen_w, screen_h):
        target_px = (target_x, target_y)
        if self.last_smooth_pos is None: self.last_smooth_pos = target_px; return target_px
        if screen_w <= 0 or screen_h <= 0: return self.last_smooth_pos 
        
        last_norm_x = self.last_smooth_pos[0]/screen_w 
        last_norm_y = self.last_smooth_pos[1]/screen_h
        target_norm_x = target_x/screen_w
        target_norm_y = target_y/screen_h

        dist_sq_norm = (target_norm_x-last_norm_x)**2 + (target_norm_y-last_norm_y)**2
        if dist_sq_norm < self.inactive_zone_sq: return self.last_smooth_pos
        
        self._update_alpha()
        smooth_x = int(self.alpha*target_x + (1-self.alpha)*self.last_smooth_pos[0])
        smooth_y = int(self.alpha*target_y + (1-self.alpha)*self.last_smooth_pos[1])
        self.last_smooth_pos = (smooth_x, smooth_y)
        return smooth_x, smooth_y

    def reset(self): self.last_smooth_pos = None; logger.debug("Smoother reset.")

# -----------------------------------------------------------------------------
# Detection Thread
# -----------------------------------------------------------------------------
class DetectionThread(threading.Thread):
    def __init__(self, config, cap, hands, data_q, stop_evt):
        super().__init__(daemon=True, name="DetectionThread"); self.config = config
        self.cap=cap; self.hands=hands; self.data_q=data_q; self.stop_evt=stop_evt
        self.proc_w=config["process_width"]; self.proc_h=config["process_height"]
        self.flip=config["flip_camera"]
        logger.info(f"DetectionThread: ProcRes={self.proc_w}x{self.proc_h}, Flip={self.flip}")

    def run(self):
        logger.info("DetectionThread starting"); t_start=time.perf_counter(); frame_n=0
        while not self.stop_evt.is_set():
            if self.cap is None or not self.cap.isOpened(): logger.error("Webcam N/A"); time.sleep(0.5); continue
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None: time.sleep(0.05); continue
                if self.flip: frame = cv.flip(frame, 1)
                proc_frame = cv.resize(frame, (self.proc_w, self.proc_h), interpolation=cv.INTER_LINEAR)
                rgb_frame = cv.cvtColor(proc_frame, cv.COLOR_BGR2RGB); rgb_frame.flags.writeable=False
                results = self.hands.process(rgb_frame); rgb_frame.flags.writeable=True
                try: self.data_q.put((frame, results, (self.proc_w, self.proc_h)), block=False)
                except queue.Full:
                    try: self.data_q.get_nowait(); self.data_q.put_nowait((frame, results, (self.proc_w, self.proc_h)))
                    except queue.Empty: pass
                    except queue.Full: pass
                frame_n += 1
            except Exception as e: logger.exception(f"DetectionThread loop error: {e}"); time.sleep(0.1)
        elapsed = time.perf_counter()-t_start; fps = frame_n/elapsed if elapsed>0 else 0
        logger.info(f"DetectionThread finished. {frame_n} frames in {elapsed:.2f}s (Avg FPS: {fps:.1f})")

# -----------------------------------------------------------------------------
# Tutorial Manager Class
# -----------------------------------------------------------------------------
# TUTORIAL MANAGER - REVISED
class TutorialManager:
    def __init__(self, handpal):
        self.handpal = handpal
        self.active = False
        self.current_step = 0
        self.step_completed_flag = False  # True if current step's action was detected and confirmed
        self.success_message_start_time = 0 # Timer for how long to show "FANTASTICO!"
        
        # Timer for specific action confirmations (e.g., holding hand in position)
        self.action_confirm_start_time = 0

        self.SUCCESS_MESSAGE_DURATION = 0.95 # seconds - slightly longer to read
        self.POSITION_HOLD_DURATION = 0.75 # seconds for right hand presence
        self.GESTURE_HOLD_DURATION = 0.35  # seconds to confirm a gesture is intentional

        self.steps = [
            {
                "title": "Benvenuto in HandPal",
                "instruction": "Posiziona la mano DESTRA nello schermo per muovere il cursore.",
                "completion_type": "position", # Right hand detected and held
                "icon": "👋"
            },
            {
                "title": "Click Sinistro",
                "instruction": "Con la mano SINISTRA, avvicina pollice e indice per fare click.",
                "completion_type": "gesture",
                "required_gesture": "click", # GestureRecognizer sets active_gesture to "click" or "double_click"
                "icon": "👆"
            },
            {
                "title": "Drag & Drop",
                "instruction": "Mano SINISTRA: chiudi a PUGNO per iniziare il Drag, muovi, poi apri per rilasciare.",
                "completion_type": "gesture",
                "required_gesture": "drag", # GestureRecognizer sets active_gesture to "drag"
                "icon": "✊"
            },
            {
                "title": "Scorrimento",
                "instruction": "Mano SINISTRA: estendi indice e medio (altre dita piegate) per scorrere.",
                "completion_type": "gesture",
                "required_gesture": "Scroll", # GestureRecognizer sets active_gesture to "Scroll"
                "icon": "✌️"
            }
        ]
    
    def start_tutorial(self):
        logger.info("Tutorial starting...")
        self.active = True
        self.current_step = 0
        self._reset_step_state() # Initialize for the first step
        self.handpal.tracking_enabled = True # Ensure mouse control is active for tutorial
        logger.info(f"Tutorial started. Current step {self.current_step}: {self.steps[self.current_step].get('title')}")
        
    def stop_tutorial(self):
        logger.info("Tutorial stopping.")
        self.active = False
        self._reset_step_state() # Clear any lingering state
        
    def _reset_step_state(self):
        """Resets state variables for the current or upcoming step."""
        self.step_completed_flag = False
        self.success_message_start_time = 0
        self.action_confirm_start_time = 0
        # logger.debug(f"Tutorial step state reset for step {self.current_step if self.active else 'N/A'}")

    def _advance_to_next_step(self):
        logger.debug(f"Advancing from tutorial step {self.current_step}")
        self.current_step += 1
        if self.current_step >= len(self.steps):
            logger.info("Tutorial completed!")
            self.stop_tutorial() # This will also call _reset_step_state
        else:
            self._reset_step_state() # Prepare for the new current step
            logger.info(f"Advanced to tutorial step {self.current_step}: {self.steps[self.current_step].get('title')}")
            
    def check_step_completion(self, results): # results from MediaPipe
        if not self.active or self.current_step >= len(self.steps):
            return # Tutorial not active or all steps done
            
        now = time.time()

        # --- Part 1: Check if the current step's action is detected and confirmed ---
        if not self.step_completed_flag: # Only check for action if step isn't already marked as completed
            current_step_config = self.steps[self.current_step]
            completion_type = current_step_config.get("completion_type")
            action_confirmed_this_frame = False

            if completion_type == "position":
                if self.handpal.last_right_hand_lm_norm is not None: # Right hand is present
                    if self.action_confirm_start_time == 0: # Start timer if hand just appeared for this step
                        self.action_confirm_start_time = now
                        logger.debug(f"Step {self.current_step} (Position): Right hand detected, confirmation timer started.")
                    elif (now - self.action_confirm_start_time) > self.POSITION_HOLD_DURATION:
                        action_confirmed_this_frame = True
                        logger.debug(f"Step {self.current_step} (Position): Right hand held, action confirmed.")
                else: # Right hand lost
                    if self.action_confirm_start_time != 0:
                        logger.debug(f"Step {self.current_step} (Position): Right hand lost, resetting confirmation timer.")
                    self.action_confirm_start_time = 0 # Reset timer

            elif completion_type == "gesture":
                required_gesture = current_step_config.get("required_gesture")
                # Get the gesture currently recognized by GestureRecognizer
                # This is set by check_thumb_index_click, check_fist_drag, check_scroll_gesture
                current_active_gesture = self.handpal.gesture_recognizer.gesture_state.get("active_gesture")
                
                # Special handling for "click" as it can be "click" or "double_click"
                gesture_match = False
                if required_gesture == "click":
                    gesture_match = current_active_gesture in ["click", "double_click"]
                else:
                    gesture_match = (current_active_gesture == required_gesture)

                if gesture_match:
                    if self.action_confirm_start_time == 0:
                        self.action_confirm_start_time = now
                        logger.debug(f"Step {self.current_step} (Gesture '{required_gesture}'): Matched '{current_active_gesture}', confirmation timer started.")
                    elif (now - self.action_confirm_start_time) > self.GESTURE_HOLD_DURATION:
                        action_confirmed_this_frame = True
                        logger.debug(f"Step {self.current_step} (Gesture '{required_gesture}'): Gesture held, action confirmed.")
                else: # Gesture not active or not matching
                    if self.action_confirm_start_time != 0:
                         logger.debug(f"Step {self.current_step} (Gesture '{required_gesture}'): Gesture no longer active/matching, resetting confirmation timer.")
                    self.action_confirm_start_time = 0

            elif completion_type == "menu_hover":
                # self.handpal.menu_trigger_active is True if hand is in the zone
                # self.handpal.check_menu_trigger() returns True if hover delay is met
                # For tutorial, we just need hand in zone for a bit.
                if self.handpal.menu_trigger_active: # Hand is in the menu circle zone
                    if self.action_confirm_start_time == 0:
                        self.action_confirm_start_time = now
                        logger.debug(f"Step {self.current_step} (Menu Hover): Hand in menu zone, confirmation timer started.")
                    # Use app's hover delay for consistency, or a tutorial-specific one
                    elif (now - self.action_confirm_start_time) > self.handpal.MENU_HOVER_DELAY:
                        action_confirmed_this_frame = True
                        logger.debug(f"Step {self.current_step} (Menu Hover): Hover confirmed.")
                else: # Hand not in menu zone
                    if self.action_confirm_start_time != 0:
                        logger.debug(f"Step {self.current_step} (Menu Hover): Hand left menu zone, resetting confirmation timer.")
                    self.action_confirm_start_time = 0
            
            if action_confirmed_this_frame:
                self.step_completed_flag = True
                self.success_message_start_time = now # Start timer for "FANTASTICO!"
                self.action_confirm_start_time = 0 # Reset general confirmation timer
                logger.info(f"Tutorial Step {self.current_step} ('{current_step_config.get('title')}') COMPLETED. Displaying success message.")

        # --- Part 2: If step is completed, check if it's time to advance ---
        if self.step_completed_flag:
            if (now - self.success_message_start_time) > self.SUCCESS_MESSAGE_DURATION:
                logger.debug(f"Step {self.current_step} success message duration ended. Advancing to next step.")
                self._advance_to_next_step()
            # Else: success message is still being displayed.
            
    def draw_tutorial_overlay(self, frame):
        if not self.active or self.current_step >= len(self.steps):
            return frame
            
        h, w = frame.shape[:2]
        if w == 0 or h == 0: return frame
            
        overlay_height = 120
        tutorial_panel_img = np.zeros((overlay_height, w, 3), dtype=np.uint8)
        cv.rectangle(tutorial_panel_img, (0, 0), (w, overlay_height), (20, 20, 20), -1)
        
        alpha = 0.85
        frame_roi = frame[h-overlay_height:h, 0:w]
        blended_roi = cv.addWeighted(frame_roi, 1-alpha, tutorial_panel_img, alpha, 0)
        frame[h-overlay_height:h, 0:w] = blended_roi
        
        current_step_config = self.steps[self.current_step]
        title = current_step_config.get("title", "Tutorial")
        instruction = current_step_config.get("instruction", "")
        icon = current_step_config.get("icon", "📚")
        
        num_steps = len(self.steps)
        dot_area_width = num_steps * 25 
        start_x_dots = (w - dot_area_width) // 2
        for i in range(num_steps):
            color = (0, 220, 220) if i == self.current_step else \
                    (0, 150, 0) if i < self.current_step else \
                    (100, 100, 100)
            cv.circle(frame, (start_x_dots + i*25, h-20), 7, color, -1)

        y_title_baseline = h - overlay_height + 30
        y_instruction_baseline = h - overlay_height + 65
        y_fantastico_baseline = h - overlay_height + 95
        y_esc_msg_baseline = h - overlay_height + 20
        y_progress_text_baseline = h - 16

        if PIL_AVAILABLE:
            pil_img = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            font_size_title = 22
            font_size_instruction = 18
            font_size_fantastico = 20
            font_size_esc = 15
            font_size_progress_text = 16
            font_size_dot_icon = 18

            default_font_family = "arial.ttf"
            if os.name == 'nt': emoji_font_family = "seguiemj.ttf"
            elif sys.platform == "darwin": emoji_font_family = "AppleColorEmoji.ttf"
            else: emoji_font_family = "NotoColorEmoji.ttf"

            def load_font(preferred_family, size, fallback_family=default_font_family):
                try: return ImageFont.truetype(preferred_family, size)
                except IOError:
                    try: return ImageFont.truetype(fallback_family, size)
                    except IOError: return ImageFont.load_default()

            font_title = load_font(emoji_font_family, font_size_title)
            font_instruction = load_font(default_font_family, font_size_instruction)
            font_fantastico = load_font(default_font_family, font_size_fantastico)
            font_esc = load_font(default_font_family, font_size_esc)
            font_progress_text = load_font(default_font_family, font_size_progress_text)
            font_dot_icon = load_font(emoji_font_family, font_size_dot_icon)

            # --- CORRECTED TEXT ANCHOR ARGS ---
            text_anchor_args = {}
            # PIL_VERSION_TUPLE is defined globally from PIL_VERSION_STR
            if PIL_VERSION_TUPLE >= (8, 0, 0):
                text_anchor_args = {"anchor": "ls"}
            # For older Pillow versions, text will be top-left aligned by default.
            # If precise baseline alignment is crucial, you'd need manual y-offset calculations.
            # ------------------------------------

            draw.text((20, y_title_baseline), f"{icon} {title}", font=font_title, fill=(255, 255, 255), **text_anchor_args)
            draw.text((20, y_instruction_baseline), instruction, font=font_instruction, fill=(220, 220, 220), **text_anchor_args)
            
            if self.step_completed_flag and self.success_message_start_time > 0:
                 draw.text((w // 2 - 100, y_fantastico_baseline), "FANTASTICO!", font=font_fantastico, fill=(60, 255, 60), **text_anchor_args)
            
            draw.text((w-220, y_esc_msg_baseline), "Premi ESC per uscire dal tutorial", font=font_esc, fill=(150,150,150), **text_anchor_args)

            for i in range(num_steps):
                dot_center_x = start_x_dots + i * 25
                if i < self.current_step:
                     draw.text((dot_center_x - 5, y_progress_text_baseline), "", font=font_progress_text, fill=(255,255,255), **text_anchor_args)
                elif i == self.current_step:
                    step_icon = self.steps[i].get("icon", "")
                    icon_x_on_dot = dot_center_x - 7 
                    icon_y_on_dot = y_progress_text_baseline - 2
                    draw.text((icon_x_on_dot, icon_y_on_dot), step_icon, font=font_dot_icon, fill=(255,255,255), **text_anchor_args)

            frame = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)
        
        else: # PIL_AVAILABLE is False
            cv.putText(frame, f"{icon} {title}", (20, y_title_baseline), cv.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(frame, instruction, (20, y_instruction_baseline), cv.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv.LINE_AA)
            
            if self.step_completed_flag and self.success_message_start_time > 0:
                 cv.putText(frame, "✓ FANTASTICO!", (w // 2 - 100, y_fantastico_baseline),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (60, 255, 60), 2, cv.LINE_AA)
            
            cv.putText(frame, "Premi ESC per uscire dal tutorial", (w-220, y_esc_msg_baseline), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150),1,cv.LINE_AA)

            for i in range(num_steps):
                dot_center_x = start_x_dots + i * 25
                if i < self.current_step:
                     cv.putText(frame, "", (dot_center_x - 5, y_progress_text_baseline), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv.LINE_AA)
                elif i == self.current_step:
                    step_icon = self.steps[i].get("icon", "")
                    cv.putText(frame, step_icon, (dot_center_x - 7, y_progress_text_baseline-2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1,cv.LINE_AA)
        
        return frame

# -----------------------------------------------------------------------------
# Menu Related Functions & Classes
# -----------------------------------------------------------------------------
APP_CSV_PATH = 'applications.csv'
def create_default_apps_csv():
    if not os.path.exists(APP_CSV_PATH):
        logger.info(f"Creating default apps file: {APP_CSV_PATH}")
        try:
            with open(APP_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f); writer.writerow(['label', 'path', 'color', 'icon'])
                writer.writerow(['Calculator', 'calc.exe', '#0078D7', '🧮'])
                writer.writerow(['Browser', 'https://duckduckgo.com', '#DE5833', '🌐'])
                writer.writerow(['Notepad', 'notepad.exe', '#FFDA63', '📝'])
                writer.writerow(['Tutorial', '@tutorial', '#00B894', '📚'])
        except Exception as e: logger.error(f"Failed to create {APP_CSV_PATH}: {e}")

def read_applications_from_csv():
    create_default_apps_csv(); apps = []; default_color, default_icon = '#555555', '🚀'
    try:
        with open(APP_CSV_PATH, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label=row.get('label','?'); path=row.get('path')
                if not path: logger.warning(f"Skipping row (no path): {row}"); continue
                color=row.get('color', default_color).strip(); icon=row.get('icon', default_icon).strip()
                if not (color.startswith('#') and len(color) == 7): color = default_color
                apps.append({'label': label, 'path': path, 'color': color, 'icon': icon})
        logger.info(f"Loaded {len(apps)} apps from {APP_CSV_PATH}")
    except Exception as e: logger.error(f"Error reading {APP_CSV_PATH}: {e}"); apps=[{'label':'Error','path':'','color':'#F00','icon':'⚠'}]
    return apps

def launch_application(path):
    global handpal_instance
    if not path: logger.warning("Launch attempt with empty path."); return
    logger.info(f"Launching: {path}")
    try:
        if path == '@tutorial':
            if handpal_instance and handpal_instance.tutorial_manager:
                handpal_instance.tutorial_manager.start_tutorial()
                if handpal_instance.floating_menu and handpal_instance.floating_menu.visible:
                    handpal_instance.floating_menu.hide()
            else: logger.warning("HandPal instance or tutorial manager not found for @tutorial.")
            return
            
        if path.startswith(('http://', 'https://')): webbrowser.open(path)
        elif os.name == 'nt': os.startfile(path)
        else: subprocess.Popen(path)
    except FileNotFoundError: logger.error(f"Launch failed: File not found '{path}'")
    except Exception as e: logger.error(f"Launch failed '{path}': {e}")

class FloatingMenu:
    def __init__(self, root):
        self.root = root; self.window = Toplevel(self.root)
        self.window.title("HandPal Menu"); self.window.attributes("-topmost", True)
        self.window.overrideredirect(True); 
        self.window.configure(bg='#222831'); self.apps = read_applications_from_csv()
        self._create_elements() # This will also set initial geometry
        self.window.withdraw(); self.visible = False
        self.window.bind("<ButtonPress-1>", self._start_move); self.window.bind("<ButtonRelease-1>", self._stop_move); self.window.bind("<B1-Motion>", self._do_move)
        self._offset_x = 0; self._offset_y = 0; logger.info("FloatingMenu initialized.")

    def _create_elements(self):
        title_f = Frame(self.window, bg='#222831', pady=15); title_f.pack(fill=X)
        Label(title_f, text="HANDPAL MENU", font=("Helvetica", 14, "bold"), bg='#222831', fg='#EEEEEE').pack()
        Label(title_f, text="Launch Application", font=("Helvetica", 10), bg='#222831', fg='#00ADB5').pack(pady=(0, 10))
        
        btn_cont = Frame(self.window, bg='#222831', padx=20); btn_cont.pack(fill=BOTH, expand=True); self.buttons = []
        
        num_apps = len(self.apps)
        menu_height = 160 + num_apps * 48 # Base height + (button height + pady) per app
        menu_height = min(max(menu_height, 250), 500)
        self.window.geometry(f"280x{menu_height}+50+50") # Set initial size and position

        for app in self.apps:
            f = Frame(btn_cont, bg='#222831', pady=5); f.pack(fill=X)
            btn = TkButton(f, text=f"{app.get('icon',' ')} {app.get('label','?')}", 
                           bg=app.get('color','#555'), fg="white", font=("Helvetica", 11), 
                           relief=tk.FLAT, borderwidth=0, padx=10, pady=6, width=20, anchor='w',
                           command=lambda p=app.get('path'): launch_application(p))
            btn.pack(fill=X); self.buttons.append(btn)
        
        bottom_f = Frame(self.window, bg='#222831', pady=10); bottom_f.pack(fill=X, side=tk.BOTTOM)
        TkButton(bottom_f, text="✖ Close Menu", bg='#393E46', fg='#EEEEEE', 
                 font=("Helvetica", 10), relief=tk.FLAT, borderwidth=0, 
                 padx=10, pady=5, width=15, command=self.hide).pack(pady=5)

    def show(self):
        if not self.visible: self.window.deiconify(); self.window.lift(); self.visible=True; logger.debug("Menu shown.")
    def hide(self):
        if self.visible: self.window.withdraw(); self.visible=False; logger.debug("Menu hidden.")
    def toggle(self): (self.hide if self.visible else self.show)()
    def _start_move(self, event): self._offset_x, self._offset_y = event.x, event.y
    def _stop_move(self, event): self._offset_x = self._offset_y = 0
    def _do_move(self, event): x=self.window.winfo_x()+event.x-self._offset_x; y=self.window.winfo_y()+event.y-self._offset_y; self.window.geometry(f"+{x}+{y}")

# -----------------------------------------------------------------------------
# HandPal Class
# -----------------------------------------------------------------------------
class HandPal:
    def __init__(self, config):
        self.config = config; self.mouse = Controller()
        self.gesture_recognizer = GestureRecognizer(config)
        try: 
            self.tk_root = tk.Tk(); self.tk_root.withdraw()
            self.screen_size = (self.tk_root.winfo_screenwidth(), self.tk_root.winfo_screenheight())
            self.floating_menu = FloatingMenu(self.tk_root); logger.info(f"Screen: {self.screen_size}. Menu OK.")
        except tk.TclError as e: logger.error(f"Tkinter init failed: {e}. Menu N/A."); self.tk_root=None; self.floating_menu=None; self.screen_size=(1920,1080)
        
        self.motion_smoother = MotionSmoother(config)
        self.tutorial_manager = TutorialManager(self)

        self.running = False; self.stop_event = threading.Event(); self.detection_thread = None
        self.data_queue = queue.Queue(maxsize=2)
        self.cap = None; self.mp_hands = mp.solutions.hands; self.hands_instance = None
        self.mp_drawing = mp.solutions.drawing_utils; self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.last_cursor_pos = None; self.last_right_hand_lm_norm = None
        self.tracking_enabled = True; self.debug_mode = False
        self.calibration_active = False; self.calibration_points = []; self.calibration_step = 0
        self.calib_corners = ["top-left", "top-right", "bottom-right", "bottom-left"]
        
        self.menu_trigger_zone_px = {"cx":0, "cy":0, "radius_sq":0, "is_valid": False}
        self.current_display_dims = (0, 0)
        self.menu_trigger_active = False; self._menu_activate_time = 0
        self.MENU_HOVER_DELAY = 0.3 

        self.fps_stats = deque(maxlen=60)
        self._last_proc_dims = tuple(self.config.get(key) for key in ["process_width", "process_height"])

        self.debug_values = {"fps":0.0, "det_fps":0.0, "last_act":time.time(), "cur_hist":deque(maxlen=50),
                             "map":{"raw":'-',"cal":'-',"map":'-',"smooth":'-'}, 
                             "L":"U", "R":"U", "gest":"N/A", "q":0, "menu":"Off"}
        logger.info("HandPal instance initialized.")

    def start(self):
        if self.running: return True
        logger.info("Starting HandPal..."); self.stop_event.clear()
        try:
            self.cap = cv.VideoCapture(self.config["device"], cv.CAP_DSHOW if os.name == 'nt' else cv.CAP_ANY)
            if not self.cap.isOpened(): self.cap = cv.VideoCapture(self.config["device"])
            if not self.cap.isOpened(): raise IOError(f"Cannot open webcam {self.config['device']}")
            
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.config["width"])
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.config["height"])
            self.cap.set(cv.CAP_PROP_FPS, self.config["max_fps"])
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            w, h = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv.CAP_PROP_FPS)
            logger.info(f"Webcam OK: {w}x{h} @ {fps:.1f} FPS (Requested: {self.config['width']}x{self.config['height']} @ {self.config['max_fps']})")

            self.hands_instance = self.mp_hands.Hands(
                static_image_mode=self.config["use_static_image_mode"], 
                max_num_hands=2, 
                min_detection_confidence=self.config["min_detection_confidence"], 
                min_tracking_confidence=self.config["min_tracking_confidence"])
            
            self.detection_thread = DetectionThread(self.config, self.cap, self.hands_instance, self.data_queue, self.stop_event)
            self.detection_thread.start()
            self.running = True
            logger.info("HandPal started successfully!")
            return True
        except Exception as e: 
            logger.exception(f"Startup failed: {e}")
            self.stop()
            return False

    def stop(self):
        if not self.running and not self.stop_event.is_set():
            self.stop_event.set()
            cv.destroyAllWindows()
            self._destroy_tkinter()
            return
        if not self.running: return
        
        logger.info("Stopping HandPal...")
        self.running = False
        self.stop_event.set()
        
        if self.detection_thread and self.detection_thread.is_alive():
            logger.debug("Joining DetectionThread...")
            self.detection_thread.join(timeout=2.0)
            if self.detection_thread.is_alive():
                logger.warning(f"DetectionThread did not terminate gracefully.")
        
        if hasattr(self.hands_instance, 'close') and self.hands_instance: 
            self.hands_instance.close()
            self.hands_instance = None
        
        if self.cap: 
            self.cap.release()
            self.cap = None
            logger.debug("Webcam released.")
        
        cv.destroyAllWindows()
        logger.info("OpenCV windows closed.")
        self._destroy_tkinter()
        logger.info("HandPal stopped.")

    def _destroy_tkinter(self):
        if self.tk_root:
            try: 
                logger.debug("Destroying Tkinter root...")
                self.tk_root.quit()
                self.tk_root.destroy()
                logger.debug("Tkinter root destroyed.")
            except Exception as e: logger.error(f"Error destroying Tkinter: {e}")
            self.tk_root = None; self.floating_menu = None

    def map_to_screen(self, x_norm, y_norm):
        sw, sh = self.screen_size
        if self.debug_mode:
            self.debug_values["map"]["raw"] = f"{x_norm:.3f},{y_norm:.3f}"
        
        x_cal, y_cal = x_norm, y_norm
        if self.config["calibration.enabled"] and not self.calibration_active :
            x_min,x_max = self.config["calibration.x_min"], self.config["calibration.x_max"]
            y_min,y_max = self.config["calibration.y_min"], self.config["calibration.y_max"]
            range_x = x_max - x_min
            range_y = y_max - y_min

            if range_x > 0.01 and range_y > 0.01:
                x_clamped = max(x_min, min(x_norm, x_max))
                y_clamped = max(y_min, min(y_norm, y_max))
                x_cal = (x_clamped - x_min) / range_x
                y_cal = (y_clamped - y_min) / range_y
        
        if self.debug_mode: self.debug_values["map"]["cal"]=f"{x_cal:.3f},{y_cal:.3f}"
        
        margin = self.config["calibration.screen_margin"]
        x_expanded = x_cal * (1 + 2 * margin) - margin
        y_expanded = y_cal * (1 + 2 * margin) - margin
        
        screen_x = int(x_expanded * sw)
        screen_y = int(y_expanded * sh)
        
        screen_x = max(0, min(screen_x, sw - 1))
        screen_y = max(0, min(screen_y, sh - 1))
        
        if self.debug_mode: self.debug_values["map"]["map"]=f"{screen_x},{screen_y}"
        return screen_x, screen_y

    def get_hand_pos_in_display_pixels(self):
        if self.last_right_hand_lm_norm is None: return None
        if self.current_display_dims[0] == 0 or self.current_display_dims[1] == 0: return None
        if self._last_proc_dims[0] == 0 or self._last_proc_dims[1] == 0: return None

        norm_x_proc, norm_y_proc = self.last_right_hand_lm_norm

        proc_w, proc_h = self._last_proc_dims
        display_w, display_h = self.current_display_dims

        pixel_x_proc = norm_x_proc * proc_w
        pixel_y_proc = norm_y_proc * proc_h

        scale_x = display_w / proc_w
        scale_y = display_h / proc_h

        display_x = int(pixel_x_proc * scale_x)
        display_y = int(pixel_y_proc * scale_y)

        display_x = max(0, min(display_x, display_w - 1))
        display_y = max(0, min(display_y, display_h - 1))

        return (display_x, display_y)

    def check_menu_trigger(self):
        if not self.menu_trigger_zone_px["is_valid"]:
            if self.menu_trigger_active: logger.debug("Menu trigger deactivated (Zone Invalid).")
            self.menu_trigger_active = False; self._menu_activate_time = 0
            if self.debug_mode: self.debug_values["menu"] = "Off (Zone Invalid)"
            return False

        hand_pos_px_display = self.get_hand_pos_in_display_pixels()

        if hand_pos_px_display is None:
            if self.menu_trigger_active: logger.debug("Menu trigger deactivated (Hand Lost).")
            self.menu_trigger_active = False; self._menu_activate_time = 0
            if self.debug_mode: self.debug_values["menu"] = "Off (No Hand)"
            return False

        hx_px, hy_px = hand_pos_px_display
        cx_px, cy_px = self.menu_trigger_zone_px["cx"], self.menu_trigger_zone_px["cy"]
        radius_sq_px = self.menu_trigger_zone_px["radius_sq"]

        dist_sq_px = (hx_px - cx_px)**2 + (hy_px - cy_px)**2
        is_inside = dist_sq_px < radius_sq_px

        now = time.time(); activated_for_menu_display = False
        if is_inside:
            if not self.menu_trigger_active: 
                self.menu_trigger_active = True; self._menu_activate_time = now
                logger.debug(f"Menu Trigger Zone Entered.")
            hover_time = now - self._menu_activate_time
            if hover_time >= self.MENU_HOVER_DELAY:
                activated_for_menu_display = True 
                if self.debug_mode: self.debug_values["menu"] = "ACTIVATE!"
            else:
                 if self.debug_mode: self.debug_values["menu"] = f"Hover {hover_time:.1f}s"
        else: 
            if self.menu_trigger_active: logger.debug("Menu Trigger Zone Exited.")
            self.menu_trigger_active = False; self._menu_activate_time = 0
            if self.debug_mode: self.debug_values["menu"] = "Off"
        return activated_for_menu_display

    def process_results(self, results, proc_dims):
        self._last_proc_dims = proc_dims 
        lm_l, lm_r = None, None; self.last_right_hand_lm_norm = None

        if results and results.multi_hand_landmarks and results.multi_handedness:
            for i, hand_lm in enumerate(results.multi_hand_landmarks):
                try: label = results.multi_handedness[i].classification[0].label
                except (IndexError, AttributeError): continue
                if label == "Right":
                    lm_r = hand_lm
                    try: self.last_right_hand_lm_norm = (hand_lm.landmark[8].x, hand_lm.landmark[8].y)
                    except IndexError: self.last_right_hand_lm_norm = None
                elif label == "Left": lm_l = hand_lm

        if lm_r and self.last_right_hand_lm_norm is not None:
             if self.check_menu_trigger():
                 if self.floating_menu and not self.floating_menu.visible and not self.tutorial_manager.active:
                     logger.info("Menu trigger activated by hand hover."); self.floating_menu.show()

             if self.tracking_enabled and not self.calibration_active:
                 try:
                     norm_x, norm_y = self.last_right_hand_lm_norm
                     target_x, target_y = self.map_to_screen(norm_x, norm_y)
                     smooth_x, smooth_y = self.motion_smoother.update(target_x, target_y, self.screen_size[0], self.screen_size[1])
                     if self.last_cursor_pos is None or smooth_x != self.last_cursor_pos[0] or smooth_y != self.last_cursor_pos[1]:
                         self.mouse.position = (smooth_x, smooth_y); self.last_cursor_pos = (smooth_x, smooth_y)
                         if self.debug_mode: self.debug_values["map"]["smooth"]=f"{smooth_x},{smooth_y}"; self.debug_values["cur_hist"].append(self.last_cursor_pos)
                 except Exception as e: logger.error(f"Cursor update error: {e}"); self.motion_smoother.reset()
        else: 
            self.motion_smoother.reset(); self.last_right_hand_lm_norm = None
            if self.menu_trigger_active: logger.debug("Menu trigger deactivated (Right Hand Lost).")
            self.menu_trigger_active = False; self._menu_activate_time = 0
            if self.debug_mode: self.debug_values["menu"] = "Off (No R Hand)"

        action_performed_this_frame = False
        if lm_l and not self.calibration_active:
            scroll_amount = self.gesture_recognizer.check_scroll_gesture(lm_l, "Left")
            if scroll_amount is not None:
                clicks_to_scroll = int(scroll_amount * -1) 
                if clicks_to_scroll != 0:
                    try: 
                        self.mouse.scroll(0, clicks_to_scroll); action_performed_this_frame = True
                        logger.debug(f"Scroll: {clicks_to_scroll}"); 
                        self.debug_values["last_act"]=time.time(); self.debug_values["gest"]="Scroll"
                    except Exception as e: logger.error(f"Scroll error: {e}")
            
            if not action_performed_this_frame:
                drag_action = self.gesture_recognizer.check_fist_drag(lm_l, "Left")
                if drag_action:
                    action_performed_this_frame = True
                    if drag_action == "drag_start":
                        try: 
                            self.mouse.press(Button.left)
                            logger.info("Fist Drag Start"); print("⭐ DRAG START ⭐")
                            self.debug_values["last_act"] = time.time(); self.debug_values["gest"] = "Drag ⛓️"
                        except Exception as e: logger.error(f"Drag start error: {e}")
                    elif drag_action == "drag_continue":
                        self.debug_values["gest"] = "Dragging ⛓️"
                    elif drag_action == "drag_end":
                        try:
                            self.mouse.release(Button.left)
                            logger.info("Drag End"); print("⭐ DRAG END ⭐")
                            self.debug_values["last_act"] = time.time(); self.debug_values["gest"] = "Idle"
                        except Exception as e: logger.error(f"Drag end error: {e}")
            
            if not action_performed_this_frame and not self.gesture_recognizer.gesture_state["fist_drag_active"]:
                click_type = self.gesture_recognizer.check_thumb_index_click(lm_l, "Left")
                if click_type:
                    button_to_click = Button.left
                    num_clicks = 1 if click_type == "click" else 2
                    action_name = "Click" if click_type == "click" else "DoubleClick"
                    try: 
                        self.mouse.click(button_to_click, num_clicks); action_performed_this_frame = True
                        logger.info(f"{action_name} Left"); 
                        self.debug_values["last_act"] = time.time(); self.debug_values["gest"] = action_name
                    except Exception as e: logger.error(f"{action_name} error: {e}")

        if self.debug_mode:
            self.debug_values["L"] = self.gesture_recognizer.detect_hand_pose(lm_l, "Left")
            self.debug_values["R"] = self.gesture_recognizer.detect_hand_pose(lm_r, "Right")
            if not action_performed_this_frame:
                internal_gest = self.gesture_recognizer.gesture_state.get("active_gesture")
                if not (self.debug_values["gest"] == "Dragging ⛓️" and internal_gest is None):
                    self.debug_values["gest"] = str(internal_gest) if internal_gest else "Idle"
            
        if self.tutorial_manager.active:
            self.tutorial_manager.check_step_completion(results)

    def draw_landmarks(self, frame, multi_hand_lm):
        if not multi_hand_lm: return frame
        for hand_lm in multi_hand_lm: 
            self.mp_drawing.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS, 
                                           self.mp_drawing_styles.get_default_hand_landmarks_style(), 
                                           self.mp_drawing_styles.get_default_hand_connections_style())
        return frame

    def draw_menu_trigger_circle(self, image):
        h, w = image.shape[:2]; radius_px = 30
        if w == 0 or h == 0: self.menu_trigger_zone_px["is_valid"] = False; return image
        
        center_px = (w - 50, h // 2)
        intensity = 150 + int(50 * np.sin(time.time() * 5))
        base_color = (255, intensity, 0)
        draw_color = (0, 255, 0) if self.menu_trigger_active else base_color
        
        cv.circle(image, center_px, radius_px, draw_color, -1)
        cv.circle(image, center_px, radius_px, (255, 255, 255), 1)
        cv.putText(image, "Menu", (center_px[0] - 20, center_px[1] + radius_px + 15), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        
        self.menu_trigger_zone_px["cx"] = center_px[0]
        self.menu_trigger_zone_px["cy"] = center_px[1]
        self.menu_trigger_zone_px["radius_sq"] = radius_px**2
        self.menu_trigger_zone_px["is_valid"] = True
        return image

    def draw_overlays(self, frame, results):
        self.current_display_dims = (frame.shape[1], frame.shape[0])
        h_disp, w_disp = self.current_display_dims[1], self.current_display_dims[0]
        if w_disp == 0 or h_disp == 0: return frame

        overlay_c, bg_c, accent_c, error_c = (255,255,255), (0,0,0), (0,255,255), (0,0,255)
        font, fsc_sml, fsc_med = cv.FONT_HERSHEY_SIMPLEX, 0.4, 0.5
        l_type, thick_n, thick_b = cv.LINE_AA, 1, 2

        if results and results.multi_hand_landmarks: 
            frame = self.draw_landmarks(frame, results.multi_hand_landmarks)
        
        if not self.tutorial_manager.active :
            frame = self.draw_menu_trigger_circle(frame) 

        if self.gesture_recognizer.gesture_state["fist_drag_active"]:
            drag_indicator = "DRAG ACTIVE"
            text_y_drag = 50
            if self.tutorial_manager.active: text_y_drag = 30
            cv.putText(frame, drag_indicator, (w_disp//2 - 80, text_y_drag), font, 0.8, error_c, thick_b, l_type)
            cv.rectangle(frame, (5, 5), (w_disp-5, h_disp-5), error_c, 3)

        if self.calibration_active:
            top_bar_h = 100
            bg = frame.copy(); cv.rectangle(bg, (0,0), (w_disp, top_bar_h), bg_c, -1); alpha = 0.7
            frame = cv.addWeighted(bg, alpha, frame, 1 - alpha, 0)
            
            cv.putText(frame, f"CALIBRATION ({self.calibration_step+1}/4): RIGHT Hand", (10, 25), font, fsc_med, overlay_c, thick_n, l_type)
            cv.putText(frame, f"Point index to {self.calib_corners[self.calibration_step]} corner", (10, 50), font, fsc_med, overlay_c, thick_n, l_type)
            cv.putText(frame, "Press SPACE to confirm", (10, 75), font, fsc_med, accent_c, thick_b, l_type)
            cv.putText(frame, "(ESC to cancel)", (w_disp - 150, 20), font, fsc_sml, overlay_c, thick_n, l_type)
            
            radius=15; inactive_color=(100,100,100); active_color=error_c; fill=-1
            corners_px = [(radius, radius), (w_disp-radius, radius), 
                          (w_disp-radius, h_disp-radius), (radius, h_disp-radius)]
            for i, p_corner in enumerate(corners_px): 
                cv.circle(frame, p_corner, radius, active_color if self.calibration_step==i else inactive_color, fill)
                cv.circle(frame, p_corner, radius, overlay_c, 1)
            
            hand_pos_px_display = self.get_hand_pos_in_display_pixels()
            if hand_pos_px_display: 
                cv.circle(frame, hand_pos_px_display, 10, (0, 255, 0), -1)
        
        elif not self.tutorial_manager.active :
            info_text = "C:Calib | D:Debug | Q:Exit | M:Menu | Hover circle for menu"
            if not self.debug_mode :
                 cv.putText(frame, info_text, (10, h_disp-10), font, fsc_sml, overlay_c, thick_n, l_type)
            
            fps_val = self.debug_values["fps"]
            cv.putText(frame, f"FPS: {fps_val:.1f}", (w_disp-80, 20), font, fsc_med, overlay_c, thick_n, l_type)

            if self.debug_mode:
                panel_h = 110 
                panel_y_start = h_disp - panel_h
                
                bg_debug = frame.copy()
                cv.rectangle(bg_debug, (0, panel_y_start), (w_disp, h_disp), bg_c, -1); alpha_debug = 0.7
                frame = cv.addWeighted(bg_debug, alpha_debug, frame, 1 - alpha_debug, 0)
                
                y_text = panel_y_start + 18; d_vals = self.debug_values; map_vals = d_vals['map']; cal_cfg = self.config['calibration']
                
                cv.putText(frame, info_text, (10, y_text), font, fsc_sml, overlay_c, thick_n, l_type); y_text += 18
                t1=f"L:{d_vals['L']} R:{d_vals['R']} Gest:{d_vals['gest']} Menu:{d_vals['menu']}"; 
                cv.putText(frame,t1,(10,y_text),font,fsc_sml,overlay_c,thick_n,l_type); y_text+=18
                t2=f"Map: Raw({map_vals['raw']}) Cal({map_vals['cal']}) -> MapPx({map_vals['map']}) -> Smooth({map_vals['smooth']})"; 
                cv.putText(frame,t2,(10,y_text),font,fsc_sml,overlay_c,thick_n,l_type); y_text+=18
                t3=f"Calib X[{cal_cfg['x_min']:.2f}-{cal_cfg['x_max']:.2f}] Y[{cal_cfg['y_min']:.2f}-{cal_cfg['y_max']:.2f}] En:{cal_cfg['enabled']}"; 
                cv.putText(frame,t3,(10,y_text),font,fsc_sml,overlay_c,thick_n,l_type); y_text+=18
                time_since_act = time.time()-d_vals['last_act']
                t4=f"LastAct:{time_since_act:.1f}s ago | QSize:{d_vals['q']}"; 
                cv.putText(frame,t4,(10,y_text),font,fsc_sml,overlay_c,thick_n,l_type); y_text+=18
                
                if len(d_vals["cur_hist"]) > 1:
                    pts_screen_coords = np.array(list(d_vals["cur_hist"]),dtype=np.int32)
                    if self.screen_size[0]>0 and self.screen_size[1]>0:
                        pts_frame_coords=pts_screen_coords.copy()
                        pts_frame_coords[:,0]=pts_frame_coords[:,0]*w_disp//self.screen_size[0]
                        pts_frame_coords[:,1]=pts_frame_coords[:,1]*h_disp//self.screen_size[1]
                        pts_frame_coords=np.clip(pts_frame_coords,[0,0],[w_disp-1,h_disp-1])
                        cv.polylines(frame,[pts_frame_coords],False,accent_c,1, lineType=cv.LINE_AA)
        
        if self.tutorial_manager.active:
            frame = self.tutorial_manager.draw_tutorial_overlay(frame)
            
        return frame

    def main_loop(self):
        logger.info("Main loop starting."); win_name="HandPal Control"; cv.namedWindow(win_name,cv.WINDOW_NORMAL)
        dw_cfg, dh_cfg = self.config.get('display_width'), self.config.get('display_height')
        if dw_cfg and dh_cfg:
            try: cv.resizeWindow(win_name, dw_cfg, dh_cfg); logger.info(f"Set display size to {dw_cfg}x{dh_cfg}")
            except Exception as e: logger.warning(f"Failed to set display size: {e}")
        
        last_valid_frame = None

        while self.running:
            loop_start_time = time.perf_counter()
            
            if self.tk_root:
                try: self.tk_root.update_idletasks(); self.tk_root.update()
                except tk.TclError as e:
                    if "application has been destroyed" in str(e).lower(): 
                        logger.info("Tkinter root destroyed. Stopping."); self.stop(); break
                    else: logger.error(f"Tkinter error: {e}")

            current_frame_raw, mp_results, proc_dims_tuple = None, None, None
            try:
                current_frame_raw, mp_results, proc_dims_tuple = self.data_queue.get(block=True, timeout=0.01)
                last_valid_frame = current_frame_raw.copy()
                if self.debug_mode: self.debug_values["q"] = self.data_queue.qsize()
            except queue.Empty:
                current_frame_raw = last_valid_frame 
                mp_results = None
                proc_dims_tuple = self._last_proc_dims
            except Exception as e: 
                logger.exception(f"Data queue error: {e}"); time.sleep(0.1); continue

            if current_frame_raw is not None:
                 if mp_results is not None and proc_dims_tuple is not None:
                     try: self.process_results(mp_results, proc_dims_tuple)
                     except Exception as e: 
                         logger.exception("process_results error")

                 display_frame = current_frame_raw.copy()
                 try: display_frame_with_overlays = self.draw_overlays(display_frame, mp_results)
                 except Exception as e: 
                     logger.exception("draw_overlays error")
                     cv.putText(display_frame,"DRAW ERR",(50,100),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                     display_frame_with_overlays = display_frame

                 final_display_frame = display_frame_with_overlays
                 if dw_cfg and dh_cfg:
                     h_curr, w_curr = final_display_frame.shape[:2]
                     if w_curr != dw_cfg or h_curr != dh_cfg:
                         try: final_display_frame = cv.resize(final_display_frame, (dw_cfg,dh_cfg), interpolation=cv.INTER_LINEAR)
                         except Exception as e: logger.error(f"Display resize error: {e}")
                 
                 cv.imshow(win_name, final_display_frame)
            else:
                time.sleep(0.01)

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'): logger.info("'q' pressed, stopping."); self.stop(); break
            elif key == ord('c'):
                if not self.calibration_active and not self.tutorial_manager.active: self.start_calibration()
            elif key == ord('d'): 
                self.debug_mode = not self.debug_mode
                logger.info(f"Debug mode toggled to: {self.debug_mode}")
                if self.debug_mode: self.debug_values["cur_hist"].clear()
            elif key == ord('m'):
                 if self.floating_menu and not self.tutorial_manager.active: 
                     logger.debug("'m' pressed, toggling menu."); self.floating_menu.toggle()
            elif key == 27: # ESC
                if self.calibration_active: self.cancel_calibration()
                elif self.tutorial_manager.active: self.tutorial_manager.stop_tutorial()
                elif self.floating_menu and self.floating_menu.visible: self.floating_menu.hide()
            elif key == 32: # Spacebar
                if self.calibration_active: self.process_calibration_step()

            elapsed_time = time.perf_counter() - loop_start_time
            self.fps_stats.append(elapsed_time)
            if self.fps_stats: 
                avg_duration = sum(self.fps_stats)/len(self.fps_stats)
                self.debug_values["fps"] = 1.0/avg_duration if avg_duration > 0 else 0

        logger.info("Main loop finished."); cv.destroyAllWindows()

    def start_calibration(self):
        if self.calibration_active: return
        logger.info("Starting calibration..."); print("\n--- CALIBRATION START ---")
        self.calibration_active = True; self.config["calibration.active"] = True
        self.calibration_points = []; self.calibration_step = 0
        self.tracking_enabled = False; self.motion_smoother.reset()
        if self.floating_menu and self.floating_menu.visible: self.floating_menu.hide()
        print(f"Step {self.calibration_step+1}/4: Point RIGHT index to {self.calib_corners[0]}. Press SPACE. (ESC to cancel)")

    def cancel_calibration(self):
        if not self.calibration_active: return
        logger.info("Calibration cancelled."); print("\n--- CALIBRATION CANCELLED ---")
        self.calibration_active = False; self.config["calibration.active"] = False
        self.calibration_points = []; self.calibration_step = 0
        self.tracking_enabled = True

    def process_calibration_step(self):
        if not self.calibration_active: return
        
        hand_pos_on_display_px = self.get_hand_pos_in_display_pixels() 
        if hand_pos_on_display_px is None: 
            print("(!) Right hand not detected or not providing landmarks!"); return
        
        display_w, display_h = self.current_display_dims
        if display_w == 0 or display_h == 0: 
            print("(!) Invalid display window dimensions for calibration!"); return

        norm_x_on_display = hand_pos_on_display_px[0] / display_w
        norm_y_on_display = hand_pos_on_display_px[1] / display_h
        
        norm_x_on_display = max(0.0, min(1.0, norm_x_on_display))
        norm_y_on_display = max(0.0, min(1.0, norm_y_on_display))

        self.calibration_points.append((norm_x_on_display, norm_y_on_display))
        logger.info(f"Calib point {self.calibration_step+1} ({self.calib_corners[self.calibration_step]}): DisplayNorm=({norm_x_on_display:.3f},{norm_y_on_display:.3f})")
        print(f"-> Point {self.calibration_step+1}/4 ({self.calib_corners[self.calibration_step]}) captured.")
        
        self.calibration_step += 1
        if self.calibration_step >= 4: self.complete_calibration()
        else: print(f"\nStep {self.calibration_step+1}/4: Point RIGHT index to {self.calib_corners[self.calibration_step]}. Press SPACE.")

    def complete_calibration(self):
        logger.info("Completing calibration...")
        if len(self.calibration_points) != 4: 
            logger.error(f"Incorrect point count for calibration: {len(self.calibration_points)}"); 
            print("(!) ERROR: Incorrect number of calibration points."); self.cancel_calibration(); return
        
        try:
            xs_norm_display=[p[0] for p in self.calibration_points] 
            ys_norm_display=[p[1] for p in self.calibration_points]
            
            xmin_disp_norm, xmax_disp_norm = min(xs_norm_display), max(xs_norm_display)
            ymin_disp_norm, ymax_disp_norm = min(ys_norm_display), max(ys_norm_display)
            
            if (xmax_disp_norm-xmin_disp_norm < 0.05 or ymax_disp_norm-ymin_disp_norm < 0.05): 
                print("(!) WARNING: Calibration area is very small. Results may be inaccurate.")
            
            self.config.set("calibration.x_min", xmin_disp_norm)
            self.config.set("calibration.x_max", xmax_disp_norm)
            self.config.set("calibration.y_min", ymin_disp_norm)
            self.config.set("calibration.y_max", ymax_disp_norm)
            self.config.set("calibration.enabled", True)
            self.config.save()
            logger.info(f"Calibration saved: X_norm_display[{xmin_disp_norm:.3f}-{xmax_disp_norm:.3f}], Y_norm_display[{ymin_disp_norm:.3f}-{ymax_disp_norm:.3f}]")
            print("\n--- CALIBRATION SAVED ---")
        except Exception as e: 
            logger.exception("Calibration completion/save error."); print("(!) ERROR saving calibration.")
            self.config.set("calibration.enabled", False)
        finally: 
            self.calibration_active = False; self.config["calibration.active"] = False
            self.calibration_points = []; self.calibration_step = 0
            self.tracking_enabled = True

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="HandPal - Mouse control with gestures & menu.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cfg = Config.DEFAULT_CONFIG 
    parser.add_argument('--device', type=int, default=None, help=f'Webcam ID (Default: {cfg["device"]})')
    parser.add_argument('--width', type=int, default=None, help=f'Webcam Width (Default: {cfg["width"]})')
    parser.add_argument('--height', type=int, default=None, help=f'Webcam Height (Default: {cfg["height"]})')
    parser.add_argument('--display-width', type=int, default=None, help='Preview window width (Default: webcam frame size)')
    parser.add_argument('--display-height', type=int, default=None, help='Preview window height (Default: webcam frame size)')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug overlay & logging.')
    parser.add_argument('--flip-camera', action='store_true', default=None, help=f'Flip webcam horizontally (Default: {cfg["flip_camera"]})')
    parser.add_argument('--no-flip', dest='flip_camera', action='store_false', help='Disable horizontal flip.')
    parser.add_argument('--calibrate', action='store_true', default=False, help='Start calibration on launch.')
    parser.add_argument('--reset-config', action='store_true', default=False, help='Delete config file and exit.')
    parser.add_argument('--config-file', type=str, default=Config.CONFIG_FILENAME, help='Path to config JSON file.')
    parser.add_argument('--cursor-file', type=str, default=None, help=f'Path to custom .cur file (Win only) (Default: {cfg["custom_cursor_path"]})')
    parser.add_argument('--tutorial', action='store_true', default=False, help='Start with the tutorial.')
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    args = parse_arguments(); cursor_set = False
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(log_level); [h.setLevel(log_level) for h in logger.handlers]
    logger.info(f"Log level set to {logging.getLevelName(log_level)}")

    if args.reset_config:
        path = args.config_file
        try:
            if os.path.exists(path): os.remove(path); logger.info(f"Removed config: {path}"); print(f"Removed config: {path}")
            else: logger.info("No config file found to reset."); print("No config file found to reset.")
            return 0
        except Exception as e: logger.error(f"Error removing config: {e}"); print(f"Error removing config: {e}"); return 1

    create_default_apps_csv()
    config = Config(args)

    if os.name == 'nt':
        cursor_path_cfg = config.get('custom_cursor_path')
        if cursor_path_cfg:
             res_path = None
             if not os.path.isabs(cursor_path_cfg) and not os.path.exists(cursor_path_cfg):
                 try: script_dir = os.path.dirname(os.path.abspath(__file__))
                 except NameError: script_dir = os.getcwd()
                 pot_path = os.path.join(script_dir, cursor_path_cfg); res_path = pot_path if os.path.exists(pot_path) else None
             else: res_path = cursor_path_cfg
             
             if res_path and os.path.exists(res_path):
                 logger.info(f"Attempting to set cursor: {res_path}"); cursor_set = set_custom_cursor(res_path)
             elif cursor_path_cfg: logger.error(f"Cursor file specified ('{cursor_path_cfg}') but not found/accessible.")
    
    app_instance = None
    global handpal_instance
    try:
        app_instance = HandPal(config)
        handpal_instance = app_instance

        if args.debug: app_instance.debug_mode = True
        if app_instance.start():
            if args.calibrate: 
                logger.info("Starting calibration due to --calibrate flag..."); time.sleep(0.75)
                app_instance.start_calibration()
            if args.tutorial and not args.calibrate:
                logger.info("Starting tutorial due to --tutorial flag..."); time.sleep(0.75)
                app_instance.tutorial_manager.start_tutorial()

            app_instance.main_loop()
        else: 
            logger.error("HandPal failed to start."); print("ERROR: HandPal failed to start. Check handpal.log for details."); return 1
    except KeyboardInterrupt: logger.info("Ctrl+C received. Stopping HandPal...")
    except Exception as e: 
        logger.exception("Unhandled error in main execution block."); 
        print(f"UNHANDLED ERROR: {e}. Check handpal.log for details."); return 1
    finally:
        logger.info("Main 'finally' block: Executing cleanup...")
        if app_instance: app_instance.stop()
        if cursor_set: logger.info("Restoring default system cursor..."); restore_default_cursor()
        logger.info("HandPal terminated."); print("\nHandPal terminated.")
        time.sleep(0.1)
    return 0

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())