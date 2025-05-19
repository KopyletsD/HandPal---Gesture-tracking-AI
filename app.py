import argparse
import threading
import os
import sys
import time
import numpy as np
import cv2 as cv
import mediapipe as mp
import tkinter as tk
from tkinter import Toplevel, Frame, Label, Button as TkButton, BOTH, X, Text, Scrollbar, RIGHT, Y, LEFT, TOP
from pynput.mouse import Controller, Button
from pynput.keyboard import Key, Controller as KeyboardController # Renamed to avoid conflict
from collections import deque
import json
import logging
import ctypes
import queue
import csv
import subprocess
import webbrowser

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
# Windows Cursor Functions (Unchanged)
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
# Debug Window Class (Unchanged)
# -----------------------------------------------------------------------------
class DebugWindow:
    def __init__(self, root):
        self.root = root
        self.window = None
        self.text_widget = None
        self.visible = False
        self.last_update = 0
        self.update_interval = 0.2  # Aggiorna ogni 0.2 secondi
        self._create_window()
        
    def _create_window(self):
        self.window = Toplevel(self.root)
        self.window.title("HandPal Debug")
        self.window.attributes("-topmost", True)
        self.window.geometry("400x300+800+100")
        self.window.configure(bg='#1E1E1E')
        
        frame = Frame(self.window, bg='#1E1E1E')
        frame.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = Scrollbar(frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        self.text_widget = Text(frame, wrap=tk.WORD, bg='#1E1E1E', fg='#CCCCCC', 
                            font=('Consolas', 9), padx=5, pady=5)
        self.text_widget.pack(side=LEFT, fill=BOTH, expand=True)
        
        self.text_widget.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.text_widget.yview)
        
        close_btn = TkButton(self.window, text="Chiudi Debug", bg='#555555', fg='white',
                         command=self.hide)
        close_btn.pack(side=TOP, pady=5)
        
        self.window.withdraw()
        
    def show(self):
        if not self.visible:
            self.window.deiconify()
            self.window.lift()
            self.visible = True
            logger.debug("Debug window shown")
            
    def hide(self):
        if self.visible:
            self.window.withdraw()
            self.visible = False
            logger.debug("Debug window hidden")
            
    def toggle(self):
        if self.visible:
            self.hide()
        else:
            self.show()
            
    def update(self, debug_values):
        now = time.time()
        if not self.visible or now - self.last_update < self.update_interval:
            return
            
        self.last_update = now
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(tk.END, "=== HANDPAL DEBUG INFO ===\n\n")
        self.text_widget.insert(tk.END, f"FPS: {debug_values['fps']:.1f}\n")
        self.text_widget.insert(tk.END, f"\nHAND POSE:\n")
        self.text_widget.insert(tk.END, f"Left: {debug_values['L']}\n")
        self.text_widget.insert(tk.END, f"Right: {debug_values['R']}\n")
        self.text_widget.insert(tk.END, f"Active Gesture: {debug_values['gest']}\n")
        
        if 'zoom' in debug_values:
            self.text_widget.insert(tk.END, f"\nZOOM INFO:\n")
            self.text_widget.insert(tk.END, f"Active: {debug_values['zoom'].get('active', False)}\n")
            self.text_widget.insert(tk.END, f"Distance: {debug_values['zoom'].get('distance', 0):.3f}\n")
            self.text_widget.insert(tk.END, f"Delta: {debug_values['zoom'].get('delta', 0):.3f}\n")
        
        self.text_widget.insert(tk.END, f"\nMENU STATUS: {debug_values['menu']}\n")
        
        self.text_widget.insert(tk.END, f"\nMAPPING:\n")
        m = debug_values['map']
        self.text_widget.insert(tk.END, f"Raw: {m['raw']}\n")
        self.text_widget.insert(tk.END, f"Calibrated: {m['cal']}\n")
        self.text_widget.insert(tk.END, f"Mapped: {m['map']}\n")
        self.text_widget.insert(tk.END, f"Smoothed: {m['smooth']}\n")
        
        t_act = time.time() - debug_values["last_act"]
        self.text_widget.insert(tk.END, f"\nLast Action: {t_act:.1f}s ago\n")
        self.text_widget.insert(tk.END, f"Queue Size: {debug_values['q']}\n")
        
        c = debug_values.get('calib', {})
        if c:
            self.text_widget.insert(tk.END, f"\nCALIBRATION:\n")
            self.text_widget.insert(tk.END, f"X Range: [{c.get('x_min', '?'):.2f} - {c.get('x_max', '?'):.2f}]\n")
            self.text_widget.insert(tk.END, f"Y Range: [{c.get('y_min', '?'):.2f} - {c.get('y_max', '?'):.2f}]\n")
            self.text_widget.insert(tk.END, f"Enabled: {c.get('enabled', False)}\n")

# -----------------------------------------------------------------------------
# Config Class
# -----------------------------------------------------------------------------
class Config:
    DEFAULT_CONFIG = {
    "device": 0,
    "width": 1920, # Preferred camera capture width
    "height": 1440, # Preferred camera capture height
    "process_width": 640, # Width for MediaPipe processing
    "process_height": 360, # Height for MediaPipe processing
    "flip_camera": True,  # MODIFIED: Default is False (not flipped)
    "display_width": None, # OpenCV window width (None for auto)
    "display_height": None, # OpenCV window height (None for auto)
    "min_detection_confidence": 0.6,
    "min_tracking_confidence": 0.5,
    "use_static_image_mode": False,
    "smoothing_factor": 0.7,
    "inactivity_zone": 0.015,
    "click_cooldown": 0.4,
    "gesture_sensitivity": 0.02,
    "gesture_settings": {
        "scroll_sensitivity": 4,
        "double_click_time": 0.35,
        "zoom_sensitivity": 15.0, # MODIFIED: Increased sensitivity for more noticeable zoom
                                 # This scales the raw pinch delta.
        "zoom_activation_dist_max": 0.15, # Normalized distance: thumb-index closer than this to start zoom
        "zoom_deactivation_dist_min": 0.015, # Deactivate if pinch gets extremely tight
        "zoom_deactivation_dist_max": 0.30, # Deactivate if fingers open too wide
        "min_zoom_delta_magnitude": 0.003 # Min smoothed normalized delta to trigger zoom action
    },
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
        self.config = json.loads(json.dumps(self.DEFAULT_CONFIG)) 
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
                # MODIFIED: Simplified flip_camera logic
                if key == 'flip_camera': 
                    self.config['flip_camera'] = value # value is True if --flip-camera is passed
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
        if key.startswith("display_"): self._validate_display_dims(); return True # MODIFIED: return True

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
            "scroll_active": False, "last_click_time": 0, "last_click_button": None,
            "scroll_history": deque(maxlen=5), "active_gesture": None,
            "last_pose": {"Left": "U", "Right": "U"}, "pose_stable_count": {"Left": 0, "Right": 0},
            "zoom_active": False, "last_pinch_distance": None, "zoom_history": deque(maxlen=5)
        }
        self.POSE_STABILITY_THRESHOLD = 3

    def _dist(self, p1, p2):
        if None in [p1, p2] or not all(hasattr(p, 'x') for p in [p1,p2]): return float('inf')
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def check_thumb_index_click(self, hand_lm, handedness):
        # MODIFIED: Allow click during zoom_pending for better UX (e.g. to close menu while zoom gesture is primed)
        if handedness != "Left" or hand_lm is None or self.gesture_state["scroll_active"] or self.gesture_state.get("active_gesture") == "zoom":
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
                if (now - self.gesture_state["last_click_time"]) < double_click_t and self.gesture_state["last_click_button"] == Button.left:
                    gesture = "double_click"
                else:
                    gesture = "click"
                self.gesture_state["last_click_time"] = now
                self.gesture_state["last_click_button"] = Button.left
                self.gesture_state["active_gesture"] = "click" # Keep this specific to click
        elif self.gesture_state["active_gesture"] == "click": # Only reset if it was a click
             self.gesture_state["active_gesture"] = None
        return gesture

    def check_scroll_gesture(self, hand_landmarks, handedness):
         if handedness != "Left":
             return None
         if self.gesture_state["active_gesture"] == "click" or self.gesture_state["zoom_active"]:
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
                 self.gesture_state["active_gesture"] = "scroll"
             
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
                         return smooth_delta * 100 # MODIFIED: Was 100, but scroll takes integer clicks. Will be converted later.
                                                    # Return the smoothed delta directly. HandPal will decide scroll clicks.
             self.last_positions[8] = (index_tip.x, index_tip.y)
         else:
             if self.gesture_state["scroll_active"]:
                 self.gesture_state["scroll_active"] = False
                 self.gesture_state["scroll_history"].clear()
                 if self.gesture_state["active_gesture"] == "scroll":
                     self.gesture_state["active_gesture"] = None
         return None

    # NEW: Zoom Gesture Implementation
    def check_zoom_gesture(self, hand_lm, handedness):
        if handedness != "Right" or hand_lm is None:
            if self.gesture_state["zoom_active"]: # Ensure reset if hand lost
                self.gesture_state["zoom_active"] = False
                self.gesture_state["active_gesture"] = None
                self.gesture_state["last_pinch_distance"] = None
                self.gesture_state["zoom_history"].clear()
            return None

        # Avoid conflict with click if left hand is also performing one
        # (though zoom is right hand, click is left, so less direct conflict)

        try:
            thumb_tip = hand_lm.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP] # LM 4
            index_tip = hand_lm.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP] # LM 8
        except (IndexError, TypeError):
            if self.gesture_state["zoom_active"]:
                self.gesture_state["zoom_active"] = False; self.gesture_state["active_gesture"] = None
                self.gesture_state["last_pinch_distance"] = None; self.gesture_state["zoom_history"].clear()
            return None

        current_dist = self._dist(thumb_tip, index_tip)
        
        gs = self.config["gesture_settings"]
        ZOOM_ACTIVATION_MAX = gs["zoom_activation_dist_max"]
        ZOOM_DEACTIVATION_MIN = gs["zoom_deactivation_dist_min"]
        ZOOM_DEACTIVATION_MAX = gs["zoom_deactivation_dist_max"]
        MIN_ZOOM_DELTA = gs["min_zoom_delta_magnitude"]
        ZOOM_SENSITIVITY = gs["zoom_sensitivity"]

        if not self.gesture_state["zoom_active"]:
            # Try to activate zoom: fingers must be pinched enough but not too much
            if ZOOM_DEACTIVATION_MIN < current_dist < ZOOM_ACTIVATION_MAX:
                self.gesture_state["zoom_active"] = True
                self.gesture_state["last_pinch_distance"] = current_dist
                self.gesture_state["zoom_history"].clear()
                self.gesture_state["active_gesture"] = "zoom_pending" # Indicates zoom mode is on, but not necessarily moving
                logger.info(f"Zoom gesture ACTIVATED. Initial distance: {current_dist:.3f}")
                return {"active": True, "distance": current_dist, "delta": 0.0}
            else:
                return None # Not in activation range
        else: # Zoom is active
            # Check for deactivation conditions (too close, too far)
            if not (ZOOM_DEACTIVATION_MIN < current_dist < ZOOM_DEACTIVATION_MAX):
                self.gesture_state["zoom_active"] = False
                self.gesture_state["active_gesture"] = None
                self.gesture_state["last_pinch_distance"] = None
                self.gesture_state["zoom_history"].clear()
                logger.info(f"Zoom gesture DEACTIVATED. Distance: {current_dist:.3f}")
                return {"active": False, "distance": current_dist, "delta": 0.0} # Deactivation signal

            raw_delta = current_dist - self.gesture_state["last_pinch_distance"]
            self.gesture_state["zoom_history"].append(raw_delta)
            
            smooth_delta = sum(self.gesture_state["zoom_history"]) / len(self.gesture_state["zoom_history"])
            self.gesture_state["last_pinch_distance"] = current_dist
            
            if abs(smooth_delta) > MIN_ZOOM_DELTA:
                scaled_delta = smooth_delta * ZOOM_SENSITIVITY
                self.gesture_state["active_gesture"] = "zoom" # Actively zooming
                # logger.debug(f"Zoom action: dist={current_dist:.3f}, raw_delta={raw_delta:.4f}, smooth_delta={smooth_delta:.4f}, scaled_delta={scaled_delta:.4f}")
                return {"active": True, "distance": current_dist, "delta": scaled_delta}
            else:
                self.gesture_state["active_gesture"] = "zoom_pending" # Still in zoom mode, but no significant movement
                return {"active": True, "distance": current_dist, "delta": 0.0}


    def detect_hand_pose(self, hand_lm, handedness):
        if hand_lm is None or handedness not in ["Left", "Right"]: return "U"
        try: lm=hand_lm.landmark; w,tt,it,mt,rt,pt=lm[0],lm[4],lm[8],lm[12],lm[16],lm[20]; im,mm,rm,pm=lm[5],lm[9],lm[13],lm[17]
        except (IndexError, TypeError): return "U"
        y_ext = 0.03; t_ext=tt.y<w.y-y_ext; i_ext=it.y<im.y-y_ext; m_ext=mt.y<mm.y-y_ext; r_ext=rt.y<rm.y-y_ext; p_ext=pt.y<pm.y-y_ext
        num_ext = sum([i_ext,m_ext,r_ext,p_ext]); pose = "U"
        
        # MODIFIED: Check active_gesture for more accurate pose display during zoom
        active_gest = self.gesture_state.get("active_gesture")
        if handedness=="Left" and self.gesture_state["scroll_active"]: pose="Scroll"
        elif handedness=="Right" and (active_gest == "zoom" or active_gest == "zoom_pending"): pose="Zoom" # MODIFIED
        elif i_ext and num_ext==1: pose="Point"
        elif i_ext and m_ext and num_ext==2: pose="Two"
        elif num_ext >= 4: pose="Open"
        elif num_ext==0 and not t_ext: pose="Fist"
        
        last=self.gesture_state["last_pose"].get(handedness,"U"); count=self.gesture_state["pose_stable_count"].get(handedness,0)
        if pose==last and pose!="U": count+=1
        else: count=0
        self.gesture_state["last_pose"][handedness]=pose; self.gesture_state["pose_stable_count"][handedness]=count
        return pose if count>=self.POSE_STABILITY_THRESHOLD or pose in ["Scroll", "Zoom"] else last+"?"


# -----------------------------------------------------------------------------
# MotionSmoother Class (Unchanged)
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
        last_norm_x = self.last_smooth_pos[0]/screen_w; last_norm_y = self.last_smooth_pos[1]/screen_h
        target_norm_x = target_x/screen_w; target_norm_y = target_y/screen_h
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
        self.flip=config["flip_camera"] # MODIFIED: Reads from config
        logger.info(f"DetectionThread: ProcRes={self.proc_w}x{self.proc_h}, Flip={self.flip}")

    def run(self):
        logger.info("DetectionThread starting"); t_start=time.perf_counter(); frame_n=0
        while not self.stop_evt.is_set():
            if self.cap is None or not self.cap.isOpened(): logger.error("Webcam N/A"); time.sleep(0.5); continue
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None: time.sleep(0.05); continue

                # MODIFIED: Apply flip to the original frame if configured
                if self.flip:
                    frame = cv.flip(frame, 1)

                proc_frame = cv.resize(frame, (self.proc_w, self.proc_h), interpolation=cv.INTER_LINEAR)
                rgb_frame = cv.cvtColor(proc_frame, cv.COLOR_BGR2RGB); rgb_frame.flags.writeable=False
                results = self.hands.process(rgb_frame); rgb_frame.flags.writeable=True
                try: 
                    # The 'frame' passed to queue is now potentially flipped
                    self.data_q.put((frame, results, (self.proc_w, self.proc_h)), block=False)
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
class TutorialManager:
    def __init__(self, handpal):
        self.handpal = handpal
        self.active = False
        self.current_step = 0
        self.step_completed = False
        self.completion_timer = 0
        
        self.last_gesture_time = 0
        self.gesture_clicks_detected = 0
        self.last_gesture_click_time = 0
        
        self.mouse_clicks_detected = 0
        self.last_mouse_click_time = 0
        self.mouse_double_click_detected = False
        
        self.zoom_activity_detected = False # Tracks if any zoom movement happened
        self.min_zoom_delta_seen = 0 # Tracks if zoom-out occurred
        self.max_zoom_delta_seen = 0 # Tracks if zoom-in occurred
        self.last_zoom_activity_time = 0 # Time of last significant zoom movement

        self.scroll_detected = False
        self.corners_visited = [False, False, False, False]
        
        # MODIFIED: Replaced emoji icons with ASCII alternatives for better OpenCV compatibility
        self.steps = [
            {
                "title": "Benvenuto in HandPal",
                "instruction": "Posiziona la mano DESTRA nello schermo per muovere il cursore",
                "completion_type": "position",
                "icon": "Hi!", # MODIFIED
                "allows_mouse": False
            },
            {
                "title": "Movimento del Cursore",
                "instruction": "Muovi il cursore verso i 4 angoli dello schermo",
                "completion_type": "corners",
                "icon": "->", # MODIFIED
                "allows_mouse": True
            },
            {
                "title": "Click Sinistro",
                "instruction": "Avvicina pollice e indice della mano SINISTRA oppure fai click col mouse",
                "completion_type": "click",
                "icon": "(C)", # MODIFIED
                "allows_mouse": True
            },
            {
                "title": "Doppio Click",
                "instruction": "Fai due click rapidi con la mano SINISTRA oppure doppio click col mouse",
                "completion_type": "double_click",
                "icon": "(CC)", # MODIFIED
                "allows_mouse": True
            },
            {
                "title": "Zoom In/Out", # Title was fine, icon was the issue
                "instruction": "Con la mano DESTRA, avvicina e allontana pollice e indice per zoom",
                "completion_type": "zoom",
                "icon": "(Zm)", # MODIFIED
                "allows_mouse": False
            },
            {
                "title": "Scorrimento",
                "instruction": "Con la mano SINISTRA, estendi indice e medio per scorrere",
                "completion_type": "scroll",
                "icon": "(S)", # MODIFIED
                "allows_mouse": False
            },
            {
                "title": "Menu",
                "instruction": "Posiziona l'indice destro sul cerchio del menu e mantieni",
                "completion_type": "menu_hover",
                "icon": "(M)", # MODIFIED
                "allows_mouse": False
            },
            {
                "title": "Complimenti!",
                "instruction": "Hai completato il tutorial! Fai click per terminare",
                "completion_type": "click",
                "icon": "Done", # MODIFIED
                "allows_mouse": True
            }
        ]
    
    def start_tutorial(self):
        self.active = True
        self.current_step = 0
        self.step_completed = False
        self.completion_timer = 0
        self.handpal.tracking_enabled = True # Ensure tracking is on for tutorial
        self.reset_step_counters()
        logger.info("Tutorial started")
        
    def stop_tutorial(self):
        self.active = False
        logger.info("Tutorial stopped")
        
    def reset_step_counters(self):
        self.last_gesture_time = time.time()
        self.gesture_clicks_detected = 0
        self.last_gesture_click_time = 0
        self.mouse_clicks_detected = 0
        self.last_mouse_click_time = 0
        self.mouse_double_click_detected = False
        
        self.zoom_activity_detected = False
        self.min_zoom_delta_seen = 0
        self.max_zoom_delta_seen = 0
        self.last_zoom_activity_time = 0

        self.scroll_detected = False
        self.corners_visited = [False, False, False, False]
        
    def next_step(self):
        self.current_step += 1
        self.step_completed = False
        self.completion_timer = 0
        self.reset_step_counters()
        
        if self.current_step >= len(self.steps):
            self.stop_tutorial()
            return True
        return False
    
    def register_mouse_click(self, is_double=False):
        if not self.active or self.step_completed: return
        current_step_def = self.steps[self.current_step] # Renamed to avoid conflict
        completion_type = current_step_def.get("completion_type", "")
        allows_mouse = current_step_def.get("allows_mouse", False)
        
        if not allows_mouse: return
        now = time.time()
        
        if is_double and completion_type == "double_click":
            self.mouse_double_click_detected = True; self.step_completed = True; return
            
        if completion_type == "click":
            self.step_completed = True; return
            
        if completion_type == "double_click":
            if now - self.last_mouse_click_time < self.handpal.config.get("gesture_settings.double_click_time", 0.5) + 0.1: # Allow a bit more time for mouse
                self.mouse_clicks_detected += 1
                if self.mouse_clicks_detected >= 2: self.step_completed = True
            else: self.mouse_clicks_detected = 1
            self.last_mouse_click_time = now
    
    def check_step_completion(self, results):
        if not self.active or self.step_completed: return
            
        current_step_def = self.steps[self.current_step] # Renamed
        completion_type = current_step_def.get("completion_type", "")
        now = time.time()
        
        if completion_type == "position" and self.handpal.last_right_hand_lm_norm is not None:
            if self.completion_timer == 0: self.completion_timer = now # Start timer when hand is first seen
            if now - self.completion_timer > 1.0: # Hand must be present for 1 second
                logger.info("Tutorial: Hand position detected")
                self.step_completed = True
                self.last_gesture_time = now # Update for auto-advance
        elif completion_type == "position" and self.handpal.last_right_hand_lm_norm is None:
            self.completion_timer = 0 # Reset if hand lost
                
        elif completion_type == "corners":
            if self.handpal.last_cursor_pos is None: return
            x, y = self.handpal.last_cursor_pos
            screen_w, screen_h = self.handpal.screen_size
            margin = int(min(screen_w, screen_h) * 0.1) # 10% margin
            
            if x < margin and y < margin: self.corners_visited[0] = True
            if x > screen_w - margin and y < margin: self.corners_visited[1] = True
            if x > screen_w - margin and y > screen_h - margin: self.corners_visited[2] = True
            if x < margin and y > screen_h - margin: self.corners_visited[3] = True
                
            if all(self.corners_visited):
                logger.info("Tutorial: All corners visited")
                self.step_completed = True; self.last_gesture_time = now
                
        elif completion_type == "click":
            last_click_time = self.handpal.gesture_recognizer.gesture_state.get("last_click_time", 0)
            if last_click_time > self.last_gesture_click_time: # New gesture click
                self.last_gesture_click_time = last_click_time
                self.step_completed = True; self.last_gesture_time = now
                
        elif completion_type == "double_click":
            if self.mouse_double_click_detected: return # Already done by mouse
            last_g_click_time = self.handpal.gesture_recognizer.gesture_state.get("last_click_time", 0)
            
            if last_g_click_time > self.last_gesture_click_time: # New gesture click
                if now - self.last_gesture_click_time < self.handpal.config.get("gesture_settings.double_click_time", 0.35) + 0.05: # Slightly more lenient for tutorial
                    self.gesture_clicks_detected += 1
                    if self.gesture_clicks_detected >= 2:
                        self.step_completed = True; self.last_gesture_time = now
                else: self.gesture_clicks_detected = 1 # First click of a potential double
                self.last_gesture_click_time = last_g_click_time
            elif now - self.last_gesture_click_time > 1.0 and self.gesture_clicks_detected > 0:
                self.gesture_clicks_detected = 0 # Reset counter
                
        # MODIFIED: Zoom step completion logic
        elif completion_type == "zoom":
            zoom_active_by_gesture = self.handpal.gesture_recognizer.gesture_state.get("zoom_active", False)
            current_zoom_delta = self.handpal.last_zoom_delta # This is the scaled delta

            if zoom_active_by_gesture and abs(current_zoom_delta) > 0.05: # Check for *any* zoom movement
                self.zoom_activity_detected = True
                self.last_zoom_activity_time = now
                if current_zoom_delta > 0:
                    self.max_zoom_delta_seen = max(self.max_zoom_delta_seen, current_zoom_delta)
                else:
                    self.min_zoom_delta_seen = min(self.min_zoom_delta_seen, current_zoom_delta)
                logger.info(f"Tutorial: Zoom movement detected: delta={current_zoom_delta:.3f}")

            # Require both zoom in and zoom out actions, and some sustained activity
            if self.zoom_activity_detected and self.min_zoom_delta_seen < -0.1 and self.max_zoom_delta_seen > 0.1:
                if now - self.last_zoom_activity_time > 0.5: # Wait for a brief pause after last zoom action
                    logger.info("Tutorial: Zoom In & Out exercise completed")
                    self.step_completed = True
                    self.last_gesture_time = now
                
        elif completion_type == "scroll":
            if self.handpal.gesture_recognizer.gesture_state.get("active_gesture") == "scroll":
                if self.handpal.gesture_recognizer.gesture_state.get("scroll_active", False):
                    # Check if scroll history has enough entries and shows some movement
                    scroll_hist = self.handpal.gesture_recognizer.gesture_state.get("scroll_history", [])
                    if len(scroll_hist) >= 3 and any(abs(s_delta) > 0.01 for s_delta in scroll_hist): # Check for actual scroll
                        logger.info("Tutorial: Scroll gesture completed")
                        self.step_completed = True; self.last_gesture_time = now
                
        elif completion_type == "menu_hover":
            if self.handpal.menu_trigger_active: # Menu circle is being hovered by hand
                if self.completion_timer == 0: self.completion_timer = now
                elif now - self.completion_timer >= self.handpal.MENU_HOVER_DELAY - 0.2: # Slightly less for tutorial
                    logger.info("Tutorial: Menu hover completed")
                    self.step_completed = True; self.last_gesture_time = now
            else: self.completion_timer = 0
                
        if self.step_completed and (now - self.last_gesture_time > 1.0): # Auto-advance with a small delay
            if not self.next_step(): # If not last step
                 self.last_gesture_time = time.time() # Reset for next step's auto-advance timer


    def handle_skip_click(self, x, y, w, h):
        if not self.active: return False
        skip_btn_x, skip_btn_y = w - 100, h - 35
        skip_btn_width, skip_btn_height = 80, 25
        
        if (skip_btn_x <= x <= skip_btn_x + skip_btn_width and 
            skip_btn_y <= y <= skip_btn_y + skip_btn_height):
            logger.info(f"Skip button clicked in tutorial step {self.current_step+1}")
            return self.next_step()
        return False
    
    def draw_tutorial_overlay(self, frame):
        if not self.active or self.current_step >= len(self.steps): # Added boundary check
            return frame, (0, 0, 0, 0)
            
        h, w = frame.shape[:2]
        if w == 0 or h == 0: return frame, (0, 0, 0, 0)
            
        overlay_height = 120
        tutorial_panel = frame.copy()
        cv.rectangle(tutorial_panel, (0, h-overlay_height), (w, h), (0, 0, 0), -1)
        frame = cv.addWeighted(tutorial_panel, 0.8, frame, 0.2, 0)
        
        current_step_def = self.steps[self.current_step] # Renamed
        title = current_step_def.get("title", "Tutorial")
        instruction = current_step_def.get("instruction", "")
        icon = current_step_def.get("icon", ":)") # MODIFIED: Default icon
        
        for i in range(len(self.steps)):
            color = (0, 255, 255) if i == self.current_step else (100, 100, 100)
            cv.circle(frame, (20 + i*30, h-20), 8, color, -1 if i <= self.current_step else 2)
        
        # MODIFIED: Display icon and title separately for better font handling
        # Icon text might be wider, adjust title position
        icon_x_start = 20
        title_x_start = icon_x_start + 50 # Estimate space for icon
        if len(icon) > 1: # If icon is a word like "Hi!"
            title_x_start = icon_x_start + len(icon) * 15 # Rough estimate

        cv.putText(frame, icon, (icon_x_start, h-95), cv.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        cv.putText(frame, title, (title_x_start, h-95), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv.putText(frame, instruction, (20, h-60), cv.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1) # Slightly smaller font
        
        if self.step_completed:
            cv.putText(frame, "Ottimo! Passaggio completato", (w//2-150, h-25), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        skip_btn_x, skip_btn_y = w - 100, h - 35
        skip_btn_width, skip_btn_height = 80, 25
        cv.rectangle(frame, (skip_btn_x, skip_btn_y), 
                    (skip_btn_x + skip_btn_width, skip_btn_y + skip_btn_height), 
                    (0, 120, 255), -1)
        cv.putText(frame, "Skip >>", (skip_btn_x + 12, skip_btn_y + 18), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        completion_type = current_step_def.get("completion_type", "")
        if self.handpal.debug_mode: # Only show specific debugs if debug_mode is on
            if completion_type == "corners":
                status = "".join(["V" if v else "O" for v in self.corners_visited])
                cv.putText(frame, f"Angoli: {status}", (20, h-35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200),1)
            elif completion_type == "double_click":
                info = f"G:{self.gesture_clicks_detected} M:{self.mouse_clicks_detected}"
                cv.putText(frame, info, (20, h-35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200),1)
            elif completion_type == "zoom":
                z_info = f"D:{self.handpal.last_zoom_delta:.2f} Act:{self.zoom_activity_detected} Min:{self.min_zoom_delta_seen:.2f} Max:{self.max_zoom_delta_seen:.2f}"
                cv.putText(frame, z_info, (w//2 - 200, h-40), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,200),1)

        return frame, (skip_btn_x, skip_btn_y, skip_btn_width, skip_btn_height)


# -----------------------------------------------------------------------------
# Menu Related Functions & Classes (Largely Unchanged)
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
                writer.writerow(['Tutorial', '@tutorial', '#00B894', '📚']) # Icon should be fine in Tkinter
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
    if not path: logger.warning("Launch attempt with empty path."); return
    logger.info(f"Launching: {path}")
    try:
        global handpal_instance
        if path == '@tutorial':
            if handpal_instance:
                handpal_instance.tutorial_manager.start_tutorial()
                if handpal_instance.floating_menu and handpal_instance.floating_menu.visible:
                    handpal_instance.floating_menu.hide()
            return
            
        if path.startswith(('http://', 'https://')): webbrowser.open(path)
        elif os.name == 'nt': os.startfile(path)
        else: subprocess.Popen(path) # May need to be subprocess.Popen([path]) for some systems
    except FileNotFoundError: logger.error(f"Launch failed: File not found '{path}'")
    except Exception as e: logger.error(f"Launch failed '{path}': {e}")

class FloatingMenu: # (Unchanged from original, assuming Tkinter handles icons well)
    def __init__(self, root):
        self.root = root; self.window = Toplevel(self.root)
        self.window.title("HandPal Menu"); self.window.attributes("-topmost", True)
        self.window.overrideredirect(True); self.window.geometry("280x400+50+50")
        self.window.configure(bg='#222831'); self.apps = read_applications_from_csv()
        self._create_elements(); self.window.withdraw(); self.visible = False
        self.window.bind("<ButtonPress-1>", self._start_move); self.window.bind("<ButtonRelease-1>", self._stop_move); self.window.bind("<B1-Motion>", self._do_move)
        self._offset_x = 0; self._offset_y = 0; logger.info("FloatingMenu initialized.")
    def _create_elements(self):
        title_f = Frame(self.window, bg='#222831', pady=15); title_f.pack(fill=X)
        Label(title_f, text="HANDPAL MENU", font=("Helvetica", 14, "bold"), bg='#222831', fg='#EEEEEE').pack()
        Label(title_f, text="Launch Application", font=("Helvetica", 10), bg='#222831', fg='#00ADB5').pack(pady=(0, 10))
        btn_cont = Frame(self.window, bg='#222831', padx=20); btn_cont.pack(fill=BOTH, expand=True); self.buttons = []
        for app in self.apps:
            f = Frame(btn_cont, bg='#222831', pady=8); f.pack(fill=X)
            btn = TkButton(f, text=f"{app.get('icon',' ')} {app.get('label','?')}", bg=app.get('color','#555'), fg="white", font=("Helvetica", 11), relief=tk.FLAT, borderwidth=0, padx=10, pady=8, width=20, anchor='w', command=lambda p=app.get('path'): launch_application(p))
            btn.pack(fill=X); self.buttons.append(btn)
        bottom_f = Frame(self.window, bg='#222831', pady=15); bottom_f.pack(fill=X, side=tk.BOTTOM)
        TkButton(bottom_f, text="✖ Close Menu", bg='#393E46', fg='#EEEEEE', font=("Helvetica", 10), relief=tk.FLAT, borderwidth=0, padx=10, pady=5, width=15, command=self.hide).pack(pady=5)
    def show(self):
        if not self.visible: self.window.deiconify(); self.window.lift(); self.visible=True; logger.debug("Menu shown.")
    def hide(self):
        if self.visible: self.window.withdraw(); self.visible=False; logger.debug("Menu hidden.")
    def toggle(self): (self.hide if self.visible else self.show)()
    def _start_move(self, event): self._offset_x, self._offset_y = event.x, event.y
    def _stop_move(self, event): self._offset_x = self._offset_y = 0
    def _do_move(self, event): x=self.window.winfo_x()+event.x-self._offset_x; y=self.window.winfo_y()+event.y-self._offset_y; self.window.geometry(f"+{x}+{y}")

# -----------------------------------------------------------------------------
# HandPal Class (Main Application)
# -----------------------------------------------------------------------------
class HandPal:
    def __init__(self, config):
        self.config = config; self.mouse = Controller()
        self.keyboard = KeyboardController() # Use the renamed controller
        self.gesture_recognizer = GestureRecognizer(config)
        try:
            self.tk_root = tk.Tk(); self.tk_root.withdraw()
            self.screen_size = (self.tk_root.winfo_screenwidth(), self.tk_root.winfo_screenheight())
            self.floating_menu = FloatingMenu(self.tk_root)
            self.debug_window = DebugWindow(self.tk_root)
            logger.info(f"Screen: {self.screen_size}. Menu and Debug Window OK.")
        except tk.TclError as e: 
            logger.error(f"Tkinter init failed: {e}. Menu N/A.")
            self.tk_root=None; self.floating_menu=None; self.debug_window=None
            self.screen_size=(1920,1080) # Fallback screen size
        
        self.motion_smoother = MotionSmoother(config)
        self.running = False; self.stop_event = threading.Event(); self.detection_thread = None
        self.data_queue = queue.Queue(maxsize=2)
        self.cap = None; self.mp_hands = mp.solutions.hands; self.hands_instance = None
        self.mp_drawing = mp.solutions.drawing_utils; self.mp_drawing_styles = mp.solutions.drawing_styles
        self.last_cursor_pos = None; self.last_right_hand_lm_norm = None
        self.tracking_enabled = True; self.debug_mode = False
        self.calibration_active = False; self.calibration_points = []; self.calibration_step = 0
        self.calib_corners = ["top-left", "top-right", "bottom-right", "bottom-left"]
        
        self.last_zoom_delta = 0.0 # Stores the SCALED delta from gesture recognizer
        self.tutorial_manager = TutorialManager(self)
        
        self.menu_trigger_zone_px = {"cx": 0, "cy": 0, "radius_sq": 0, "is_valid": False}
        self.menu_trigger_active = False; self._menu_activate_time = 0
        self.MENU_HOVER_DELAY = 1.0
        
        self.tutorial_skip_btn = {"x": 0, "y": 0, "width": 0, "height": 0, "visible": False}
        
        self.fps_stats = deque(maxlen=30)
        self.debug_values = {
            "fps": 0, "L": "U", "R": "U", "gest": "Idle", "menu": "Off",
            "map": {"raw": "", "cal": "", "map": "", "smooth": ""},
            "q": 0, "last_act": time.time(), "cur_hist": deque(maxlen=50),
            "calib": {}, "zoom": {"active": False, "distance": 0, "delta": 0}
        }
        self._last_proc_dims = (config.get("process_width",640), config.get("process_height",360)) # Init with config
        self.current_display_dims = (0, 0) # Will be updated from frame
        
        logger.info("HandPal instance initialized.")

    def handle_cv_mouse_event(self, event, x, y, flags, param):
        if event not in [cv.EVENT_LBUTTONDOWN, cv.EVENT_LBUTTONDBLCLK]: return
        logger.debug(f"Mouse event detected: {event} at ({x}, {y})")
        
        if self.tutorial_manager.active and self.tutorial_skip_btn["visible"]:
            btn = self.tutorial_skip_btn
            if (btn["x"] <= x <= btn["x"] + btn["width"] and 
                btn["y"] <= y <= btn["y"] + btn["height"]):
                logger.info("Skip button clicked via mouse")
                self.tutorial_manager.next_step()
                return
            is_double = (event == cv.EVENT_LBUTTONDBLCLK)
            self.tutorial_manager.register_mouse_click(is_double)

    def map_to_screen(self, norm_x_proc, norm_y_proc):
        if self.debug_mode: self.debug_values["map"]["raw"] = f"{norm_x_proc:.2f},{norm_y_proc:.2f}"
        
        if self.config["calibration.enabled"] and not self.calibration_active:
            x_min, x_max = self.config["calibration.x_min"], self.config["calibration.x_max"]
            y_min, y_max = self.config["calibration.y_min"], self.config["calibration.y_max"]
            self.debug_values["calib"] = {"x_min":x_min, "x_max":x_max, "y_min":y_min, "y_max":y_max, "enabled":True}
            
            norm_x_cal = (norm_x_proc - x_min) / (x_max - x_min) if (x_max - x_min) != 0 else 0.5
            norm_y_cal = (norm_y_proc - y_min) / (y_max - y_min) if (y_max - y_min) != 0 else 0.5
            norm_x_cal = max(0.0, min(1.0, norm_x_cal))
            norm_y_cal = max(0.0, min(1.0, norm_y_cal))
            if self.debug_mode: self.debug_values["map"]["cal"] = f"{norm_x_cal:.2f},{norm_y_cal:.2f}"
        else:
            norm_x_cal = max(0.0, min(1.0, norm_x_proc))
            norm_y_cal = max(0.0, min(1.0, norm_y_proc))
            if self.debug_mode: self.debug_values["map"]["cal"] = f"N/A ({norm_x_cal:.2f},{norm_y_cal:.2f})"
            self.debug_values["calib"] = {"enabled": False}


        # Use norm_x_cal, norm_y_cal for mapping to screen
        display_w, display_h = self.screen_size # Use actual screen size for final mapping
        
        # Apply screen margin from calibration settings if calibration is used
        screen_margin = self.config.get("calibration.screen_margin", 0.0) if self.config["calibration.enabled"] else 0.0
        
        final_x = int((norm_x_cal * (1 - 2 * screen_margin) + screen_margin) * display_w)
        final_y = int((norm_y_cal * (1 - 2 * screen_margin) + screen_margin) * display_h)

        final_x = max(0, min(final_x, display_w - 1))
        final_y = max(0, min(final_y, display_h - 1))
        
        if self.debug_mode: self.debug_values["map"]["map"] = f"{final_x},{final_y}"
        return (final_x, final_y)


    def get_hand_pos_in_display_pixels(self):
        if self.last_right_hand_lm_norm is None: return None
        norm_x, norm_y = self.last_right_hand_lm_norm
        
        # MODIFIED: Use map_to_screen which now directly maps to screen_size with calibration
        # This gives position in screen coordinates, not OpenCV window coordinates.
        # This is what we need for menu trigger logic related to cursor position.
        # However, the menu trigger circle itself is drawn on the OpenCV window.
        # Let's assume the OpenCV window is full-screen or large enough that its coordinates
        # are a good proxy for screen coordinates for the trigger.
        # A more robust solution would involve knowing the OpenCV window's position and size on screen.
        # For now, we'll use the OpenCV window's dimensions (self.current_display_dims) for the circle's pixel coords
        # and map hand to screen_size. This might have a slight mismatch if OpenCV window isn't full screen.
        
        # For menu trigger, we need hand position relative to the DISPLAYED OpenCV window
        # not the whole screen. So, map_to_screen should use self.current_display_dims.
        # Let's adjust map_to_screen to take target_dims.
        
        # Simplified: map_to_screen already maps to self.screen_size, which is fine for cursor.
        # For drawing the menu circle and checking against it, we need consistency.
        # The menu circle is drawn at (w-50, h//2) of the *display_frame*.
        # So hand position should be mapped to *display_frame* dimensions for menu trigger.
        
        # Let's use a local mapping for menu trigger to display_frame dims
        proc_w, proc_h = self._last_proc_dims
        disp_w, disp_h = self.current_display_dims
        if disp_w <=0 or disp_h <= 0: return None

        # Apply calibration if enabled (similar to map_to_screen but for display_frame)
        if self.config["calibration.enabled"] and not self.calibration_active:
            x_min, x_max = self.config["calibration.x_min"], self.config["calibration.x_max"]
            y_min, y_max = self.config["calibration.y_min"], self.config["calibration.y_max"]
            cal_norm_x = (norm_x - x_min) / (x_max - x_min) if (x_max-x_min)!=0 else 0.5
            cal_norm_y = (norm_y - y_min) / (y_max - y_min) if (y_max-y_min)!=0 else 0.5
            cal_norm_x = max(0.0, min(1.0, cal_norm_x))
            cal_norm_y = max(0.0, min(1.0, cal_norm_y))
        else:
            cal_norm_x = norm_x
            cal_norm_y = norm_y
            
        # Map calibrated normalized coords to display_frame pixel coords
        hand_disp_x = int(cal_norm_x * disp_w)
        hand_disp_y = int(cal_norm_y * disp_h)
        hand_disp_x = max(0, min(hand_disp_x, disp_w - 1))
        hand_disp_y = max(0, min(hand_disp_y, disp_h - 1))
        return (hand_disp_x, hand_disp_y)


    def check_menu_trigger(self):
        if not self.menu_trigger_zone_px["is_valid"]:
            if self.menu_trigger_active: logger.debug("Menu trigger deactivated (Zone Invalid).")
            self.menu_trigger_active = False; self._menu_activate_time = 0
            if self.debug_mode: self.debug_values["menu"] = "Off (Zone Invalid)"
            return False

        hand_pos_disp_px = self.get_hand_pos_in_display_pixels() # Gets hand pos in display_frame pixels

        if hand_pos_disp_px is None:
            if self.menu_trigger_active: logger.debug("Menu trigger deactivated (Hand Lost).")
            self.menu_trigger_active = False; self._menu_activate_time = 0
            if self.debug_mode: self.debug_values["menu"] = "Off (No Hand)"
            return False

        hx_px, hy_px = hand_pos_disp_px
        cx_px, cy_px = self.menu_trigger_zone_px["cx"], self.menu_trigger_zone_px["cy"] # These are in display_frame pixels
        radius_sq_px = self.menu_trigger_zone_px["radius_sq"]

        dist_sq_px = (hx_px - cx_px)**2 + (hy_px - cy_px)**2
        is_inside = dist_sq_px < radius_sq_px

        now = time.time(); activated_to_show = False # Renamed for clarity
        hover_time = 0
        if is_inside:
            if not self.menu_trigger_active: 
                self.menu_trigger_active = True; self._menu_activate_time = now
            hover_time = now - self._menu_activate_time
            if hover_time >= self.MENU_HOVER_DELAY:
                activated_to_show = True 
                if self.debug_mode: self.debug_values["menu"] = "ACTIVATE!"
            else:
                 if self.debug_mode: self.debug_values["menu"] = f"Hover {hover_time:.1f}s"
        else: 
            if self.menu_trigger_active: logger.debug("Menu Trigger Zone Exited.")
            self.menu_trigger_active = False; self._menu_activate_time = 0
            if self.debug_mode: self.debug_values["menu"] = "Off"
        
        # logger.debug(f"MenuCheckPx: Hand({hx_px},{hy_px}) Zone({cx_px},{cy_px}, Rsq={radius_sq_px:.0f}) DistSq={dist_sq_px:.0f} Inside={is_inside} Active={self.menu_trigger_active} Hover={hover_time:.2f}")
        return activated_to_show


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

        action_this_frame = False # Flag to see if any gesture caused an action

        # --- Right Hand: Cursor, Menu & Zoom ---
        if lm_r and self.last_right_hand_lm_norm is not None:
             if self.check_menu_trigger(): # This uses get_hand_pos_in_display_pixels
                 if self.floating_menu and not self.floating_menu.visible:
                     logger.info("Menu trigger activated by hand hover."); self.floating_menu.show()
                     action_this_frame = True

             # MODIFIED: Zoom gesture processing
             zoom_result = self.gesture_recognizer.check_zoom_gesture(lm_r, "Right")
             if zoom_result and zoom_result["active"]:
                 self.debug_values["zoom"]["active"] = True
                 self.debug_values["zoom"]["distance"] = zoom_result.get("distance", 0)
                 self.debug_values["zoom"]["delta"] = zoom_result.get("delta", 0) # This is scaled delta
                 self.last_zoom_delta = zoom_result.get("delta", 0) 

                 # Perform zoom action if scaled delta is significant
                 # The threshold here should be small as last_zoom_delta is already scaled by sensitivity
                 if abs(self.last_zoom_delta) > 0.1 : # e.g. if scaled delta means "move 0.1 scroll units"
                     try:
                         # Convert scaled delta to integer scroll clicks for Ctrl+Scroll
                         # Positive delta = fingers apart = zoom in = scroll wheel up
                         # Negative delta = fingers together = zoom out = scroll wheel down
                         scroll_amount = int(self.last_zoom_delta) 
                         
                         if scroll_amount != 0:
                             with self.keyboard.pressed(Key.ctrl):
                                 self.mouse.scroll(0, scroll_amount)
                             
                             self.debug_values["last_act"] = time.time()
                             # Gesture recognizer sets active_gesture to "zoom"
                             logger.debug(f"Zoom action: scaled_delta={self.last_zoom_delta:.3f}, scroll_clicks={scroll_amount}")
                             action_this_frame = True
                     except Exception as e:
                         logger.error(f"Zoom error: {e}")
             elif self.debug_values["zoom"]["active"]: # Was active, now zoom_result is None or not active
                 self.debug_values["zoom"]["active"] = False
                 self.debug_values["zoom"]["delta"] = 0
                 self.last_zoom_delta = 0
                 self.gesture_recognizer.gesture_state["active_gesture"] = None # Ensure reset if it was zoom/zoom_pending

             # Move cursor (only if not actively performing a gesture that might conflict, e.g. menu activation)
             # And if zoom is not in its "active movement" phase. Allow cursor move during "zoom_pending".
             can_move_cursor = self.tracking_enabled and not self.calibration_active and \
                               (self.gesture_recognizer.gesture_state.get("active_gesture") != "zoom")
             
             if can_move_cursor:
                 try:
                     norm_x, norm_y = self.last_right_hand_lm_norm
                     target_x, target_y = self.map_to_screen(norm_x, norm_y) # Maps to full screen_size
                     smooth_x, smooth_y = self.motion_smoother.update(target_x, target_y, self.screen_size[0], self.screen_size[1])
                     if self.last_cursor_pos is None or smooth_x != self.last_cursor_pos[0] or smooth_y != self.last_cursor_pos[1]:
                         self.mouse.position = (smooth_x, smooth_y); self.last_cursor_pos = (smooth_x, smooth_y)
                         if self.debug_mode: self.debug_values["map"]["smooth"]=f"{smooth_x},{smooth_y}"; self.debug_values["cur_hist"].append(self.last_cursor_pos)
                 except Exception as e: logger.error(f"Cursor update error: {e}"); self.motion_smoother.reset()
        else: 
            self.motion_smoother.reset(); self.last_right_hand_lm_norm = None
            if self.menu_trigger_active: self.menu_trigger_active = False; self._menu_activate_time = 0
            if self.debug_values["zoom"]["active"]: # Ensure zoom state is reset if right hand lost
                self.debug_values["zoom"]["active"] = False; self.debug_values["zoom"]["delta"] = 0
                self.last_zoom_delta = 0
                self.gesture_recognizer.gesture_state["zoom_active"] = False
                if self.gesture_recognizer.gesture_state.get("active_gesture") in ["zoom", "zoom_pending"]:
                    self.gesture_recognizer.gesture_state["active_gesture"] = None
            if self.debug_mode: self.debug_values["menu"] = "Off (No Hand)"


        # --- Left Hand: Gestures --- 
        if lm_l and not self.calibration_active:
            # Prioritize click over scroll if both conditions met (less likely but good to order)
            click_type = self.gesture_recognizer.check_thumb_index_click(lm_l, "Left")
            if click_type:
                btn = Button.left; count = 1 if click_type=="click" else 2; name=f"{click_type.capitalize()}"
                try: 
                    self.mouse.click(btn, count); action_this_frame=True
                    self.debug_values["last_act"]=time.time()
                    # Gesture recognizer sets active_gesture to "click"
                    logger.info(f"{name} Left")
                except Exception as e: logger.error(f"{name} error: {e}")
            
            if not action_this_frame: # Only check scroll if click didn't happen
                scroll_delta = self.gesture_recognizer.check_scroll_gesture(lm_l, "Left") # Returns smoothed delta
                if scroll_delta is not None and abs(scroll_delta) > 0.01: # Threshold the delta
                    # Convert delta to integer clicks. Positive delta = scroll up, negative = scroll down.
                    # scroll_delta from gesture is usually small (e.g., 0.0x to 0.x).
                    # mouse.scroll expects integer clicks. Need scaling.
                    # Original scroll was smooth_delta * 100. Let's use a smaller multiplier.
                    scroll_clicks = int(scroll_delta * self.config.get("gesture_settings.scroll_sensitivity", 4) * 0.25) # Tune this multiplier
                    if scroll_clicks != 0:
                        try: 
                            self.mouse.scroll(0, scroll_clicks); action_this_frame = True
                            self.debug_values["last_act"]=time.time()
                            # Gesture recognizer sets active_gesture to "scroll"
                            logger.debug(f"Scroll: {scroll_clicks} (delta: {scroll_delta:.3f})")
                        except Exception as e: logger.error(f"Scroll error: {e}")

        if self.debug_mode:
            self.debug_values["L"] = self.gesture_recognizer.detect_hand_pose(lm_l, "Left")
            self.debug_values["R"] = self.gesture_recognizer.detect_hand_pose(lm_r, "Right")
            current_gest = self.gesture_recognizer.gesture_state.get("active_gesture")
            self.debug_values["gest"] = str(current_gest) if current_gest else "Idle"
            
        if self.tutorial_manager.active:
            self.tutorial_manager.check_step_completion(results)
            
        if self.debug_mode and self.debug_window:
            self.debug_window.update(self.debug_values)

    def draw_landmarks(self, frame, multi_hand_lm):
        if not multi_hand_lm: return frame
        for hand_lm in multi_hand_lm: self.mp_drawing.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS, self.mp_drawing_styles.get_default_hand_landmarks_style(), self.mp_drawing_styles.get_default_hand_connections_style())
        return frame

    def draw_menu_trigger_circle(self, image):
        h, w = image.shape[:2]; radius_px = int(min(w,h) * 0.05) # Relative radius
        if w == 0 or h == 0: self.menu_trigger_zone_px["is_valid"] = False; return image
        
        center_px = (w - int(w*0.08), h // 2) # Position relative to width
        intensity = 150 + int(50 * np.sin(time.time() * 5))
        base_color = (255, intensity, 0); draw_color = (0, 255, 0) if self.menu_trigger_active else base_color
        
        cv.circle(image, center_px, radius_px, draw_color, -1)
        cv.circle(image, center_px, radius_px, (255, 255, 255), 1)
        cv.putText(image, "Menu", (center_px[0] - radius_px//2 -5 , center_px[1] + radius_px + 15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv.LINE_AA)
        
        self.menu_trigger_zone_px["cx"] = center_px[0]
        self.menu_trigger_zone_px["cy"] = center_px[1]
        self.menu_trigger_zone_px["radius_sq"] = radius_px**2
        self.menu_trigger_zone_px["is_valid"] = True
        return image

    def draw_zoom_indicator(self, image):
        # MODIFIED: Only draw if zoom is "active" OR "pending" (i.e. gesture is primed)
        if not (self.gesture_recognizer.gesture_state.get("active_gesture") in ["zoom", "zoom_pending"]):
            return image
            
        h, w = image.shape[:2]
        if w == 0 or h == 0: return image
            
        bar_width_total = int(w * 0.15) # Indicator width relative to screen
        bar_height = int(h * 0.04)
        start_x = w - bar_width_total - int(w*0.08) # Position near menu circle
        start_y = int(h*0.03)
        
        # Background
        cv.rectangle(image, (start_x-2, start_y-2), (start_x+bar_width_total+2, start_y+bar_height+2), (0,0,0), -1)
        cv.rectangle(image, (start_x, start_y), (start_x+bar_width_total, start_y+bar_height), (50,50,50), -1)
        
        # Text "ZOOM"
        text_scale = max(0.4, h/1200) # Scale text with height
        cv.putText(image, "ZOOM", (start_x+5, start_y + int(bar_height*0.65)), cv.FONT_HERSHEY_SIMPLEX, text_scale, (255,255,255), 1)
        
        # Zoom delta visualization (simple +/-)
        current_delta = self.last_zoom_delta # This is the scaled delta
        zoom_char = "="
        char_color = (150,150,150)
        if current_delta > 0.05: zoom_char = "+"; char_color = (0,255,0)
        elif current_delta < -0.05: zoom_char = "-"; char_color = (0,80,255)
        
        cv.putText(image, zoom_char, (start_x + bar_width_total - int(bar_width_total*0.2), start_y + int(bar_height*0.7)), 
                  cv.FONT_HERSHEY_SIMPLEX, text_scale * 1.2, char_color, 2)
        
        # Pinch distance bar (optional, can be noisy)
        # current_pinch_dist = self.gesture_recognizer.gesture_state.get("last_pinch_distance", 0)
        # if current_pinch_dist > 0:
        #     fill_ratio = min(current_pinch_dist / self.config.get("gesture_settings.zoom_deactivation_dist_max", 0.3), 1.0)
        #     fill_width = int( (bar_width_total * 0.5) * fill_ratio ) # Half of the bar for this viz
        #     bar_viz_x = start_x + int(bar_width_total * 0.3)
        #     cv.rectangle(image, (bar_viz_x, start_y+int(bar_height*0.2)), 
        #                  (bar_viz_x + fill_width, start_y+int(bar_height*0.8)), char_color, -1)
        return image


    def draw_overlays(self, frame, results):
        self.current_display_dims = (frame.shape[1], frame.shape[0]) # (width, height)
        h, w = self.current_display_dims[1], self.current_display_dims[0]
        if w == 0 or h == 0: return frame

        overlay_c, bg_c, accent_c, error_c = (255,255,255), (0,0,0), (0,255,255), (0,0,255)
        font, fsc_sml, fsc_med = cv.FONT_HERSHEY_SIMPLEX, 0.4, 0.5 # Adjusted for typical display sizes
        if h < 480: fsc_sml, fsc_med = 0.3, 0.4 # Smaller fonts for smaller windows

        if results and results.multi_hand_landmarks: frame = self.draw_landmarks(frame, results.multi_hand_landmarks)
        
        if not self.tutorial_manager.active: # Don't draw menu/zoom indicators during tutorial to reduce clutter
            frame = self.draw_menu_trigger_circle(frame)
            frame = self.draw_zoom_indicator(frame)

        if self.calibration_active:
            bg = frame.copy(); cv.rectangle(bg, (0,0), (w, int(h*0.2) if h > 400 else 100 ), bg_c, -1); alpha = 0.7 # Relative height for calib overlay
            frame = cv.addWeighted(bg, alpha, frame, 1 - alpha, 0)
            text_y_offset = int(h*0.03)
            cv.putText(frame, f"CALIBRATION ({self.calibration_step+1}/4): RIGHT Hand", (10, text_y_offset), font, fsc_med, overlay_c, 1)
            cv.putText(frame, f"Point index to {self.calib_corners[self.calibration_step]} corner", (10, text_y_offset + int(h*0.04)), font, fsc_med, overlay_c, 1)
            cv.putText(frame, "Press SPACE to confirm", (10, text_y_offset + int(h*0.08)), font, fsc_med, accent_c, 1)
            cv.putText(frame, "(ESC to cancel)", (w - int(w*0.25) if w > 500 else 100, text_y_offset), font, fsc_sml, overlay_c, 1)
            
            radius=int(min(w,h)*0.03); inactive=(100,100,100); active_clr=error_c; fill=-1
            corners_px = [(radius, radius), (w-radius, radius), (w-radius, h-radius), (radius, h-radius)]
            for i, p in enumerate(corners_px): cv.circle(frame, p, radius, active_clr if self.calibration_step==i else inactive, fill); cv.circle(frame, p, radius, overlay_c, 1)
            
            # Use get_hand_pos_in_display_pixels for calibration point display
            hand_pos_disp_px = self.get_hand_pos_in_display_pixels()
            if hand_pos_disp_px: cv.circle(frame, hand_pos_disp_px, 10, (0, 255, 0), -1)
        else:
            if not self.tutorial_manager.active: # No info text during tutorial
                info = "C:Calib | D:Debug | Q:Exit | M:Menu | Circle:Menu"
                cv.putText(frame, info, (10, h-10), font, fsc_sml, overlay_c, 1)
            fps = self.debug_values["fps"]; cv.putText(frame, f"FPS: {fps:.1f}", (w - int(w*0.12) if w > 600 else 70, int(h*0.04)), font, fsc_med, overlay_c, 1)
                        
        if self.tutorial_manager.active:
            frame_with_btn, skip_btn_info = self.tutorial_manager.draw_tutorial_overlay(frame)
            self.tutorial_skip_btn = {
                "x": skip_btn_info[0], "y": skip_btn_info[1],
                "width": skip_btn_info[2], "height": skip_btn_info[3],
                "visible": True
            }
            frame = frame_with_btn
        else:
            self.tutorial_skip_btn["visible"] = False
        return frame

    def main_loop(self):
        logger.info("Main loop starting."); win_name="HandPal Control"; cv.namedWindow(win_name,cv.WINDOW_NORMAL)
        cv.setMouseCallback(win_name, self.handle_cv_mouse_event)
        
        dw, dh = self.config.get('display_width'), self.config.get('display_height')
        if dw and dh:
            try: cv.resizeWindow(win_name, dw, dh); logger.info(f"Set display size {dw}x{dh}")
            except Exception as e: logger.warning(f"Failed to set display size: {e}") # Corrected Exception
        last_valid_frame = np.zeros((self.config.get("process_height", 360), self.config.get("process_width", 640), 3), dtype=np.uint8)


        while self.running:
            loop_start = time.perf_counter()
            if self.tk_root:
                try: self.tk_root.update_idletasks(); self.tk_root.update()
                except tk.TclError as e:
                    if "application has been destroyed" in str(e).lower(): logger.info("Tkinter root destroyed. Stopping."); self.stop(); break
                    else: logger.error(f"Tkinter error: {e}")

            current_frame, results, proc_dims = None, None, None
            try:
                frame_data, results, proc_dims_from_q = self.data_queue.get(block=True, timeout=0.01)
                current_frame = frame_data.copy() # Make a mutable copy
                last_valid_frame = current_frame # Store last good frame
                self._last_proc_dims = proc_dims_from_q # Update proc_dims used by results
                if self.debug_mode: self.debug_values["q"] = self.data_queue.qsize()
            except queue.Empty:
                current_frame = last_valid_frame.copy() if last_valid_frame is not None else None
                results = None 
                # proc_dims remains self._last_proc_dims (already set)
            except Exception as e: logger.exception(f"Data queue error: {e}"); time.sleep(0.01); continue # MODIFIED timeout

            if current_frame is not None:
                 if results is not None: # Process only if new results arrived
                     try: self.process_results(results, self._last_proc_dims)
                     except Exception as e: logger.exception("process_results error"); cv.putText(current_frame,"PROC ERR",(50,50),1,1,(0,0,255),2)

                 try: display_frame = self.draw_overlays(current_frame, results)
                 except Exception as e: logger.exception("draw_overlays error"); cv.putText(current_frame,"DRAW ERR",(50,100),1,1,(0,0,255),2); display_frame = current_frame

                 final_frame = display_frame
                 disp_w_cfg, disp_h_cfg = self.config.get('display_width'), self.config.get('display_height')
                 if disp_w_cfg and disp_h_cfg: 
                     try:
                         h_curr, w_curr = display_frame.shape[:2]
                         if w_curr!=disp_w_cfg or h_curr!=disp_h_cfg: final_frame=cv.resize(display_frame,(disp_w_cfg,disp_h_cfg),interpolation=cv.INTER_LINEAR)
                     except Exception as e: logger.error(f"Resize error: {e}")
                 cv.imshow(win_name, final_frame)
            else: # If current_frame is None (e.g. very first frames)
                time.sleep(0.01) # wait a bit

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'): logger.info("'q' pressed, stopping."); self.stop(); break
            elif key == ord('c'):
                if not self.calibration_active: self.start_calibration()
            elif key == ord('d'): 
                self.debug_mode = not self.debug_mode
                logger.info(f"Debug mode toggled to: {self.debug_mode}")
                if self.debug_window: (self.debug_window.show if self.debug_mode else self.debug_window.hide)()
                if not self.debug_mode: self.debug_values["cur_hist"].clear()
            elif key == ord('m'):
                 if self.floating_menu: logger.debug("'m' pressed, toggling menu."); self.floating_menu.toggle()
            elif key == 27: # ESC
                if self.calibration_active: self.cancel_calibration()
                elif self.tutorial_manager.active: self.tutorial_manager.stop_tutorial()
                elif self.floating_menu and self.floating_menu.visible: self.floating_menu.hide()
                elif self.debug_window and self.debug_window.visible: self.debug_window.hide()
            elif key == 32: # Spacebar
                if self.calibration_active: self.process_calibration_step()
                elif self.tutorial_manager.active: 
                    if not self.tutorial_manager.step_completed: # If step not done, Space = Skip
                        logger.info("Space pressed during tutorial, skipping step.")
                        self.tutorial_manager.next_step()
                    # If step is completed, space might be used by an app, so don't intercept for tutorial.
                    # The auto-advance should handle completed steps.

            elapsed = time.perf_counter() - loop_start; self.fps_stats.append(elapsed)
            if self.fps_stats: avg_dur = sum(self.fps_stats)/len(self.fps_stats); self.debug_values["fps"] = 1.0/avg_dur if avg_dur>0 else 0

        logger.info("Main loop finished."); cv.destroyAllWindows()

    def start_calibration(self):
        if self.calibration_active: return
        logger.info("Starting calibration..."); print("\n--- CALIBRATION START ---")
        self.calibration_active = True; self.calibration_points = []; self.calibration_step = 0
        self.config["calibration.active"] = True # Signal to config (though not directly used by it)

    def cancel_calibration(self):
        if not self.calibration_active: return
        logger.info("Calibration cancelled."); print("--- CALIBRATION CANCELLED ---")
        self.calibration_active = False; self.config["calibration.active"] = False

    def process_calibration_step(self):
        if not self.calibration_active: return
        # Use last_right_hand_lm_norm which is (norm_x, norm_y) from INDEX finger tip, relative to processed frame
        if self.last_right_hand_lm_norm is None:
            logger.warning("No hand detected for calibration point."); print("No hand for calib point!")
            return
        
        point = self.last_right_hand_lm_norm # This is (norm_x, norm_y)
        print(f"Calibration point {self.calibration_step+1}: {point[0]:.3f}, {point[1]:.3f}")
        self.calibration_points.append(point)
        
        self.calibration_step += 1
        if self.calibration_step >= 4: self.finish_calibration()
        
    def finish_calibration(self):
        if not self.calibration_active or len(self.calibration_points) < 4: 
            logger.warning("Incomplete calibration points."); self.cancel_calibration(); return
        
        x_vals = [p[0] for p in self.calibration_points]
        y_vals = [p[1] for p in self.calibration_points]
        
        x_min_raw, x_max_raw = min(x_vals), max(x_vals)
        y_min_raw, y_max_raw = min(y_vals), max(y_vals)

        # Add a small safety margin to avoid issues if calibration points are too close to edge
        margin = 0.02 
        x_min = max(0.0, x_min_raw - margin)
        x_max = min(1.0, x_max_raw + margin)
        y_min = max(0.0, y_min_raw - margin)
        y_max = min(1.0, y_max_raw + margin)

        # Ensure min < max and range is not too small
        if x_max - x_min < 0.1: x_min, x_max = 0.1, 0.9; logger.warning("Calibrated X range too small, reset to default-like.")
        if y_max - y_min < 0.1: y_min, y_max = 0.1, 0.9; logger.warning("Calibrated Y range too small, reset to default-like.")

        self.config["calibration.x_min"] = x_min; self.config["calibration.x_max"] = x_max
        self.config["calibration.y_min"] = y_min; self.config["calibration.y_max"] = y_max
        self.config["calibration.enabled"] = True; self.config["calibration.active"] = False
        
        self.calibration_active = False
        logger.info(f"Calibration complete: X:[{x_min:.3f}-{x_max:.3f}] Y:[{y_min:.3f}-{y_max:.3f}]")
        print(f"--- CALIBRATION COMPLETE ---\nX range: {x_min:.3f} - {x_max:.3f}\nY range: {y_min:.3f} - {y_max:.3f}")
        self.config.save()

    def start(self):
        if self.running: logger.warning("start() called but already running."); return True
        logger.info("Starting HandPal...")
        
        try:
            # Use CAP_DSHOW for better performance/stability on Windows
            cap_backend = cv.CAP_DSHOW if os.name == 'nt' else cv.CAP_ANY
            self.cap = cv.VideoCapture(self.config["device"], cap_backend)
            if not self.cap.isOpened(): raise Exception(f"Failed to open camera {self.config['device']}")
            
            # Set desired resolution (camera might not support it exactly)
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.config["width"])
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.config["height"])
            self.cap.set(cv.CAP_PROP_FPS, self.config["max_fps"]) # Request FPS
            
            w_actual = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            h_actual = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            fps_actual = self.cap.get(cv.CAP_PROP_FPS)
            logger.info(f"Camera initialized: Actual {w_actual}x{h_actual} @ {fps_actual:.1f}fps (Requested {self.config['width']}x{self.config['height']} @ {self.config['max_fps']}fps)")
            # Update config if process_width/height are larger than actual camera feed
            if self.config["process_width"] > w_actual: self.config["process_width"] = w_actual
            if self.config["process_height"] > h_actual: self.config["process_height"] = h_actual
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}"); return False
        
        try:
            self.hands_instance = self.mp_hands.Hands(
                static_image_mode=self.config["use_static_image_mode"], max_num_hands=2,
                min_detection_confidence=self.config["min_detection_confidence"],
                min_tracking_confidence=self.config["min_tracking_confidence"]
            )
        except Exception as e:
            logger.error(f"MediaPipe init failed: {e}"); self.cap.release(); self.cap = None; return False
        
        try:
            self.stop_event.clear()
            self.detection_thread = DetectionThread(self.config, self.cap, self.hands_instance, self.data_queue, self.stop_event)
            self.detection_thread.start()
        except Exception as e:
            logger.error(f"Detection thread start failed: {e}")
            if self.hands_instance: self.hands_instance.close(); self.hands_instance = None
            if self.cap: self.cap.release(); self.cap = None
            return False
        
        self.running = True
        return True

    def stop(self):
        if not self.running: logger.warning("stop() called but not running."); return
        logger.info("Stopping HandPal...")
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.stop_event.set()
            self.detection_thread.join(timeout=1.0) # Reduced timeout
            if self.detection_thread.is_alive(): logger.warning("Detection thread join timeout.")
        
        if self.hands_instance:
            try: self.hands_instance.close(); logger.info("MediaPipe Hands closed")
            except Exception as e: logger.error(f"Error closing MediaPipe: {e}")
        self.hands_instance = None
        
        if self.cap:
            try: self.cap.release(); logger.info("Camera released")
            except Exception as e: logger.error(f"Error releasing camera: {e}")
        self.cap = None
        
        self.running = False
        logger.info("HandPal stopped.")

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="HandPal - Hand Gesture Control")
    parser.add_argument("-d", "--device", type=int, help="Camera device index")
    parser.add_argument("-f", "--flip-camera", action="store_true", help="Flip camera horizontally", default=True)
    parser.add_argument("--display-width", type=int, help="Display window width")
    parser.add_argument("--display-height", type=int, help="Display window height")
    parser.add_argument("--cursor-file", type=str, help="Path to custom cursor file (.cur)")
    args = parser.parse_args()

    config = Config(args)

    if os.name == 'nt' and config["custom_cursor_path"]:
        set_custom_cursor(config["custom_cursor_path"])

    global handpal_instance
    handpal = HandPal(config)
    handpal_instance = handpal

    if handpal.start():
        try: handpal.main_loop()
        except KeyboardInterrupt: print("\nExiting (Ctrl+C)...")
        except Exception as e: logger.exception(f"Unhandled error in main execution: {e}")
        finally:
            handpal.stop()
            if os.name == 'nt' and config["custom_cursor_path"]:
                restore_default_cursor()
            if handpal.tk_root: # Destroy Tkinter root if it exists
                try: handpal.tk_root.destroy()
                except Exception: pass
        logger.info("HandPal finished."); print("\nHandPal terminated.")
    else:
        logger.error("HandPal failed to start.")
        print("HandPal failed to start. Check logs for details.")
    return 0

if __name__ == "__main__":
    sys.exit(main())