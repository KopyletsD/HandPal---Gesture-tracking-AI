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
from pynput.keyboard import Key, Controller as KeyboardController 
from collections import deque
import json
import logging
import ctypes
import queue
import csv
import subprocess
import webbrowser
import math 

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
        self.update_interval = 0.2
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
            self.text_widget.insert(tk.END, f"Distance (norm): {debug_values['zoom'].get('distance', 0):.3f}\n")
            self.text_widget.insert(tk.END, f"Delta (scaled): {debug_values['zoom'].get('delta', 0):.3f}\n") 
        
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
    "width": 1920, 
    "height": 1440,
    "process_width": 640,
    "process_height": 360,
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
    "gesture_settings": {
        "scroll_sensitivity": 4,    
        "scroll_click_scaler": 25.0,
        "double_click_time": 0.35,
        "zoom_sensitivity": 8.0,   # MODIFIED: Reduced significantly (was 25.0). Amplifies raw pinch delta.
        "min_actionable_scaled_delta_for_zoom": 0.4, # MODIFIED: New. Threshold for the *scaled* delta to trigger action.
        "zoom_activation_dist_max": 0.15, 
        "zoom_deactivation_dist_min": 0.015, 
        "zoom_deactivation_dist_max": 0.30, 
        "min_zoom_delta_magnitude": 0.003 
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
                if key == 'flip_camera': 
                    self.config['flip_camera'] = value 
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
        if key.startswith("display_"): self._validate_display_dims(); return True 

    def save(self):
        try:
            save_cfg = json.loads(json.dumps(self.config)); save_cfg.get("calibration", {}).pop("active", None)
            with open(self.CONFIG_FILENAME, 'w') as f: json.dump(save_cfg, f, indent=2)
            logger.info(f"Config saved to {self.CONFIG_FILENAME}")
        except Exception as e: logger.error(f"Error saving config: {e}")

    def __getitem__(self, key): return self.get(key)
    def __setitem__(self, key, value): self.set(key, value)

# -----------------------------------------------------------------------------
# GestureRecognizer Class (Largely Unchanged, config changes affect behavior)
# -----------------------------------------------------------------------------
class GestureRecognizer:
    def __init__(self, config):
        self.config = config
        self.last_positions = {}
        self.gesture_state = {
            "scroll_active": False, "last_click_time": 0, "last_click_button": None,
            "scroll_history": deque(maxlen=5), "active_gesture": None,
            "last_pose": {"Left": "U", "Right": "U"}, "pose_stable_count": {"Left": 0, "Right": 0},
            "zoom_active": False, "last_pinch_distance": None, "zoom_history": deque(maxlen=5) # zoom_history is for raw_delta
        }
        self.POSE_STABILITY_THRESHOLD = 3

    def _dist(self, p1, p2):
        if None in [p1, p2] or not all(hasattr(p, 'x') for p in [p1,p2]): return float('inf')
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def check_thumb_index_click(self, hand_lm, handedness):
        is_actively_zooming = self.gesture_state.get("active_gesture") == "zoom"
        
        if handedness != "Left" or hand_lm is None or self.gesture_state["scroll_active"] or is_actively_zooming:
            return None
        try:
            thumb = hand_lm.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            index = hand_lm.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
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
                else:
                    gesture = "click"
                self.gesture_state["last_click_time"] = now
                self.gesture_state["last_click_button"] = Button.left
                self.gesture_state["active_gesture"] = "click" 
        elif self.gesture_state["active_gesture"] == "click": 
             self.gesture_state["active_gesture"] = None
        return gesture

    def check_scroll_gesture(self, hand_landmarks, handedness):
         if handedness != "Left":
             return None
         if self.gesture_state["active_gesture"] == "click" or \
            self.gesture_state.get("active_gesture") in ["zoom", "zoom_pending"]:
             return None
             
         try:
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
            index_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
            middle_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
            ring_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
         except (IndexError, TypeError):
             return None

         index_extended = index_tip.y < index_mcp.y
         middle_extended = middle_tip.y < middle_mcp.y
         fingers_close_x = abs(index_tip.x - middle_tip.x) < 0.08 
         fingers_close_y = abs(index_tip.y - middle_tip.y) < 0.05 
         ring_pinky_folded = (ring_tip.y > middle_mcp.y) and \
                             (pinky_tip.y > middle_mcp.y + 0.02) 

         dist_thumb_index = self._dist(thumb_tip, index_tip)
         min_thumb_dist_for_scroll = self.config.get("gesture_sensitivity", 0.02) * 2.5 
         thumb_far_enough = dist_thumb_index > min_thumb_dist_for_scroll
         
         if index_extended and middle_extended and fingers_close_x and fingers_close_y and ring_pinky_folded and thumb_far_enough:
             if not self.gesture_state["scroll_active"]:
                 self.gesture_state["scroll_active"] = True
                 self.gesture_state["scroll_history"].clear()
                 self.gesture_state["active_gesture"] = "scroll"
             
             tracked_y_lm_idx = mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP
             curr_y_coord = hand_landmarks.landmark[tracked_y_lm_idx].y
             
             if tracked_y_lm_idx in self.last_positions:
                 prev_y = self.last_positions[tracked_y_lm_idx][1]
                 delta_y = (curr_y_coord - prev_y) * self.config.get("gesture_settings.scroll_sensitivity", 4)
                 
                 if abs(delta_y) > 0.0005: 
                     self.gesture_state["scroll_history"].append(delta_y)
                 
                 if len(self.gesture_state["scroll_history"]) > 0:
                     smooth_delta = sum(self.gesture_state["scroll_history"]) / len(self.gesture_state["scroll_history"])
                     if abs(smooth_delta) > 0.001: 
                         self.last_positions[tracked_y_lm_idx] = (hand_landmarks.landmark[tracked_y_lm_idx].x, curr_y_coord)
                         return smooth_delta 
             self.last_positions[tracked_y_lm_idx] = (hand_landmarks.landmark[tracked_y_lm_idx].x, curr_y_coord)
         else:
             if self.gesture_state["scroll_active"]:
                 self.gesture_state["scroll_active"] = False
                 self.gesture_state["scroll_history"].clear()
                 if self.gesture_state["active_gesture"] == "scroll":
                     self.gesture_state["active_gesture"] = None
         return None

    def check_zoom_gesture(self, hand_lm, handedness):
        if handedness != "Right" or hand_lm is None:
            if self.gesture_state["zoom_active"]: 
                self.gesture_state["zoom_active"] = False
                self.gesture_state["active_gesture"] = None
                self.gesture_state["last_pinch_distance"] = None
                self.gesture_state["zoom_history"].clear()
            return None

        try:
            thumb_tip = hand_lm.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP] 
            index_tip = hand_lm.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP] 
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
        MIN_ZOOM_DELTA_MAGNITUDE_RAW = gs["min_zoom_delta_magnitude"] # For smoothed raw pinch delta before scaling
        ZOOM_SENSITIVITY = gs["zoom_sensitivity"] # Amplifies smoothed raw delta

        if not self.gesture_state["zoom_active"]:
            if ZOOM_DEACTIVATION_MIN < current_dist < ZOOM_ACTIVATION_MAX:
                self.gesture_state["zoom_active"] = True
                self.gesture_state["last_pinch_distance"] = current_dist
                self.gesture_state["zoom_history"].clear()
                self.gesture_state["active_gesture"] = "zoom_pending" 
                logger.info(f"Zoom gesture ACTIVATED. Initial distance: {current_dist:.3f}")
                return {"active": True, "distance": current_dist, "delta": 0.0} # Return 0 scaled delta on activation
            else:
                return None 
        else: 
            if not (ZOOM_DEACTIVATION_MIN < current_dist < ZOOM_DEACTIVATION_MAX):
                self.gesture_state["zoom_active"] = False
                self.gesture_state["active_gesture"] = None
                self.gesture_state["last_pinch_distance"] = None
                self.gesture_state["zoom_history"].clear()
                logger.info(f"Zoom gesture DEACTIVATED. Distance: {current_dist:.3f}")
                return {"active": False, "distance": current_dist, "delta": 0.0} 

            raw_delta_dist = current_dist - self.gesture_state["last_pinch_distance"]
            self.gesture_state["zoom_history"].append(raw_delta_dist) # History of raw distance changes
            
            smooth_raw_delta = sum(self.gesture_state["zoom_history"]) / len(self.gesture_state["zoom_history"])
            self.gesture_state["last_pinch_distance"] = current_dist # Update last distance
            
            if abs(smooth_raw_delta) > MIN_ZOOM_DELTA_MAGNITUDE_RAW: # Check if smoothed raw delta is significant
                scaled_delta = smooth_raw_delta * ZOOM_SENSITIVITY # Amplify it
                self.gesture_state["active_gesture"] = "zoom" 
                # logger.debug(f"Zoom: smooth_raw_delta={smooth_raw_delta:.4f}, scaled_delta={scaled_delta:.4f}")
                return {"active": True, "distance": current_dist, "delta": scaled_delta}
            else:
                self.gesture_state["active_gesture"] = "zoom_pending" 
                return {"active": True, "distance": current_dist, "delta": 0.0} # No significant movement, delta is 0


    def detect_hand_pose(self, hand_lm, handedness):
        if hand_lm is None or handedness not in ["Left", "Right"]: return "U"
        try: lm=hand_lm.landmark; w,tt,it,mt,rt,pt=lm[0],lm[4],lm[8],lm[12],lm[16],lm[20]; im,mm,rm,pm=lm[5],lm[9],lm[13],lm[17]
        except (IndexError, TypeError): return "U"
        y_ext = 0.03; t_ext=tt.y<w.y-y_ext; i_ext=it.y<im.y-y_ext; m_ext=mt.y<mm.y-y_ext; r_ext=rt.y<rm.y-y_ext; p_ext=pt.y<pm.y-y_ext
        num_ext = sum([i_ext,m_ext,r_ext,p_ext]); pose = "U"
        
        active_gest = self.gesture_state.get("active_gesture")
        if handedness=="Left" and self.gesture_state["scroll_active"]: pose="Scroll"
        elif handedness=="Right" and (active_gest == "zoom" or active_gest == "zoom_pending"): pose="Zoom" 
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
# Detection Thread (Unchanged)
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
                if self.flip:
                    frame = cv.flip(frame, 1)
                proc_frame = cv.resize(frame, (self.proc_w, self.proc_h), interpolation=cv.INTER_LINEAR)
                rgb_frame = cv.cvtColor(proc_frame, cv.COLOR_BGR2RGB); rgb_frame.flags.writeable=False
                results = self.hands.process(rgb_frame); rgb_frame.flags.writeable=True
                try: 
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
# Tutorial Manager Class (Unchanged)
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
        
        self.zoom_activity_detected = False 
        self.min_zoom_delta_seen = 0 
        self.max_zoom_delta_seen = 0 
        self.last_zoom_activity_time = 0 

        self.scroll_detected = False
        self.corners_visited = [False, False, False, False]
        
        self.steps = [
            {"title": "Benvenuto in HandPal", "instruction": "Posiziona la mano DESTRA nello schermo per muovere il cursore", "completion_type": "position", "icon": "Hi!", "allows_mouse": False },
            {"title": "Movimento del Cursore", "instruction": "Muovi il cursore verso i 4 angoli dello schermo", "completion_type": "corners", "icon": "->", "allows_mouse": True },
            {"title": "Click Sinistro", "instruction": "Avvicina pollice e indice della mano SINISTRA oppure fai click col mouse", "completion_type": "click", "icon": "(C)", "allows_mouse": True },
            {"title": "Doppio Click", "instruction": "Fai due click rapidi con la mano SINISTRA oppure doppio click col mouse", "completion_type": "double_click", "icon": "(CC)", "allows_mouse": True },
            {"title": "Zoom In/Out", "instruction": "Con la mano DESTRA, avvicina e allontana pollice e indice per zoom", "completion_type": "zoom", "icon": "(Zm)", "allows_mouse": False },
            {"title": "Scorrimento", "instruction": "Con la mano SINISTRA, estendi indice e medio per scorrere", "completion_type": "scroll", "icon": "(S)", "allows_mouse": False },
            {"title": "Menu", "instruction": "Posiziona l'indice destro sul cerchio del menu e mantieni", "completion_type": "menu_hover", "icon": "(M)", "allows_mouse": False },
            {"title": "Complimenti!", "instruction": "Hai completato il tutorial! Fai click per terminare", "completion_type": "click", "icon": "Done", "allows_mouse": True }
        ]
    
    def start_tutorial(self):
        self.active = True; self.current_step = 0; self.step_completed = False; self.completion_timer = 0
        self.handpal.tracking_enabled = True; self.reset_step_counters(); logger.info("Tutorial started")
        
    def stop_tutorial(self): self.active = False; logger.info("Tutorial stopped")
        
    def reset_step_counters(self):
        self.last_gesture_time = time.time(); self.gesture_clicks_detected = 0; self.last_gesture_click_time = 0
        self.mouse_clicks_detected = 0; self.last_mouse_click_time = 0; self.mouse_double_click_detected = False
        self.zoom_activity_detected = False; self.min_zoom_delta_seen = 0; self.max_zoom_delta_seen = 0; self.last_zoom_activity_time = 0
        self.scroll_detected = False; self.corners_visited = [False, False, False, False]
        
    def next_step(self):
        self.current_step += 1; self.step_completed = False; self.completion_timer = 0; self.reset_step_counters()
        if self.current_step >= len(self.steps): self.stop_tutorial(); return True
        return False
    
    def register_mouse_click(self, is_double=False):
        if not self.active or self.step_completed: return
        current_step_def = self.steps[self.current_step]; completion_type = current_step_def.get("completion_type", ""); allows_mouse = current_step_def.get("allows_mouse", False)
        if not allows_mouse: return
        now = time.time()
        if is_double and completion_type == "double_click": self.mouse_double_click_detected = True; self.step_completed = True; return
        if completion_type == "click": self.step_completed = True; return
        if completion_type == "double_click":
            if now - self.last_mouse_click_time < self.handpal.config.get("gesture_settings.double_click_time", 0.5) + 0.1: 
                self.mouse_clicks_detected += 1
                if self.mouse_clicks_detected >= 2: self.step_completed = True
            else: self.mouse_clicks_detected = 1
            self.last_mouse_click_time = now
    
    def check_step_completion(self, results):
        if not self.active or self.step_completed: return
        current_step_def = self.steps[self.current_step]; completion_type = current_step_def.get("completion_type", ""); now = time.time()
        
        if completion_type == "position" and self.handpal.last_right_hand_lm_norm is not None:
            if self.completion_timer == 0: self.completion_timer = now 
            if now - self.completion_timer > 1.0: logger.info("Tutorial: Hand position detected"); self.step_completed = True; self.last_gesture_time = now 
        elif completion_type == "position" and self.handpal.last_right_hand_lm_norm is None: self.completion_timer = 0 
        elif completion_type == "corners":
            if self.handpal.last_cursor_pos is None: return
            x, y = self.handpal.last_cursor_pos; screen_w, screen_h = self.handpal.screen_size; margin = int(min(screen_w, screen_h) * 0.1) 
            if x < margin and y < margin: self.corners_visited[0] = True
            if x > screen_w - margin and y < margin: self.corners_visited[1] = True
            if x > screen_w - margin and y > screen_h - margin: self.corners_visited[2] = True
            if x < margin and y > screen_h - margin: self.corners_visited[3] = True
            if all(self.corners_visited): logger.info("Tutorial: All corners visited"); self.step_completed = True; self.last_gesture_time = now
        elif completion_type == "click":
            last_click_time = self.handpal.gesture_recognizer.gesture_state.get("last_click_time", 0)
            if last_click_time > self.last_gesture_click_time: self.last_gesture_click_time = last_click_time; self.step_completed = True; self.last_gesture_time = now
        elif completion_type == "double_click":
            if self.mouse_double_click_detected: return 
            last_g_click_time = self.handpal.gesture_recognizer.gesture_state.get("last_click_time", 0)
            if last_g_click_time > self.last_gesture_click_time: 
                if now - self.last_gesture_click_time < self.handpal.config.get("gesture_settings.double_click_time", 0.35) + 0.05: 
                    self.gesture_clicks_detected += 1
                    if self.gesture_clicks_detected >= 2: self.step_completed = True; self.last_gesture_time = now
                else: self.gesture_clicks_detected = 1 
                self.last_gesture_click_time = last_g_click_time
            elif now - self.last_gesture_click_time > 1.0 and self.gesture_clicks_detected > 0: self.gesture_clicks_detected = 0 
        elif completion_type == "zoom":
            zoom_active_by_gesture = self.handpal.gesture_recognizer.gesture_state.get("zoom_active", False); current_zoom_delta = self.handpal.last_zoom_delta 
            if zoom_active_by_gesture and abs(current_zoom_delta) > 0.05: # Check scaled delta for any movement
                self.zoom_activity_detected = True; self.last_zoom_activity_time = now
                if current_zoom_delta > 0: self.max_zoom_delta_seen = max(self.max_zoom_delta_seen, current_zoom_delta)
                else: self.min_zoom_delta_seen = min(self.min_zoom_delta_seen, current_zoom_delta)
            if self.zoom_activity_detected and self.min_zoom_delta_seen < -0.1 and self.max_zoom_delta_seen > 0.1: # Ensure both in and out (based on scaled delta)
                if now - self.last_zoom_activity_time > 0.5: logger.info("Tutorial: Zoom In & Out exercise completed"); self.step_completed = True; self.last_gesture_time = now
        elif completion_type == "scroll":
            if self.handpal.gesture_recognizer.gesture_state.get("active_gesture") == "scroll" and self.handpal.gesture_recognizer.gesture_state.get("scroll_active", False):
                scroll_hist = self.handpal.gesture_recognizer.gesture_state.get("scroll_history", [])
                if len(scroll_hist) >= 3 and any(abs(s_delta) > 0.01 for s_delta in scroll_hist): logger.info("Tutorial: Scroll gesture completed"); self.step_completed = True; self.last_gesture_time = now
        elif completion_type == "menu_hover":
            if self.handpal.menu_trigger_active: 
                if self.completion_timer == 0: self.completion_timer = now
                elif now - self.completion_timer >= self.handpal.MENU_HOVER_DELAY - 0.2: logger.info("Tutorial: Menu hover completed"); self.step_completed = True; self.last_gesture_time = now
            else: self.completion_timer = 0
        if self.step_completed and (now - self.last_gesture_time > 1.0): 
            if not self.next_step(): self.last_gesture_time = time.time() 

    def handle_skip_click(self, x, y, w, h):
        if not self.active: return False
        skip_btn_x, skip_btn_y = w - 100, h - 35; skip_btn_width, skip_btn_height = 80, 25
        if (skip_btn_x <= x <= skip_btn_x + skip_btn_width and skip_btn_y <= y <= skip_btn_y + skip_btn_height):
            logger.info(f"Skip button clicked in tutorial step {self.current_step+1}"); return self.next_step()
        return False
    
    def draw_tutorial_overlay(self, frame):
        if not self.active or self.current_step >= len(self.steps): return frame, (0,0,0,0)
        h, w = frame.shape[:2];
        if w == 0 or h == 0: return frame, (0,0,0,0)
        overlay_height = 120; tutorial_panel = frame.copy(); cv.rectangle(tutorial_panel, (0, h-overlay_height), (w, h), (0,0,0), -1); frame = cv.addWeighted(tutorial_panel, 0.8, frame, 0.2, 0)
        current_step_def = self.steps[self.current_step]; title = current_step_def.get("title", "Tutorial"); instruction = current_step_def.get("instruction", ""); icon = current_step_def.get("icon", ":)") 
        for i in range(len(self.steps)): cv.circle(frame, (20 + i*30, h-20), 8, (0,255,255) if i == self.current_step else (100,100,100), -1 if i <= self.current_step else 2)
        icon_x_start = 20; title_x_start = icon_x_start + (50 if len(icon) <= 2 else len(icon)*15)
        cv.putText(frame, icon, (icon_x_start, h-95), cv.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2); cv.putText(frame, title, (title_x_start, h-95), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv.putText(frame, instruction, (20, h-60), cv.FONT_HERSHEY_SIMPLEX, 0.65, (230,230,230), 1) 
        if self.step_completed: cv.putText(frame, "Ottimo! Passaggio completato", (w//2-150, h-25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        skip_btn_x, skip_btn_y = w-100, h-35; skip_btn_width, skip_btn_height = 80, 25
        cv.rectangle(frame, (skip_btn_x, skip_btn_y), (skip_btn_x + skip_btn_width, skip_btn_y + skip_btn_height), (0,120,255), -1)
        cv.putText(frame, "Skip >>", (skip_btn_x+12, skip_btn_y+18), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        if self.handpal.debug_mode:
            completion_type = current_step_def.get("completion_type", "")
            if completion_type == "corners": cv.putText(frame, f"Angoli: {''.join(['V' if v else 'O' for v in self.corners_visited])}", (20, h-35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200),1)
            elif completion_type == "double_click": cv.putText(frame, f"G:{self.gesture_clicks_detected} M:{self.mouse_clicks_detected}", (20, h-35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200),1)
            elif completion_type == "zoom": cv.putText(frame, f"D:{self.handpal.last_zoom_delta:.2f} Act:{self.zoom_activity_detected} Min:{self.min_zoom_delta_seen:.2f} Max:{self.max_zoom_delta_seen:.2f}", (w//2 - 200, h-40), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,200),1)
        return frame, (skip_btn_x, skip_btn_y, skip_btn_width, skip_btn_height)

# -----------------------------------------------------------------------------
# Menu Related Functions & Classes (Unchanged)
# -----------------------------------------------------------------------------
APP_CSV_PATH = 'applications.csv'
def create_default_apps_csv():
    if not os.path.exists(APP_CSV_PATH):
        logger.info(f"Creating default apps file: {APP_CSV_PATH}")
        try:
            with open(APP_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f); writer.writerow(['label', 'path', 'color', 'icon'])
                writer.writerow(['Calculator', 'calc.exe', '#0078D7', '🧮']); writer.writerow(['Browser', 'https://duckduckgo.com', '#DE5833', '🌐'])
                writer.writerow(['Notepad', 'notepad.exe', '#FFDA63', '📝']); writer.writerow(['Tutorial', '@tutorial', '#00B894', '📚']) 
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
            if handpal_instance: handpal_instance.tutorial_manager.start_tutorial()
            if handpal_instance.floating_menu and handpal_instance.floating_menu.visible: handpal_instance.floating_menu.hide()
            return
        if path.startswith(('http://', 'https://')): webbrowser.open(path)
        elif os.name == 'nt': os.startfile(path)
        else: subprocess.Popen(path) 
    except FileNotFoundError: logger.error(f"Launch failed: File not found '{path}'")
    except Exception as e: logger.error(f"Launch failed '{path}': {e}")

class FloatingMenu: 
    def __init__(self, root):
        self.root = root; self.window = Toplevel(self.root); self.window.title("HandPal Menu"); self.window.attributes("-topmost", True)
        self.window.overrideredirect(True); self.window.geometry("280x400+50+50"); self.window.configure(bg='#222831'); self.apps = read_applications_from_csv()
        self._create_elements(); self.window.withdraw(); self.visible = False
        self.window.bind("<ButtonPress-1>", self._start_move); self.window.bind("<ButtonRelease-1>", self._stop_move); self.window.bind("<B1-Motion>", self._do_move)
        self._offset_x = 0; self._offset_y = 0; logger.info("FloatingMenu initialized.")
    def _create_elements(self):
        title_f = Frame(self.window, bg='#222831', pady=15); title_f.pack(fill=X)
        Label(title_f, text="HANDPAL MENU", font=("Helvetica", 14, "bold"), bg='#222831', fg='#EEEEEE').pack(); Label(title_f, text="Launch Application", font=("Helvetica", 10), bg='#222831', fg='#00ADB5').pack(pady=(0,10))
        btn_cont = Frame(self.window, bg='#222831', padx=20); btn_cont.pack(fill=BOTH, expand=True); self.buttons = []
        for app in self.apps:
            f = Frame(btn_cont, bg='#222831', pady=8); f.pack(fill=X)
            btn = TkButton(f, text=f"{app.get('icon',' ')} {app.get('label','?')}", bg=app.get('color','#555'), fg="white", font=("Helvetica",11), relief=tk.FLAT, borderwidth=0, padx=10,pady=8,width=20,anchor='w', command=lambda p=app.get('path'):launch_application(p))
            btn.pack(fill=X); self.buttons.append(btn)
        bottom_f = Frame(self.window, bg='#222831', pady=15); bottom_f.pack(fill=X, side=tk.BOTTOM)
        TkButton(bottom_f, text="✖ Close Menu", bg='#393E46', fg='#EEEEEE',font=("Helvetica",10), relief=tk.FLAT, borderwidth=0, padx=10,pady=5,width=15, command=self.hide).pack(pady=5)
    def show(self):
        if not self.visible: self.window.deiconify(); self.window.lift(); self.visible=True; logger.debug("Menu shown.")
    def hide(self):
        if self.visible: self.window.withdraw(); self.visible=False; logger.debug("Menu hidden.")
    def toggle(self): (self.hide if self.visible else self.show)()
    def _start_move(self,event): self._offset_x, self._offset_y = event.x, event.y
    def _stop_move(self,event): self._offset_x = self._offset_y = 0
    def _do_move(self,event): x=self.window.winfo_x()+event.x-self._offset_x; y=self.window.winfo_y()+event.y-self._offset_y; self.window.geometry(f"+{x}+{y}")

# -----------------------------------------------------------------------------
# HandPal Class (Main Application)
# -----------------------------------------------------------------------------
class HandPal:
    def __init__(self, config):
        self.config = config; self.mouse = Controller(); self.keyboard = KeyboardController() 
        self.gesture_recognizer = GestureRecognizer(config)
        try:
            self.tk_root = tk.Tk(); self.tk_root.withdraw()
            self.screen_size = (self.tk_root.winfo_screenwidth(), self.tk_root.winfo_screenheight())
            self.floating_menu = FloatingMenu(self.tk_root); self.debug_window = DebugWindow(self.tk_root)
            logger.info(f"Screen: {self.screen_size}. Menu and Debug Window OK.")
        except tk.TclError as e: 
            logger.error(f"Tkinter init failed: {e}. Menu N/A.")
            self.tk_root=None; self.floating_menu=None; self.debug_window=None; self.screen_size=(1920,1080) 
        
        self.motion_smoother = MotionSmoother(config)
        self.running = False; self.stop_event = threading.Event(); self.detection_thread = None
        self.data_queue = queue.Queue(maxsize=2)
        self.cap = None; self.mp_hands = mp.solutions.hands; self.hands_instance = None
        self.mp_drawing = mp.solutions.drawing_utils; self.mp_drawing_styles = mp.solutions.drawing_styles
        self.last_cursor_pos = None; self.last_right_hand_lm_norm = None
        self.tracking_enabled = True; self.debug_mode = False
        self.calibration_active = False; self.calibration_points = []; self.calibration_step = 0
        self.calib_corners = ["top-left", "top-right", "bottom-right", "bottom-left"]
        self.last_zoom_delta = 0.0; self.tutorial_manager = TutorialManager(self)
        self.menu_trigger_zone_px = {"cx": 0, "cy": 0, "radius_sq": 0, "is_valid": False}
        self.menu_trigger_active = False; self._menu_activate_time = 0; self.MENU_HOVER_DELAY = 1.0
        self.tutorial_skip_btn = {"x":0,"y":0,"width":0,"height":0,"visible":False}
        self.fps_stats = deque(maxlen=30)
        self.debug_values = {
            "fps":0, "L":"U", "R":"U", "gest":"Idle", "menu":"Off",
            "map":{"raw":"","cal":"","map":"","smooth":""}, "q":0, "last_act":time.time(),
            "cur_hist":deque(maxlen=50), "calib":{}, "zoom":{"active":False,"distance":0,"delta":0}
        }
        self._last_proc_dims = (config.get("process_width",640), config.get("process_height",360)) 
        self.current_display_dims = (0,0) 
        logger.info("HandPal instance initialized.")

    def handle_cv_mouse_event(self, event, x, y, flags, param):
        if event not in [cv.EVENT_LBUTTONDOWN, cv.EVENT_LBUTTONDBLCLK]: return
        logger.debug(f"Mouse event detected: {event} at ({x}, {y})")
        if self.tutorial_manager.active and self.tutorial_skip_btn["visible"]:
            btn = self.tutorial_skip_btn
            if (btn["x"] <= x <= btn["x"] + btn["width"] and btn["y"] <= y <= btn["y"] + btn["height"]):
                logger.info("Skip button clicked via mouse"); self.tutorial_manager.next_step(); return
            self.tutorial_manager.register_mouse_click(is_double=(event == cv.EVENT_LBUTTONDBLCLK))

    def map_to_screen(self, norm_x_proc, norm_y_proc):
        if self.debug_mode: self.debug_values["map"]["raw"] = f"{norm_x_proc:.2f},{norm_y_proc:.2f}"
        if self.config["calibration.enabled"] and not self.calibration_active:
            x_min,x_max=self.config["calibration.x_min"],self.config["calibration.x_max"]; y_min,y_max=self.config["calibration.y_min"],self.config["calibration.y_max"]
            self.debug_values["calib"] = {"x_min":x_min,"x_max":x_max,"y_min":y_min,"y_max":y_max,"enabled":True}
            norm_x_cal=(norm_x_proc-x_min)/(x_max-x_min) if (x_max-x_min)!=0 else 0.5; norm_y_cal=(norm_y_proc-y_min)/(y_max-y_min) if (y_max-y_min)!=0 else 0.5
            norm_x_cal=max(0.0,min(1.0,norm_x_cal)); norm_y_cal=max(0.0,min(1.0,norm_y_cal))
            if self.debug_mode: self.debug_values["map"]["cal"] = f"{norm_x_cal:.2f},{norm_y_cal:.2f}"
        else:
            norm_x_cal=max(0.0,min(1.0,norm_x_proc)); norm_y_cal=max(0.0,min(1.0,norm_y_proc))
            if self.debug_mode: self.debug_values["map"]["cal"] = f"N/A ({norm_x_cal:.2f},{norm_y_cal:.2f})"
            self.debug_values["calib"] = {"enabled":False}
        display_w,display_h = self.screen_size 
        screen_margin = self.config.get("calibration.screen_margin",0.0) if self.config["calibration.enabled"] else 0.0
        final_x = int((norm_x_cal*(1-2*screen_margin)+screen_margin)*display_w); final_y = int((norm_y_cal*(1-2*screen_margin)+screen_margin)*display_h)
        final_x = max(0,min(final_x,display_w-1)); final_y = max(0,min(final_y,display_h-1))
        if self.debug_mode: self.debug_values["map"]["map"] = f"{final_x},{final_y}"
        return (final_x, final_y)

    def get_hand_pos_in_display_pixels(self): # This returns pixel coords relative to the OpenCV display window
        if self.last_right_hand_lm_norm is None: return None
        norm_x, norm_y = self.last_right_hand_lm_norm
        disp_w, disp_h = self.current_display_dims
        if disp_w <=0 or disp_h <= 0: return None

        # Apply calibration if enabled (relative to processed frame normalized coords)
        if self.config["calibration.enabled"] and not self.calibration_active:
            x_min, x_max = self.config["calibration.x_min"], self.config["calibration.x_max"]
            y_min, y_max = self.config["calibration.y_min"], self.config["calibration.y_max"]
            cal_norm_x = (norm_x - x_min) / (x_max-x_min) if (x_max-x_min)!=0 else 0.5
            cal_norm_y = (norm_y - y_min) / (y_max-y_min) if (y_max-y_min)!=0 else 0.5
            cal_norm_x = max(0.0, min(1.0, cal_norm_x))
            cal_norm_y = max(0.0, min(1.0, cal_norm_y))
        else: # No calibration or calibration active, use raw normalized coords
            cal_norm_x = norm_x
            cal_norm_y = norm_y
            
        # Map calibrated normalized coords to display_frame pixel coords
        # This mapping assumes that the aspect ratio of the normalized space (from MediaPipe)
        # is similar to the aspect ratio of the display window.
        # If they differ significantly, this can lead to visual discrepancies.
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

        hand_pos_disp_px = self.get_hand_pos_in_display_pixels() 
        if hand_pos_disp_px is None:
            if self.menu_trigger_active: logger.debug("Menu trigger deactivated (Hand Lost).")
            self.menu_trigger_active = False; self._menu_activate_time = 0
            if self.debug_mode: self.debug_values["menu"] = "Off (No Hand)"
            return False

        hx_px, hy_px = hand_pos_disp_px
        cx_px, cy_px = self.menu_trigger_zone_px["cx"], self.menu_trigger_zone_px["cy"] 
        
        # MODIFIED: Increase effective trigger radius slightly for better usability
        # Original radius_sq is for drawing. Trigger area is slightly larger.
        # (1.20**2) is approx 1.44. So, 44% larger area, 20% larger radius.
        effective_radius_sq = self.menu_trigger_zone_px["radius_sq"] * (1.20**2) 

        dist_sq_px = (hx_px - cx_px)**2 + (hy_px - cy_px)**2
        is_inside = dist_sq_px < effective_radius_sq # Use effective radius for check

        now = time.time(); activated_to_show = False 
        hover_time = 0
        if is_inside:
            if not self.menu_trigger_active: self.menu_trigger_active = True; self._menu_activate_time = now
            hover_time = now - self._menu_activate_time
            if hover_time >= self.MENU_HOVER_DELAY: activated_to_show = True; 
            if self.debug_mode: self.debug_values["menu"] = "ACTIVATE!" if activated_to_show else f"Hover {hover_time:.1f}s"
        else: 
            if self.menu_trigger_active: logger.debug("Menu Trigger Zone Exited.")
            self.menu_trigger_active = False; self._menu_activate_time = 0
            if self.debug_mode: self.debug_values["menu"] = "Off"
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
                    try: self.last_right_hand_lm_norm = (hand_lm.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x, 
                                                          hand_lm.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y)
                    except IndexError: self.last_right_hand_lm_norm = None
                elif label == "Left": lm_l = hand_lm
        action_this_frame = False 

        # --- Right Hand: Cursor, Menu & Zoom ---
        if lm_r and self.last_right_hand_lm_norm is not None:
             if self.check_menu_trigger(): 
                 if self.floating_menu and not self.floating_menu.visible:
                     logger.info("Menu trigger activated by hand hover."); self.floating_menu.show(); action_this_frame = True
             
             zoom_result = self.gesture_recognizer.check_zoom_gesture(lm_r, "Right")
             if zoom_result and zoom_result["active"]:
                 self.debug_values["zoom"]["active"] = True
                 self.debug_values["zoom"]["distance"] = zoom_result.get("distance", 0)
                 self.debug_values["zoom"]["delta"] = zoom_result.get("delta", 0) 
                 self.last_zoom_delta = zoom_result.get("delta", 0) # This is scaled_delta from recognizer

                 # MODIFIED: Zoom action logic based on new config and rounding behavior
                 scroll_amount = 0
                 min_actionable_scaled_delta = self.config.get("gesture_settings.min_actionable_scaled_delta_for_zoom", 0.4)
                 
                 if abs(self.last_zoom_delta) > min_actionable_scaled_delta:
                     scroll_amount = int(round(self.last_zoom_delta))
                     if scroll_amount == 0: # Passed threshold but rounded to 0, ensure at least 1 click
                         scroll_amount = 1 if self.last_zoom_delta > 0 else -1
                 
                 if scroll_amount != 0:
                     try:
                         with self.keyboard.pressed(Key.ctrl): self.mouse.scroll(0, scroll_amount)
                         self.debug_values["last_act"] = time.time()
                         logger.debug(f"Zoom action: scaled_delta={self.last_zoom_delta:.3f}, scroll_clicks={scroll_amount}")
                         action_this_frame = True
                     except Exception as e: logger.error(f"Zoom error: {e}")
             elif self.debug_values["zoom"]["active"]: 
                 self.debug_values["zoom"]["active"] = False; self.debug_values["zoom"]["delta"] = 0; self.last_zoom_delta = 0
                 if self.gesture_recognizer.gesture_state.get("active_gesture") in ["zoom", "zoom_pending"]:
                    self.gesture_recognizer.gesture_state["active_gesture"] = None

             can_move_cursor = self.tracking_enabled and not self.calibration_active and \
                               (self.gesture_recognizer.gesture_state.get("active_gesture") != "zoom")
             if can_move_cursor:
                 try:
                     norm_x, norm_y = self.last_right_hand_lm_norm
                     target_x,target_y = self.map_to_screen(norm_x,norm_y) 
                     smooth_x,smooth_y = self.motion_smoother.update(target_x,target_y,self.screen_size[0],self.screen_size[1])
                     if self.last_cursor_pos is None or smooth_x!=self.last_cursor_pos[0] or smooth_y!=self.last_cursor_pos[1]:
                         self.mouse.position=(smooth_x,smooth_y); self.last_cursor_pos=(smooth_x,smooth_y)
                         if self.debug_mode: self.debug_values["map"]["smooth"]=f"{smooth_x},{smooth_y}"; self.debug_values["cur_hist"].append(self.last_cursor_pos)
                 except Exception as e: logger.error(f"Cursor update error: {e}"); self.motion_smoother.reset()
        else: 
            self.motion_smoother.reset(); self.last_right_hand_lm_norm = None
            if self.menu_trigger_active: self.menu_trigger_active = False; self._menu_activate_time = 0
            if self.debug_values["zoom"]["active"]: 
                self.debug_values["zoom"]["active"]=False; self.debug_values["zoom"]["delta"]=0; self.last_zoom_delta=0
                self.gesture_recognizer.gesture_state["zoom_active"]=False
                if self.gesture_recognizer.gesture_state.get("active_gesture") in ["zoom","zoom_pending"]: self.gesture_recognizer.gesture_state["active_gesture"]=None
            if self.debug_mode: self.debug_values["menu"] = "Off (No Hand)"

        # --- Left Hand: Gestures --- 
        if lm_l and not self.calibration_active:
            click_type = self.gesture_recognizer.check_thumb_index_click(lm_l, "Left")
            if click_type:
                btn=Button.left; count=1 if click_type=="click" else 2; name=f"{click_type.capitalize()}"
                try: self.mouse.click(btn,count); action_this_frame=True; self.debug_values["last_act"]=time.time(); logger.info(f"{name} Left")
                except Exception as e: logger.error(f"{name} error: {e}")
            
            if not action_this_frame: 
                scroll_delta_from_recognizer = self.gesture_recognizer.check_scroll_gesture(lm_l, "Left")
                if scroll_delta_from_recognizer is not None:
                    scroll_clicks = 0; min_effective_scroll_delta = 0.015 
                    scroll_click_scaler = self.config.get("gesture_settings.scroll_click_scaler", 25.0)
                    if abs(scroll_delta_from_recognizer) > min_effective_scroll_delta:
                        calculated_clicks = scroll_delta_from_recognizer * scroll_click_scaler
                        if calculated_clicks > 0: scroll_clicks = max(1, int(round(calculated_clicks)))
                        elif calculated_clicks < 0: scroll_clicks = min(-1, int(round(calculated_clicks)))
                    if scroll_clicks != 0:
                        try: self.mouse.scroll(0,scroll_clicks); action_this_frame=True; self.debug_values["last_act"]=time.time(); logger.debug(f"Scroll: {scroll_clicks} clicks (delta_rec: {scroll_delta_from_recognizer:.4f})")
                        except Exception as e: logger.error(f"Scroll error: {e}")

        if self.debug_mode:
            self.debug_values["L"] = self.gesture_recognizer.detect_hand_pose(lm_l, "Left")
            self.debug_values["R"] = self.gesture_recognizer.detect_hand_pose(lm_r, "Right")
            current_gest = self.gesture_recognizer.gesture_state.get("active_gesture"); self.debug_values["gest"] = str(current_gest) if current_gest else "Idle"
        if self.tutorial_manager.active: self.tutorial_manager.check_step_completion(results)
        if self.debug_mode and self.debug_window: self.debug_window.update(self.debug_values)

    def draw_landmarks(self, frame, multi_hand_lm): # Unchanged
        if not multi_hand_lm: return frame
        for hand_lm in multi_hand_lm: self.mp_drawing.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS, self.mp_drawing_styles.get_default_hand_landmarks_style(), self.mp_drawing_styles.get_default_hand_connections_style())
        return frame

    def draw_menu_trigger_circle(self, image): # Unchanged visual, trigger logic modified in check_menu_trigger
        h, w = image.shape[:2]; radius_px = int(min(w,h) * 0.05) 
        if w == 0 or h == 0: self.menu_trigger_zone_px["is_valid"] = False; return image
        center_px = (w - int(w*0.08), h // 2); intensity = 150 + int(50 * np.sin(time.time() * 5))
        base_color = (255, intensity, 0); draw_color = (0,255,0) if self.menu_trigger_active else base_color
        cv.circle(image, center_px, radius_px, draw_color, -1); cv.circle(image, center_px, radius_px, (255,255,255), 1)
        cv.putText(image, "Menu", (center_px[0]-radius_px//2-5, center_px[1]+radius_px+15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv.LINE_AA)
        self.menu_trigger_zone_px.update({"cx":center_px[0],"cy":center_px[1],"radius_sq":radius_px**2,"is_valid":True})
        return image

    def draw_zoom_indicator(self, image): # Unchanged
        if not (self.gesture_recognizer.gesture_state.get("active_gesture") in ["zoom", "zoom_pending"]): return image
        h, w = image.shape[:2];
        if w == 0 or h == 0: return image
        bar_width_total = int(w*0.15); bar_height = int(h*0.04); start_x = w-bar_width_total-int(w*0.08); start_y = int(h*0.03)
        cv.rectangle(image, (start_x-2,start_y-2), (start_x+bar_width_total+2,start_y+bar_height+2), (0,0,0),-1)
        cv.rectangle(image, (start_x,start_y), (start_x+bar_width_total,start_y+bar_height), (50,50,50),-1)
        text_scale = max(0.4, h/1200); cv.putText(image, "ZOOM", (start_x+5, start_y+int(bar_height*0.65)), cv.FONT_HERSHEY_SIMPLEX, text_scale, (255,255,255),1)
        current_delta = self.last_zoom_delta; zoom_char="="; char_color=(150,150,150); min_indicator_delta=0.05
        if current_delta > min_indicator_delta: zoom_char="+"; char_color=(0,255,0)
        elif current_delta < -min_indicator_delta: zoom_char="-"; char_color=(0,80,255)
        cv.putText(image, zoom_char, (start_x+bar_width_total-int(bar_width_total*0.2), start_y+int(bar_height*0.7)), cv.FONT_HERSHEY_SIMPLEX, text_scale*1.2, char_color,2)
        return image

    def draw_overlays(self, frame, results):
        self.current_display_dims = (frame.shape[1], frame.shape[0]) 
        h, w = self.current_display_dims[1], self.current_display_dims[0]
        if w == 0 or h == 0: return frame
        overlay_c, bg_c, accent_c, error_c = (255,255,255), (0,0,0), (0,255,255), (0,0,255)
        font, fsc_sml, fsc_med = cv.FONT_HERSHEY_SIMPLEX, 0.4, 0.5 
        if h < 480: fsc_sml, fsc_med = 0.3, 0.4 

        if results and results.multi_hand_landmarks: frame = self.draw_landmarks(frame, results.multi_hand_landmarks)
        
        if not self.tutorial_manager.active: 
            frame = self.draw_menu_trigger_circle(frame)
            frame = self.draw_zoom_indicator(frame)

        # MODIFIED: Add debug dot for hand position in display coordinates
        if self.debug_mode and not self.calibration_active:
            hand_debug_pos_disp = self.get_hand_pos_in_display_pixels() # Already gets from index_finger_tip
            if hand_debug_pos_disp:
                cv.circle(frame, hand_debug_pos_disp, 6, (0, 0, 255), -1) # Bright red dot
                cv.putText(frame, f"H:({hand_debug_pos_disp[0]},{hand_debug_pos_disp[1]})", 
                           (hand_debug_pos_disp[0] + 10, hand_debug_pos_disp[1]), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        if self.calibration_active:
            bg = frame.copy(); cv.rectangle(bg, (0,0), (w, int(h*0.2) if h > 400 else 100 ), bg_c, -1); alpha = 0.7 
            frame = cv.addWeighted(bg, alpha, frame, 1-alpha, 0)
            text_y_offset = int(h*0.03)
            cv.putText(frame, f"CALIBRATION ({self.calibration_step+1}/4): RIGHT Hand", (10,text_y_offset), font, fsc_med, overlay_c,1)
            cv.putText(frame, f"Point index to {self.calib_corners[self.calibration_step]} corner", (10,text_y_offset+int(h*0.04)), font, fsc_med, overlay_c,1)
            cv.putText(frame, "Press SPACE to confirm", (10,text_y_offset+int(h*0.08)), font, fsc_med, accent_c,1)
            cv.putText(frame, "(ESC to cancel)", (w-int(w*0.25) if w>500 else 100,text_y_offset), font, fsc_sml, overlay_c,1)
            radius=int(min(w,h)*0.03); inactive=(100,100,100); active_clr=error_c; fill=-1
            corners_px = [(radius,radius),(w-radius,radius),(w-radius,h-radius),(radius,h-radius)]
            for i,p in enumerate(corners_px): cv.circle(frame,p,radius, active_clr if self.calibration_step==i else inactive, fill); cv.circle(frame,p,radius,overlay_c,1)
            hand_pos_disp_px = self.get_hand_pos_in_display_pixels() # For calibration target visual
            if hand_pos_disp_px: cv.circle(frame, hand_pos_disp_px, 10, (0,255,0), -1) # Green dot for live position
        else:
            if not self.tutorial_manager.active: 
                info = "C:Calib | D:Debug | Q:Exit | M:Menu | Circle:Menu"
                cv.putText(frame, info, (10,h-10), font, fsc_sml, overlay_c,1)
            fps = self.debug_values["fps"]; cv.putText(frame, f"FPS: {fps:.1f}", (w-int(w*0.12) if w>600 else 70,int(h*0.04)), font, fsc_med, overlay_c,1)
                        
        if self.tutorial_manager.active:
            frame, skip_btn_info = self.tutorial_manager.draw_tutorial_overlay(frame)
            self.tutorial_skip_btn.update({"x":skip_btn_info[0],"y":skip_btn_info[1],"width":skip_btn_info[2],"height":skip_btn_info[3],"visible":True})
        else: self.tutorial_skip_btn["visible"] = False
        return frame

    def main_loop(self): # Unchanged
        logger.info("Main loop starting."); win_name="HandPal Control"; cv.namedWindow(win_name,cv.WINDOW_NORMAL)
        cv.setMouseCallback(win_name, self.handle_cv_mouse_event)
        dw,dh = self.config.get('display_width'), self.config.get('display_height')
        if dw and dh:
            try: cv.resizeWindow(win_name, dw, dh); logger.info(f"Set display size {dw}x{dh}")
            except Exception as e: logger.warning(f"Failed to set display size: {e}") 
        last_valid_frame = np.zeros((self.config.get("process_height",360), self.config.get("process_width",640),3), dtype=np.uint8)
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
                current_frame = frame_data.copy(); last_valid_frame = current_frame; self._last_proc_dims = proc_dims_from_q
                if self.debug_mode: self.debug_values["q"] = self.data_queue.qsize()
            except queue.Empty: current_frame = last_valid_frame.copy() if last_valid_frame is not None else None; results = None 
            except Exception as e: logger.exception(f"Data queue error: {e}"); time.sleep(0.01); continue 
            if current_frame is not None:
                 if results is not None: 
                     try: self.process_results(results, self._last_proc_dims)
                     except Exception as e: logger.exception("process_results error"); cv.putText(current_frame,"PROC ERR",(50,50),1,1,(0,0,255),2)
                 try: display_frame = self.draw_overlays(current_frame, results)
                 except Exception as e: logger.exception("draw_overlays error"); cv.putText(current_frame,"DRAW ERR",(50,100),1,1,(0,0,255),2); display_frame = current_frame
                 final_frame = display_frame
                 disp_w_cfg,disp_h_cfg = self.config.get('display_width'),self.config.get('display_height')
                 if disp_w_cfg and disp_h_cfg: 
                     try:
                         h_curr,w_curr = display_frame.shape[:2]
                         if w_curr!=disp_w_cfg or h_curr!=disp_h_cfg: final_frame=cv.resize(display_frame,(disp_w_cfg,disp_h_cfg),interpolation=cv.INTER_LINEAR)
                     except Exception as e: logger.error(f"Resize error: {e}")
                 cv.imshow(win_name, final_frame)
            else: time.sleep(0.01) 
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'): logger.info("'q' pressed, stopping."); self.stop(); break
            elif key == ord('c'):
                if not self.calibration_active: self.start_calibration()
            elif key == ord('d'): 
                self.debug_mode = not self.debug_mode; logger.info(f"Debug mode toggled to: {self.debug_mode}")
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
                elif self.tutorial_manager.active and not self.tutorial_manager.step_completed: 
                    logger.info("Space pressed during tutorial, skipping step."); self.tutorial_manager.next_step()
            elapsed = time.perf_counter()-loop_start; self.fps_stats.append(elapsed)
            if self.fps_stats: avg_dur = sum(self.fps_stats)/len(self.fps_stats); self.debug_values["fps"] = 1.0/avg_dur if avg_dur>0 else 0
        logger.info("Main loop finished."); cv.destroyAllWindows()

    def start_calibration(self): # Unchanged
        if self.calibration_active: return; logger.info("Starting calibration..."); print("\n--- CALIBRATION START ---")
        self.calibration_active=True; self.calibration_points=[]; self.calibration_step=0; self.config["calibration.active"]=True
    def cancel_calibration(self): # Unchanged
        if not self.calibration_active: return; logger.info("Calibration cancelled."); print("--- CALIBRATION CANCELLED ---")
        self.calibration_active=False; self.config["calibration.active"]=False
    def process_calibration_step(self): # Unchanged
        if not self.calibration_active: return
        if self.last_right_hand_lm_norm is None: logger.warning("No hand detected for calib point."); print("No hand for calib point!"); return
        point = self.last_right_hand_lm_norm; print(f"Calib point {self.calibration_step+1}: {point[0]:.3f}, {point[1]:.3f}"); self.calibration_points.append(point)
        self.calibration_step+=1;
        if self.calibration_step>=4: self.finish_calibration()
    def finish_calibration(self): # Unchanged
        if not self.calibration_active or len(self.calibration_points)<4: logger.warning("Incomplete calib points."); self.cancel_calibration(); return
        x_vals=[p[0] for p in self.calibration_points]; y_vals=[p[1] for p in self.calibration_points]
        x_min_raw,x_max_raw=min(x_vals),max(x_vals); y_min_raw,y_max_raw=min(y_vals),max(y_vals)
        margin=0.02; x_min=max(0.0,x_min_raw-margin); x_max=min(1.0,x_max_raw+margin); y_min=max(0.0,y_min_raw-margin); y_max=min(1.0,y_max_raw+margin)
        if x_max-x_min<0.1: x_min,x_max=0.1,0.9; logger.warning("Calibrated X range too small, reset.")
        if y_max-y_min<0.1: y_min,y_max=0.1,0.9; logger.warning("Calibrated Y range too small, reset.")
        self.config.set("calibration.x_min",x_min); self.config.set("calibration.x_max",x_max)
        self.config.set("calibration.y_min",y_min); self.config.set("calibration.y_max",y_max)
        self.config.set("calibration.enabled",True); self.config.set("calibration.active",False); self.calibration_active=False
        logger.info(f"Calibration complete: X:[{x_min:.3f}-{x_max:.3f}] Y:[{y_min:.3f}-{y_max:.3f}]"); print(f"--- CALIBRATION COMPLETE ---\nX range: {x_min:.3f}-{x_max:.3f}\nY range: {y_min:.3f}-{y_max:.3f}")
        self.config.save()

    def start(self): # Unchanged
        if self.running: logger.warning("start() called but already running."); return True
        logger.info("Starting HandPal...")
        try:
            cap_backend = cv.CAP_DSHOW if os.name == 'nt' else cv.CAP_ANY
            self.cap = cv.VideoCapture(self.config["device"], cap_backend)
            if not self.cap.isOpened(): raise Exception(f"Failed to open camera {self.config['device']}")
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.config["width"]); self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.config["height"]); self.cap.set(cv.CAP_PROP_FPS, self.config["max_fps"]) 
            w_actual = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)); h_actual = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)); fps_actual = self.cap.get(cv.CAP_PROP_FPS)
            logger.info(f"Camera initialized: Actual {w_actual}x{h_actual} @ {fps_actual:.1f}fps (Requested {self.config['width']}x{self.config['height']} @ {self.config['max_fps']}fps)")
            if self.config["process_width"] > w_actual: self.config["process_width"]=w_actual
            if self.config["process_height"] > h_actual: self.config["process_height"]=h_actual
        except Exception as e: logger.error(f"Camera initialization failed: {e}"); return False
        try:
            self.hands_instance = self.mp_hands.Hands(static_image_mode=self.config["use_static_image_mode"],max_num_hands=2,min_detection_confidence=self.config["min_detection_confidence"],min_tracking_confidence=self.config["min_tracking_confidence"])
        except Exception as e: logger.error(f"MediaPipe init failed: {e}"); self.cap.release(); self.cap=None; return False
        try:
            self.stop_event.clear(); self.detection_thread = DetectionThread(self.config,self.cap,self.hands_instance,self.data_queue,self.stop_event); self.detection_thread.start()
        except Exception as e:
            logger.error(f"Detection thread start failed: {e}")
            if self.hands_instance: self.hands_instance.close(); self.hands_instance=None
            if self.cap: self.cap.release(); self.cap=None; return False
        self.running = True; return True

    def stop(self): # Unchanged
        if not self.running: logger.warning("stop() called but not running."); return
        logger.info("Stopping HandPal...")
        if self.detection_thread and self.detection_thread.is_alive():
            self.stop_event.set(); self.detection_thread.join(timeout=1.0) 
            if self.detection_thread.is_alive(): logger.warning("Detection thread join timeout.")
        if self.hands_instance:
            try: self.hands_instance.close(); logger.info("MediaPipe Hands closed")
            except Exception as e: logger.error(f"Error closing MediaPipe: {e}")
        self.hands_instance=None
        if self.cap:
            try: self.cap.release(); logger.info("Camera released")
            except Exception as e: logger.error(f"Error releasing camera: {e}")
        self.cap=None; self.running=False; logger.info("HandPal stopped.")

# -----------------------------------------------------------------------------
# Main Function (Unchanged)
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
    if os.name == 'nt' and config["custom_cursor_path"]: set_custom_cursor(config["custom_cursor_path"])
    global handpal_instance; handpal = HandPal(config); handpal_instance = handpal
    if handpal.start():
        try: handpal.main_loop()
        except KeyboardInterrupt: print("\nExiting (Ctrl+C)...")
        except Exception as e: logger.exception(f"Unhandled error in main execution: {e}")
        finally:
            handpal.stop()
            if os.name == 'nt' and config["custom_cursor_path"]: restore_default_cursor()
            if handpal.tk_root: 
                try: handpal.tk_root.destroy()
                except Exception: pass
        logger.info("HandPal finished."); print("\nHandPal terminated.")
    else: logger.error("HandPal failed to start."); print("HandPal failed to start. Check logs for details.")
    return 0

if __name__ == "__main__":
    sys.exit(main())