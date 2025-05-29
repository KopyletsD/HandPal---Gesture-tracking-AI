import argparse
import threading
import os
import sys
import time
import numpy as np
import cv2 as cv
import mediapipe as mp
import tkinter as tk
from tkinter import Toplevel, Frame, Label, Button as TkButton, BOTH, X # TkButton alias
# from pynput.mouse import Controller, Button # Client does NOT control mouse directly
from pynput.mouse import Button # Only for Button enum if needed in gesture recognizer
from collections import deque
import json
import logging
# import ctypes # Client does not do system cursor
import queue
import csv
# import subprocess # Client does not launch apps directly
# import webbrowser # Client does not launch apps directly
import websockets # For client
import asyncio # For client ws

try:
    from PIL import Image, ImageDraw, ImageFont
    from PIL import __version__ as PIL_VERSION_STR
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    PIL_VERSION_STR = "0.0.0"
    print("Warning: Pillow library (PIL) not found. Emojis in tutorial might not render correctly.")

PIL_VERSION_TUPLE = (0,0,0)
if PIL_AVAILABLE:
    try:
        PIL_VERSION_TUPLE = tuple(map(int, PIL_VERSION_STR.split('.')))
    except ValueError:
        print(f"Warning: Could not parse Pillow version string: {PIL_VERSION_STR}")
        PIL_VERSION_TUPLE = (0,0,0)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("handpal_client.log"), logging.StreamHandler()]
)
logger = logging.getLogger("HandPalClient")

# Variabile globale per l'istanza di HandPal (per il tutorial)
handpal_client_instance = None # Changed name
websocket_connection = None # Global for easy access, or pass around

# -----------------------------------------------------------------------------
# Config Class (Client-Focused)
# -----------------------------------------------------------------------------
class Config:
    DEFAULT_CONFIG = { # Client specific defaults
        "device": 0, "width": 1920, "height": 1440,
        "process_width": 1920, "process_height": 1440,
        "flip_camera": True, "display_width": None, "display_height": None,
        "min_detection_confidence": 0.6, "min_tracking_confidence": 0.5,
        "use_static_image_mode": False, "smoothing_factor": 0.7,
        "inactivity_zone": 0.015, "click_cooldown": 0.4,
        "gesture_sensitivity": 0.02,
        "gesture_settings": {"scroll_sensitivity": 4, "double_click_time": 0.35},
        "calibration": {
            "enabled": True, "screen_margin": 0.1,
            "x_min": 0.15, "x_max": 0.85, "y_min": 0.15, "y_max": 0.85, "active": False
        },
        "max_fps": 60,
        # "custom_cursor_path": "red_cursor.cur", # Server handles this
        "server_host": "localhost", # New: server address
        "server_port": 8765        # New: server port
    }
    CONFIG_FILENAME = os.path.expanduser("~/.handpal_config.json") # Shared with server

    def _deep_update(self, target, source):
        for k, v in source.items():
            if isinstance(v, dict) and k in target and isinstance(target[k], dict):
                self._deep_update(target[k], v)
            else:
                target[k] = v

    def __init__(self, args=None):
        self.config = json.loads(json.dumps(self.DEFAULT_CONFIG)) # Deep copy
        if os.path.exists(self.CONFIG_FILENAME):
            try:
                with open(self.CONFIG_FILENAME, 'r') as f:
                    self._deep_update(self.config, json.load(f)) # Load all, client uses what it needs
                logger.info(f"Client config loaded from {self.CONFIG_FILENAME}")
            except Exception as e:
                logger.error(f"Error loading client config: {e}")

        if args: self._apply_cli_args(args)

        self.config["calibration"]["active"] = False # Runtime state
        self._validate_calibration(); self._validate_display_dims()
        logger.debug(f"Final client config: {self.config}")

    def _apply_cli_args(self, args):
        cli_args = vars(args)
        for key, value in cli_args.items():
            if value is not None:
                if key == 'flip_camera' and value is False: self.config['flip_camera'] = False
                # elif key == 'cursor_file': self.config['custom_cursor_path'] = value # Server handles
                elif key == 'server_host_cli': self.config['server_host'] = value # Special name to avoid conflict
                elif key == 'server_port_cli': self.config['server_port'] = value # Special name
                elif '.' in key: logger.warning(f"Nested CLI arg '{key}' ignored (not supported by simple client parse).")
                elif key in self.config:
                    if not isinstance(self.config[key], dict):
                        try: self.config[key] = type(self.config[key])(value)
                        except Exception: self.config[key] = value
                    else: logger.warning(f"Ignoring CLI arg '{key}' for dict field in client config.")


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

    def save(self): # Client saves its part of config
        try:
            save_cfg = json.loads(json.dumps(self.config))
            save_cfg.get("calibration", {}).pop("active", None) # Don't save runtime state

            # Load existing full config to not overwrite server parts
            full_config_to_save = {}
            if os.path.exists(self.CONFIG_FILENAME):
                with open(self.CONFIG_FILENAME, 'r') as f:
                    full_config_to_save = json.load(f)

            # Update with client-specific values from save_cfg
            for k, v in save_cfg.items():
                if k in self.DEFAULT_CONFIG: # Only save keys client owns by default
                    full_config_to_save[k] = v

            with open(self.CONFIG_FILENAME, 'w') as f:
                json.dump(full_config_to_save, f, indent=2)
            logger.info(f"Config (with client updates) saved to {self.CONFIG_FILENAME}")
        except Exception as e:
            logger.error(f"Error saving client config: {e}")

    def __getitem__(self, key): return self.get(key)
    def __setitem__(self, key, value): self.set(key, value)


# -----------------------------------------------------------------------------
# GestureRecognizer Class (Client-Side, uses pynput.Button for enum type)
# -----------------------------------------------------------------------------
class GestureRecognizer:
    def __init__(self, config):
        self.config = config
        self.last_positions = {}
        self.gesture_state = {
            "scroll_active": False, "last_click_time": 0,
            "last_click_button": None, "scroll_history": deque(maxlen=5),
            "active_gesture": None, "last_pose": {"Left": "U", "Right": "U"},
            "pose_stable_count": {"Left": 0, "Right": 0}, "fist_drag_active": False
        }
        self.POSE_STABILITY_THRESHOLD = 3

    def _dist(self, p1, p2):
        if None in [p1, p2] or not all(hasattr(p, 'x') for p in [p1,p2]): return float('inf')
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def check_thumb_index_click(self, hand_lm, handedness): # Returns action string
        if handedness != "Left" or hand_lm is None or self.gesture_state["scroll_active"] or self.gesture_state["fist_drag_active"]:
            return None
        try: thumb, index = hand_lm.landmark[4], hand_lm.landmark[8]
        except (IndexError, TypeError): return None

        dist = self._dist(thumb, index)
        now = time.time()
        thresh = self.config["gesture_sensitivity"]
        gesture_action = None

        if dist < thresh:
            cooldown = self.config["click_cooldown"]
            if (now - self.gesture_state["last_click_time"]) > cooldown:
                double_click_t = self.config["gesture_settings"]["double_click_time"]
                # Using pynput.mouse.Button for type consistency if comparing later
                if (now - self.gesture_state["last_click_time"]) < double_click_t and \
                   self.gesture_state["last_click_button"] == Button.left: # pynput.mouse.Button.left
                    gesture_action = "double_click"
                    self.gesture_state["active_gesture"] = "double_click"
                else:
                    gesture_action = "click"
                    self.gesture_state["active_gesture"] = "click"
                self.gesture_state["last_click_time"] = now
                self.gesture_state["last_click_button"] = Button.left # pynput.mouse.Button.left
        elif self.gesture_state["active_gesture"] in ["click", "double_click"]:
            self.gesture_state["active_gesture"] = None
        return gesture_action


    def check_fist_drag(self, hand_lm, handedness): # Returns action string: "drag_start", "drag_continue", "drag_end"
        if handedness != "Left" or hand_lm is None or self.gesture_state["scroll_active"]:
            return None
        raw_pose = self.detect_raw_pose(hand_lm)
        is_fist = raw_pose == "Fist"

        if self.gesture_state["fist_drag_active"]:
            try:
                lm=hand_lm.landmark; it,mt,rt,pt = lm[8],lm[12],lm[16],lm[20]
                im,mm,rm,pm = lm[5],lm[9],lm[13],lm[17]; y_ext = 0.03
                i_ext = it.y < im.y - y_ext; m_ext = mt.y < mm.y - y_ext
                r_ext = rt.y < rm.y - y_ext ; p_ext = pt.y < pm.y - y_ext
                num_ext = sum([i_ext,m_ext,r_ext,p_ext]); still_dragging = num_ext <= 1
            except (IndexError, TypeError): still_dragging = False

            if still_dragging: return "drag_continue"
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
            lm = hand_lm.landmark; it,mt,rt,pt = lm[8],lm[12],lm[16],lm[20]
            im,mm,rm,pm = lm[5],lm[9],lm[13],lm[17]; y_ext = 0.03
            i_ext = it.y < im.y - y_ext; m_ext = mt.y < mm.y - y_ext
            r_ext = rt.y < rm.y - y_ext; p_ext = pt.y < pm.y - y_ext
            num_ext = sum([i_ext,m_ext,r_ext,p_ext])
            if num_ext == 0: return "Fist"
            if num_ext >= 4: return "Open"
            if i_ext and num_ext == 1: return "Point"
            if i_ext and m_ext and num_ext == 2: return "Two"
            return "Other"
        except (IndexError, TypeError): return "U"

    def check_scroll_gesture(self, hand_landmarks, handedness): # Returns scroll_amount (float) or None
         if handedness != "Left": return None
         if self.gesture_state["active_gesture"] in ["click", "double_click"] or self.gesture_state["fist_drag_active"]:
             return None
         try: # Added try-except for safety
            index_tip = hand_landmarks.landmark[8]; middle_tip = hand_landmarks.landmark[12]
            index_mcp = hand_landmarks.landmark[5]; middle_mcp = hand_landmarks.landmark[9]
            ring_tip = hand_landmarks.landmark[16]; pinky_tip = hand_landmarks.landmark[20]
         except (IndexError, TypeError): return None

         index_extended = index_tip.y < index_mcp.y
         middle_extended = middle_tip.y < middle_mcp.y
         fingers_close = abs(index_tip.x - middle_tip.x) < 0.08
         ring_pinky_folded = (ring_tip.y > index_mcp.y) and (pinky_tip.y > index_mcp.y)

         if index_extended and middle_extended and fingers_close and ring_pinky_folded:
             if not self.gesture_state["scroll_active"]:
                 self.gesture_state["scroll_active"] = True
                 self.gesture_state["scroll_history"].clear()
                 self.gesture_state["active_gesture"] = "Scroll"

             if 8 in self.last_positions: # Index finger tip (landmark 8)
                 prev_y = self.last_positions[8][1]; curr_y = index_tip.y
                 # scroll_sensitivity is pixels per unit change, so higher is more sensitive
                 # delta_y is normalized change, multiply by sensitivity
                 delta_y = (curr_y - prev_y) * self.config["gesture_settings"]["scroll_sensitivity"]

                 if abs(delta_y) > 0.0005: # Threshold to add to history (normalized units)
                     self.gesture_state["scroll_history"].append(delta_y)

                 if len(self.gesture_state["scroll_history"]) > 0:
                     smooth_delta = sum(self.gesture_state["scroll_history"]) / len(self.gesture_state["scroll_history"])
                     if abs(smooth_delta) > 0.001: # Threshold for actual scroll action
                         self.last_positions[8] = (index_tip.x, index_tip.y)
                         # Return the smoothed delta, server will interpret this magnitude
                         return smooth_delta * 100 # Scale for server (arbitrary, server can re-scale)

             self.last_positions[8] = (index_tip.x, index_tip.y) # Update last pos for next frame
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
        i_ext=it.y<im.y-y_ext; m_ext=mt.y<mm.y-y_ext
        r_ext=rt.y<rm.y-y_ext; p_ext=pt.y<pm.y-y_ext
        num_ext = sum([i_ext,m_ext,r_ext,p_ext]); pose = "U"

        if handedness=="Left" and self.gesture_state["scroll_active"]: pose="Scroll"
        elif num_ext == 0: pose="Fist"
        elif i_ext and num_ext == 1: pose="Point"
        elif i_ext and m_ext and num_ext == 2: pose="Two"
        elif num_ext >= 4: pose="Open"

        last_known_pose=self.gesture_state["last_pose"].get(handedness,"U")
        stable_count=self.gesture_state["pose_stable_count"].get(handedness,0)

        if pose == last_known_pose and pose != "U": stable_count += 1
        else: stable_count = 0

        self.gesture_state["last_pose"][handedness] = pose
        self.gesture_state["pose_stable_count"][handedness] = stable_count

        if pose == "Fist" and stable_count >= 1: return pose # Fist is more immediate
        if pose == "Scroll" or (pose != "U" and stable_count >= self.POSE_STABILITY_THRESHOLD) : return pose
        else: return last_known_pose + "?"


# -----------------------------------------------------------------------------
# MotionSmoother Class (Client-Side, works with normalized 0-1 coords)
# -----------------------------------------------------------------------------
class MotionSmoother:
    def __init__(self, config):
        self.config = config; self.last_smooth_pos_norm = None; self._update_alpha()
        # Inactivity zone is applied to normalized (0-1) coordinates
        self.inactive_zone_sq_norm = self.config["inactivity_zone"]**2

    def _update_alpha(self):
        factor = max(0.01, min(0.99, self.config["smoothing_factor"]))
        self.alpha = 1.0 - factor # Inverted: higher factor means more smoothing (less alpha)

    def update(self, target_norm_x, target_norm_y): # Takes and returns normalized 0-1
        target_pos_norm = (target_norm_x, target_norm_y)
        if self.last_smooth_pos_norm is None:
            self.last_smooth_pos_norm = target_pos_norm
            return target_pos_norm

        dist_sq_norm = (target_norm_x - self.last_smooth_pos_norm[0])**2 + \
                       (target_norm_y - self.last_smooth_pos_norm[1])**2

        if dist_sq_norm < self.inactive_zone_sq_norm:
            return self.last_smooth_pos_norm # No change if within inactivity zone

        self._update_alpha() # Update alpha based on config (could be dynamic)

        # Corrected based on typical EMA: alpha for current, (1-alpha) for previous
        # If smoothing_factor = 0.7 (high smoothing), then less weight on current.
        # Let alpha_ema be the direct weight for current value.
        # smoothing_factor in config is "how much to smooth"
        # If smoothing_factor = 0.1 (low smoothing), current value has high weight (0.9)
        # If smoothing_factor = 0.9 (high smoothing), current value has low weight (0.1)
        current_weight = 1.0 - self.config["smoothing_factor"] # This is the 'alpha' in standard EMA

        smooth_norm_x = current_weight * target_norm_x + (1.0 - current_weight) * self.last_smooth_pos_norm[0]
        smooth_norm_y = current_weight * target_norm_y + (1.0 - current_weight) * self.last_smooth_pos_norm[1]

        self.last_smooth_pos_norm = (smooth_norm_x, smooth_norm_y)
        return smooth_norm_x, smooth_norm_y

    def reset(self): self.last_smooth_pos_norm = None; logger.debug("Client Smoother reset.")


# -----------------------------------------------------------------------------
# Detection Thread (Client-Side)
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
            if self.cap is None or not self.cap.isOpened():
                logger.error("Webcam N/A in DetectionThread"); time.sleep(0.5); continue
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None: time.sleep(0.05); continue
                if self.flip: frame = cv.flip(frame, 1)

                proc_frame = cv.resize(frame, (self.proc_w, self.proc_h), interpolation=cv.INTER_LINEAR)
                rgb_frame = cv.cvtColor(proc_frame, cv.COLOR_BGR2RGB); rgb_frame.flags.writeable=False
                results = self.hands.process(rgb_frame); rgb_frame.flags.writeable=True # Allow writes for drawing later

                try: self.data_q.put((frame, results, (self.proc_w, self.proc_h)), block=False, timeout=0.01) # Added timeout
                except queue.Full: # If full, try to clear one and add new
                    try: self.data_q.get_nowait(); self.data_q.put_nowait((frame, results, (self.proc_w, self.proc_h)))
                    except queue.Empty: pass # Should not happen if Full was true
                    except queue.Full: pass # Still full, drop frame
                frame_n += 1
            except Exception as e: logger.exception(f"DetectionThread loop error: {e}"); time.sleep(0.1)
        elapsed = time.perf_counter()-t_start; fps = frame_n/elapsed if elapsed>0 else 0
        logger.info(f"DetectionThread finished. {frame_n} frames in {elapsed:.2f}s (Avg FPS: {fps:.1f})")


# -----------------------------------------------------------------------------
# Tutorial Manager Class (Client-Side, with handpal_client_instance)
# -----------------------------------------------------------------------------
class TutorialManager:
    def __init__(self, handpal_cli): # Takes HandPalClient instance
        self.handpal_client = handpal_cli # Renamed
        self.active = False; self.current_step = 0
        self.step_completed_flag = False; self.success_message_start_time = 0
        self.action_confirm_start_time = 0
        self.SUCCESS_MESSAGE_DURATION = 0.95; self.POSITION_HOLD_DURATION = 0.75
        self.GESTURE_HOLD_DURATION = 0.35
        self.steps = [
            {"title": "Benvenuto in HandPal", "instruction": "Posiziona la mano DESTRA nello schermo per muovere il cursore.", "completion_type": "position", "icon": "👋"},
            {"title": "Click Sinistro", "instruction": "Con la mano SINISTRA, avvicina pollice e indice per fare click.", "completion_type": "gesture", "required_gesture": "click", "icon": "👆"},
            {"title": "Drag & Drop", "instruction": "Mano SINISTRA: chiudi a PUGNO per iniziare il Drag, muovi, poi apri per rilasciare.", "completion_type": "gesture", "required_gesture": "drag", "icon": "✊"},
            {"title": "Scorrimento", "instruction": "Mano SINISTRA: estendi indice e medio (altre dita piegate) per scorrere.", "completion_type": "gesture", "required_gesture": "Scroll", "icon": "✌️"}
        ]

    def start_tutorial(self):
        logger.info("Tutorial starting on client...")
        self.active = True; self.current_step = 0
        self._reset_step_state()
        self.handpal_client.tracking_enabled = True # Client controls its local tracking flag
        # Schedule the message sending through the client's asyncio mechanism
        if self.handpal_client:
            self.handpal_client.schedule_server_message({"type": "control", "command": "enable_tracking"})
        logger.info(f"Tutorial started. Step {self.current_step}: {self.steps[self.current_step].get('title')}")

    def stop_tutorial(self):
        logger.info("Tutorial stopping on client.")
        self.active = False; self._reset_step_state()

    def _reset_step_state(self):
        self.step_completed_flag = False; self.success_message_start_time = 0
        self.action_confirm_start_time = 0

    def _advance_to_next_step(self):
        logger.debug(f"Advancing from tutorial step {self.current_step}")
        self.current_step += 1
        if self.current_step >= len(self.steps):
            logger.info("Tutorial completed on client!"); self.stop_tutorial()
        else:
            self._reset_step_state()
            logger.info(f"Advanced to tutorial step {self.current_step}: {self.steps[self.current_step].get('title')}")

    def check_step_completion(self, results_mp): # results_mp from MediaPipe
        if not self.active or self.current_step >= len(self.steps): return
        now = time.time()

        if not self.step_completed_flag:
            current_step_config = self.steps[self.current_step]
            completion_type = current_step_config.get("completion_type")
            action_confirmed_this_frame = False

            if completion_type == "position": # Right hand presence
                if self.handpal_client.last_right_hand_lm_norm is not None:
                    if self.action_confirm_start_time == 0: self.action_confirm_start_time = now
                    elif (now - self.action_confirm_start_time) > self.POSITION_HOLD_DURATION:
                        action_confirmed_this_frame = True
                else: self.action_confirm_start_time = 0

            elif completion_type == "gesture":
                required_gesture = current_step_config.get("required_gesture")
                current_active_gesture = self.handpal_client.gesture_recognizer.gesture_state.get("active_gesture")
                gesture_match = (current_active_gesture == required_gesture) or \
                                (required_gesture == "click" and current_active_gesture == "double_click")

                if gesture_match:
                    if self.action_confirm_start_time == 0: self.action_confirm_start_time = now
                    elif (now - self.action_confirm_start_time) > self.GESTURE_HOLD_DURATION:
                        action_confirmed_this_frame = True
                else: self.action_confirm_start_time = 0

            # Removed menu_hover type for client, as menu trigger is local visualization for now.
            # Server doesn't need to know about hover for tutorial step completion here.

            if action_confirmed_this_frame:
                self.step_completed_flag = True
                self.success_message_start_time = now
                self.action_confirm_start_time = 0
                logger.info(f"Client Tutorial Step {self.current_step} COMPLETED. Displaying success.")

        if self.step_completed_flag:
            if (now - self.success_message_start_time) > self.SUCCESS_MESSAGE_DURATION:
                self._advance_to_next_step()

    def draw_tutorial_overlay(self, frame):
        # This function remains largely the same as original, using PIL if available
        # It will use self.handpal_client where it previously used self.handpal
        if not self.active or self.current_step >= len(self.steps): return frame

        h, w = frame.shape[:2]
        if w == 0 or h == 0: return frame

        overlay_height = 120
        tutorial_panel_img = np.zeros((overlay_height, w, 3), dtype=np.uint8)
        cv.rectangle(tutorial_panel_img, (0, 0), (w, overlay_height), (20, 20, 20), -1)

        alpha = 0.85
        try: # Ensure ROI is valid
            frame_roi = frame[h-overlay_height:h, 0:w]
            if frame_roi.shape[0] != overlay_height or frame_roi.shape[1] != w :
                # Fallback if frame is smaller than expected overlay
                logger.warning("Frame too small for full tutorial overlay, drawing at top.")
                frame_roi = frame[0:overlay_height, 0:w]
                tutorial_panel_img = np.zeros((frame_roi.shape[0], frame_roi.shape[1], 3), dtype=np.uint8)
                cv.rectangle(tutorial_panel_img, (0,0), (frame_roi.shape[1], frame_roi.shape[0]), (20,20,20),-1)


            blended_roi = cv.addWeighted(frame_roi, 1-alpha, tutorial_panel_img, alpha, 0)

            if frame_roi.shape[0] != overlay_height or frame_roi.shape[1] != w :
                 frame[0:overlay_height, 0:w] = blended_roi
            else:
                 frame[h-overlay_height:h, 0:w] = blended_roi

        except Exception as e:
            logger.error(f"Error creating tutorial blended ROI: {e}")
            # Don't draw overlay if error to prevent crash
            return frame

        current_step_config = self.steps[self.current_step]
        title = current_step_config.get("title", "Tutorial")
        instruction = current_step_config.get("instruction", "")
        icon = current_step_config.get("icon", "📚")

        num_steps = len(self.steps)
        dot_area_width = num_steps * 25
        start_x_dots = (w - dot_area_width) // 2

        y_base = h - overlay_height if (frame.shape[0] >= overlay_height) else 0

        y_title_baseline = y_base + 30
        y_instruction_baseline = y_base + 65
        y_fantastico_baseline = y_base + 95
        y_esc_msg_baseline = y_base + 20 # Relative to overlay top
        y_progress_text_baseline = y_base + overlay_height - 20 # Relative to overlay top, near bottom of overlay


        if PIL_AVAILABLE:
            pil_img = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            font_size_title = 22; font_size_instruction = 18; font_size_fantastico = 20
            font_size_esc = 15; font_size_progress_text = 16; font_size_dot_icon = 18
            default_font_family = "arial.ttf"
            if os.name == 'nt': emoji_font_family = "seguiemj.ttf"
            elif sys.platform == "darwin": emoji_font_family = "AppleColorEmoji.ttf"
            else: emoji_font_family = "NotoColorEmoji.ttf"

            def load_font(preferred, size, fallback=default_font_family):
                try: return ImageFont.truetype(preferred, size)
                except IOError:
                    try: return ImageFont.truetype(fallback, size)
                    except IOError: return ImageFont.load_default()

            font_title = load_font(emoji_font_family, font_size_title)
            font_instruction = load_font(default_font_family, font_size_instruction)
            font_fantastico = load_font(default_font_family, font_size_fantastico)
            font_esc = load_font(default_font_family, font_size_esc)
            font_dot_icon = load_font(emoji_font_family, font_size_dot_icon)

            text_anchor_args = {"anchor": "ls"} if PIL_VERSION_TUPLE >= (8,0,0) else {}

            draw.text((20, y_title_baseline), f"{icon} {title}", font=font_title, fill=(255,255,255), **text_anchor_args)
            draw.text((20, y_instruction_baseline), instruction, font=font_instruction, fill=(220,220,220), **text_anchor_args)
            if self.step_completed_flag and self.success_message_start_time > 0:
                 draw.text((w // 2 - 100, y_fantastico_baseline), "FANTASTICO!", font=font_fantastico, fill=(60,255,60), **text_anchor_args)
            draw.text((w-220 if w > 220 else 20, y_esc_msg_baseline), "Premi ESC per uscire dal tutorial", font=font_esc, fill=(150,150,150), **text_anchor_args)

            for i in range(num_steps):
                color = (0, 220, 220) if i == self.current_step else \
                        (0, 150, 0) if i < self.current_step else \
                        (100, 100, 100)
                # Convert color from BGR (cv) to RGB (PIL)
                pil_color = (color[2], color[1], color[0])
                dot_x = start_x_dots + i*25
                dot_y = y_progress_text_baseline + 7 # Adjust for circle center from text baseline
                draw.ellipse([(dot_x-7, dot_y-7), (dot_x+7, dot_y+7)], fill=pil_color)

                if i == self.current_step:
                    step_icon = self.steps[i].get("icon", "")
                    # Adjust icon position to be centered on the dot
                    # Adjust icon position to be centered on the dot
                    # Calculate actual width and height of the step_icon
                    if hasattr(draw, 'textbbox'): # Preferred for Pillow >= 9.0
                        # textbbox(xy, text, ...) returns (left, top, right, bottom)
                        # We pass (0,0) for xy as we only need the box dimensions, not its position yet.
                        bbox = draw.textbbox((0,0), step_icon, font=font_dot_icon)
                        icon_w = bbox[2] - bbox[0]
                        icon_h = bbox[3] - bbox[1]
                    elif hasattr(draw, 'textsize'): # Fallback for older Pillow versions
                        size_tuple = draw.textsize(step_icon, font=font_dot_icon)
                        icon_w = size_tuple[0]
                        icon_h = size_tuple[1]
                    else:
                        # Absolute fallback if neither textbbox nor textsize is available (very unlikely for modern Pillow)
                        # or if using a very old font object that only has getsize.
                        try:
                            getsize_result = font_dot_icon.getsize(step_icon)
                            icon_w = getsize_result[0]
                            icon_h = getsize_result[1]
                        except AttributeError: # Ultimate fallback
                            icon_w = 14 * len(step_icon) # Rough estimate based on char count
                            icon_h = 14                  # Fixed estimate

                    icon_x_on_dot = dot_x - (icon_w / 2)
                    # The -2 is an empirical vertical adjustment that was in your original code.
                    # You might need to fine-tune it based on the font and icons.
                    icon_y_on_dot = dot_y - (icon_h / 2) - 2
                    draw.text((icon_x_on_dot, icon_y_on_dot), step_icon, font=font_dot_icon, fill=(0,0,0), **text_anchor_args)


            frame = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)

        else: # PIL_AVAILABLE is False
            cv.putText(frame, f"{icon} {title}", (20, y_title_baseline), cv.FONT_HERSHEY_TRIPLEX, 0.8, (255,255,255), 1, cv.LINE_AA)
            cv.putText(frame, instruction, (20, y_instruction_baseline), cv.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1, cv.LINE_AA)
            if self.step_completed_flag and self.success_message_start_time > 0:
                 cv.putText(frame, "✓ FANTASTICO!", (w // 2 - 100, y_fantastico_baseline), cv.FONT_HERSHEY_SIMPLEX, 0.7, (60,255,60), 2, cv.LINE_AA)
            cv.putText(frame, "Premi ESC per uscire dal tutorial", (w-220 if w > 220 else 20, y_esc_msg_baseline), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150),1,cv.LINE_AA)
            for i in range(num_steps):
                color = (0, 220, 220) if i == self.current_step else \
                        (0, 150, 0) if i < self.current_step else \
                        (100, 100, 100)
                cv.circle(frame, (start_x_dots + i*25, y_progress_text_baseline + 7), 7, color, -1)
                if i == self.current_step:
                    step_icon_cv = self.steps[i].get("icon", "") # Emojis might not render well in OpenCV
                    cv.putText(frame, step_icon_cv, (start_x_dots + i*25 -7, y_progress_text_baseline+7+5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1,cv.LINE_AA)


        return frame

# -----------------------------------------------------------------------------
# Menu Related (Client-Side)
# -----------------------------------------------------------------------------
APP_CSV_PATH = 'applications.csv' # Client reads this
def create_default_apps_csv(): # Client creates this if not exists
    if not os.path.exists(APP_CSV_PATH):
        logger.info(f"Client creating default apps file: {APP_CSV_PATH}")
        try:
            with open(APP_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f); writer.writerow(['label', 'path', 'color', 'icon'])
                writer.writerow(['Calculator', 'calc.exe', '#0078D7', '🧮']) # Path is for server
                writer.writerow(['Browser', 'https://duckduckgo.com', '#DE5833', '🌐'])
                writer.writerow(['Notepad', 'notepad.exe', '#FFDA63', '📝'])
                writer.writerow(['Tutorial', '@tutorial', '#00B894', '📚']) # Special path
        except Exception as e: logger.error(f"Client failed to create {APP_CSV_PATH}: {e}")

def read_applications_from_csv(): # Client reads this
    create_default_apps_csv(); apps = []
    default_color, default_icon = '#555555', '🚀'
    try:
        with open(APP_CSV_PATH, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label=row.get('label','?'); path=row.get('path')
                if not path: logger.warning(f"Skipping row (no path): {row}"); continue
                color=row.get('color', default_color).strip()
                icon=row.get('icon', default_icon).strip()
                if not (color.startswith('#') and len(color) == 7): color = default_color
                apps.append({'label': label, 'path': path, 'color': color, 'icon': icon})
        logger.info(f"Client loaded {len(apps)} apps from {APP_CSV_PATH}")
    except Exception as e:
        logger.error(f"Client error reading {APP_CSV_PATH}: {e}")
        apps=[{'label':'Error','path':'','color':'#F00','icon':'⚠'}]
    return apps

# This global function is replaced by HandPalClient.schedule_server_launch
# and HandPalClient._async_launch_app_on_server
# async def launch_app_on_server(path_or_label): # Client sends command to server
#     global handpal_client_instance
#     if not path_or_label:
#         logger.warning("Client: Launch attempt with empty path/label.")
#         return
#
#     payload = {"type": "launch", "path": path_or_label}
#     if handpal_client_instance and handpal_client_instance.websocket:
#         await handpal_client_instance.send_to_server(payload)
#         logger.info(f"Client sent launch request for: {path_or_label}")
#         if handpal_client_instance.floating_menu and handpal_client_instance.floating_menu.visible:
#              handpal_client_instance.floating_menu.hide() # Hide menu after sending
#     else:
#         logger.error("Client: Cannot send launch request, no WebSocket or HandPal instance.")


class FloatingMenu: # Client-Side Tkinter Menu
    def __init__(self, root_tk, client_app_instance): # MODIFIED: Added client_app_instance
        self.root = root_tk
        self.client_app = client_app_instance # MODIFIED: Store client app instance
        self.window = Toplevel(self.root)
        self.window.title("HandPal Menu (Client)"); self.window.attributes("-topmost", True)
        self.window.overrideredirect(True); self.window.configure(bg='#222831')
        self.apps = read_applications_from_csv() # Client reads its local CSV
        self._create_elements()
        self.window.withdraw(); self.visible = False
        self.window.bind("<ButtonPress-1>", self._start_move)
        self.window.bind("<ButtonRelease-1>", self._stop_move)
        self.window.bind("<B1-Motion>", self._do_move)
        self._offset_x = 0; self._offset_y = 0
        logger.info("Client FloatingMenu initialized.")

    def _create_elements(self):
        title_f = Frame(self.window, bg='#222831', pady=15); title_f.pack(fill=X)
        Label(title_f, text="HANDPAL MENU", font=("Helvetica",14,"bold"),bg='#222831',fg='#EEEEEE').pack()
        Label(title_f, text="Launch Application", font=("Helvetica",10),bg='#222831',fg='#00ADB5').pack(pady=(0,10))

        btn_cont = Frame(self.window, bg='#222831', padx=20); btn_cont.pack(fill=BOTH, expand=True)
        num_apps = len(self.apps)
        calc_ideal_h = 160 + num_apps * 48; MIN_H = 250
        screen_h = self.root.winfo_screenheight()
        MAX_H_RATIO = 0.85; dynamic_max_h = int(screen_h * MAX_H_RATIO)
        menu_h = min(max(calc_ideal_h, MIN_H), dynamic_max_h)
        self.window.geometry(f"280x{menu_h}+50+50")

        for app_data in self.apps:
            f = Frame(btn_cont, bg='#222831', pady=5); f.pack(fill=X)
            # MODIFIED: Command now calls client_app's method
            btn = TkButton(f, text=f"{app_data.get('icon','')} {app_data.get('label','?')}",
                           bg=app_data.get('color','#555'), fg="white", font=("Helvetica",11),
                           relief=tk.FLAT, borderwidth=0, padx=10, pady=6, width=20, anchor='w',
                           command=lambda p=app_data.get('path'): self.client_app.schedule_server_launch(p))
            btn.pack(fill=X)

        bottom_f = Frame(self.window, bg='#222831', pady=10); bottom_f.pack(fill=X, side=tk.BOTTOM)
        TkButton(bottom_f, text="✖ Close Menu", bg='#393E46', fg='#EEEEEE',
                 font=("Helvetica",10), relief=tk.FLAT, borderwidth=0,
                 padx=10, pady=5, width=15, command=self.hide).pack(pady=5)

    def show(self):
        if not self.visible: self.window.deiconify(); self.window.lift(); self.visible=True; logger.debug("Client menu shown.")
    def hide(self):
        if self.visible: self.window.withdraw(); self.visible=False; logger.debug("Client menu hidden.")
    def toggle(self): (self.hide if self.visible else self.show)()
    def _start_move(self, event): self._offset_x, self._offset_y = event.x, event.y
    def _stop_move(self, event): self._offset_x = self._offset_y = 0
    def _do_move(self,event): x=self.window.winfo_x()+event.x-self._offset_x; y=self.window.winfo_y()+event.y-self._offset_y; self.window.geometry(f"+{x}+{y}")


# -----------------------------------------------------------------------------
# HandPalClient Class
# -----------------------------------------------------------------------------
class HandPalClient:
    def __init__(self, config_obj, ws_uri): # Takes Config object and server URI
        self.config = config_obj
        self.ws_uri = ws_uri
        self.websocket = None # WebSocket connection instance
        self.is_connected = False
        self.message_send_queue = asyncio.Queue() # For sending messages from sync threads
        self.message_receive_queue = asyncio.Queue() # For receiving messages in sync main loop
        self.async_loop = None # MODIFIED: To store asyncio event loop from ws_thread

        self.gesture_recognizer = GestureRecognizer(self.config)
        try:
            self.tk_root = tk.Tk(); self.tk_root.withdraw()
            self.client_screen_size = (self.tk_root.winfo_screenwidth(), self.tk_root.winfo_screenheight())
            # MODIFIED: Pass self (HandPalClient instance) to FloatingMenu
            self.floating_menu = FloatingMenu(self.tk_root, self)
            logger.info(f"Client screen: {self.client_screen_size}. Menu OK.")
        except tk.TclError as e:
            logger.error(f"Tkinter init failed for client: {e}. Menu N/A.")
            self.tk_root=None; self.floating_menu=None
            self.client_screen_size=(1920,1080) # Fallback

        self.motion_smoother = MotionSmoother(self.config) # Works with normalized 0-1
        self.tutorial_manager = TutorialManager(self)

        self.running = False; self.stop_event = threading.Event(); self.detection_thread = None
        self.data_queue = queue.Queue(maxsize=2) # For frames from detection thread
        self.cap = None; self.mp_hands = mp.solutions.hands; self.hands_instance = None
        self.mp_drawing = mp.solutions.drawing_utils; self.mp_drawing_styles = mp.solutions.drawing_styles

        self.last_sent_cursor_pos_norm = None # Store normalized pos
        self.last_right_hand_lm_norm = None # Raw normalized from mediapipe
        self.tracking_enabled = True; self.debug_mode = False
        self.calibration_active = False; self.calibration_points = []; self.calibration_step = 0
        self.calib_corners = ["top-left", "top-right", "bottom-right", "bottom-left"]

        self.menu_trigger_zone_px = {"cx":0, "cy":0, "radius_sq":0, "is_valid": False}
        self.current_display_dims = (0,0) # For drawing on client's OpenCV window
        self.menu_trigger_active = False; self._menu_activate_time = 0
        self.MENU_HOVER_DELAY = 0.3

        self.fps_stats = deque(maxlen=60)
        self._last_proc_dims = tuple(self.config.get(key) for key in ["process_width", "process_height"])
        self.debug_values = {"fps":0.0, "q":0, "L":"U", "R":"U", "gest":"N/A", "map_norm_sent":"-", "menu":"Off", "ws_stat":"Offline"}
        logger.info("HandPalClient instance initialized.")

    async def send_to_server(self, data_dict):
        if self.websocket and self.is_connected:
            try:
                await self.websocket.send(json.dumps(data_dict))
            except websockets.exceptions.ConnectionClosed:
                logger.error("WS connection closed while trying to send. Marking as disconnected.")
                self.is_connected = False
                self.debug_values["ws_stat"] = "Disconnected"
            except Exception as e:
                logger.error(f"Error sending to server: {e}")
        else:
            # If trying to send while not connected, put in queue to be picked up by ws_sender_task
            # This is useful for messages originating from non-async contexts that need to be sent
            # once connection is (re-)established.
            # However, for direct calls to send_to_server from async contexts, usually means ws is down.
            # Let's assume ws_sender_task handles queuing for robustness.
            # logger.warning(f"WS not connected. Cannot send directly: {data_dict.get('type')}")
            # Add to general message queue if appropriate, or handle error
            # For now, ws_sender_task is the primary sender.
            # This direct send_to_server is used by _async_launch_app_on_server,
            # which runs in ws_thread, so it expects websocket to be live.
            # If called from other contexts, consider queuing.
            logger.warning(f"WS not connected. Failed to send: {data_dict.get('type')}")


    async def ws_sender_task(self):
        while self.running:
            try:
                message_to_send = await asyncio.wait_for(self.message_send_queue.get(), timeout=0.1)
                if self.websocket and self.is_connected:
                    await self.send_to_server(message_to_send) # Uses the modified send_to_server
                    self.message_send_queue.task_done()
                else: # Put it back if not connected
                    await self.message_send_queue.put(message_to_send)
                    await asyncio.sleep(0.5) # Wait before retrying
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in ws_sender_task: {e}")
                await asyncio.sleep(0.1)


    async def ws_receiver_task(self):
        while self.running and self.websocket and self.is_connected:
            try:
                message = await asyncio.wait_for(self.websocket.recv(), timeout=0.1)
                # logger.debug(f"Client received from server: {message}")
                await self.message_receive_queue.put(message)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WS connection closed by server in receiver task.")
                self.is_connected = False
                self.debug_values["ws_stat"] = "Disconnected"
                break
            except Exception as e:
                logger.error(f"Error in ws_receiver_task: {e}")
                self.is_connected = False # Assume connection is problematic
                self.debug_values["ws_stat"] = "Error"
                break


    async def connect_and_manage_ws(self):
        # MODIFIED: Capture the asyncio event loop for this thread
        try:
            self.async_loop = asyncio.get_running_loop() # Python 3.7+
        except RuntimeError: # Fallback for older Python or if called unexpectedly
            self.async_loop = asyncio.get_event_loop()
            
        reconnect_delay = 5
        while self.running:
            try:
                async with websockets.connect(self.ws_uri, ping_interval=20, ping_timeout=20) as ws:
                    self.websocket = ws
                    self.is_connected = True
                    self.debug_values["ws_stat"] = "Online"
                    logger.info(f"Connected to server: {self.ws_uri}")

                    await self.send_to_server({"type": "request_initial_cursor"})

                    sender = asyncio.create_task(self.ws_sender_task())
                    receiver = asyncio.create_task(self.ws_receiver_task())
                    await asyncio.gather(sender, receiver) # Keep running these
            except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError, OSError) as e:
                logger.error(f"WebSocket connection failed or closed: {e}. Retrying in {reconnect_delay}s...")
                self.is_connected = False
                self.debug_values["ws_stat"] = f"Offline ({type(e).__name__})"
            except Exception as e:
                logger.error(f"Unexpected error in WebSocket management: {e}. Retrying in {reconnect_delay}s...")
                self.is_connected = False
                self.debug_values["ws_stat"] = "Error"

            self.websocket = None # Clear instance on disconnect
            if not self.running: break
            await asyncio.sleep(reconnect_delay)

    # MODIFIED: New method to schedule launch from Tkinter via ws_thread
    def schedule_server_launch(self, path_to_launch):
        if self.async_loop:
            coro = self._async_launch_app_on_server(path_to_launch)
            future = asyncio.run_coroutine_threadsafe(coro, self.async_loop)
            # Optionally, handle future result/exceptions if needed:
            # future.add_done_callback(self._handle_launch_result)
            logger.info(f"Scheduled app launch for '{path_to_launch}' on async loop.")
        else:
            logger.error("Client: Cannot schedule app launch. Asyncio loop not available.")

    # MODIFIED: New async helper for launching app, runs in ws_thread
    async def _async_launch_app_on_server(self, path_or_label):
        if not path_or_label:
            logger.warning("Client: Launch attempt with empty path/label.")
            return

        payload = {"type": "launch", "path": path_or_label}
        await self.send_to_server(payload) # This runs in ws_thread, websocket should be available
        logger.info(f"Client sent launch request (from async helper) for: {path_or_label}")

        # Hide menu: Tkinter ops should be from main thread.
        if self.floating_menu and self.floating_menu.visible:
            if self.tk_root:
                # Schedule hide() to be called by Tkinter's main loop
                self.tk_root.after(0, self.floating_menu.hide)
            else: # Fallback if tk_root is somehow not available
                try:
                    self.floating_menu.hide()
                except Exception as e:
                    logger.warning(f"Error in fallback menu hide from ws_thread: {e}")

    # New method for TutorialManager to send messages via asyncio loop
    def schedule_server_message(self, message_dict):
        if self.async_loop:
            coro = self.send_to_server(message_dict)
            future = asyncio.run_coroutine_threadsafe(coro, self.async_loop)
            logger.info(f"Scheduled server message {message_dict.get('type')} on async loop.")
        else:
            # Fallback: try putting on the message_send_queue directly (less ideal for single messages)
            # Or log error if async_loop is critical path
            logger.warning(f"Async loop not available for scheduling message: {message_dict.get('type')}. Queuing directly.")
            try:
                self.message_send_queue.put_nowait(message_dict)
            except Exception as e:
                logger.error(f"Failed to queue message: {e}")


    def start_client_services(self): # Was 'start'
        if self.running: return True
        logger.info("Starting HandPalClient services..."); self.stop_event.clear()
        try:
            self.cap = cv.VideoCapture(self.config["device"], cv.CAP_DSHOW if os.name == 'nt' else cv.CAP_ANY)
            if not self.cap.isOpened(): self.cap = cv.VideoCapture(self.config["device"]) # Try again
            if not self.cap.isOpened(): raise IOError(f"Cannot open webcam {self.config['device']}")

            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.config["width"])
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.config["height"])
            self.cap.set(cv.CAP_PROP_FPS, self.config["max_fps"])
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1) # Keep buffer small for low latency
            w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)); h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv.CAP_PROP_FPS)
            logger.info(f"Webcam OK: {w}x{h} @ {fps:.1f} FPS (Req: {self.config['width']}x{self.config['height']} @ {self.config['max_fps']})")

            self.hands_instance = self.mp_hands.Hands(
                static_image_mode=self.config["use_static_image_mode"], max_num_hands=2,
                min_detection_confidence=self.config["min_detection_confidence"],
                min_tracking_confidence=self.config["min_tracking_confidence"])

            self.detection_thread = DetectionThread(self.config, self.cap, self.hands_instance, self.data_queue, self.stop_event)
            self.detection_thread.start()

            self.running = True # Set running before starting async tasks
            self.ws_thread = threading.Thread(target=lambda: asyncio.run(self.connect_and_manage_ws()), daemon=True, name="WSThread")
            self.ws_thread.start()

            logger.info("HandPalClient services started successfully!")
            return True
        except Exception as e:
            logger.exception(f"Client startup failed: {e}")
            self.stop_client_services() # Cleanup
            return False

    def stop_client_services(self): # Was 'stop'
        if not self.running and not self.stop_event.is_set(): # Already stopped or stopping
            self.stop_event.set() # Ensure it's set
            # cv.destroyAllWindows() # Usually called at end of main_loop or here if main_loop not reached
            # self._destroy_tkinter() # Same
            return
        if not self.running: return # Already stopped

        logger.info("Stopping HandPalClient services...")
        self.running = False # Signal loops to stop
        self.stop_event.set() # Signal detection thread

        if self.ws_thread and self.ws_thread.is_alive():
            logger.debug("Waiting for WebSocket thread to join...")
            self.ws_thread.join(timeout=3.0)
            if self.ws_thread.is_alive():
                logger.warning("WebSocket thread did not terminate gracefully.")

        if self.detection_thread and self.detection_thread.is_alive():
            logger.debug("Joining DetectionThread...")
            self.detection_thread.join(timeout=2.0)
            if self.detection_thread.is_alive():
                logger.warning(f"DetectionThread did not terminate gracefully.")

        if hasattr(self.hands_instance, 'close') and self.hands_instance:
            self.hands_instance.close(); self.hands_instance = None

        if self.cap: self.cap.release(); self.cap = None; logger.debug("Webcam released.")

        cv.destroyAllWindows(); logger.info("OpenCV windows closed by client.")
        self._destroy_tkinter()
        logger.info("HandPalClient services stopped.")

    def _destroy_tkinter(self):
        if self.tk_root:
            try:
                logger.debug("Destroying Tkinter root for client..."); self.tk_root.quit(); self.tk_root.destroy()
                logger.debug("Client Tkinter root destroyed.")
            except Exception as e: logger.error(f"Error destroying client Tkinter: {e}")
            self.tk_root = None; self.floating_menu = None

    def _map_hand_to_normalized_interaction(self, x_norm_cam, y_norm_cam):
        # Takes raw normalized camera coords (0-1), applies client calibration
        # and margin, returns normalized interaction coords (0-1) for sending.
        x_cal, y_cal = x_norm_cam, y_norm_cam

        if self.config["calibration.enabled"] and not self.calibration_active :
            x_min, x_max = self.config["calibration.x_min"], self.config["calibration.x_max"]
            y_min, y_max = self.config["calibration.y_min"], self.config["calibration.y_max"]
            range_x = x_max - x_min; range_y = y_max - y_min

            if range_x > 0.01 and range_y > 0.01: # Avoid division by zero or tiny range
                x_clamped = max(x_min, min(x_norm_cam, x_max))
                y_clamped = max(y_min, min(y_norm_cam, y_max))
                x_cal = (x_clamped - x_min) / range_x
                y_cal = (y_clamped - y_min) / range_y

        margin = self.config["calibration.screen_margin"]
        x_final_norm = x_cal * (1 + 2 * margin) - margin
        y_final_norm = y_cal * (1 + 2 * margin) - margin

        # Clamp final normalized coordinates to 0-1 strictly
        x_final_norm = max(0.0, min(x_final_norm, 1.0))
        y_final_norm = max(0.0, min(y_final_norm, 1.0))

        return x_final_norm, y_final_norm


    def get_hand_pos_in_display_pixels(self): # For client's local display drawing
        if self.last_right_hand_lm_norm is None: return None
        # current_display_dims is the OpenCV window size on client
        if self.current_display_dims[0] == 0 or self.current_display_dims[1] == 0: return None
        # _last_proc_dims is the MediaPipe processing resolution
        if self._last_proc_dims[0] == 0 or self._last_proc_dims[1] == 0: return None

        norm_x_proc, norm_y_proc = self.last_right_hand_lm_norm # Raw 0-1 from MP

        # This function now maps the raw 0-1 from MP directly to client's display window pixels
        # It does NOT use the calibration data intended for server interaction.
        display_w, display_h = self.current_display_dims

        # Simple scaling of raw landmark to display window
        display_x = int(norm_x_proc * display_w)
        display_y = int(norm_y_proc * display_h)

        display_x = max(0, min(display_x, display_w - 1))
        display_y = max(0, min(display_y, display_h - 1))
        return (display_x, display_y)

    def check_menu_trigger(self): # For client's local menu visualization
        if not self.menu_trigger_zone_px["is_valid"]:
            if self.menu_trigger_active: logger.debug("Menu trigger deactivated (Zone Invalid).")
            self.menu_trigger_active = False; self._menu_activate_time = 0
            if self.debug_mode: self.debug_values["menu"] = "Off (Zone Invalid)"
            return False

        # Use raw hand landmark for menu trigger visualization on client screen
        hand_pos_px_display = self.get_hand_pos_in_display_pixels()

        if hand_pos_px_display is None:
            if self.menu_trigger_active: logger.debug("Menu trigger deactivated (Hand Lost).")
            self.menu_trigger_active = False; self._menu_activate_time = 0
            if self.debug_mode: self.debug_values["menu"] = "Off (No Hand)"
            return False

        hx_px, hy_px = hand_pos_px_display
        cx_px, cy_px = self.menu_trigger_zone_px["cx"], self.menu_trigger_zone_px["cy"]
        radius_sq_px = self.menu_trigger_zone_px["radius_sq"]
        is_inside = ((hx_px - cx_px)**2 + (hy_px - cy_px)**2) < radius_sq_px

        now = time.time(); activated_for_menu_display = False
        if is_inside:
            if not self.menu_trigger_active:
                self.menu_trigger_active = True; self._menu_activate_time = now
            if (now - self._menu_activate_time) >= self.MENU_HOVER_DELAY:
                activated_for_menu_display = True
                if self.debug_mode: self.debug_values["menu"] = "ACTIVATE!"
            else:
                 if self.debug_mode: self.debug_values["menu"] = f"Hover {(now-self._menu_activate_time):.1f}s"
        else:
            if self.menu_trigger_active: logger.debug("Menu Trigger Zone Exited.")
            self.menu_trigger_active = False; self._menu_activate_time = 0
            if self.debug_mode: self.debug_values["menu"] = "Off"
        return activated_for_menu_display

    def process_results_and_send(self, mp_results, proc_dims_tuple):
        self._last_proc_dims = proc_dims_tuple # Store processing dimensions
        lm_l, lm_r = None, None; self.last_right_hand_lm_norm = None # Raw MP norm

        if mp_results and mp_results.multi_hand_landmarks and mp_results.multi_handedness:
            for i, hand_lm_mp in enumerate(mp_results.multi_hand_landmarks):
                try: label = mp_results.multi_handedness[i].classification[0].label
                except (IndexError, AttributeError): continue
                if label == "Right":
                    lm_r = hand_lm_mp
                    try: # Get landmark for cursor (e.g. index finger tip)
                        self.last_right_hand_lm_norm = (hand_lm_mp.landmark[8].x, hand_lm_mp.landmark[8].y)
                    except IndexError: self.last_right_hand_lm_norm = None
                elif label == "Left": lm_l = hand_lm_mp

        # Right hand for cursor movement
        if lm_r and self.last_right_hand_lm_norm is not None:
            if self.check_menu_trigger(): # Local visual check for menu
                 if self.floating_menu and not self.floating_menu.visible and not self.tutorial_manager.active:
                     logger.info("Client menu trigger activated by hand hover."); self.floating_menu.show()

            if self.tracking_enabled and not self.calibration_active:
                # 1. Get raw normalized camera coordinates from MediaPipe
                raw_norm_x_cam, raw_norm_y_cam = self.last_right_hand_lm_norm

                # 2. Apply client's calibration and margin to get normalized interaction coordinates
                norm_x_interaction, norm_y_interaction = self._map_hand_to_normalized_interaction(raw_norm_x_cam, raw_norm_y_cam)

                # 3. Smooth these normalized interaction coordinates
                smooth_norm_x, smooth_norm_y = self.motion_smoother.update(norm_x_interaction, norm_y_interaction)

                # 4. Send smoothed normalized coordinates to server
                if self.last_sent_cursor_pos_norm is None or \
                   abs(smooth_norm_x - self.last_sent_cursor_pos_norm[0]) > 1e-4 or \
                   abs(smooth_norm_y - self.last_sent_cursor_pos_norm[1]) > 1e-4: # Send if changed significantly

                    # Queue the message for the sender task
                    try:
                        self.message_send_queue.put_nowait(
                            {"type": "move", "x": smooth_norm_x, "y": smooth_norm_y}
                        )
                    except queue.Full:
                        logger.warning("Client message_send_queue full, dropping move command.")

                    self.last_sent_cursor_pos_norm = (smooth_norm_x, smooth_norm_y)
                    if self.debug_mode: self.debug_values["map_norm_sent"] = f"{smooth_norm_x:.3f},{smooth_norm_y:.3f}"
        else:
            self.motion_smoother.reset(); self.last_right_hand_lm_norm = None
            # Visual menu trigger logic for client display
            if self.menu_trigger_active: logger.debug("Client menu trigger deactivated (Right Hand Lost).")
            self.menu_trigger_active = False; self._menu_activate_time = 0
            if self.debug_mode: self.debug_values["menu"] = "Off (No R Hand)"

        # Left hand for gestures
        action_performed_this_frame = False # To prioritize gestures
        if lm_l and not self.calibration_active:
            # Scroll
            scroll_val = self.gesture_recognizer.check_scroll_gesture(lm_l, "Left")
            if scroll_val is not None:
                try:
                    self.message_send_queue.put_nowait({"type": "scroll", "dy": int(scroll_val * -1)})
                except queue.Full: logger.warning("Client message_send_queue full, dropping scroll command.")
                action_performed_this_frame = True; logger.debug(f"Client sending scroll: {int(scroll_val * -1)}")
                if self.debug_mode: self.debug_values["gest"] = "Scroll"

            # Fist Drag
            if not action_performed_this_frame:
                drag_action_str = self.gesture_recognizer.check_fist_drag(lm_l, "Left")
                if drag_action_str:
                    action_performed_this_frame = True
                    try:
                        if drag_action_str == "drag_start":
                            self.message_send_queue.put_nowait({"type": "drag", "action": "start"})
                            logger.info("Client sending Drag Start")
                            if self.debug_mode: self.debug_values["gest"] = "Drag Start ⛓️"
                        elif drag_action_str == "drag_end":
                            self.message_send_queue.put_nowait({"type": "drag", "action": "end"})
                            logger.info("Client sending Drag End")
                            if self.debug_mode: self.debug_values["gest"] = "Drag End"
                        elif drag_action_str == "drag_continue":
                             if self.debug_mode: self.debug_values["gest"] = "Dragging ⛓️"
                    except queue.Full: logger.warning("Client message_send_queue full, dropping drag command.")


            # Click / Double Click
            if not action_performed_this_frame and not self.gesture_recognizer.gesture_state["fist_drag_active"]:
                click_action_str = self.gesture_recognizer.check_thumb_index_click(lm_l, "Left")
                if click_action_str:
                    num_clicks = 2 if click_action_str == "double_click" else 1
                    try:
                        self.message_send_queue.put_nowait(
                            {"type": "click", "button": "left", "count": num_clicks}
                        )
                    except queue.Full: logger.warning("Client message_send_queue full, dropping click command.")

                    action_performed_this_frame = True
                    logger.info(f"Client sending {click_action_str}")
                    if self.debug_mode: self.debug_values["gest"] = click_action_str.capitalize()

        if self.debug_mode:
            self.debug_values["L"] = self.gesture_recognizer.detect_hand_pose(lm_l, "Left")
            self.debug_values["R"] = self.gesture_recognizer.detect_hand_pose(lm_r, "Right")
            if not action_performed_this_frame:
                internal_gest = self.gesture_recognizer.gesture_state.get("active_gesture")
                # Avoid overwriting "Dragging" if drag is active but no specific event this frame
                if not (self.debug_values["gest"].startswith("Drag") and internal_gest is None):
                    self.debug_values["gest"] = str(internal_gest) if internal_gest else "Idle"

        if self.tutorial_manager.active:
            self.tutorial_manager.check_step_completion(mp_results)


    def draw_landmarks(self, frame, multi_hand_lm_mp): # Draws on client's frame
        if not multi_hand_lm_mp: return frame
        for hand_lm_data in multi_hand_lm_mp:
            self.mp_drawing.draw_landmarks(frame, hand_lm_data, self.mp_hands.HAND_CONNECTIONS,
                                           self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                           self.mp_drawing_styles.get_default_hand_connections_style())
        return frame

    def draw_menu_trigger_circle(self, image): # Draws on client's frame
        h_img, w_img = image.shape[:2]; radius_px = 30
        if w_img == 0 or h_img == 0:
            self.menu_trigger_zone_px["is_valid"] = False; return image

        center_px = (w_img - 50, h_img // 2) # Position on client's display
        intensity = 150 + int(50 * np.sin(time.time() * 5))
        base_color = (255, intensity, 0) # BGR
        draw_color = (0, 255, 0) if self.menu_trigger_active else base_color # Green if active

        cv.circle(image, center_px, radius_px, draw_color, -1)
        cv.circle(image, center_px, radius_px, (255,255,255), 1) # White border
        cv.putText(image, "Menu", (center_px[0]-20, center_px[1]+radius_px+15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)

        # Store pix_coords for hit-testing with hand position on display
        self.menu_trigger_zone_px["cx"] = center_px[0]
        self.menu_trigger_zone_px["cy"] = center_px[1]
        self.menu_trigger_zone_px["radius_sq"] = radius_px**2
        self.menu_trigger_zone_px["is_valid"] = True
        return image


    def draw_overlays(self, frame, mp_results): # Draws on client's frame
        self.current_display_dims = (frame.shape[1], frame.shape[0]) # Update client display dims
        h_disp, w_disp = self.current_display_dims[1], self.current_display_dims[0]
        if w_disp == 0 or h_disp == 0: return frame # Safety check

        overlay_c, bg_c, accent_c, error_c = (255,255,255), (0,0,0), (0,255,255), (0,0,255)
        font, fsc_sml, fsc_med = cv.FONT_HERSHEY_SIMPLEX, 0.4, 0.5
        l_type, thick_n, thick_b = cv.LINE_AA, 1, 2

        if mp_results and mp_results.multi_hand_landmarks:
            frame = self.draw_landmarks(frame, mp_results.multi_hand_landmarks)

        if not self.tutorial_manager.active : # Don't draw menu circle during tutorial
            frame = self.draw_menu_trigger_circle(frame)

        if self.gesture_recognizer.gesture_state["fist_drag_active"]:
            drag_indicator = "DRAG ACTIVE (CLIENT)" # Indicate it's client-side detection
            text_y_drag = 50 if not self.tutorial_manager.active else 30
            cv.putText(frame, drag_indicator, (w_disp//2 - 100, text_y_drag), font, 0.7, error_c, thick_b, l_type)
            cv.rectangle(frame, (5,5), (w_disp-5, h_disp-5), error_c, 3) # Border

        if self.calibration_active:
            top_bar_h = 100
            bg = frame.copy(); cv.rectangle(bg, (0,0), (w_disp, top_bar_h), bg_c, -1); alpha = 0.7
            frame = cv.addWeighted(bg, alpha, frame, 1 - alpha, 0)
            cv.putText(frame, f"CALIBRATION ({self.calibration_step+1}/4): RIGHT Hand", (10,25), font, fsc_med, overlay_c, thick_n, l_type)
            cv.putText(frame, f"Point index to {self.calib_corners[self.calibration_step]} corner", (10,50), font, fsc_med, overlay_c, thick_n, l_type)
            cv.putText(frame, "Press SPACE to confirm", (10,75), font, fsc_med, accent_c, thick_b, l_type)
            cv.putText(frame, "(ESC to cancel)", (w_disp-150 if w_disp > 150 else 10, 20), font, fsc_sml, overlay_c, thick_n, l_type)

            radius=15; inactive_color=(100,100,100); active_color=error_c; fill=-1
            corners_px = [(radius,radius), (w_disp-radius,radius), (w_disp-radius,h_disp-radius), (radius,h_disp-radius)]
            for i, p_corner in enumerate(corners_px):
                cv.circle(frame, p_corner, radius, active_color if self.calibration_step==i else inactive_color, fill)
                cv.circle(frame, p_corner, radius, overlay_c, 1)

            hand_pos_px_display = self.get_hand_pos_in_display_pixels() # For visual cue
            if hand_pos_px_display:
                cv.circle(frame, hand_pos_px_display, 10, (0,255,0), -1) # Green dot for hand

        elif not self.tutorial_manager.active : # Standard info text if not calibrating or tutorial
            ws_status_text = f"WS: {self.debug_values['ws_stat']}"
            info_text = f"C:Calib | D:Debug | Q:Exit | M:Menu | {ws_status_text}"
            if not self.debug_mode :
                 cv.putText(frame, info_text, (10, h_disp-10), font, fsc_sml, overlay_c, thick_n, l_type)

            fps_val = self.debug_values["fps"]
            cv.putText(frame, f"FPS: {fps_val:.1f}", (w_disp-150 if w_disp > 150 else 10, 20), font, fsc_med, overlay_c, thick_n, l_type)

            if self.debug_mode:
                panel_h = 110; panel_y_start = h_disp - panel_h
                bg_debug = frame.copy()
                cv.rectangle(bg_debug, (0, panel_y_start), (w_disp,h_disp), bg_c, -1); alpha_debug = 0.7
                frame = cv.addWeighted(bg_debug, alpha_debug, frame, 1-alpha_debug, 0)

                y_text = panel_y_start + 18; d_vals = self.debug_values; cal_cfg = self.config['calibration']
                cv.putText(frame, info_text, (10,y_text), font, fsc_sml, overlay_c,thick_n,l_type); y_text+=18
                t1=f"L:{d_vals['L']} R:{d_vals['R']} Gest:{d_vals['gest']} Menu:{d_vals['menu']}";
                cv.putText(frame,t1,(10,y_text),font,fsc_sml,overlay_c,thick_n,l_type); y_text+=18
                t2=f"SentNorm: {d_vals['map_norm_sent']}"
                cv.putText(frame,t2,(10,y_text),font,fsc_sml,overlay_c,thick_n,l_type); y_text+=18
                t3=f"Calib X[{cal_cfg['x_min']:.2f}-{cal_cfg['x_max']:.2f}] Y[{cal_cfg['y_min']:.2f}-{cal_cfg['y_max']:.2f}] En:{cal_cfg['enabled']}";
                cv.putText(frame,t3,(10,y_text),font,fsc_sml,overlay_c,thick_n,l_type); y_text+=18
                t4=f"QSize:{d_vals['q']}";
                cv.putText(frame,t4,(10,y_text),font,fsc_sml,overlay_c,thick_n,l_type); y_text+=18

        if self.tutorial_manager.active:
            frame = self.tutorial_manager.draw_tutorial_overlay(frame)

        return frame

    def process_server_messages(self):
        try:
            while not self.message_receive_queue.empty():
                msg_str = self.message_receive_queue.get_nowait()
                try:
                    data = json.loads(msg_str)
                    logger.debug(f"Client processing server message: {data}")
                    if data.get("type") == "command":
                        if data.get("command") == "start_tutorial":
                            logger.info("Received start_tutorial command from server.")
                            if not self.tutorial_manager.active: # Avoid re-starting if already active
                                if self.tk_root: # Ensure Tkinter calls are safe
                                    self.tk_root.after(0, self.tutorial_manager.start_tutorial)
                                else:
                                    self.tutorial_manager.start_tutorial()
                        # Add other server commands here
                except json.JSONDecodeError:
                    logger.error(f"Client received invalid JSON from server: {msg_str}")
                except Exception as e:
                    logger.error(f"Client error processing server message data: {e}")
                self.message_receive_queue.task_done()
        except queue.Empty:
            pass # No messages to process

    def main_loop(self): # This is a SYNC loop using OpenCV
        logger.info("Client main_loop starting."); win_name="HandPal Client"; cv.namedWindow(win_name,cv.WINDOW_NORMAL)
        dw_cfg, dh_cfg = self.config.get('display_width'), self.config.get('display_height')
        if dw_cfg and dh_cfg:
            try: cv.resizeWindow(win_name, dw_cfg, dh_cfg)
            except Exception as e: logger.warning(f"Failed to set client display size: {e}")

        last_valid_frame = np.zeros((self.config.get('height', 480), self.config.get('width', 640), 3), dtype=np.uint8)
        cv.putText(last_valid_frame, "Waiting for camera...", (50,50), cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)


        while self.running:
            loop_start_time = time.perf_counter()

            if self.tk_root: # Update Tkinter UI (menu)
                try: self.tk_root.update_idletasks(); self.tk_root.update()
                except tk.TclError as e:
                    if "application has been destroyed" in str(e).lower():
                        logger.info("Client Tkinter root destroyed. Stopping."); self.stop_client_services(); break
                    else: logger.error(f"Client Tkinter error: {e}")

            # Process messages received from server (non-blocking check)
            self.process_server_messages()


            current_frame_raw, mp_results, proc_dims_tuple = None, None, None
            try: # Get data from detection thread
                current_frame_raw, mp_results, proc_dims_tuple = self.data_queue.get(block=True, timeout=0.005) # Shorter timeout
                last_valid_frame = current_frame_raw.copy() # Keep last good frame
                if self.debug_mode: self.debug_values["q"] = self.data_queue.qsize()
            except queue.Empty:
                current_frame_raw = last_valid_frame # Use last good frame if queue is empty
                mp_results = None # No new results
                proc_dims_tuple = self._last_proc_dims # Use last known proc dims
            except Exception as e:
                logger.exception(f"Client data queue error: {e}"); time.sleep(0.01); continue

            if current_frame_raw is not None:
                 if mp_results is not None and proc_dims_tuple is not None:
                     try: self.process_results_and_send(mp_results, proc_dims_tuple)
                     except Exception as e: logger.exception("Client process_results_and_send error")

                 display_frame = current_frame_raw.copy()
                 try: display_frame_with_overlays = self.draw_overlays(display_frame, mp_results)
                 except Exception as e:
                     logger.exception("Client draw_overlays error")
                     cv.putText(display_frame,"DRAW ERR",(50,100),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                     display_frame_with_overlays = display_frame

                 final_display_frame = display_frame_with_overlays
                 if dw_cfg and dh_cfg: # Resize if specific display size set for OpenCV window
                     h_curr, w_curr = final_display_frame.shape[:2]
                     if w_curr!=dw_cfg or h_curr!=dh_cfg:
                         try: final_display_frame = cv.resize(final_display_frame,(dw_cfg,dh_cfg), interpolation=cv.INTER_LINEAR)
                         except Exception as e: logger.error(f"Client display resize error: {e}")

                 cv.imshow(win_name, final_display_frame)
            else: # Should not happen if last_valid_frame logic is working
                time.sleep(0.01)


            key = cv.waitKey(1) & 0xFF
            if key == ord('q'): logger.info("'q' pressed, client stopping."); self.stop_client_services(); break
            elif key == ord('c'):
                if not self.calibration_active and not self.tutorial_manager.active: self.start_calibration()
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                logger.info(f"Client debug mode toggled to: {self.debug_mode}")
            elif key == ord('m'):
                 if self.floating_menu and not self.tutorial_manager.active:
                     logger.debug("Client 'm' pressed, toggling menu."); self.floating_menu.toggle()
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

        logger.info("Client main_loop finished."); cv.destroyAllWindows()

    # --- Calibration Methods (Client-Side: determines mapping from camera to normalized interaction space) ---
    def start_calibration(self):
        if self.calibration_active: return
        logger.info("Client starting calibration..."); print("\n--- CLIENT CALIBRATION START ---")
        self.calibration_active = True; self.config.set("calibration.active", True)
        self.calibration_points = []; self.calibration_step = 0
        self.tracking_enabled = False # Disable sending move commands during calibration
        self.motion_smoother.reset()
        if self.floating_menu and self.floating_menu.visible: self.floating_menu.hide()
        print(f"Step {self.calibration_step+1}/4: Point RIGHT index to {self.calib_corners[0]} of preview window. Press SPACE.")

    def cancel_calibration(self):
        if not self.calibration_active: return
        logger.info("Client calibration cancelled."); print("\n--- CLIENT CALIBRATION CANCELLED ---")
        self.calibration_active = False; self.config.set("calibration.active", False)
        self.calibration_points = []; self.calibration_step = 0
        self.tracking_enabled = True

    def process_calibration_step(self):
        if not self.calibration_active: return

        if self.last_right_hand_lm_norm is None:
            print("(!) Right hand not detected or not providing landmarks for calibration!"); return

        norm_x_cam, norm_y_cam = self.last_right_hand_lm_norm # These are 0-1 relative to camera frame

        self.calibration_points.append((norm_x_cam, norm_y_cam))
        logger.info(f"Calib point {self.calibration_step+1} ({self.calib_corners[self.calibration_step]}): CamNorm=({norm_x_cam:.3f},{norm_y_cam:.3f})")
        print(f"-> Point {self.calibration_step+1}/4 ({self.calib_corners[self.calibration_step]}) captured (camera coords).")

        self.calibration_step += 1
        if self.calibration_step >= 4: self.complete_calibration()
        else: print(f"\nStep {self.calibration_step+1}/4: Point RIGHT index to {self.calib_corners[self.calibration_step]}. Press SPACE.")

    def complete_calibration(self):
        logger.info("Client completing calibration...")
        if len(self.calibration_points) != 4:
            logger.error(f"Incorrect point count for calibration: {len(self.calibration_points)}")
            print("(!) ERROR: Incorrect number of calibration points."); self.cancel_calibration(); return

        try:
            xs_cam_norm = [p[0] for p in self.calibration_points]
            ys_cam_norm = [p[1] for p in self.calibration_points]

            xmin_cam_norm, xmax_cam_norm = min(xs_cam_norm), max(xs_cam_norm)
            ymin_cam_norm, ymax_cam_norm = min(ys_cam_norm), max(ys_cam_norm)

            if (xmax_cam_norm-xmin_cam_norm < 0.05 or ymax_cam_norm-ymin_cam_norm < 0.05):
                print("(!) WARNING: Calibration area (in camera view) is very small. Results may be inaccurate.")

            self.config.set("calibration.x_min", xmin_cam_norm)
            self.config.set("calibration.x_max", xmax_cam_norm)
            self.config.set("calibration.y_min", ymin_cam_norm)
            self.config.set("calibration.y_max", ymax_cam_norm)
            self.config.set("calibration.enabled", True)
            self.config.save() # Save to shared config file
            logger.info(f"Client calibration saved: CamX_norm[{xmin_cam_norm:.3f}-{xmax_cam_norm:.3f}], CamY_norm[{ymin_cam_norm:.3f}-{ymax_cam_norm:.3f}]")
            print("\n--- CLIENT CALIBRATION SAVED ---")
        except Exception as e:
            logger.exception("Client calibration completion/save error."); print("(!) ERROR saving client calibration.")
            self.config.set("calibration.enabled", False) # Disable on error
        finally:
            self.calibration_active = False; self.config.set("calibration.active", False)
            self.calibration_points = []; self.calibration_step = 0
            self.tracking_enabled = True

# --- Argument Parsing (Client) ---
def parse_client_arguments():
    parser = argparse.ArgumentParser(description="HandPal Client", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cfg_defaults = Config.DEFAULT_CONFIG
    parser.add_argument('--device', type=int, default=None, help=f'Webcam ID (Default: {cfg_defaults["device"]})')
    parser.add_argument('--width', type=int, default=None, help=f'Webcam Width (Default: {cfg_defaults["width"]})')
    parser.add_argument('--height', type=int, default=None, help=f'Webcam Height (Default: {cfg_defaults["height"]})')
    parser.add_argument('--display-width', type=int, default=None, help='Client preview window width')
    parser.add_argument('--display-height', type=int, default=None, help='Client preview window height')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable client debug overlay & logging.')
    parser.add_argument('--flip-camera', action='store_true', default=None, help=f'Flip webcam (Def: {cfg_defaults["flip_camera"]})')
    parser.add_argument('--no-flip', dest='flip_camera', action='store_false', help='Disable horizontal flip.')
    parser.add_argument('--calibrate', action='store_true', default=False, help='Start client calibration on launch.')
    parser.add_argument('--reset-config', action='store_true', default=False, help='Reset client-relevant parts of config and exit.')
    parser.add_argument('--server-host', dest='server_host_cli', type=str, default=None, help=f'Server host (Default: {cfg_defaults["server_host"]})')
    parser.add_argument('--server-port', dest='server_port_cli', type=int, default=None, help=f'Server port (Default: {cfg_defaults["server_port"]})')
    parser.add_argument('--tutorial', action='store_true', default=False, help='Start with the tutorial (client-side).')
    return parser.parse_args()

# --- Main Function (Client) ---
def main_client():
    global handpal_client_instance # For tutorial and menu actions
    args = parse_client_arguments()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(log_level); [h.setLevel(log_level) for h in logger.handlers]
    logger.info(f"Client log level set to {logging.getLevelName(log_level)}")

    client_config = Config(args) # Client loads/manages its config

    if args.reset_config:
        logger.info("Resetting client-specific keys in config to default.")
        default_client_conf_keys = Config.DEFAULT_CONFIG.keys()
        existing_conf_path = Config.CONFIG_FILENAME

        current_full_config = {}
        if os.path.exists(existing_conf_path):
            try:
                with open(existing_conf_path, 'r') as f:
                    current_full_config = json.load(f)
            except Exception as e: logger.error(f"Could not load existing config for client reset: {e}")

        for key in default_client_conf_keys: # Iterate over client default keys
            current_full_config[key] = Config.DEFAULT_CONFIG[key] # Set to client default

        try:
            with open(existing_conf_path, 'w') as f:
                json.dump(current_full_config, f, indent=2)
            logger.info(f"Client-specific config keys reset/set to default in {existing_conf_path}")
        except Exception as e: logger.error(f"Failed to save reset client config: {e}")
        return 0


    create_default_apps_csv() # Client ensures its app CSV exists

    ws_uri = f"ws://{client_config.get('server_host')}:{client_config.get('server_port')}"
    app_instance = None

    try:
        app_instance = HandPalClient(client_config, ws_uri)
        handpal_client_instance = app_instance # Set global for access

        if args.debug: app_instance.debug_mode = True

        if app_instance.start_client_services(): # Starts camera, detection, and WS connection attempt
            if args.calibrate:
                logger.info("Client starting calibration due to --calibrate flag..."); time.sleep(0.75) # Ensure window ready
                app_instance.start_calibration()

            if args.tutorial and not args.calibrate: 
                logger.info("Client starting tutorial due to --tutorial flag..."); time.sleep(0.75)
                if app_instance.tk_root: # Ensure Tkinter calls are safe
                    app_instance.tk_root.after(100, app_instance.tutorial_manager.start_tutorial) # Delay slightly
                else:
                    app_instance.tutorial_manager.start_tutorial()


            app_instance.main_loop() # OpenCV main loop (sync)
        else:
            logger.error("HandPalClient failed to start."); print("ERROR: HandPalClient failed to start. Check client log."); return 1

    except KeyboardInterrupt: logger.info("Ctrl+C received. Stopping HandPalClient...")
    except Exception as e:
        logger.exception("Unhandled error in client main execution block.")
        print(f"CLIENT UNHANDLED ERROR: {e}. Check client log."); return 1
    finally:
        logger.info("Client main 'finally' block: Executing cleanup...")
        if app_instance: app_instance.stop_client_services()
        logger.info("HandPalClient terminated."); print("\nHandPalClient terminated.")
        time.sleep(0.1) # Allow logs to flush
    return 0

if __name__ == "__main__":
    sys.exit(main_client())