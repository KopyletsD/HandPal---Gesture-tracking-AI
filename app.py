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
from pynput.mouse import Controller, Button, Listener
from pynput.keyboard import Key, Controller as KeyboardController
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
# Debug Window Class
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
        
        # Aggiungi un widget di testo scrollabile
        frame = Frame(self.window, bg='#1E1E1E')
        frame.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = Scrollbar(frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        self.text_widget = Text(frame, wrap=tk.WORD, bg='#1E1E1E', fg='#CCCCCC', 
                            font=('Consolas', 9), padx=5, pady=5)
        self.text_widget.pack(side=LEFT, fill=BOTH, expand=True)
        
        self.text_widget.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.text_widget.yview)
        
        # Bottone per chiudere la finestra
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
        
        # Pulisci il testo
        self.text_widget.delete(1.0, tk.END)
        
        # Aggiungi i valori di debug
        self.text_widget.insert(tk.END, "=== HANDPAL DEBUG INFO ===\n\n")
        
        # FPS
        self.text_widget.insert(tk.END, f"FPS: {debug_values['fps']:.1f}\n")
        
        # Pose della mano
        self.text_widget.insert(tk.END, f"\nHAND POSE:\n")
        self.text_widget.insert(tk.END, f"Left: {debug_values['L']}\n")
        self.text_widget.insert(tk.END, f"Right: {debug_values['R']}\n")
        self.text_widget.insert(tk.END, f"Active Gesture: {debug_values['gest']}\n")
        
        # Zoom info se disponibile
        if 'zoom' in debug_values:
            self.text_widget.insert(tk.END, f"\nZOOM INFO:\n")
            self.text_widget.insert(tk.END, f"Active: {debug_values['zoom'].get('active', False)}\n")
            self.text_widget.insert(tk.END, f"Distance: {debug_values['zoom'].get('distance', 0):.3f}\n")
            self.text_widget.insert(tk.END, f"Delta: {debug_values['zoom'].get('delta', 0):.3f}\n")
        
        # Menu
        self.text_widget.insert(tk.END, f"\nMENU STATUS: {debug_values['menu']}\n")
        
        # Mapping coordinates
        self.text_widget.insert(tk.END, f"\nMAPPING:\n")
        m = debug_values['map']
        self.text_widget.insert(tk.END, f"Raw: {m['raw']}\n")
        self.text_widget.insert(tk.END, f"Calibrated: {m['cal']}\n")
        self.text_widget.insert(tk.END, f"Mapped: {m['map']}\n")
        self.text_widget.insert(tk.END, f"Smoothed: {m['smooth']}\n")
        
        # Tempi
        t_act = time.time() - debug_values["last_act"]
        self.text_widget.insert(tk.END, f"\nLast Action: {t_act:.1f}s ago\n")
        self.text_widget.insert(tk.END, f"Queue Size: {debug_values['q']}\n")
        
        # Calibrazione
        c = debug_values.get('calib', {})
        if c:
            self.text_widget.insert(tk.END, f"\nCALIBRATION:\n")
            self.text_widget.insert(tk.END, f"X Range: [{c.get('x_min', '?'):.2f} - {c.get('x_max', '?'):.2f}]\n")
            self.text_widget.insert(tk.END, f"Y Range: [{c.get('y_min', '?'):.2f} - {c.get('y_max', '?'):.2f}]\n")
            self.text_widget.insert(tk.END, f"Enabled: {c.get('enabled', False)}\n")

# -----------------------------------------------------------------------------
# Config Class (MODIFICATA: flip_camera = False di default)
# -----------------------------------------------------------------------------
class Config:
    DEFAULT_CONFIG = {
    "device": 0,
    "width": 1920,
    "height": 1440,
    "process_width": 1920,
    "process_height": 1440,
    "flip_camera": False,  # MODIFICATO: default non flippato
    "display_width": None,
    "display_height": None,
    "min_detection_confidence": 0.6,
    "min_tracking_confidence": 0.5,
    "use_static_image_mode": False,
    "smoothing_factor": 0.7,
    "inactivity_zone": 0.015,
    "click_cooldown": 0.4,
    "gesture_sensitivity": 0.02,  # Valore ridotto per richiedere un contatto quasi totale
    "gesture_settings": {"scroll_sensitivity": 4, "double_click_time": 0.35, "zoom_sensitivity": 8.0},
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
                if key == 'flip_camera' and value is True: self.config['flip_camera'] = True  # MODIFICATO per impostare flip_camera solo se True
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
# GestureRecognizer Class
# -----------------------------------------------------------------------------
class GestureRecognizer:
    def __init__(self, config):
        self.config = config; self.last_positions = {}
        self.gesture_state = {"scroll_active": False, "last_click_time": 0, "last_click_button": None,
                              "scroll_history": deque(maxlen=5), "active_gesture": None,
                              "last_pose": {"Left": "U", "Right": "U"}, "pose_stable_count": {"Left": 0, "Right": 0},
                              "zoom_active": False, "last_pinch_distance": None, "zoom_history": deque(maxlen=5)}
        self.POSE_STABILITY_THRESHOLD = 3

    def _dist(self, p1, p2):
        if None in [p1, p2] or not all(hasattr(p, 'x') for p in [p1,p2]): return float('inf')
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def check_thumb_index_click(self, hand_lm, handedness):
        if handedness != "Left" or hand_lm is None or self.gesture_state["scroll_active"] or self.gesture_state["zoom_active"]:
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
            # gestisci il cooldown e il double click
            cooldown = self.config["click_cooldown"]
            if (now - self.gesture_state["last_click_time"]) > cooldown:
                double_click_t = self.config["gesture_settings"]["double_click_time"]
                if (now - self.gesture_state["last_click_time"]) < double_click_t and self.gesture_state["last_click_button"] == Button.left:
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
         """Rileva il gesto di scorrimento (indice e medio estesi verticalmente), solo per la mano sinistra."""
         # Esegui il riconoscimento solo se è la mano sinistra
         if handedness != "Left":
             return None
         
         # Verifica che non ci siano altri gesti attivi
         if self.gesture_state["active_gesture"] == "click" or self.gesture_state["zoom_active"]:
             return None
             
         index_tip = hand_landmarks.landmark[8]
         middle_tip = hand_landmarks.landmark[12]
         index_mcp = hand_landmarks.landmark[5]  # Base dell'indice
         middle_mcp = hand_landmarks.landmark[9]  # Base del medio
         ring_tip = hand_landmarks.landmark[16]  # Punta dell'anulare
         pinky_tip = hand_landmarks.landmark[20]  # Punta del mignolo
         
         # Verifica che entrambe le dita siano estese e vicine, e che anulare e mignolo siano piegati
         index_extended = index_tip.y < index_mcp.y
         middle_extended = middle_tip.y < middle_mcp.y
         fingers_close = abs(index_tip.x - middle_tip.x) < 0.08
         ring_pinky_folded = (ring_tip.y > index_mcp.y) and (pinky_tip.y > index_mcp.y)
         
         if index_extended and middle_extended and fingers_close and ring_pinky_folded:
             # Attiva modalità scroll
             if not self.gesture_state["scroll_active"]:
                 # Inizia nuovo scorrimento, resetta storia
                 self.gesture_state["scroll_active"] = True
                 self.gesture_state["scroll_history"].clear()
                 self.gesture_state["active_gesture"] = "scroll"
             
             # Calcola movimento verticale
             if 8 in self.last_positions:
                 prev_y = self.last_positions[8][1]
                 curr_y = index_tip.y
                 delta_y = (curr_y - prev_y) * self.config["gesture_settings"]["scroll_sensitivity"]
                 
                 # Aggiungi alla storia solo se movimento significativo
                 if abs(delta_y) > 0.0005:
                     self.gesture_state["scroll_history"].append(delta_y)
                 
                 # Calcola la media per smoothing
                 if len(self.gesture_state["scroll_history"]) > 0:
                     smooth_delta = sum(self.gesture_state["scroll_history"]) / len(self.gesture_state["scroll_history"])
                     
                     # Aggiorna posizione e restituisci solo se delta significativo
                     if abs(smooth_delta) > 0.001:
                         self.last_positions[8] = (index_tip.x, index_tip.y)
                         return smooth_delta * 100
                 
             # Sempre aggiorna l'ultima posizione per il riferimento
             self.last_positions[8] = (index_tip.x, index_tip.y)
         else:
             # Disattiva scroll quando le dita non sono più in posizione
             if self.gesture_state["scroll_active"]:
                 self.gesture_state["scroll_active"] = False
                 self.gesture_state["scroll_history"].clear()
                 
                 # Resetta il gesto attivo se necessario
                 if self.gesture_state["active_gesture"] == "scroll":
                     self.gesture_state["active_gesture"] = None
             
         return None

    def detect_hand_pose(self, hand_lm, handedness):
        if hand_lm is None or handedness not in ["Left", "Right"]: return "U"
        try: lm=hand_lm.landmark; w,tt,it,mt,rt,pt=lm[0],lm[4],lm[8],lm[12],lm[16],lm[20]; im,mm,rm,pm=lm[5],lm[9],lm[13],lm[17]
        except (IndexError, TypeError): return "U"
        y_ext = 0.03; t_ext=tt.y<w.y-y_ext; i_ext=it.y<im.y-y_ext; m_ext=mt.y<mm.y-y_ext; r_ext=rt.y<rm.y-y_ext; p_ext=pt.y<pm.y-y_ext
        num_ext = sum([i_ext,m_ext,r_ext,p_ext]); pose = "U"
        
        # Pose speciali
        if handedness=="Left" and self.gesture_state["scroll_active"]: pose="Scroll"
        elif handedness=="Right" and self.gesture_state["zoom_active"]: pose="Zoom"
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
        if screen_w <= 0 or screen_h <= 0: return self.last_smooth_pos # Avoid div by zero
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
        self.flip=config["flip_camera"]
        logger.info(f"DetectionThread: ProcRes={self.proc_w}x{self.proc_h}, Flip={self.flip}")

    def run(self):
        logger.info("DetectionThread starting"); t_start=time.perf_counter(); frame_n=0
        while not self.stop_evt.is_set():
            if self.cap is None or not self.cap.isOpened(): logger.error("Webcam N/A"); time.sleep(0.5); continue
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None: time.sleep(0.05); continue
                proc_frame = cv.resize(frame, (self.proc_w, self.proc_h), interpolation=cv.INTER_LINEAR)
                rgb_frame = cv.cvtColor(proc_frame, cv.COLOR_BGR2RGB); rgb_frame.flags.writeable=False
                results = self.hands.process(rgb_frame); rgb_frame.flags.writeable=True
                try: self.data_q.put((frame, results, (self.proc_w, self.proc_h)), block=False) # Pass process dims
                except queue.Full:
                    try: self.data_q.get_nowait(); self.data_q.put_nowait((frame, results, (self.proc_w, self.proc_h)))
                    except queue.Empty:
                        pass
                    except queue.Full:
                        pass
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
        
        # Gesture tracking
        self.last_gesture_time = 0
        self.gesture_clicks_detected = 0
        self.last_gesture_click_time = 0
        
        # Mouse click tracking
        self.mouse_clicks_detected = 0
        self.last_mouse_click_time = 0
        self.mouse_double_click_detected = False
        
        # Zoom and scroll tracking
        self.zoom_activity_detected = False
        self.last_zoom_change = 0
        self.scroll_detected = False
        
        # Corner tracking for cursor movement exercise
        self.corners_visited = [False, False, False, False]
        
        # Define all tutorial steps
        self.steps = [
            {
                "title": "Benvenuto in HandPal",
                "instruction": "Posiziona la mano DESTRA nello schermo per muovere il cursore",
                "completion_type": "position",
                "icon": "👋",
                "allows_mouse": False
            },
            {
                "title": "Movimento del Cursore",
                "instruction": "Muovi il cursore verso i 4 angoli dello schermo",
                "completion_type": "corners",
                "icon": "👉",
                "allows_mouse": True
            },
            {
                "title": "Click Sinistro",
                "instruction": "Avvicina pollice e indice della mano SINISTRA oppure fai click col mouse",
                "completion_type": "click",
                "icon": "👆",
                "allows_mouse": True
            },
            {
                "title": "Doppio Click",
                "instruction": "Fai due click rapidi con la mano SINISTRA oppure doppio click col mouse",
                "completion_type": "double_click",
                "icon": "✌",
                "allows_mouse": True
            },
            {
                "title": "Zoom In/Out",
                "instruction": "Con la mano DESTRA, avvicina e allontana pollice e indice per zoom",
                "completion_type": "zoom",
                "icon": "🔍",
                "allows_mouse": False
            },
            {
                "title": "Scorrimento",
                "instruction": "Con la mano SINISTRA, estendi indice e medio per scorrere",
                "completion_type": "scroll",
                "icon": "👆👆",
                "allows_mouse": False
            },
            {
                "title": "Menu",
                "instruction": "Posiziona l'indice destro sul cerchio del menu e mantieni",
                "completion_type": "menu_hover",
                "icon": "📋",
                "allows_mouse": False
            },
            {
                "title": "Complimenti!",
                "instruction": "Hai completato il tutorial! Fai click per terminare",
                "completion_type": "click",
                "icon": "🎉",
                "allows_mouse": True
            }
        ]
    
    def start_tutorial(self):
        """Avvia il tutorial resettando tutti i contatori."""
        self.active = True
        self.current_step = 0
        self.step_completed = False
        self.completion_timer = 0
        self.handpal.tracking_enabled = True
        self.reset_step_counters()
        logger.info("Tutorial started")
        
    def stop_tutorial(self):
        """Ferma il tutorial."""
        self.active = False
        logger.info("Tutorial stopped")
        
    def reset_step_counters(self):
        """Resetta tutti i contatori per il passo corrente."""
        self.last_gesture_time = time.time()
        self.gesture_clicks_detected = 0
        self.last_gesture_click_time = 0
        self.mouse_clicks_detected = 0
        self.last_mouse_click_time = 0
        self.mouse_double_click_detected = False
        self.zoom_activity_detected = False
        self.last_zoom_change = 0
        self.scroll_detected = False
        self.corners_visited = [False, False, False, False]
        
    def next_step(self):
        """Avanza al prossimo passo del tutorial."""
        self.current_step += 1
        self.step_completed = False
        self.completion_timer = 0
        self.reset_step_counters()
        
        if self.current_step >= len(self.steps):
            self.stop_tutorial()
            return True
        return False
    
    def register_mouse_click(self, is_double=False):
        """Registra un click del mouse per il tutorial."""
        if not self.active or self.step_completed:
            return
            
        current_step = self.steps[self.current_step]
        completion_type = current_step.get("completion_type", "")
        allows_mouse = current_step.get("allows_mouse", False)
        
        if not allows_mouse:
            return
            
        now = time.time()
        
        # Gestisci doppio click diretto da sistema
        if is_double and completion_type == "double_click":
            logger.info("Tutorial: Mouse double-click detected directly")
            self.mouse_double_click_detected = True
            self.step_completed = True
            return
            
        # Gestisci click singolo
        if completion_type == "click":
            logger.info("Tutorial: Mouse click detected for click step")
            self.step_completed = True
            return
            
        # Gestisci doppio click tramite click singoli ravvicinati
        if completion_type == "double_click":
            if now - self.last_mouse_click_time < 0.5:
                self.mouse_clicks_detected += 1
                logger.info(f"Tutorial: Mouse click #{self.mouse_clicks_detected} for double click")
                
                if self.mouse_clicks_detected >= 2:
                    logger.info("Tutorial: Mouse double-click completed via two quick clicks")
                    self.step_completed = True
            else:
                # Primo click o click dopo timeout
                self.mouse_clicks_detected = 1
                logger.info("Tutorial: First mouse click for potential double click")
                
            self.last_mouse_click_time = now
    
    def check_step_completion(self, results):
        """Controlla se il passo corrente è stato completato."""
        if not self.active or self.step_completed:
            return
            
        current = self.steps[self.current_step]
        completion_type = current.get("completion_type", "")
        now = time.time()
        
        # Step 1: Detect hand position
        if completion_type == "position" and self.handpal.last_right_hand_lm_norm is not None:
            if now - self.last_gesture_time > 1.5:
                logger.info("Tutorial: Hand position detected")
                self.step_completed = True
                
        # Step 2: Check cursor visited all corners
        elif completion_type == "corners":
            if self.handpal.last_cursor_pos is None:
                return
                
            # Check if cursor has visited all corners
            x, y = self.handpal.last_cursor_pos
            screen_w, screen_h = self.handpal.screen_size
            margin = 100  # pixels from edge
            
            # Update corners visited
            if x < margin and y < margin:
                self.corners_visited[0] = True  # Top-left
            if x > screen_w - margin and y < margin:
                self.corners_visited[1] = True  # Top-right
            if x > screen_w - margin and y > screen_h - margin:
                self.corners_visited[2] = True  # Bottom-right
            if x < margin and y > screen_h - margin:
                self.corners_visited[3] = True  # Bottom-left
                
            if all(self.corners_visited):
                logger.info("Tutorial: All corners visited")
                self.step_completed = True
                
        # Step 3 & 8: Click detection
        elif completion_type == "click":
            # Check for gesture click (if not already completed by mouse)
            last_click_time = self.handpal.gesture_recognizer.gesture_state.get("last_click_time", 0)
            if last_click_time > self.last_gesture_click_time:
                logger.info("Tutorial: Gesture click detected")
                self.last_gesture_click_time = last_click_time
                self.step_completed = True
                
        # Step 4: Double click detection
        elif completion_type == "double_click":
            # Check if already completed by mouse
            if self.mouse_double_click_detected:
                return
                
            # Check for gesture double click
            last_click_time = self.handpal.gesture_recognizer.gesture_state.get("last_click_time", 0)
            
            # If we detect a new click
            if last_click_time > self.last_gesture_click_time:
                logger.info(f"Tutorial: Gesture click #{self.gesture_clicks_detected+1} detected")
                
                # Check if this could be part of a double click
                if now - self.last_gesture_click_time < 0.5 and self.gesture_clicks_detected > 0:
                    self.gesture_clicks_detected += 1
                    logger.info(f"Tutorial: Potential double click (clicks={self.gesture_clicks_detected})")
                    
                    if self.gesture_clicks_detected >= 2:
                        logger.info("Tutorial: Gesture double click completed")
                        self.step_completed = True
                else:
                    # First click of a potential double
                    self.gesture_clicks_detected = 1
                    
                self.last_gesture_click_time = last_click_time
                
            # Reset after timeout
            elif now - self.last_gesture_click_time > 1.0 and self.gesture_clicks_detected > 0:
                logger.info("Tutorial: Reset click counter due to timeout")
                self.gesture_clicks_detected = 0
                
        # Step 5: Zoom detection
        elif completion_type == "zoom":
            # Check if zoom is active
            zoom_active = self.handpal.gesture_recognizer.gesture_state.get("zoom_active", False)
            
            if zoom_active:
                # Check for significant zoom movements
                if self.handpal.last_zoom_delta and abs(self.handpal.last_zoom_delta) > 0.1:
                    self.last_zoom_change = now
                    self.zoom_activity_detected = True
                    logger.info(f"Tutorial: Zoom movement detected: {self.handpal.last_zoom_delta:.3f}")
            
            # Complete if we've detected zoom activity and it's been stable for a bit
            if self.zoom_activity_detected and now - self.last_zoom_change > 1.0:
                logger.info("Tutorial: Zoom exercise completed")
                self.step_completed = True
                
        # Step 6: Scroll detection
        elif completion_type == "scroll":
            # Check if scroll is active
            if self.handpal.gesture_recognizer.gesture_state.get("active_gesture") == "scroll":
                if self.handpal.gesture_recognizer.gesture_state.get("scroll_active", False):
                    if len(self.handpal.gesture_recognizer.gesture_state.get("scroll_history", [])) > 2:
                        logger.info("Tutorial: Scroll gesture completed")
                        self.step_completed = True
                
        # Step 7: Menu hover detection
        elif completion_type == "menu_hover":
            # Check if menu trigger is active
            if self.handpal.menu_trigger_active:
                if self.completion_timer == 0:
                    self.completion_timer = now
                elif now - self.completion_timer >= self.handpal.MENU_HOVER_DELAY:
                    logger.info("Tutorial: Menu hover completed")
                    self.step_completed = True
            else:
                self.completion_timer = 0
                
        # Auto advance after delay when step is completed
        if self.step_completed and now - self.last_gesture_time > 1.5:
            self.next_step()
    
    def handle_skip_click(self, x, y, w, h):
        """Gestisce i click sul pulsante Skip."""
        if not self.active:
            return False
            
        # Verifica se il click è sul pulsante Skip
        skip_btn_x, skip_btn_y = w - 100, h - 35
        skip_btn_width, skip_btn_height = 80, 25
        
        if (skip_btn_x <= x <= skip_btn_x + skip_btn_width and 
            skip_btn_y <= y <= skip_btn_y + skip_btn_height):
            logger.info(f"Skip button clicked in tutorial step {self.current_step+1}")
            return self.next_step()
            
        return False
    
    def draw_tutorial_overlay(self, frame):
        """Disegna l'overlay del tutorial sul frame."""
        if not self.active:
            return frame, (0, 0, 0, 0)
            
        h, w = frame.shape[:2]
        if w == 0 or h == 0:
            return frame, (0, 0, 0, 0)
            
        # Crea un pannello semi-trasparente nella parte inferiore dello schermo
        overlay_height = 120
        tutorial_panel = frame.copy()
        cv.rectangle(tutorial_panel, (0, h-overlay_height), (w, h), (0, 0, 0), -1)
        frame = cv.addWeighted(tutorial_panel, 0.8, frame, 0.2, 0)
        
        current = self.steps[self.current_step]
        title = current.get("title", "Tutorial")
        instruction = current.get("instruction", "")
        icon = current.get("icon", "📋")
        
        # Indicatori di progresso
        for i in range(len(self.steps)):
            color = (0, 255, 255) if i == self.current_step else (100, 100, 100)
            cv.circle(frame, (20 + i*30, h-20), 8, color, -1 if i <= self.current_step else 2)
        
        # Contenuto principale
        cv.putText(frame, f"{icon} {title}", (20, h-95), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv.putText(frame, instruction, (20, h-60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Stato di completamento
        if self.step_completed:
            cv.putText(frame, "✓ Ottimo! Passaggio completato", (w//2-150, h-25), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Pulsante Skip
        skip_btn_x, skip_btn_y = w - 100, h - 35
        skip_btn_width, skip_btn_height = 80, 25
        cv.rectangle(frame, (skip_btn_x, skip_btn_y), 
                    (skip_btn_x + skip_btn_width, skip_btn_y + skip_btn_height), 
                    (0, 120, 255), -1)
        cv.putText(frame, "Skip >>", (skip_btn_x + 15, skip_btn_y + 18), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Feedback specifico per tipo di esercizio
        completion_type = current.get("completion_type", "")
        
        if completion_type == "corners" and self.handpal.debug_mode:
            status = "".join(["✓" if v else "○" for v in self.corners_visited])
            cv.putText(frame, f"Angoli: {status}", (20, h-35), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)
                       
        elif completion_type == "double_click" and self.handpal.debug_mode:
            click_info = f"Gesture: {self.gesture_clicks_detected}, Mouse: {self.mouse_clicks_detected}"
            cv.putText(frame, click_info, (20, h-35), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)
                       
        elif completion_type == "zoom" and self.handpal.debug_mode:
            zoom_info = f"Zoom active: {self.handpal.gesture_recognizer.gesture_state.get('zoom_active', False)}"
            cv.putText(frame, zoom_info, (20, h-35), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)
        
        return frame, (skip_btn_x, skip_btn_y, skip_btn_width, skip_btn_height)

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
                # Aggiungiamo l'opzione Tutorial
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
    if not path: logger.warning("Launch attempt with empty path."); return
    logger.info(f"Launching: {path}")
    try:
        # Controlla se è il comando speciale del tutorial
        if path == '@tutorial':
            # Accedi all'istanza globale di HandPal
            global handpal_instance
            if handpal_instance:
                handpal_instance.tutorial_manager.start_tutorial()
                # Nascondi automaticamente il menu quando inizia il tutorial
                if handpal_instance.floating_menu and handpal_instance.floating_menu.visible:
                    handpal_instance.floating_menu.hide()
            return
            
        # Implementazione originale
        if path.startswith(('http://', 'https://')): webbrowser.open(path)
        elif os.name == 'nt': os.startfile(path)
        else: subprocess.Popen(path)
    except FileNotFoundError: logger.error(f"Launch failed: File not found '{path}'")
    except Exception as e: logger.error(f"Launch failed '{path}': {e}")

class FloatingMenu:
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
        self.keyboard = KeyboardController()
        self.gesture_recognizer = GestureRecognizer(config)
        try: # Init Tkinter
            self.tk_root = tk.Tk(); self.tk_root.withdraw()
            self.screen_size = (self.tk_root.winfo_screenwidth(), self.tk_root.winfo_screenheight())
            self.floating_menu = FloatingMenu(self.tk_root)
            self.debug_window = DebugWindow(self.tk_root)
            logger.info(f"Screen: {self.screen_size}. Menu and Debug Window OK.")
        except tk.TclError as e: 
            logger.error(f"Tkinter init failed: {e}. Menu N/A.")
            self.tk_root=None
            self.floating_menu=None
            self.debug_window=None
            self.screen_size=(1920,1080)
        
        self.motion_smoother = MotionSmoother(config)
        self.running = False; self.stop_event = threading.Event(); self.detection_thread = None
        self.data_queue = queue.Queue(maxsize=2)
        self.cap = None; self.mp_hands = mp.solutions.hands; self.hands_instance = None
        self.mp_drawing = mp.solutions.drawing_utils; self.mp_drawing_styles = mp.solutions.drawing_styles
        self.last_cursor_pos = None; self.last_right_hand_lm_norm = None # Store raw normalized landmark
        self.tracking_enabled = True; self.debug_mode = False
        self.calibration_active = False; self.calibration_points = []; self.calibration_step = 0
        self.calib_corners = ["top-left", "top-right", "bottom-right", "bottom-left"]
        
        # Zoom tracking
        self.last_zoom_delta = 0.0
        
        # Inizializza il tutorial manager
        self.tutorial_manager = TutorialManager(self)
        
        # Menu trigger variables
        self.menu_trigger_zone_px = {"cx": 0, "cy": 0, "radius_sq": 0, "is_valid": False}
        self.menu_trigger_active = False
        self._menu_activate_time = 0
        self.MENU_HOVER_DELAY = 1.0  # seconds to activate menu
        
        # Skip button di tutorial
        self.tutorial_skip_btn = {"x": 0, "y": 0, "width": 0, "height": 0, "visible": False}
        
        # FPS and debug values initialization
        self.fps_stats = deque(maxlen=30)
        self.debug_values = {
            "fps": 0, "L": "U", "R": "U", "gest": "Idle", "menu": "Off",
            "map": {"raw": "", "cal": "", "map": "", "smooth": ""},
            "q": 0, "last_act": time.time(), "cur_hist": deque(maxlen=50),
            "calib": {}, "zoom": {"active": False, "distance": 0, "delta": 0}
        }
        self._last_proc_dims = (640, 360)
        self.current_display_dims = (0, 0)
        
        logger.info("HandPal instance initialized.")

    # Funzione per gestire il click del mouse nell'OpenCV window
    def handle_cv_mouse_event(self, event, x, y, flags, param):
        if event not in [cv.EVENT_LBUTTONDOWN, cv.EVENT_LBUTTONDBLCLK]:
            return
            
        logger.debug(f"Mouse event detected: {event} at ({x}, {y})")
        
        # Se il tutorial è attivo
        if self.tutorial_manager.active and self.tutorial_skip_btn["visible"]:
            btn = self.tutorial_skip_btn
            
            # Controlla click sul pulsante Skip
            if (btn["x"] <= x <= btn["x"] + btn["width"] and 
                btn["y"] <= y <= btn["y"] + btn["height"]):
                logger.info("Skip button clicked via mouse")
                self.tutorial_manager.next_step()
                return
            
            # Altrimenti, registra il click per il completamento dell'esercizio
            is_double = (event == cv.EVENT_LBUTTONDBLCLK)
            self.tutorial_manager.register_mouse_click(is_double)

    def map_to_screen(self, norm_x_proc, norm_y_proc):
        """Maps normalized coordinates to screen coordinates with optional calibration."""
        if self.debug_mode: self.debug_values["map"]["raw"] = f"{norm_x_proc:.2f},{norm_y_proc:.2f}"
        
        # Apply calibration if enabled
        if self.config["calibration.enabled"] and not self.calibration_active:
            x_min = self.config["calibration.x_min"]
            x_max = self.config["calibration.x_max"]
            y_min = self.config["calibration.y_min"]
            y_max = self.config["calibration.y_max"]
            
            # Aggiorna info calibrazione per debug window
            self.debug_values["calib"] = {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "enabled": True
            }
            
            # Normalize to calibration range
            norm_x_cal = (norm_x_proc - x_min) / (x_max - x_min)
            norm_y_cal = (norm_y_proc - y_min) / (y_max - y_min)
            
            # Clamp to [0,1]
            norm_x_cal = max(0.0, min(1.0, norm_x_cal))
            norm_y_cal = max(0.0, min(1.0, norm_y_cal))
            
            if self.debug_mode: self.debug_values["map"]["cal"] = f"{norm_x_cal:.2f},{norm_y_cal:.2f}"
        else:
            # No calibration, just clamp
            norm_x_cal = max(0.0, min(1.0, norm_x_proc))
            norm_y_cal = max(0.0, min(1.0, norm_y_proc))
            if self.debug_mode: self.debug_values["map"]["cal"] = f"N/A"
            
        # Method 2: Convert to pixels on process frame, then scale pixels (more robust to aspect ratio changes)
        proc_w, proc_h = self._last_proc_dims
        display_w, display_h = self.current_display_dims

        pixel_x_proc = norm_x_proc * proc_w
        pixel_y_proc = norm_y_proc * proc_h

        scale_x = display_w / proc_w
        scale_y = display_h / proc_h

        display_x = int(pixel_x_proc * scale_x)
        display_y = int(pixel_y_proc * scale_y)

        # Clamp to display bounds
        display_x = max(0, min(display_x, display_w - 1))
        display_y = max(0, min(display_y, display_h - 1))

        return (display_x, display_y)

    def get_hand_pos_in_display_pixels(self):
        """Helper to get the right hand position in display pixel coordinates."""
        if self.last_right_hand_lm_norm is None:
            return None
        
        norm_x, norm_y = self.last_right_hand_lm_norm
        display_w, display_h = self.current_display_dims
        
        # If no valid display dims, can't convert
        if display_w <= 0 or display_h <= 0:
            return None
            
        # Convert using map_to_screen (this properly handles calibration if enabled)
        target_x, target_y = self.map_to_screen(norm_x, norm_y)
        
        return (target_x, target_y)

    def check_menu_trigger(self):
        """Checks if right hand index (in display pixels) is inside the trigger zone (also in display pixels)."""
        if not self.menu_trigger_zone_px["is_valid"]:
            if self.menu_trigger_active: logger.debug("Menu trigger deactivated (Zone Invalid).")
            self.menu_trigger_active = False; self._menu_activate_time = 0
            if self.debug_mode: self.debug_values["menu"] = "Off (Zone Invalid)"
            return False

        hand_pos_px = self.get_hand_pos_in_display_pixels()

        if hand_pos_px is None:
            if self.menu_trigger_active: logger.debug("Menu trigger deactivated (Hand Lost).")
            self.menu_trigger_active = False; self._menu_activate_time = 0
            if self.debug_mode: self.debug_values["menu"] = "Off (No Hand)"
            return False

        # Compare in pixel space
        hx_px, hy_px = hand_pos_px
        cx_px, cy_px = self.menu_trigger_zone_px["cx"], self.menu_trigger_zone_px["cy"]
        radius_sq_px = self.menu_trigger_zone_px["radius_sq"]

        dist_sq_px = (hx_px - cx_px)**2 + (hy_px - cy_px)**2
        is_inside = dist_sq_px < radius_sq_px

        now = time.time(); activated = False
        if is_inside:
            if not self.menu_trigger_active: # Just entered
                self.menu_trigger_active = True; self._menu_activate_time = now
                logger.debug(f"Menu Trigger Zone Entered (HandPx: {hand_pos_px} ZonePx Center: ({cx_px},{cy_px}))")
            hover_time = now - self._menu_activate_time
            if hover_time >= self.MENU_HOVER_DELAY:
                activated = True # Signal show
                if self.debug_mode: self.debug_values["menu"] = "ACTIVATE!"
                logger.debug(f"Menu Trigger HOVER MET ({hover_time:.2f}s)")
            else:
                 if self.debug_mode: self.debug_values["menu"] = f"Hover {hover_time:.1f}s"
        else: # Not inside
            if self.menu_trigger_active: logger.debug("Menu Trigger Zone Exited.")
            self.menu_trigger_active = False; self._menu_activate_time = 0
            if self.debug_mode: self.debug_values["menu"] = "Off"

        # Log pixel coords for debugging
        if self.debug_mode:
           logger.debug(f"MenuCheckPx: Hand({hx_px},{hy_px}) DistSq={dist_sq_px:.0f} ZoneRsq={radius_sq_px:.0f} Inside={is_inside} Active={self.menu_trigger_active} Hover={hover_time if is_inside else 0:.2f}")

        return activated

    def process_results(self, results, proc_dims):
        """Processes landmarks for cursor, gestures, and menu trigger."""
        self._last_proc_dims = proc_dims # Store process dims used for these results
        lm_l, lm_r = None, None; self.last_right_hand_lm_norm = None # Reset raw norm storage

        if results and results.multi_hand_landmarks and results.multi_handedness:
            for i, hand_lm in enumerate(results.multi_hand_landmarks):
                try: label = results.multi_handedness[i].classification[0].label
                except (IndexError, AttributeError): continue
                if label == "Right":
                    lm_r = hand_lm
                    # Store raw normalized landmark for right hand if found
                    try: self.last_right_hand_lm_norm = (hand_lm.landmark[8].x, hand_lm.landmark[8].y)
                    except IndexError: self.last_right_hand_lm_norm = None
                elif label == "Left": lm_l = hand_lm

        # --- Right Hand: Cursor, Menu & Zoom ---
        if lm_r and self.last_right_hand_lm_norm is not None:
             # Check menu trigger (uses self.last_right_hand_lm_norm and display dims)
             if self.check_menu_trigger():
                 if self.floating_menu and not self.floating_menu.visible:
                     logger.info("Menu trigger activated by hand hover."); self.floating_menu.show()

             # Check zoom gesture
             zoom_result = False
             if zoom_result:
                 # Aggiorna info debug
                 self.debug_values["zoom"]["active"] = True
                 self.debug_values["zoom"]["distance"] = zoom_result["distance"]
                 self.debug_values["zoom"]["delta"] = zoom_result["delta"]
                 self.last_zoom_delta = zoom_result["delta"]
                 
                 # Esegui lo zoom simulando Ctrl+Rotellina
                 if abs(zoom_result["delta"]) > 0.05:
                     try:
                         # Prima opzione: simulare combinazioni di tasti per zoom (Ctrl+[+/-])
                         delta = zoom_result["delta"]
                         if delta > 0:  # Zoom in (ingrandisci)
                             with self.keyboard.pressed(Key.ctrl):
                                 self.keyboard.press('+')
                                 self.keyboard.release('+')
                         else:  # Zoom out (rimpicciolisci)
                             with self.keyboard.pressed(Key.ctrl):
                                 self.keyboard.press('-')
                                 self.keyboard.release('-')
                                 
                         self.debug_values["last_act"] = time.time()
                         self.debug_values["gest"] = "Zoom"
                         logger.debug(f"Zoom: {zoom_result['delta']:.3f}")
                     except Exception as e:
                         logger.error(f"Zoom error: {e}")
             else:
                 # Reset dello stato di zoom
                 if self.debug_values["zoom"]["active"]:
                     self.debug_values["zoom"]["active"] = False
                     self.debug_values["zoom"]["delta"] = 0
                     self.last_zoom_delta = 0

             # Move cursor (uses self.last_right_hand_lm_norm for mapping)
             if self.tracking_enabled and not self.calibration_active:
                 try:
                     norm_x, norm_y = self.last_right_hand_lm_norm
                     target_x, target_y = self.map_to_screen(norm_x, norm_y)
                     smooth_x, smooth_y = self.motion_smoother.update(target_x, target_y, self.screen_size[0], self.screen_size[1])
                     if self.last_cursor_pos is None or smooth_x != self.last_cursor_pos[0] or smooth_y != self.last_cursor_pos[1]:
                         self.mouse.position = (smooth_x, smooth_y); self.last_cursor_pos = (smooth_x, smooth_y)
                         if self.debug_mode: self.debug_values["map"]["smooth"]=f"{smooth_x},{smooth_y}"; self.debug_values["cur_hist"].append(self.last_cursor_pos)
                 except Exception as e: logger.error(f"Cursor update error: {e}"); self.motion_smoother.reset()
        else: # No right hand or index tip issue
            self.motion_smoother.reset(); self.last_right_hand_lm_norm = None
            # Reset menu trigger state explicitly if hand lost
            if self.menu_trigger_active: logger.debug("Menu trigger deactivated (Hand Lost).")
            self.menu_trigger_active = False; self._menu_activate_time = 0
            # Reset zoom state
            self.debug_values["zoom"]["active"] = False
            self.debug_values["zoom"]["delta"] = 0
            self.last_zoom_delta = 0
            if self.debug_mode: self.debug_values["menu"] = "Off (No Hand)"


        # --- Left Hand: Gestures --- 
        action = False
        if lm_l and not self.calibration_active:
            scroll = self.gesture_recognizer.check_scroll_gesture(lm_l, "Left")
            if scroll is not None:
                clicks = int(scroll * -1)
                if clicks != 0:
                    try: self.mouse.scroll(0, clicks); action = True; logger.debug(f"Scroll: {clicks}"); self.debug_values["last_act"]=time.time(); self.debug_values["gest"]="Scroll"
                    except Exception as e: logger.error(f"Scroll error: {e}")
            if not action:
                click_type = self.gesture_recognizer.check_thumb_index_click(lm_l, "Left")
                if click_type:
                    btn = Button.left; count = 1 if click_type=="click" else 2; name=f"{click_type.capitalize()}"
                    try: self.mouse.click(btn, count); action=True; logger.info(f"{name} Left"); self.debug_values["last_act"]=time.time(); self.debug_values["gest"]=name
                    except Exception as e: logger.error(f"{name} error: {e}")

        # --- Update Debug Poses/Gestures ---
        if self.debug_mode:
            self.debug_values["L"] = self.gesture_recognizer.detect_hand_pose(lm_l, "Left")
            self.debug_values["R"] = self.gesture_recognizer.detect_hand_pose(lm_r, "Right")
            if not action: internal_gest = self.gesture_recognizer.gesture_state.get("active_gesture"); self.debug_values["gest"] = str(internal_gest) if internal_gest else "Idle"
            
        # --- Check Tutorial Step Completion ---
        if self.tutorial_manager.active:
            self.tutorial_manager.check_step_completion(results)
            
        # Aggiorna la finestra di debug se necessario
        if self.debug_mode and self.debug_window:
            self.debug_window.update(self.debug_values)

    def draw_landmarks(self, frame, multi_hand_lm):
        if not multi_hand_lm: return frame
        for hand_lm in multi_hand_lm: self.mp_drawing.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS, self.mp_drawing_styles.get_default_hand_landmarks_style(), self.mp_drawing_styles.get_default_hand_connections_style())
        return frame

    def draw_menu_trigger_circle(self, image):
        """Draws the menu trigger circle and calculates its zone in display pixels."""
        h, w = image.shape[:2]; radius_px = 30
        if w == 0 or h == 0: self.menu_trigger_zone_px["is_valid"] = False; return image
        center_px = (w - 50, h // 2); intensity = 150 + int(50 * np.sin(time.time() * 5))
        base_color = (255, intensity, 0); draw_color = (0, 255, 0) if self.menu_trigger_active else base_color
        cv.circle(image, center_px, radius_px, draw_color, -1)
        cv.circle(image, center_px, radius_px, (255, 255, 255), 1)
        cv.putText(image, "Menu", (center_px[0] - 20, center_px[1] + radius_px + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        # --- Update Pixel Zone ---
        self.menu_trigger_zone_px["cx"] = center_px[0]
        self.menu_trigger_zone_px["cy"] = center_px[1]
        self.menu_trigger_zone_px["radius_sq"] = radius_px**2
        self.menu_trigger_zone_px["is_valid"] = True
        return image

    def draw_zoom_indicator(self, image):
        """Disegna un indicatore di zoom se il gesto è attivo."""
        if not self.gesture_recognizer.gesture_state.get("zoom_active", False):
            return image
            
        h, w = image.shape[:2]
        if w == 0 or h == 0: 
            return image
            
        # Posizione e dimensioni
        width = 200
        height = 30
        x = w - width - 60
        y = 20
        
        # Sfondo
        cv.rectangle(image, (x-5, y-5), (x+width+5, y+height+5), (0,0,0), -1)
        cv.rectangle(image, (x, y), (x+width, y+height), (50,50,50), -1)
        
        # Testo
        cv.putText(image, "ZOOM", (x+5, y+20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        # Livello di zoom (solo per visivo)
        delta = self.last_zoom_delta
        color = (0, 255, 0) if delta > 0 else (0, 80, 255) if delta < 0 else (150, 150, 150)
        
        cv.putText(image, "+" if delta > 0 else "-" if delta < 0 else "=", (x+width-30, y+20), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Barra di intensità
        center_x = x + 80
        bar_width = 60
        cv.rectangle(image, (center_x, y+5), (center_x+bar_width, y+height-5), (70,70,70), -1)
        
        # Colore della barra in base alla direzione e intensità
        intensity = min(abs(delta * 20), bar_width)
        bar_x = center_x + (bar_width//2 - int(intensity//2)) if delta < 0 else center_x + (bar_width//2)
        bar_w = int(intensity) if delta < 0 else int(intensity)
        
        if intensity > 1:
            cv.rectangle(image, (bar_x, y+7), (bar_x+bar_w, y+height-7), color, -1)
        
        return image

    def draw_overlays(self, frame, results):
        """Draws all overlays."""
        # Store current display dimensions for other methods
        self.current_display_dims = (frame.shape[1], frame.shape[0])
        h, w = self.current_display_dims[1], self.current_display_dims[0] # Use stored h, w
        if w == 0 or h == 0: return frame

        overlay_c, bg_c, accent_c, error_c = (255,255,255), (0,0,0), (0,255,255), (0,0,255)
        font, fsc_sml, fsc_med = cv.FONT_HERSHEY_SIMPLEX, 0.4, 0.5
        l_type, thick_n, thick_b = 1, 1, 2

        if results and results.multi_hand_landmarks: frame = self.draw_landmarks(frame, results.multi_hand_landmarks)
        frame = self.draw_menu_trigger_circle(frame) # Updates pixel zone
        frame = self.draw_zoom_indicator(frame) # Disegna indicatore zoom se attivo

        # --- Calibration Overlay ---
        if self.calibration_active:
            bg = frame.copy(); cv.rectangle(bg, (0,0), (w, 100), bg_c, -1); alpha = 0.7
            frame = cv.addWeighted(bg, alpha, frame, 1 - alpha, 0)
            cv.putText(frame, f"CALIBRATION ({self.calibration_step+1}/4): RIGHT Hand", (10, 25), font, fsc_med, overlay_c, thick_n)
            cv.putText(frame, f"Point index to {self.calib_corners[self.calibration_step]} corner", (10, 50), font, fsc_med, overlay_c, thick_n)
            cv.putText(frame, "Press SPACE to confirm", (10, 75), font, fsc_med, accent_c, thick_b)
            cv.putText(frame, "(ESC to cancel)", (w - 150, 20), font, fsc_sml, overlay_c, thick_n)
            radius=15; inactive=(100,100,100); active=error_c; fill=-1
            corners_px = [(radius, radius), (w-radius, radius), (w-radius, h-radius), (radius, h-radius)]
            for i, p in enumerate(corners_px): cv.circle(frame, p, radius, active if self.calibration_step==i else inactive, fill); cv.circle(frame, p, radius, overlay_c, 1)
            hand_pos_px = self.get_hand_pos_in_display_pixels() # Use helper
            if hand_pos_px: cv.circle(frame, hand_pos_px, 10, (0, 255, 0), -1)
        # --- Normal Overlay ---
        else:
            info = "C:Calib | D:Debug | Q:Exit | M:Menu | Hover circle for menu"
            cv.putText(frame, info, (10, h-10), font, fsc_sml, overlay_c, l_type)
            fps = self.debug_values["fps"]; cv.putText(frame, f"FPS: {fps:.1f}", (w-80, 20), font, fsc_med, overlay_c, l_type)
                        
        # --- Tutorial Overlay (se attivo) ---
        if self.tutorial_manager.active:
            frame_with_btn, skip_btn_info = self.tutorial_manager.draw_tutorial_overlay(frame)
            # Salva le info del pulsante Skip per i click del mouse
            self.tutorial_skip_btn = {
                "x": skip_btn_info[0],
                "y": skip_btn_info[1],
                "width": skip_btn_info[2],
                "height": skip_btn_info[3],
                "visible": True
            }
            frame = frame_with_btn
        else:
            self.tutorial_skip_btn["visible"] = False
            
        return frame

    def main_loop(self):
        """Main application loop."""
        logger.info("Main loop starting."); win_name="HandPal Control"; cv.namedWindow(win_name,cv.WINDOW_NORMAL)
        
        # Registra il callback per i click del mouse
        cv.setMouseCallback(win_name, self.handle_cv_mouse_event)
        
        dw, dh = self.config.get('display_width'), self.config.get('display_height')
        if dw and dh:
            try:
                cv.resizeWindow(win_name, dw, dh)
                logger.info(f"Set display size {dw}x{dh}")
            except:
                logger.warning("Failed to set display size.")
        last_valid_frame = None # For display continuity

        while self.running:
            loop_start = time.perf_counter()
            # --- Update Tkinter ---
            if self.tk_root:
                try: self.tk_root.update_idletasks(); self.tk_root.update()
                except tk.TclError as e:
                    if "application has been destroyed" in str(e).lower(): logger.info("Tkinter root destroyed. Stopping."); self.stop(); break
                    else: logger.error(f"Tkinter error: {e}")

            # --- Get Data ---
            current_frame, results, proc_dims = None, None, None
            try:
                # Get frame, results, AND process dims from queue
                frame_data, results, proc_dims = self.data_queue.get(block=True, timeout=0.01)
                current_frame = frame_data.copy()
                last_valid_frame = current_frame # Store last good frame received
                if self.debug_mode: self.debug_values["q"] = self.data_queue.qsize()
            except queue.Empty:
                current_frame = last_valid_frame # Use last good frame if queue empty
                # Need to decide if we process old results with new frame or skip processing
                results = None # Skip processing if frame is old? Safer.
                proc_dims = self._last_proc_dims # Use last known proc_dims
            except Exception as e: logger.exception(f"Data queue error: {e}"); time.sleep(0.1); continue

            # --- Process Data ---
            if current_frame is not None: # Ensure we have a frame to work with
                 if results is not None and proc_dims is not None: # Process only if new results arrived
                     try: self.process_results(results, proc_dims)
                     except Exception as e: logger.exception("process_results error"); cv.putText(current_frame,"PROC ERR",(50,50),1,1,(0,0,255),2)

                 # --- Draw Overlays --- (Always draw on the current frame)
                 try: display_frame = self.draw_overlays(current_frame, results) # Pass results for landmarks
                 except Exception as e: logger.exception("draw_overlays error"); cv.putText(current_frame,"DRAW ERR",(50,100),1,1,(0,0,255),2); display_frame = current_frame # Fallback

                 # --- Resize & Display ---
                 final_frame = display_frame
                 disp_w, disp_h = self.config.get('display_width'), self.config.get('display_height')
                 if disp_w and disp_h: # Resize if specified
                     try:
                         h, w = display_frame.shape[:2]
                         if w!=disp_w or h!=disp_h: final_frame=cv.resize(display_frame,(disp_w,disp_h),interpolation=cv.INTER_LINEAR)
                     except Exception as e: logger.error(f"Resize error: {e}")
                 cv.imshow(win_name, final_frame)

            # --- Handle Input ---
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'): logger.info("'q' pressed, stopping."); self.stop(); break
            elif key == ord('c'):
                if not self.calibration_active: self.start_calibration()
            elif key == ord('d'): 
                self.debug_mode = not self.debug_mode
                logger.info(f"Debug mode toggled to: {self.debug_mode}")
                # Mostra/nascondi finestra di debug
                if self.debug_window:
                    if self.debug_mode:
                        self.debug_window.show()
                    else:
                        self.debug_window.hide()
                self.debug_values["cur_hist"].clear() if self.debug_mode else None
            elif key == ord('m'):
                 if self.floating_menu: logger.debug("'m' pressed, toggling menu."); self.floating_menu.toggle()
            elif key == 27: # ESC
                if self.calibration_active: self.cancel_calibration()
                elif self.tutorial_manager.active: self.tutorial_manager.stop_tutorial()
                elif self.floating_menu and self.floating_menu.visible: self.floating_menu.hide()
                elif self.debug_window and self.debug_window.visible: self.debug_window.hide()
            elif key == 32: # Spacebar
                if self.calibration_active: self.process_calibration_step()
                # Se tutorial attivo, spacebar salta al prossimo passaggio
                elif self.tutorial_manager.active: self.tutorial_manager.next_step()

            # --- FPS Calc ---
            elapsed = time.perf_counter() - loop_start; self.fps_stats.append(elapsed)
            if self.fps_stats: avg_dur = sum(self.fps_stats)/len(self.fps_stats); self.debug_values["fps"] = 1.0/avg_dur if avg_dur>0 else 0

        logger.info("Main loop finished."); cv.destroyAllWindows()

    # --- Calibration Methods ---
    def start_calibration(self):
        if self.calibration_active: return
        logger.info("Starting calibration..."); print("\n--- CALIBRATION START ---")
        self.calibration_active = True
        self.calibration_points = []
        self.calibration_step = 0
        self.config["calibration.active"] = True

    def cancel_calibration(self):
        if not self.calibration_active: return
        logger.info("Calibration cancelled."); print("--- CALIBRATION CANCELLED ---")
        self.calibration_active = False
        self.config["calibration.active"] = False

    def process_calibration_step(self):
        if not self.calibration_active: return
        if self.last_right_hand_lm_norm is None:
            logger.warning("No hand detected for calibration point."); return
        
        # Add the current point
        point = self.last_right_hand_lm_norm
        print(f"Calibration point {self.calibration_step+1}: {point}")
        self.calibration_points.append(point)
        
        # Move to next step or finish
        self.calibration_step += 1
        if self.calibration_step >= 4:
            self.finish_calibration()
        
    def finish_calibration(self):
        if not self.calibration_active or len(self.calibration_points) < 4: 
            logger.warning("Incomplete calibration points."); return
        
        # Calculate min/max from the 4 corners
        x_vals = [p[0] for p in self.calibration_points]
        y_vals = [p[1] for p in self.calibration_points]
        
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        
        # Update config
        self.config["calibration.x_min"] = x_min
        self.config["calibration.x_max"] = x_max
        self.config["calibration.y_min"] = y_min
        self.config["calibration.y_max"] = y_max
        self.config["calibration.enabled"] = True
        self.config["calibration.active"] = False
        
        # Reset state
        self.calibration_active = False
        logger.info(f"Calibration complete: X:[{x_min:.3f}-{x_max:.3f}] Y:[{y_min:.3f}-{y_max:.3f}]")
        print(f"--- CALIBRATION COMPLETE ---\nX range: {x_min:.3f} - {x_max:.3f}\nY range: {y_min:.3f} - {y_max:.3f}")
        
        # Save config
        self.config.save()

    def start(self):
        """Initializes and starts the webcam capture and processing."""
        if self.running: logger.warning("start() called but already running."); return True
        logger.info("Starting HandPal...")
        
        # Initialize camera
        try:
            self.cap = cv.VideoCapture(self.config["device"], cv.CAP_DSHOW if os.name == 'nt' else cv.CAP_ANY)
            if not self.cap.isOpened(): raise Exception(f"Failed to open camera device {self.config['device']}")
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.config["width"])
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.config["height"])
            w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv.CAP_PROP_FPS)
            logger.info(f"Camera initialized: {w}x{h} @ {fps:.1f}fps")
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
        
        # Initialize MediaPipe
        try:
            self.hands_instance = self.mp_hands.Hands(
                static_image_mode=self.config["use_static_image_mode"],
                max_num_hands=2,
                min_detection_confidence=self.config["min_detection_confidence"],
                min_tracking_confidence=self.config["min_tracking_confidence"]
            )
            logger.info(f"MediaPipe Hands initialized")
        except Exception as e:
            logger.error(f"MediaPipe initialization failed: {e}")
            self.cap.release(); self.cap = None
            return False
        
        # Start detection thread
        try:
            self.stop_event.clear()
            self.detection_thread = DetectionThread(
                self.config, self.cap, self.hands_instance, self.data_queue, self.stop_event
            )
            self.detection_thread.start()
            logger.info(f"Detection thread started")
        except Exception as e:
            logger.error(f"Detection thread start failed: {e}")
            self.hands_instance.close(); self.hands_instance = None
            self.cap.release(); self.cap = None
            return False
        
        self.running = True
        return True

    def stop(self):
        """Stops all processing and releases resources."""
        if not self.running: logger.warning("stop() called but not running."); return
        logger.info("Stopping HandPal...")
        
        # Stop detection thread
        if self.detection_thread and self.detection_thread.is_alive():
            logger.info("Stopping detection thread...")
            self.stop_event.set()
            self.detection_thread.join(timeout=2.0)
            if self.detection_thread.is_alive():
                logger.warning("Detection thread did not stop gracefully.")
        
        # Release MediaPipe
        if self.hands_instance:
            try: self.hands_instance.close(); logger.info("MediaPipe Hands closed")
            except Exception as e: logger.error(f"Error closing MediaPipe: {e}")
            self.hands_instance = None
        
        # Release camera
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
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="HandPal - Hand Gesture Control")
    parser.add_argument("-d", "--device", type=int, help="Camera device index")
    parser.add_argument("-f", "--flip-camera", action="store_true", help="Flip camera horizontally")
    parser.add_argument("--display-width", type=int, help="Display window width")
    parser.add_argument("--display-height", type=int, help="Display window height")
    parser.add_argument("--cursor-file", type=str, help="Path to custom cursor file (.cur)")
    args = parser.parse_args()

    # Load config with command line args
    config = Config(args)

    # Apply custom cursor if configured
    if os.name == 'nt' and config["custom_cursor_path"]:
        set_custom_cursor(config["custom_cursor_path"])

    # Initialize HandPal
    global handpal_instance  # To access from menu functions
    handpal = HandPal(config)
    handpal_instance = handpal

    # Start processing
    if handpal.start():
        try:
            # Main processing loop
            handpal.main_loop()
        except KeyboardInterrupt:
            print("\nExiting (Ctrl+C)...")
        except Exception as e:
            logger.exception(f"Error in main loop: {e}")
        finally:
            # Clean shutdown
            handpal.stop()
            # Restore cursor
            if os.name == 'nt' and config["custom_cursor_path"]:
                restore_default_cursor()
            
        logger.info("HandPal finished."); print("\nHandPal terminated.")
        time.sleep(0.1)
    return 0

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
