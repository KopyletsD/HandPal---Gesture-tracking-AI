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

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("handpal.log"), logging.StreamHandler()]
)
logger = logging.getLogger("HandPal")

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
# Config Class (Unchanged)
# -----------------------------------------------------------------------------
class Config:
    DEFAULT_CONFIG = {
    "device": 0,
    "width": 1280,
    "height": 720,
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
    "gesture_sensitivity": 0.02,  # Valore ridotto per richiedere un contatto quasi totale
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
# GestureRecognizer Class (Unchanged)
# -----------------------------------------------------------------------------
class GestureRecognizer:
    def __init__(self, config):
        self.config = config; self.last_positions = {}
        self.gesture_state = {"scroll_active": False, "last_click_time": 0, "last_click_button": None,
                              "scroll_history": deque(maxlen=5), "active_gesture": None,
                              "last_pose": {"Left": "U", "Right": "U"}, "pose_stable_count": {"Left": 0, "Right": 0},
                              "fist_drag_active": False}  # Use fist for drag instead of pinch
        self.POSE_STABILITY_THRESHOLD = 3

    def _dist(self, p1, p2):
        if None in [p1, p2] or not all(hasattr(p, 'x') for p in [p1,p2]): return float('inf')
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def check_thumb_index_click(self, hand_lm, handedness):
        if handedness != "Left" or hand_lm is None or self.gesture_state["scroll_active"]:
            return None
        try:
            thumb, index = hand_lm.landmark[4], hand_lm.landmark[8]
        except (IndexError, TypeError):
            return None
        dist = self._dist(thumb, index)
        now = time.time()
        thresh = self.config["gesture_sensitivity"]
        gesture = None
        
        # Check if pinch is active (thumb and index finger close)
        if dist < thresh:
            # Check for click (when not in cooldown period)
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
        else:
            # Normal click release
            if self.gesture_state["active_gesture"] == "click":
                self.gesture_state["active_gesture"] = None
        
        return gesture

    def check_fist_drag(self, hand_lm, handedness):
        """Detect drag and drop using left hand fist pose"""
        if handedness != "Left" or hand_lm is None or self.gesture_state["scroll_active"]:
            return None
            
        # Check the current pose without stabilizzazione
        raw_pose = self.detect_raw_pose(hand_lm)
        is_fist = raw_pose == "Fist"
        
        # Se il pugno è già attivo, usa una soglia più permissiva per mantenerlo
        if self.gesture_state["fist_drag_active"]:
            # Verifica che non ci siano più di 1 dita estese
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
                still_dragging = num_ext <= 1  # Più permissivo: ok anche con 1 dito
            except (IndexError, TypeError):
                still_dragging = False
            
            if still_dragging:
                return "drag_continue"
            else:
                self.gesture_state["fist_drag_active"] = False
                self.gesture_state["active_gesture"] = None
                return "drag_end"
        
        # Check per nuovo pugno
        if is_fist and not self.gesture_state["fist_drag_active"]:
            self.gesture_state["fist_drag_active"] = True
            self.gesture_state["active_gesture"] = "drag"
            return "drag_start"
        
        # No drag action
        return None

    def detect_raw_pose(self, hand_lm):
        """Versione semplificata e immediata del rilevamento postura senza stabilizzazione"""
        if hand_lm is None: return "U"
        try: 
            lm = hand_lm.landmark
            w = lm[0]  # wrist
            it,mt,rt,pt = lm[8],lm[12],lm[16],lm[20]  # finger tips
            im,mm,rm,pm = lm[5],lm[9],lm[13],lm[17]   # finger mcp (base)
            
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
         """Rileva il gesto di scorrimento (indice e medio estesi verticalmente), solo per la mano sinistra."""
         # Esegui il riconoscimento solo se è la mano sinistra
         if handedness != "Left":
             return None
         
         # Verifica che non ci sia un gesto di click attivo
         if self.gesture_state["active_gesture"] == "click":
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
        if handedness=="Left" and self.gesture_state["scroll_active"]: pose="Scroll"
        # Rileva fist anche quando il pollice è esteso (più permissivo)
        elif num_ext==0: pose="Fist" 
        elif i_ext and num_ext==1: pose="Point"
        elif i_ext and m_ext and num_ext==2: pose="Two"
        elif num_ext >= 4: pose="Open"
        
        last=self.gesture_state["last_pose"].get(handedness,"U"); count=self.gesture_state["pose_stable_count"].get(handedness,0)
        if pose==last and pose!="U": count+=1
        else: count=0
        self.gesture_state["last_pose"][handedness]=pose; self.gesture_state["pose_stable_count"][handedness]=count
        # Riduciamo la soglia di stabilità per il pugno
        if pose == "Fist" and count >= 1:
            return pose
        return pose if count>=self.POSE_STABILITY_THRESHOLD or pose=="Scroll" else last+"?"

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
                if self.flip: frame = cv.flip(frame, 1)
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
# Menu Related Functions & Classes (Unchanged)
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
        if path.startswith(('http://', 'https://')): webbrowser.open(path)
        elif os.name == 'nt': os.startfile(path)
        else: subprocess.Popen(path)
    except FileNotFoundError: logger.error(f"Launch failed: File not found '{path}'")
    except Exception as e: logger.error(f"Launch failed '{path}': {e}")
class FloatingMenu: # (Unchanged from v2)
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
# HandPal Class (Main Application - Modified Trigger Logic)
# -----------------------------------------------------------------------------
class HandPal:
    def __init__(self, config):
        self.config = config; self.mouse = Controller()
        self.gesture_recognizer = GestureRecognizer(config)
        try: # Init Tkinter
            self.tk_root = tk.Tk(); self.tk_root.withdraw()
            self.screen_size = (self.tk_root.winfo_screenwidth(), self.tk_root.winfo_screenheight())
            self.floating_menu = FloatingMenu(self.tk_root); logger.info(f"Screen: {self.screen_size}. Menu OK.")
        except tk.TclError as e: logger.error(f"Tkinter init failed: {e}. Menu N/A."); self.tk_root=None; self.floating_menu=None; self.screen_size=(1920,1080)
        self.motion_smoother = MotionSmoother(config)
        self.running = False; self.stop_event = threading.Event(); self.detection_thread = None
        self.data_queue = queue.Queue(maxsize=2)
        self.cap = None; self.mp_hands = mp.solutions.hands; self.hands_instance = None
        self.mp_drawing = mp.solutions.drawing_utils; self.mp_drawing_styles = mp.solutions.drawing_styles
        self.last_cursor_pos = None; self.last_right_hand_lm_norm = None # Store raw normalized landmark
        self.tracking_enabled = True; self.debug_mode = False
        self.calibration_active = False; self.calibration_points = []; self.calibration_step = 0
        self.calib_corners = ["top-left", "top-right", "bottom-right", "bottom-left"]
        # Store trigger zone in PIXELS relative to display frame
        self.menu_trigger_zone_px = {"cx":0, "cy":0, "radius_sq":0, "is_valid": False}
        self.current_display_dims = (0, 0) # Store last known display dims (w, h)
        self.menu_trigger_active = False; self._menu_activate_time = 0; self.MENU_HOVER_DELAY = 0.3
        self.debug_values = {"fps":0.0, "det_fps":0.0, "last_act":time.time(), "cur_hist":deque(maxlen=50),
                             "map":{"raw":'-',"cal":'-',"map":'-',"smooth":'-}', "L":"U", "R":"U", "gest":"N/A", "q":0, "menu":"Off"
                             }
        }
                                    
        self.fps_stats = deque(maxlen=60); self._last_proc_dims = (0,0) # Store last process dims (w, h)

    def start(self): # (Unchanged from v2)
        if self.running: return True
        logger.info("Starting HandPal..."); self.stop_event.clear()
        try:
            self.cap = cv.VideoCapture(self.config["device"], cv.CAP_DSHOW)
            if not self.cap.isOpened(): self.cap = cv.VideoCapture(self.config["device"])
            if not self.cap.isOpened(): raise IOError(f"Cannot open webcam {self.config['device']}")
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.config["width"]); self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.config["height"])
            self.cap.set(cv.CAP_PROP_FPS, self.config["max_fps"]); self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            w, h, fps = int(self.cap.get(3)), int(self.cap.get(4)), self.cap.get(5); logger.info(f"Webcam OK: {w}x{h} @ {fps:.1f} FPS")
            self.hands_instance = self.mp_hands.Hands(static_image_mode=self.config["use_static_image_mode"], max_num_hands=2, min_detection_confidence=self.config["min_detection_confidence"], min_tracking_confidence=self.config["min_tracking_confidence"])
            self.detection_thread = DetectionThread(self.config, self.cap, self.hands_instance, self.data_queue, self.stop_event); self.detection_thread.start()
            self.running = True; logger.info("HandPal started successfully!")
            return True
        except Exception as e: logger.exception(f"Startup failed: {e}"); self.stop(); return False

    def stop(self): # (Unchanged from v2)
        if not self.running and not self.stop_event.is_set(): self.stop_event.set(); cv.destroyAllWindows(); self._destroy_tkinter(); return
        if not self.running: return
        logger.info("Stopping HandPal..."); self.running = False; self.stop_event.set()
        if self.detection_thread and self.detection_thread.is_alive(): logger.debug("Joining DetectionThread..."); self.detection_thread.join(timeout=2.0); logger.info(f"DetectionThread alive after join: {self.detection_thread.is_alive()}")
        if hasattr(self.hands_instance, 'close'): self.hands_instance.close()
        if self.cap: self.cap.release(); logger.debug("Webcam released.")
        cv.destroyAllWindows(); logger.info("OpenCV windows closed.")
        self._destroy_tkinter(); logger.info("HandPal stopped.")

    def _destroy_tkinter(self): # (Unchanged from v2)
        if self.tk_root:
            try: logger.debug("Destroying Tkinter root..."); self.tk_root.quit(); self.tk_root.destroy(); logger.debug("Tkinter root destroyed.")
            except Exception as e: logger.error(f"Error destroying Tkinter: {e}")
            self.tk_root = None; self.floating_menu = None

    def map_to_screen(self, x_norm, y_norm): # (Unchanged from v2)
        sw, sh = self.screen_size
        if self.debug_mode:
            self.debug_values["map"]["raw"] = f"{x_norm:.3f},{y_norm:.3f}"
        x_cal, y_cal = x_norm, y_norm
        if self.config["calibration.enabled"]:
            x_min,x_max = self.config["calibration.x_min"], self.config["calibration.x_max"]
            y_min,y_max = self.config["calibration.y_min"], self.config["calibration.y_max"]
            xr, yr = x_max-x_min, y_max-y_min
            if xr>0.01 and yr>0.01:
                x_cl=max(x_min, min(x_norm, x_max)); y_cl=max(y_min, min(y_norm, y_max))
                x_cal=(x_cl-x_min)/xr; y_cal=(y_cl-y_min)/yr
        if self.debug_mode: self.debug_values["map"]["cal"]=f"{x_cal:.3f},{y_cal:.3f}"
        margin = self.config["calibration.screen_margin"]
        x_exp=x_cal*(1+2*margin)-margin; y_exp=y_cal*(1+2*margin)-margin
        sx=int(x_exp*sw); sy=int(y_exp*sh); sx=max(0,min(sx,sw-1)); sy=max(0,min(sy,sh-1))
        if self.debug_mode: self.debug_values["map"]["map"]=f"{sx},{sy}"
        return sx, sy

    def get_hand_pos_in_display_pixels(self):
        """ Calculates the estimated pixel coordinates of the right hand index tip
            on the *current display frame*. Returns (x, y) or None. """
        if self.last_right_hand_lm_norm is None: return None
        if self.current_display_dims[0] == 0 or self.current_display_dims[1] == 0: return None
        if self._last_proc_dims[0] == 0 or self._last_proc_dims[1] == 0: return None

        # Normalized coords relative to process frame
        norm_x_proc, norm_y_proc = self.last_right_hand_lm_norm

        # --- Convert to Pixel Coords on Display Frame ---
        # Method 1: Scale normalized coords directly (assumes aspect ratios match)
        # display_x = int(norm_x_proc * self.current_display_dims[0])
        # display_y = int(norm_y_proc * self.current_display_dims[1])

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

        # --- Right Hand: Cursor & Menu ---
        if lm_r and self.last_right_hand_lm_norm is not None:
             # Check menu trigger (uses self.last_right_hand_lm_norm and display dims)
             if self.check_menu_trigger():
                 if self.floating_menu and not self.floating_menu.visible:
                     logger.info("Menu trigger activated by hand hover."); self.floating_menu.show()

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
            
            # Check for drag with fist first
            if not action:
                drag_action = self.gesture_recognizer.check_fist_drag(lm_l, "Left")
                if drag_action:
                    if drag_action == "drag_start":
                        try: 
                            self.mouse.press(Button.left)
                            action = True
                            logger.info("Fist Drag Start")
                            print("⭐ DRAG START ⭐") # Output visibile direttamente sul terminale
                            self.debug_values["last_act"] = time.time()
                            self.debug_values["gest"] = "Drag ⛓️"
                        except Exception as e:
                            logger.error(f"Drag start error: {e}")
                    elif drag_action == "drag_continue":
                        action = True
                        self.debug_values["gest"] = "Dragging ⛓️"
                    elif drag_action == "drag_end":
                        try:
                            self.mouse.release(Button.left)
                            action = True
                            logger.info("Drag End")
                            print("⭐ DRAG END ⭐") # Output visibile direttamente sul terminale
                            self.debug_values["last_act"] = time.time()
                            self.debug_values["gest"] = "Idle"
                        except Exception as e:
                            logger.error(f"Drag end error: {e}")
            
            # Check for clicks only if no drag action is happening
            if not action and not self.gesture_recognizer.gesture_state["fist_drag_active"]:
                click_type = self.gesture_recognizer.check_thumb_index_click(lm_l, "Left")
                if click_type:
                    if click_type == "double_click":
                        btn = Button.left; count = 2; name="DoubleClick"
                        try: 
                            self.mouse.click(btn, count)
                            action = True
                            logger.info(f"{name} Left")
                            self.debug_values["last_act"] = time.time()
                            self.debug_values["gest"] = name
                        except Exception as e: 
                            logger.error(f"{name} error: {e}")
                    elif click_type == "click":
                        btn = Button.left; count = 1; name="Click"
                        try: 
                            self.mouse.click(btn, count)
                            action = True
                            logger.info(f"{name} Left")
                            self.debug_values["last_act"] = time.time()
                            self.debug_values["gest"] = name
                        except Exception as e:
                            logger.error(f"{name} error: {e}")

        # --- Update Debug Poses/Gestures --- (Unchanged from v2)
        if self.debug_mode:
            self.debug_values["L"] = self.gesture_recognizer.detect_hand_pose(lm_l, "Left")
            self.debug_values["R"] = self.gesture_recognizer.detect_hand_pose(lm_r, "Right")
            if not action: internal_gest = self.gesture_recognizer.gesture_state.get("active_gesture"); self.debug_values["gest"] = str(internal_gest) if internal_gest else "Idle"


    def draw_landmarks(self, frame, multi_hand_lm): # (Unchanged)
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

        # Aggiungiamo indicatore di drag
        if self.gesture_recognizer.gesture_state["fist_drag_active"]:
            drag_indicator = "DRAG ACTIVE"
            cv.putText(frame, drag_indicator, (w//2 - 80, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # Disegna un bordo rosso intorno al frame
            cv.rectangle(frame, (5, 5), (w-5, h-5), (0, 0, 255), 3)

        # --- Calibration Overlay --- (Unchanged)
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
        # --- Normal / Debug Overlay --- (Unchanged)
        else:
            info = "C:Calib | D:Debug | Q:Exit | M:Menu | Hover circle for menu"
            cv.putText(frame, info, (10, h-10), font, fsc_sml, overlay_c, l_type)
            fps = self.debug_values["fps"]; cv.putText(frame, f"FPS: {fps:.1f}", (w-80, 20), font, fsc_med, overlay_c, l_type)
            if self.debug_mode:
                panel_h=110; bg=frame.copy(); cv.rectangle(bg,(0,h-panel_h),(w,h),bg_c,-1); alpha=0.7
                frame=cv.addWeighted(bg,alpha,frame,1-alpha,0); y=h-panel_h+15; d=self.debug_values; m=d['map']; c=self.config['calibration']
                t1=f"L:{d['L']} R:{d['R']} Gest:{d['gest']} Menu:{d['menu']}"; cv.putText(frame,t1,(10,y),font,fsc_sml,overlay_c,l_type); y+=18
                t2=f"Map: Raw({m['raw']}) Cal({m['cal']}) -> MapPx({m['map']}) -> Smooth({m['smooth']})"; cv.putText(frame,t2,(10,y),font,fsc_sml,overlay_c,l_type); y+=18
                t3=f"Calib X[{c['x_min']:.2f}-{c['x_max']:.2f}] Y[{c['y_min']:.2f}-{c['y_max']:.2f}] En:{c['enabled']}"; cv.putText(frame,t3,(10,y),font,fsc_sml,overlay_c,l_type); y+=18
                t4=f"LastAct:{time.time()-d['last_act']:.1f}s ago | QSize:{d['q']}"; cv.putText(frame,t4,(10,y),font,fsc_sml,overlay_c,l_type); y+=18
                if len(d["cur_hist"]) > 1:
                    pts_screen=np.array(list(d["cur_hist"]),dtype=np.int32)
                    if self.screen_size[0]>0 and self.screen_size[1]>0:
                        pts_frame=pts_screen.copy(); pts_frame[:,0]=pts_frame[:,0]*w//self.screen_size[0]; pts_frame[:,1]=pts_frame[:,1]*h//self.screen_size[1]
                        pts_frame=np.clip(pts_frame,[0,0],[w-1,h-1]); cv.polylines(frame,[pts_frame],False,accent_c,1)
        return frame

    def main_loop(self):
        """Main application loop."""
        logger.info("Main loop starting."); win_name="HandPal Control"; cv.namedWindow(win_name,cv.WINDOW_NORMAL)
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

            # --- Handle Input --- (Unchanged from v2)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'): logger.info("'q' pressed, stopping."); self.stop(); break
            elif key == ord('c'):
                if not self.calibration_active: self.start_calibration()
            elif key == ord('d'): self.debug_mode = not self.debug_mode; logger.info(f"Debug mode toggled to: {self.debug_mode}"); self.debug_values["cur_hist"].clear() if self.debug_mode else None
            elif key == ord('m'):
                 if self.floating_menu: logger.debug("'m' pressed, toggling menu."); self.floating_menu.toggle()
            elif key == 27: # ESC
                if self.calibration_active: self.cancel_calibration()
                elif self.floating_menu and self.floating_menu.visible: self.floating_menu.hide()
            elif key == 32: # Spacebar
                if self.calibration_active: self.process_calibration_step()

            # --- FPS Calc --- (Unchanged from v2)
            elapsed = time.perf_counter() - loop_start; self.fps_stats.append(elapsed)
            if self.fps_stats: avg_dur = sum(self.fps_stats)/len(self.fps_stats); self.debug_values["fps"] = 1.0/avg_dur if avg_dur>0 else 0

        logger.info("Main loop finished."); cv.destroyAllWindows()

    # --- Calibration Methods --- (Unchanged from v2)
    def start_calibration(self):
        if self.calibration_active: return
        logger.info("Starting calibration..."); print("\n--- CALIBRATION START ---")
        self.calibration_active = True; self.config["calibration.active"] = True
        self.calibration_points = []; self.calibration_step = 0; self.tracking_enabled = False; self.motion_smoother.reset()
        if self.floating_menu and self.floating_menu.visible: self.floating_menu.hide()
        print(f"Step {self.calibration_step+1}/4: Point RIGHT index to {self.calib_corners[0]}. Press SPACE. (ESC to cancel)")
    def cancel_calibration(self):
        if not self.calibration_active: return
        logger.info("Calibration cancelled."); print("\n--- CALIBRATION CANCELLED ---")
        self.calibration_active = False; self.config["calibration.active"] = False
        self.calibration_points = []; self.calibration_step = 0; self.tracking_enabled = True
    def process_calibration_step(self):
        if not self.calibration_active: return
        # Use the pixel helper to get position on display, then normalize for storage
        hand_pos_px = self.get_hand_pos_in_display_pixels()
        if hand_pos_px is None: print("(!) Right hand not detected!"); return
        disp_w, disp_h = self.current_display_dims
        if disp_w == 0 or disp_h == 0: print("(!) Invalid display dimensions!"); return
        norm_x = hand_pos_px[0] / disp_w
        norm_y = hand_pos_px[1] / disp_h
        pos_norm_for_calib = (norm_x, norm_y) # Store normalized relative to display

        self.calibration_points.append(pos_norm_for_calib) # Append normalized point
        logger.info(f"Calib point {self.calibration_step+1} ({self.calib_corners[self.calibration_step]}): {pos_norm_for_calib[0]:.3f},{pos_norm_for_calib[1]:.3f}")
        print(f"-> Point {self.calibration_step+1}/4 ({self.calib_corners[self.calibration_step]}) captured.")
        self.calibration_step += 1
        if self.calibration_step >= 4: self.complete_calibration()
        else: print(f"\nStep {self.calibration_step+1}/4: Point RIGHT index to {self.calib_corners[self.calibration_step]}. Press SPACE.")
    def complete_calibration(self):
        logger.info("Completing calibration...")
        if len(self.calibration_points) != 4: logger.error("Incorrect point count."); print("(!) ERROR: Incorrect points."); self.cancel_calibration(); return
        try:
            xs=[p[0] for p in self.calibration_points]; ys=[p[1] for p in self.calibration_points] # Already normalized
            xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys)
            if (xmax-xmin<0.05 or ymax-ymin<0.05): print("(!) WARNING: Calibration area small, may be inaccurate.")
            self.config.set("calibration.x_min", xmin); self.config.set("calibration.x_max", xmax)
            self.config.set("calibration.y_min", ymin); self.config.set("calibration.y_max", ymax)
            self.config.set("calibration.enabled", True); self.config.save()
            logger.info(f"Calibration saved: X[{xmin:.3f}-{xmax:.3f}], Y[{ymin:.3f}-{ymax:.3f}]"); print("\n--- CALIBRATION SAVED ---")
        except Exception as e: logger.exception("Calibration completion/save error."); print("(!) ERROR saving calibration."); self.config.set("calibration.enabled", False)
        finally: self.calibration_active = False; self.config["calibration.active"] = False; self.calibration_points = []; self.calibration_step = 0; self.tracking_enabled = True

# -----------------------------------------------------------------------------
# Argument Parsing (Unchanged)
# -----------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="HandPal - Mouse control with gestures & menu.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cfg = Config.DEFAULT_CONFIG
    parser.add_argument('--device', type=int, default=None, help=f'Webcam ID (Default: {cfg["device"]})')
    parser.add_argument('--width', type=int, default=None, help=f'Webcam Width (Default: {cfg["width"]})')
    parser.add_argument('--height', type=int, default=None, help=f'Webcam Height (Default: {cfg["height"]})')
    parser.add_argument('--display-width', type=int, default=None, help='Preview window width (Default: frame size)')
    parser.add_argument('--display-height', type=int, default=None, help='Preview window height (Default: frame size)')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug overlay & logging.')
    parser.add_argument('--flip-camera', action='store_true', default=None, help=f'Flip webcam horizontally (Default: {cfg["flip_camera"]})')
    parser.add_argument('--no-flip', dest='flip_camera', action='store_false', help='Disable horizontal flip.')
    parser.add_argument('--calibrate', action='store_true', default=False, help='Start calibration on launch.')
    parser.add_argument('--reset-config', action='store_true', default=False, help='Delete config file and exit.')
    parser.add_argument('--config-file', type=str, default=Config.CONFIG_FILENAME, help='Path to config JSON file.')
    parser.add_argument('--cursor-file', type=str, default=None, help=f'Path to custom .cur file (Win only) (Default: {cfg["custom_cursor_path"]})')
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Main Function (Unchanged)
# -----------------------------------------------------------------------------
def main():
    args = parse_arguments(); cursor_set = False
    log_level = logging.DEBUG if args.debug else logging.INFO; logger.setLevel(log_level); [h.setLevel(log_level) for h in logger.handlers]; logger.info(f"Log level set to {logging.getLevelName(log_level)}")
    if args.reset_config:
        path = args.config_file
        try:
            if os.path.exists(path): os.remove(path); logger.info(f"Removed config: {path}"); print(f"Removed config: {path}")
            else: logger.info("No config file found to reset."); print("No config file found to reset.")
            return 0
        except Exception as e: logger.error(f"Error removing config: {e}"); print(f"Error removing config: {e}"); return 1
    create_default_apps_csv(); config = Config(args)
    if os.name == 'nt': # Custom Cursor
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
             elif cursor_path_cfg: logger.error(f"Cursor file specified but not found/accessible: {cursor_path_cfg}")
    app = None
    try:
        app = HandPal(config)
        if args.debug: app.debug_mode = True
        if app.start():
            if args.calibrate: logger.info("Starting calibration (--calibrate)..."); time.sleep(0.75); app.start_calibration()
            app.main_loop()
        else: logger.error("HandPal failed to start."); print("ERROR: HandPal failed to start. Check log."); return 1
    except KeyboardInterrupt: logger.info("Ctrl+C received. Stopping...")
    except Exception as e: logger.exception("Unhandled error in main block."); print(f"UNHANDLED ERROR: {e}. Check log."); return 1
    finally:
        logger.info("Main finally block executing cleanup...")
        if app: app.stop()
        if cursor_set: logger.info("Restoring default cursor..."); restore_default_cursor()
        logger.info("HandPal finished."); print("\nHandPal terminated.")
        time.sleep(0.1)
    return 0

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())