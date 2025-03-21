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
        "width": 1980,
        "height": 1440,
        "process_width": 640,
        "process_height": 360,
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
            "drag_threshold": 0.15,
            "double_click_time": 0.4
        },
        "calibration": {
            "enabled": True,
            "screen_margin": 0.15,
            "x_min": 0.2,  # Migliorati i valori di default per evitare errori
            "x_max": 0.8,
            "y_min": 0.2,
            "y_max": 0.8,
            "active": False
        }
    }
    
    def __init__(self, args=None):
        """Inizializza la configurazione combinando valori predefiniti e argomenti CLI."""
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Carica config dal file se esiste
        config_path = os.path.expanduser("~/.handpal_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
                    
                    # Verifica che i valori di calibrazione non causino divisione per zero
                    if abs(self.config["calibration"]["x_max"] - self.config["calibration"]["x_min"]) < 0.1:
                        logger.warning("Valori di calibrazione X troppo vicini, ripristino valori di default")
                        self.config["calibration"]["x_min"] = self.DEFAULT_CONFIG["calibration"]["x_min"]
                        self.config["calibration"]["x_max"] = self.DEFAULT_CONFIG["calibration"]["x_max"]
                    
                    if abs(self.config["calibration"]["y_max"] - self.config["calibration"]["y_min"]) < 0.1:
                        logger.warning("Valori di calibrazione Y troppo vicini, ripristino valori di default")
                        self.config["calibration"]["y_min"] = self.DEFAULT_CONFIG["calibration"]["y_min"]
                        self.config["calibration"]["y_max"] = self.DEFAULT_CONFIG["calibration"]["y_max"]
                    
                    logger.info("Configurazione caricata dal file")
            except Exception as e:
                logger.error(f"Errore nel caricamento della configurazione: {e}")
        
        # Aggiorna con gli argomenti CLI se presenti
        if args:
            for key, value in vars(args).items():
                if key in self.config and value is not None:
                    self.config[key] = value
    
    def save(self):
        """Salva la configurazione in un file."""
        config_path = os.path.expanduser("~/.handpal_config.json")
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
                logger.info("Configurazione salvata")
        except Exception as e:
            logger.error(f"Errore nel salvataggio della configurazione: {e}")
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __setitem__(self, key, value):
        self.config[key] = value


class GestureRecognizer:
    """Classe specializzata per il riconoscimento dei gesti."""
    
    def __init__(self, config):
        self.config = config
        self.last_positions = {}
        self.gesture_state = {
            "drag_active": False,
            "scroll_active": False,
            "last_click_time": 0,
            "last_click_position": (0, 0),
            "scroll_history": deque(maxlen=5),
            "last_gesture": None,
            "gesture_stable_count": 0,
            "active_gesture": None  # Per tenere traccia del gesto attualmente attivo
        }
    
    def check_thumb_index_click(self, hand_landmarks, handedness):
        """Rileva il gesto di click (indice e pollice che si toccano), solo per la mano sinistra."""
        # Esegui il riconoscimento solo se è la mano sinistra
        if handedness != "Left":
            return None
            
        # Se c'è già un gesto di scroll attivo, ignora il click per evitare conflitti
        if self.gesture_state["scroll_active"]:
            return None
            
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        # Calcola distanza solo in 2D (ignorando z)
        distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        
        current_time = time.time()
        if distance < self.config["gesture_sensitivity"]:
            if (current_time - self.gesture_state["last_click_time"]) > self.config["click_cooldown"]:
                # Controllo doppio click
                if (current_time - self.gesture_state["last_click_time"]) < self.config["gesture_settings"]["double_click_time"]:
                    self.gesture_state["last_click_time"] = current_time
                    self.gesture_state["active_gesture"] = "click"
                    return "double_click"
                
                self.gesture_state["last_click_time"] = current_time
                self.gesture_state["active_gesture"] = "click"
                return "click"
        else:
            # Resetta il gesto attivo se non stiamo più facendo click
            if self.gesture_state["active_gesture"] == "click":
                self.gesture_state["active_gesture"] = None
                
        return None
    
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
    
    def detect_hand_pose(self, hand_landmarks, handedness):
        """Analizza la posa complessiva della mano, considerando solo la mano sinistra per i gesti."""
        # Se è la mano destra, riconosci solo il pointing per il movimento del cursore
        if handedness == "Right":
            index_tip = hand_landmarks.landmark[8]
            index_mcp = hand_landmarks.landmark[5]
            index_extended = index_tip.y < index_mcp.y
            
            if index_extended:
                return "pointing"
            else:
                return "unknown"
        
        # Estrai posizioni chiave delle dita
        wrist = hand_landmarks.landmark[0]
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        # Ottiene le basi delle dita per confronto
        thumb_ip = hand_landmarks.landmark[3]  # Falange prossimale del pollice
        index_mcp = hand_landmarks.landmark[5]  # Base dell'indice
        middle_mcp = hand_landmarks.landmark[9]  # Base del medio
        ring_mcp = hand_landmarks.landmark[13]  # Base dell'anulare
        pinky_mcp = hand_landmarks.landmark[17]  # Base del mignolo
        
        # Controlla se ogni dito è esteso
        thumb_extended = thumb_tip.x < thumb_ip.x if handedness == "Right" else thumb_tip.x > thumb_ip.x
        index_extended = index_tip.y < index_mcp.y
        middle_extended = middle_tip.y < middle_mcp.y
        ring_extended = ring_tip.y < ring_mcp.y
        pinky_extended = pinky_tip.y < pinky_mcp.y
        
        # Riconoscimento delle pose comuni
        current_pose = "unknown"
        
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            current_pose = "pointing"
        elif index_extended and middle_extended and not ring_extended and not pinky_extended:
            current_pose = "peace"
        elif index_extended and middle_extended and ring_extended and pinky_extended:
            current_pose = "open_palm"
        elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            current_pose = "closed_fist"
        
        # Stabilizzazione dei gesti
        if current_pose == self.gesture_state["last_gesture"]:
            self.gesture_state["gesture_stable_count"] += 1
        else:
            self.gesture_state["gesture_stable_count"] = 0
            
        # Aggiorna solo se il gesto è stabile per più frame
        if self.gesture_state["gesture_stable_count"] > 5:
            self.gesture_state["last_gesture"] = current_pose
        
        return self.gesture_state["last_gesture"] if self.gesture_state["last_gesture"] else current_pose


class MotionSmoother:
    """Classe per smoothing del movimento del mouse con filtro di media mobile."""
    
    def __init__(self, config, history_size=8):  # Aumentato history_size per smoothing più forte
        self.config = config
        self.history_size = history_size
        self.position_history = deque(maxlen=history_size)
        self.last_position = None
        self.inactivity_zone_size = config["inactivity_zone"]
        self.smoothing_factor = config["smoothing_factor"]
    
    def update(self, target_x, target_y):
        """Aggiorna la cronologia delle posizioni e calcola la nuova posizione smussata."""
        # Aggiungi nuova posizione al buffer
        self.position_history.append((target_x, target_y))
        
        if len(self.position_history) < 2:
            return target_x, target_y
        
        # Applica media ponderata esponenziale
        weights = np.exp(np.linspace(-2, 0, len(self.position_history)))
        weights = weights / np.sum(weights)
        
        x = int(sum(p[0] * w for p, w in zip(self.position_history, weights)))
        y = int(sum(p[1] * w for p, w in zip(self.position_history, weights)))
        
        # Se esiste una posizione precedente, applica anche interpolazione
        if self.last_position:
            # Applica zona morta per piccoli movimenti
            dx = abs(x - self.last_position[0]) / self.config["width"]
            dy = abs(y - self.last_position[1]) / self.config["height"]
            
            if dx < self.inactivity_zone_size and dy < self.inactivity_zone_size:
                return self.last_position
                
            # Applica smoothing tra posizione precedente e attuale
            x = int(self.last_position[0] * (1-self.smoothing_factor) + x * self.smoothing_factor)
            y = int(self.last_position[1] * (1-self.smoothing_factor) + y * self.smoothing_factor)
        
        self.last_position = (x, y)
        return x, y


class HandPal:
    """Classe principale che gestisce l'applicazione HandPal."""
    
    def __init__(self, config):
        self.config = config
        self.mouse = Controller()
        self.gesture_recognizer = GestureRecognizer(config)
        self.motion_smoother = MotionSmoother(config)
        self.running = False
        self.cap = None
        self.mp_hands = mp.solutions.hands
        self.hands = None
        self.last_cursor_pos = None
        self.last_right_hand_pos = None
        self.screen_size = (tk.Tk().winfo_screenwidth(), tk.Tk().winfo_screenheight())
        self.fps_stats = deque(maxlen=30)
        
        # Inizializzazione della calibrazione
        self.calibration_active = False
        self.calibration_points = []
        self.calibration_step = 0
        self.calibration_corners = [
            "in alto a sinistra dello SCHERMO",
            "in alto a destra dello SCHERMO",
            "in basso a destra dello SCHERMO",
            "in basso a sinistra dello SCHERMO"
        ]
        
        # Flag per stato dell'app
        self.debug_mode = False
        self.tracking_enabled = True
        
        # Debug info
        self.debug_values = {
            "hand_distances": [],
            "gesture_confidence": 0.0,
            "last_action_time": time.time(),
            "cursor_history": deque(maxlen=10),
            "screen_mapping": {"raw": (0, 0), "mapped": (0, 0)}
        }
        
    def start(self):
        """Avvia l'applicazione."""
        # Inizializza la webcam
        if self.cap is None:
            self.cap = cv.VideoCapture(self.config["device"])
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.config["width"])
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.config["height"])
            
            if not self.cap.isOpened():
                logger.error("Impossibile aprire la webcam.")
                return False
        
        # Inizializza MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.config["use_static_image_mode"],
            max_num_hands=2,  # Rileva entrambe le mani
            min_detection_confidence=self.config["min_detection_confidence"],
            min_tracking_confidence=self.config["min_tracking_confidence"]
        )
        
        # Avvia il thread principale
        self.running = True
        threading.Thread(target=self.main_loop, daemon=True).start()
        
        logger.info("HandPal avviato con successo!")
        return True
    
    def stop(self):
        """Ferma l'applicazione."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.hands is not None:
            self.hands.close()
            self.hands = None
        cv.destroyAllWindows()
        logger.info("HandPal fermato.")
    
    def map_to_screen(self, x, y):
        """Mappa le coordinate normalizzate della mano allo schermo."""
        try:
            # Per debug, salva le coordinate raw
            if self.debug_mode:
                self.debug_values["screen_mapping"]["raw"] = (x, y)
                
            # Applica calibrazione se attiva
            if self.config["calibration"]["enabled"]:
                # Adatta al range di calibrazione
                x_min = self.config["calibration"]["x_min"]
                x_max = self.config["calibration"]["x_max"]
                y_min = self.config["calibration"]["y_min"]
                y_max = self.config["calibration"]["y_max"]
                
                # Previeni divisione per zero
                x_range = max(0.1, x_max - x_min)  # Almeno 0.1 di differenza
                y_range = max(0.1, y_max - y_min)  # Almeno 0.1 di differenza
                
                # Normalizza all'interno dei limiti di calibrazione
                x_norm = max(0, min(1, (x - x_min) / x_range))
                y_norm = max(0, min(1, (y - y_min) / y_range))
            else:
                # Usa semplice mappatura 0-1
                x_norm = max(0, min(1, x))
                y_norm = max(0, min(1, y))
            
            # Aggiungi margine per facilitare l'accesso agli angoli
            margin = self.config["calibration"]["screen_margin"]
            screen_x = int((x_norm * (1 + 2 * margin) - margin) * self.screen_size[0])
            screen_y = int((y_norm * (1 + 2 * margin) - margin) * self.screen_size[1])
            
            # Limita alle dimensioni effettive dello schermo
            screen_x = max(0, min(screen_x, self.screen_size[0] - 1))
            screen_y = max(0, min(screen_y, self.screen_size[1] - 1))
            
            # Per debug, salva le coordinate mappate
            if self.debug_mode:
                self.debug_values["screen_mapping"]["mapped"] = (screen_x, screen_y)
                self.debug_values["cursor_history"].append((screen_x, screen_y))
                
            return screen_x, screen_y
        except Exception as e:
            logger.error(f"Errore durante la mappatura: {e}")
            # In caso di errore, usa una mappatura semplice
            screen_x = int(x * self.screen_size[0])
            screen_y = int(y * self.screen_size[1])
            return max(0, min(screen_x, self.screen_size[0] - 1)), max(0, min(screen_y, self.screen_size[1] - 1))
    
    def process_frame(self, frame):
        """Processa un singolo frame della webcam."""
        # Ridimensiona il frame per velocizzare l'elaborazione
        process_frame = cv.resize(frame, (self.config["process_width"], self.config["process_height"]))
        
        # Converti da BGR a RGB per MediaPipe
        rgb_frame = cv.cvtColor(process_frame, cv.COLOR_BGR2RGB)
        
        # Elabora il frame con MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Crea una copia del frame per disegnare
        display_frame = process_frame.copy()
        
        # Aggiungi informazioni sulla calibrazione o istruzioni generali
        if self.calibration_active:
            # Sfondo semi-trasparente per le istruzioni
            overlay = display_frame.copy()
            cv.rectangle(overlay, (0, 0), (display_frame.shape[1], 100), (0, 0, 0), -1)
            cv.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            # Istruzioni per la calibrazione
            cv.putText(display_frame, f"CALIBRAZIONE: Muovi la mano DESTRA {self.calibration_corners[self.calibration_step]}", 
                       (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv.putText(display_frame, "Poi premi SPAZIO per confermare la posizione", 
                       (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv.putText(display_frame, "(ESC per annullare la calibrazione)", 
                       (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            
            # Disegna gli angoli dello schermo sulla finestra per riferimento
            h, w = display_frame.shape[:2]
            # Angolo in alto a sinistra
            cv.circle(display_frame, (0, 0), 15, (0, 0, 255) if self.calibration_step == 0 else (200, 200, 200), -1)
            # Angolo in alto a destra
            cv.circle(display_frame, (w, 0), 15, (0, 0, 255) if self.calibration_step == 1 else (200, 200, 200), -1)
            # Angolo in basso a destra
            cv.circle(display_frame, (w, h), 15, (0, 0, 255) if self.calibration_step == 2 else (200, 200, 200), -1)
            # Angolo in basso a sinistra
            cv.circle(display_frame, (0, h), 15, (0, 0, 255) if self.calibration_step == 3 else (200, 200, 200), -1)
        else:
            # Istruzioni normali
            cv.putText(display_frame, "C: Calibrazione | Q: Esci | D: Debug", (10, display_frame.shape[0] - 10), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Stato dei gesti attivi
            active_gesture = self.gesture_recognizer.gesture_state["active_gesture"]
            if active_gesture:
                cv.putText(display_frame, f"Gesto attivo: {active_gesture}", (10, 30), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if results.multi_hand_landmarks:
            # Variabili per tracciare la posizione delle mani
            right_hand_pos = None
            left_hand_landmarks = None
            right_hand_landmarks = None
            frame_height, frame_width = process_frame.shape[:2]
            
            # Identifica quale landmark appartiene a quale mano
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Determina se è mano destra o sinistra
                handedness = results.multi_handedness[idx].classification[0].label
                
                if handedness == "Right":
                    right_hand_landmarks = hand_landmarks
                    # Usa la punta dell'indice per tracciare
                    index_tip = hand_landmarks.landmark[8]
                    right_hand_pos = (index_tip.x, index_tip.y)
                elif handedness == "Left":
                    left_hand_landmarks = hand_landmarks
            
            # Gestione della calibrazione (se attiva)
            if self.calibration_active and right_hand_pos:
                # Disegna un cerchio dove punta l'indice per la calibrazione
                tip_x = int(right_hand_pos[0] * frame_width)
                tip_y = int(right_hand_pos[1] * frame_height)
                cv.circle(display_frame, (tip_x, tip_y), 10, (0, 255, 0), -1)
                
                # Memorizza la posizione per la calibrazione
                self.last_right_hand_pos = right_hand_pos
            
            # Gestione del cursore (solo con mano destra)
            elif right_hand_pos and self.tracking_enabled and not self.calibration_active:
                try:
                    # Mappa la posizione della mano allo schermo e applica smoothing
                    screen_x, screen_y = self.map_to_screen(right_hand_pos[0], right_hand_pos[1])
                    smooth_x, smooth_y = self.motion_smoother.update(screen_x, screen_y)
                    
                    # Aggiorna posizione del cursore
                    if self.last_cursor_pos != (smooth_x, smooth_y):
                        self.mouse.position = (smooth_x, smooth_y)
                        self.last_cursor_pos = (smooth_x, smooth_y)
                    
                    # Memorizza l'ultima posizione della mano destra
                    self.last_right_hand_pos = right_hand_pos
                    
                    # In modalità debug, visualizza un tracciamento del cursore
                    if self.debug_mode:
                        # Disegna il percorso del cursore
                        if len(self.debug_values["cursor_history"]) > 1:
                            for i in range(1, len(self.debug_values["cursor_history"])):
                                p1 = self.debug_values["cursor_history"][i-1]
                                p2 = self.debug_values["cursor_history"][i]
                                # Converte le coordinate dello schermo in coordinate del frame
                                p1_frame = (int(p1[0] * frame_width / self.screen_size[0]), 
                                            int(p1[1] * frame_height / self.screen_size[1]))
                                p2_frame = (int(p2[0] * frame_width / self.screen_size[0]), 
                                            int(p2[1] * frame_height / self.screen_size[1]))
                                cv.line(display_frame, p1_frame, p2_frame, (0, 255, 255), 1)
                except Exception as e:
                    logger.error(f"Errore nell'aggiornamento del cursore: {e}")
            
            # Gestione dei gesti (solo con mano sinistra)
            if left_hand_landmarks and not self.calibration_active:
                # Rileva il gesto della mano
                hand_pose = self.gesture_recognizer.detect_hand_pose(left_hand_landmarks, "Left")
                
                # Controlla i gesti specifici in ordine di priorità
                scroll_amount = self.gesture_recognizer.check_scroll_gesture(left_hand_landmarks, "Left")
                if scroll_amount:
                    # Converti in click dello scroll wheel
                    scroll_clicks = int(scroll_amount / 5)  # Valore arbitrario per sensibilità
                    if scroll_clicks != 0:
                        self.mouse.scroll(0, scroll_clicks)
                        logger.debug(f"Scroll: {scroll_clicks}")
                else:
                    # Solo se non stiamo facendo scroll, controlla il click
                    click_gesture = self.gesture_recognizer.check_thumb_index_click(left_hand_landmarks, "Left")
                    if click_gesture == "click":
                        self.mouse.click(Button.left)
                        self.debug_values["last_action_time"] = time.time()
                        logger.info("Click sinistro")
                    elif click_gesture == "double_click":
                        self.mouse.click(Button.left, 2)
                        self.debug_values["last_action_time"] = time.time()
                        logger.info("Doppio click")
            
            # Disegna i landmark delle mani
            self.draw_landmarks(display_frame, results.multi_hand_landmarks)
            
            # Visualizza lo stato del riconoscimento gesti
            if left_hand_landmarks and not self.calibration_active:
                hand_pose = self.gesture_recognizer.detect_hand_pose(left_hand_landmarks, "Left")
                cv.putText(display_frame, f"Gesture: {hand_pose}", (10, 50), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Visualizza FPS
            fps = len(self.fps_stats) / sum(self.fps_stats) if self.fps_stats else 0
            cv.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1] - 80, 20), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Informazioni aggiuntive in modalità debug
            if self.debug_mode and not self.calibration_active:
                # Sfondo semitrasparente per le informazioni di debug
                debug_overlay = display_frame.copy()
                cv.rectangle(debug_overlay, (0, display_frame.shape[0] - 160), 
                            (display_frame.shape[1], display_frame.shape[0]), (0, 0, 0), -1)
                cv.addWeighted(debug_overlay, 0.7, display_frame, 0.3, 0, display_frame)
                
                # Valori di calibrazione
                calibration_info = f"Calibrazione: X[{self.config['calibration']['x_min']:.2f}-{self.config['calibration']['x_max']:.2f}] "
                calibration_info += f"Y[{self.config['calibration']['y_min']:.2f}-{self.config['calibration']['y_max']:.2f}]"
                cv.putText(display_frame, calibration_info, (10, display_frame.shape[0] - 130), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Informazioni di configurazione
                config_info = f"Sensibilità: {self.config['gesture_sensitivity']:.3f} | "
                config_info += f"Cooldown: {self.config['click_cooldown']:.2f}s | "
                config_info += f"Smoothing: {self.config['smoothing_factor']:.2f}"
                cv.putText(display_frame, config_info, (10, display_frame.shape[0] - 110), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Coordinate attuali del cursore
                cursor_pos = self.mouse.position
                cursor_info = f"Posizione Cursore: ({cursor_pos[0]}, {cursor_pos[1]})"
                cv.putText(display_frame, cursor_info, (10, display_frame.shape[0] - 90), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Tempo dall'ultimo evento
                time_since_action = time.time() - self.debug_values["last_action_time"]
                event_info = f"Ultimo evento: {time_since_action:.2f}s fa"
                cv.putText(display_frame, event_info, (10, display_frame.shape[0] - 70), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Stato dei gesti
                gesture_info = f"Stato Gesti: "
                gesture_info += f"Scroll attivo: {self.gesture_recognizer.gesture_state['scroll_active']} | "
                gesture_info += f"Click cooldown: {max(0, self.config['click_cooldown'] - (time.time() - self.gesture_recognizer.gesture_state['last_click_time'])):.2f}s"
                cv.putText(display_frame, gesture_info, (10, display_frame.shape[0] - 50), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Informazioni sulla mappatura
                if right_hand_pos:
                    mapping_info = f"Mano: ({right_hand_pos[0]:.3f}, {right_hand_pos[1]:.3f}) -> "
                    mapping_info += f"Schermo: ({self.debug_values['screen_mapping']['mapped'][0]}, {self.debug_values['screen_mapping']['mapped'][1]})"
                    cv.putText(display_frame, mapping_info, (10, display_frame.shape[0] - 30), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return display_frame
    
    def draw_landmarks(self, frame, multi_hand_landmarks):
        """Disegna i landmark delle mani sul frame."""
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
    
    def main_loop(self):
        """Loop principale dell'applicazione."""
        last_time = time.time()
        
        while self.running:
            try:
                start_time = time.time()
                
                # Limita FPS
                frame_time = 1.0 / self.config["max_fps"]
                if start_time - last_time < frame_time:
                    time.sleep(frame_time - (start_time - last_time))
                    start_time = time.time()
                
                # Acquisisci frame dalla webcam
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Errore nella lettura del frame dalla webcam")
                    time.sleep(0.1)
                    continue
                
                # Ribalta orizzontalmente il frame se necessario
                if self.config["flip_camera"]:
                    frame = cv.flip(frame, 1)
                
                # Processa il frame
                processed_frame = self.process_frame(frame)
                
                # Mostra sempre il frame elaborato
                cv.imshow("HandPal", processed_frame)
                key = cv.waitKey(1) & 0xFF
                
                # Gestione dei tasti
                if key == ord('q'):
                    self.stop()
                elif key == ord('c') and not self.calibration_active:
                    # Attiva modalità calibrazione
                    self.start_calibration()
                elif key == ord('d'):
                    # Attiva/disattiva modalità debug
                    self.debug_mode = not self.debug_mode
                    logger.info(f"Modalità debug: {'attivata' if self.debug_mode else 'disattivata'}")
                elif key == 27 and self.calibration_active:  # ESC per annullare calibrazione
                    logger.info("Calibrazione annullata dall'utente")
                    self.calibration_active = False
                    self.calibration_step = 0
                    self.calibration_points = []
                elif key == 32 and self.calibration_active:  # Spazio per confermare posizione calibrazione
                    try:
                        self.process_calibration_step()
                    except Exception as e:
                        logger.error(f"Errore nella calibrazione: {e}")
                        self.calibration_active = False
                        self.calibration_step = 0
                        self.calibration_points = []
                
                # Calcola FPS per statistiche
                frame_duration = time.time() - start_time
                self.fps_stats.append(frame_duration)
                last_time = start_time
            except Exception as e:
                logger.error(f"Errore nel main loop: {e}")
                time.sleep(0.1)  # Pausa breve in caso di errore per evitare cicli troppo rapidi
    
    def start_calibration(self):
        """Avvia il processo di calibrazione."""
        if self.calibration_active:
            return
        
        self.calibration_active = True
        self.config["calibration"]["active"] = True
        self.calibration_points = []
        self.calibration_step = 0
        
        logger.info("Calibrazione avviata. Segui le istruzioni sullo schermo.")
        print("\n--- CALIBRAZIONE ---")
        print("Per una corretta calibrazione:")
        print("1. Usa SOLO la mano DESTRA (quella usata per il puntatore)")
        print("2. Posiziona l'indice negli angoli dello SCHERMO (non della finestra webcam)")
        print("3. Premi SPAZIO per confermare ogni posizione")
        print("4. Premi ESC per annullare la calibrazione in qualsiasi momento")
        print("\nSegui le indicazioni sullo schermo...\n")
    
    def process_calibration_step(self):
        """Processa un singolo passo della calibrazione."""
        if not self.calibration_active or not self.last_right_hand_pos:
            return
        
        # Aggiungi la posizione corrente ai punti di calibrazione
        self.calibration_points.append(self.last_right_hand_pos)
        logger.info(f"Punto di calibrazione {self.calibration_step + 1} registrato: {self.last_right_hand_pos}")
        
        # Avanza alla prossima posizione o completa la calibrazione
        self.calibration_step += 1
        if self.calibration_step >= 4:  # Abbiamo 4 angoli
            self.complete_calibration()
        else:
            print(f"Ora posiziona la mano {self.calibration_corners[self.calibration_step]} e premi SPAZIO.")
    
    def complete_calibration(self):
        """Completa la calibrazione usando i punti registrati."""
        try:
            if len(self.calibration_points) == 4:
                # Calcola i limiti di calibrazione dai punti registrati
                x_values = [p[0] for p in self.calibration_points]
                y_values = [p[1] for p in self.calibration_points]
                
                # Assicura che ci sia una differenza minima tra i valori per evitare divisione per zero
                x_min = min(x_values)
                x_max = max(x_values)
                y_min = min(y_values)
                y_max = max(y_values)
                
                # Verifica che i valori non siano troppo vicini
                if (x_max - x_min) < 0.1:
                    logger.warning("Valori di calibrazione X troppo vicini")
                    x_min = max(0.0, x_min - 0.05)
                    x_max = min(1.0, x_max + 0.05)
                
                if (y_max - y_min) < 0.1:
                    logger.warning("Valori di calibrazione Y troppo vicini")
                    y_min = max(0.0, y_min - 0.05)
                    y_max = min(1.0, y_max + 0.05)
                
                self.config["calibration"]["x_min"] = x_min
                self.config["calibration"]["x_max"] = x_max
                self.config["calibration"]["y_min"] = y_min
                self.config["calibration"]["y_max"] = y_max
                
                # Salva la configurazione
                self.config.save()
                logger.info(f"Calibrazione completata e salvata con valori: X: {x_min}-{x_max}, Y: {y_min}-{y_max}")
                print("\nCalibrazione completata con successo!\n")
            else:
                logger.error("Calibrazione fallita: numero insufficiente di punti.")
                print("\nCalibrazione fallita. Riprova.\n")
        except Exception as e:
            logger.error(f"Errore durante il completamento della calibrazione: {e}")
            print("\nCalibrazione fallita a causa di un errore. Riprova.\n")
            
            # Ripristina i valori di default per sicurezza
            self.config["calibration"]["x_min"] = self.config.DEFAULT_CONFIG["calibration"]["x_min"]
            self.config["calibration"]["x_max"] = self.config.DEFAULT_CONFIG["calibration"]["x_max"]
            self.config["calibration"]["y_min"] = self.config.DEFAULT_CONFIG["calibration"]["y_min"]
            self.config["calibration"]["y_max"] = self.config.DEFAULT_CONFIG["calibration"]["y_max"]
            self.config.save()
        finally:
            self.calibration_active = False
            self.config["calibration"]["active"] = False


def parse_arguments():
    """Analizza gli argomenti della linea di comando."""
    parser = argparse.ArgumentParser(description="HandPal - Controllo del mouse con i gesti delle mani")
    
    parser.add_argument('--device', type=int, help='ID del dispositivo webcam (default: 0)')
    parser.add_argument('--width', type=int, help='Larghezza acquisizione webcam')
    parser.add_argument('--height', type=int, help='Altezza acquisizione webcam')
    parser.add_argument('--debug', action='store_true', help='Abilita modalità debug')
    parser.add_argument('--no-flip', dest='flip_camera', action='store_false', help='Disabilita ribaltamento immagine webcam')
    parser.add_argument('--calibrate', action='store_true', help='Avvia la calibrazione all\'avvio')
    parser.add_argument('--reset-config', action='store_true', help='Ripristina configurazione predefinita')
    
    return parser.parse_args()


def main():
    """Funzione principale dell'applicazione."""
    try:
        # Analizza argomenti
        args = parse_arguments()
        
        # Ripristina configurazione se richiesto
        if hasattr(args, 'reset_config') and args.reset_config:
            config_path = os.path.expanduser("~/.handpal_config.json")
            if os.path.exists(config_path):
                os.remove(config_path)
                print("Configurazione ripristinata ai valori predefiniti.")
        
        # Inizializza configurazione
        config = Config(args)
        
        # Crea e avvia l'app
        app = HandPal(config)
        
        # Imposta modalità debug se richiesto
        if hasattr(args, 'debug') and args.debug:
            app.debug_mode = True
            logger.info("Modalità debug attivata")
        
        # Avvia l'app
        if app.start():
            # Avvia calibrazione se richiesto
            if hasattr(args, 'calibrate') and args.calibrate:
                app.start_calibration()
            
            # Mantieni in esecuzione finché non viene fermato
            try:
                while app.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            finally:
                app.stop()
        
    except Exception as e:
        logger.error(f"Errore nell'esecuzione dell'applicazione: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())