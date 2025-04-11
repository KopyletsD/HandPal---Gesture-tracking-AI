# handpal/ui/ui_manager.py
import cv2 as cv
import numpy as np
import time
import logging
import mediapipe as mp
import tkinter as tk # Needed for TclError check

# Import other UI components
from .menu import FloatingMenu
# Optional type hinting imports
# from ..config import Config
# from ..apps.manager import AppManager
# from ..calibration.manager import CalibrationManager

logger = logging.getLogger(__name__)

class UIManager:
    WINDOW_NAME = "HandPal Control"

    def __init__(self, config, tk_root, app_manager):
        self.config = config
        self.tk_root = tk_root # May be None if Tkinter failed
        self.app_manager = app_manager

        # Initialize MediaPipe drawing utils
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands # Needed for HAND_CONNECTIONS

        # UI State
        self.window_created = False
        self.last_display_dims = (0, 0) # w, h of the OpenCV window
        self.menu_trigger_zone_px = {"cx":0, "cy":0, "radius_sq":0, "is_valid": False}
        self.menu_trigger_is_active_display = False # For visual feedback only

        # Debug overlay data cache
        self.debug_values_cache = {
            "fps": 0.0, "q_size": 0, "last_activity": time.time(),
            "cursor_history": [],
            "map_info": {}, "pose_info": {}, "gesture_info": {}, "menu_info": {}
        }
        self.MAX_CURSOR_HISTORY = 50

        # Create Floating Menu instance
        self.floating_menu = None
        if self.tk_root:
             try:
                  self.floating_menu = FloatingMenu(self.tk_root, self.app_manager, self.config)
                  logger.info("FloatingMenu created successfully.")
             except Exception as e:
                  logger.error(f"Failed to create FloatingMenu: {e}")
                  self.floating_menu = None # Ensure it's None on failure
        else:
             logger.warning("No Tkinter root provided, FloatingMenu disabled.")

    def create_window(self):
        """Creates the OpenCV display window."""
        if self.window_created:
            return True
        try:
            cv.namedWindow(self.WINDOW_NAME, cv.WINDOW_NORMAL | cv.WINDOW_GUI_EXPANDED) # Flags for better GUI behavior
            cv.setWindowProperty(self.WINDOW_NAME, cv.WND_PROP_TOPMOST, 1) # Try to keep on top

            disp_w = self.config.get('display_width')
            disp_h = self.config.get('display_height')
            if disp_w and disp_h:
                 try:
                      cv.resizeWindow(self.WINDOW_NAME, disp_w, disp_h)
                      logger.info(f"Set initial display window size to {disp_w}x{disp_h}")
                      self.last_display_dims = (disp_w, disp_h)
                 except cv.error as e:
                      logger.warning(f"Failed to set initial window size ({disp_w}x{disp_h}): {e}")
            else:
                 logger.info("Display window created with default size.")

            self.window_created = True
            return True
        except Exception as e:
            logger.exception(f"Failed to create OpenCV window '{self.WINDOW_NAME}': {e}")
            self.window_created = False
            return False

    def _draw_landmarks(self, image, landmarks_list):
        """Draws hand landmarks on the image."""
        if not landmarks_list: return image
        for hand_landmarks in landmarks_list:
            if hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
        return image

    def _draw_menu_trigger_circle(self, image):
        """Draws the menu trigger circle and updates its pixel zone info."""
        h, w = image.shape[:2]
        if w <= 0 or h <= 0:
             self.menu_trigger_zone_px["is_valid"] = False; return image

        radius_px = 30
        center_x_px = w - 50
        center_y_px = h // 2
        center_px = (center_x_px, center_y_px)

        # Use internal state for visual feedback color
        intensity = 150 + int(50 * np.sin(time.time() * 5))
        base_color = (255, intensity, 0) # Pulsating blue/orange
        draw_color = (0, 255, 0) if self.menu_trigger_is_active_display else base_color # Green when active

        cv.circle(image, center_px, radius_px, draw_color, -1) # Filled
        cv.circle(image, center_px, radius_px, (255, 255, 255), 1) # Outline
        cv.putText(image, "Menu", (center_x_px - 25, center_y_px + radius_px + 15),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

        # Update the pixel zone info used by InputProcessor
        self.menu_trigger_zone_px["cx"] = center_x_px
        self.menu_trigger_zone_px["cy"] = center_y_px
        self.menu_trigger_zone_px["radius_sq"] = radius_px**2
        self.menu_trigger_zone_px["is_valid"] = True
        return image

    def _draw_calibration_overlay(self, image, calib_state, hand_pos_display_px):
         """Draws calibration instructions and target markers."""
         h, w = image.shape[:2]
         if not calib_state or not calib_state.get('active'): return image

         overlay_color=(255,255,255); bg_color=(0,0,0); accent_color=(0,255,255)
         error_color=(0,0,255); inactive_target=(100,100,100); active_target=error_color
         font=cv.FONT_HERSHEY_SIMPLEX; fsc_med=0.5; fsc_sml=0.4; thick_n=1; thick_b=2

         bg = image.copy(); cv.rectangle(bg, (0,0), (w, 100), bg_color, -1); alpha = 0.7
         image = cv.addWeighted(bg, alpha, image, 1 - alpha, 0)

         step = calib_state.get('step', 0); total_steps = calib_state.get('total_steps', 4)
         corner = calib_state.get('corner_name', 'Unknown')
         cv.putText(image, f"CALIBRATION ({step+1}/{total_steps}): RIGHT Hand", (10, 25), font, fsc_med, overlay_color, thick_n)
         cv.putText(image, f"Point index to [{corner.upper()}] corner", (10, 50), font, fsc_med, overlay_color, thick_n)
         cv.putText(image, "Press SPACEBAR to confirm", (10, 75), font, fsc_med, accent_color, thick_b)
         cv.putText(image, "(ESC to cancel)", (w - 150, 20), font, fsc_sml, overlay_color, thick_n)

         radius=15; corners_px = [(radius, radius), (w-radius, radius), (w-radius, h-radius), (radius, h-radius)]
         for i, p in enumerate(corners_px):
             color = active_target if step == i else inactive_target
             cv.circle(image, p, radius, color, -1); cv.circle(image, p, radius, overlay_color, 1)

         if hand_pos_display_px: cv.circle(image, hand_pos_display_px, 10, (0, 255, 0), -1)
         return image

    def _draw_debug_overlay(self, image, debug_data):
        """Draws the detailed debug information panel."""
        h, w = image.shape[:2]; overlay_c=(255,255,255); bg_c=(0,0,0); accent_c=(0,255,255)
        font=cv.FONT_HERSHEY_SIMPLEX; fsc=0.4; line_t=cv.LINE_AA; thick=1; line_h=18
        panel_h=110; start_y = h - panel_h

        bg = image.copy(); cv.rectangle(bg,(0,start_y),(w,h),bg_c,-1); alpha=0.7
        image=cv.addWeighted(bg,alpha,image,1-alpha,0); y = start_y + 15

        fps=debug_data.get('fps',0.0); q_size=debug_data.get('q_size',0)
        last_act_t=debug_data.get('last_activity',time.time()); map_i=debug_data.get('map_info',{})
        pose_i=debug_data.get('pose_info',{}); gest_i=debug_data.get('gesture_info',{})
        menu_i=debug_data.get('menu_info',{}); cur_hist=debug_data.get('cursor_history',[])
        calib_cfg=self.config.get('calibration',{})

        pose_l=pose_i.get('L','U'); pose_r=pose_i.get('R','U'); gest_s=gest_i.get('state','Idle'); menu_s=menu_i.get('check','N/A')
        t1=f"L:{pose_l} R:{pose_r} Gest:{gest_s} Menu:{menu_s}"; cv.putText(image,t1,(10,y),font,fsc,overlay_c,thick,line_t); y+=line_h

        map_r=map_i.get('raw','-'); map_c=map_i.get('cal','-'); map_p=map_i.get('px','-'); map_s=map_i.get('smooth','-')
        t2=f"Map: Raw({map_r}) Cal({map_c}) -> Px({map_p}) -> Smooth({map_s})"; cv.putText(image,t2,(10,y),font,fsc,overlay_c,thick,line_t); y+=line_h

        cx_min=calib_cfg.get('x_min',0.0); cx_max=calib_cfg.get('x_max',1.0); cy_min=calib_cfg.get('y_min',0.0); cy_max=calib_cfg.get('y_max',1.0); cal_en=calib_cfg.get('enabled',False)
        t3=f"Calib X[{cx_min:.2f}-{cx_max:.2f}] Y[{cy_min:.2f}-{cy_max:.2f}] En:{cal_en}"; cv.putText(image,t3,(10,y),font,fsc,overlay_c,thick,line_t); y+=line_h

        t_last_act=time.time()-last_act_t
        t4=f"LastAct:{t_last_act:.1f}s ago | QSize:{q_size}"; cv.putText(image,t4,(10,y),font,fsc,overlay_c,thick,line_t); y+=line_h

        # Draw Cursor History Path
        if len(cur_hist) > 1:
            screen_w, screen_h = self.config.get('screen_width', 1920), self.config.get('screen_height', 1080)
            if screen_w > 0 and screen_h > 0 and w > 0 and h > 0:
                 pts_screen = np.array(list(cur_hist),dtype=np.int32)
                 pts_frame = pts_screen.copy()
                 pts_frame[:,0] = pts_frame[:,0] * w // screen_w
                 pts_frame[:,1] = pts_frame[:,1] * h // screen_h
                 pts_frame = np.clip(pts_frame,[0,0],[w-1,h-1])
                 cv.polylines(image,[pts_frame],False,accent_c,1)
        return image


    def update(self, frame, processed_data, calib_state, debug_mode=False, fps=0.0, queue_size=0):
        """Updates the display with the latest frame and overlays."""
        if not self.window_created: return None, (0,0), self.menu_trigger_zone_px
        if frame is None: return None, self.last_display_dims, self.menu_trigger_zone_px

        landmarks_l = processed_data.get('landmarks_l')
        landmarks_r = processed_data.get('landmarks_r')
        proc_debug_info = processed_data.get('debug_info', {})
        hand_pos_display_px = proc_debug_info.get("hand_disp_px") # Hand pos on display
        menu_check_state = proc_debug_info.get("menu_check", "Off")
        self.menu_trigger_is_active_display = not menu_check_state.startswith("Off") # Update visual state

        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        self.last_display_dims = (w, h)

        # --- Draw Overlays ---
        all_landmarks = [lm for lm in [landmarks_l, landmarks_r] if lm is not None]
        display_frame = self._draw_landmarks(display_frame, all_landmarks)
        display_frame = self._draw_menu_trigger_circle(display_frame) # Updates zone info

        if calib_state and calib_state.get('active'):
            display_frame = self._draw_calibration_overlay(display_frame, calib_state, hand_pos_display_px)
        else:
            info_text = "C:Calib | D:Debug | R:Reload | Q:Exit | M:Menu | Hover->Menu"
            cv.putText(display_frame, info_text, (10, h-10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv.LINE_AA)
            cv.putText(display_frame, f"FPS: {fps:.1f}", (w-80, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)

            if debug_mode:
                self.debug_values_cache['fps'] = fps; self.debug_values_cache['q_size'] = queue_size
                if processed_data.get('click_event') or processed_data.get('scroll_amount'): self.debug_values_cache['last_activity'] = time.time()
                if 'cursor_smooth_px' in processed_data and processed_data['cursor_smooth_px']:
                     hist = self.debug_values_cache['cursor_history']; hist.append(processed_data['cursor_smooth_px'])
                     if len(hist) > self.MAX_CURSOR_HISTORY: hist.pop(0)
                self.debug_values_cache['map_info'] = proc_debug_info
                self.debug_values_cache['pose_info'] = {"L": proc_debug_info.get("pose_L","U"), "R": proc_debug_info.get("pose_R","U")}
                self.debug_values_cache['gesture_info'] = {"state": proc_debug_info.get("gest_state","Idle"), "output": proc_debug_info.get("gest_output")}
                self.debug_values_cache['menu_info'] = {"check": menu_check_state, "dist_sq": proc_debug_info.get("menu_dist_sq",-1.0), "zone_r_sq": self.menu_trigger_zone_px.get("radius_sq", -1.0)}
                display_frame = self._draw_debug_overlay(display_frame, self.debug_values_cache)

        # --- Resize if Necessary ---
        final_frame = display_frame
        target_w = self.config.get('display_width'); target_h = self.config.get('display_height')
        if target_w and target_h and (w != target_w or h != target_h):
            try:
                final_frame = cv.resize(display_frame, (target_w, target_h), interpolation=cv.INTER_LINEAR)
                self.last_display_dims = (target_w, target_h)
            except cv.error as e:
                logger.error(f"Failed to resize frame: {e}"); self.last_display_dims = (w, h)

        # --- Show Frame ---
        try:
             # Check if window still exists before showing
             if cv.getWindowProperty(self.WINDOW_NAME, cv.WND_PROP_VISIBLE) >= 1:
                  cv.imshow(self.WINDOW_NAME, final_frame)
             else:
                  # This case might mean the user closed the window manually
                  logger.warning(f"OpenCV window '{self.WINDOW_NAME}' not visible or closed.")
                  # Option 1: Recreate the window? (might be annoying)
                  # self.window_created = False; self.create_window()
                  # Option 2: Signal app controller to stop? (Safer)
                  # raise Exception("OpenCV window closed by user") # Let controller handle stop
                  # Option 3: Just log and continue (current behavior)
                  pass
        except cv.error as e:
             logger.error(f"cv.imshow error: {e}")
             # Attempt to handle specific errors if window handle becomes invalid
             if "NULL window" in str(e) or "Invalid window handle" in str(e):
                  logger.warning("OpenCV window handle invalid. Marking as not created.")
                  self.window_created = False # Prevent further attempts until recreated
             # Propagate others? Or just log? For now, just log.
             return None, self.last_display_dims, self.menu_trigger_zone_px # Return no frame on error

        return final_frame, self.last_display_dims, self.menu_trigger_zone_px

    def show_menu(self):
        if self.floating_menu: self.floating_menu.show()
        else: logger.warning("Tried to show menu, but floating_menu is None.")

    def hide_menu(self):
        if self.floating_menu: self.floating_menu.hide()
        else: logger.warning("Tried to hide menu, but floating_menu is None.")

    def toggle_menu(self):
        if self.floating_menu: self.floating_menu.toggle()
        else: logger.warning("Tried to toggle menu, but floating_menu is None.")

    # --- ADDED METHOD ---
    def reload_menu_apps(self):
        """Triggers the floating menu to update its application list."""
        if self.floating_menu:
            logger.debug("UIManager triggering floating_menu.update_apps()")
            try:
                self.floating_menu.update_apps()
            except Exception as e:
                logger.exception("Error occurred during floating_menu.update_apps()")
        else:
            logger.warning("Cannot reload menu apps: floating_menu is None.")
    # --- END OF ADDED METHOD ---

    def destroy(self):
        """Cleans up UI resources."""
        logger.info("Destroying UI Manager...")
        if self.floating_menu:
             self.floating_menu.destroy()
             self.floating_menu = None
        if self.window_created:
             try:
                  # Check property before destroying to avoid error if already closed
                  if cv.getWindowProperty(self.WINDOW_NAME, cv.WND_PROP_VISIBLE) >= 1:
                       cv.destroyWindow(self.WINDOW_NAME)
                       logger.debug(f"OpenCV window '{self.WINDOW_NAME}' destroyed.")
                  else:
                       logger.debug(f"OpenCV window '{self.WINDOW_NAME}' already closed or not visible.")
             except Exception as e:
                  logger.error(f"Error destroying OpenCV window: {e}")
        self.window_created = False
        # Consider calling cv.destroyAllWindows() once in main controller's final cleanup