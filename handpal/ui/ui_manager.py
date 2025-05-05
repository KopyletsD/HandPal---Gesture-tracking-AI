import cv2 as cv
import numpy as np
import time
import logging
import mediapipe as mp # For drawing utilities

# Import other UI components
from .menu import FloatingMenu

logger = logging.getLogger(__name__)

class UIManager:
    WINDOW_NAME = "HandPal Control"

    def __init__(self, config, tk_root, app_manager):
        self.config = config
        self.tk_root = tk_root # Required for FloatingMenu
        self.app_manager = app_manager # Required for FloatingMenu

        # Initialize MediaPipe drawing utils
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # UI State
        self.window_created = False
        self.last_display_dims = (0, 0) # w, h of the OpenCV window
        self.menu_trigger_zone_px = {"cx":0, "cy":0, "radius_sq":0, "is_valid": False} # In display px
        self.menu_trigger_is_active_display = False # For visual feedback

        # Debug overlay data cache
        self.debug_values_cache = {
            "fps": 0.0, "q_size": 0, "last_activity": time.time(),
            "cursor_history": [], # Store last N smoothed cursor points (screen coords)
            "map_info": {}, "pose_info": {}, "gesture_info": {}, "menu_info": {}
        }
        self.MAX_CURSOR_HISTORY = 50

        # Create Floating Menu instance (requires tk_root)
        if self.tk_root:
             try:
                  self.floating_menu = FloatingMenu(self.tk_root, self.app_manager, self.config)
                  logger.info("FloatingMenu created successfully.")
             except Exception as e:
                  logger.error(f"Failed to create FloatingMenu: {e}")
                  self.floating_menu = None
        else:
             logger.warning("No Tkinter root provided, FloatingMenu disabled.")
             self.floating_menu = None

    def create_window(self):
        """Creates the OpenCV display window."""
        if self.window_created:
            return True
        try:
            cv.namedWindow(self.WINDOW_NAME, cv.WINDOW_NORMAL) # Allow resizing
            # Attempt to set initial size from config if specified
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
                 # If not specified, get size after first show? Or leave as is?
                 logger.info("Display window created with default size.")
                 # We'll get the actual size later in the update method
            self.window_created = True
            return True
        except Exception as e:
            logger.exception(f"Failed to create OpenCV window: {e}")
            self.window_created = False
            return False

    def _draw_landmarks(self, image, landmarks_list):
        """Draws hand landmarks on the image."""
        if not landmarks_list:
            return image
        for hand_landmarks in landmarks_list:
            if hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS, # Use standard connections
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
        return image

    def _draw_menu_trigger_circle(self, image):
        """
        Draws the menu trigger circle on the display image and updates
        its position/size in pixel coordinates (self.menu_trigger_zone_px).
        """
        h, w = image.shape[:2]
        if w == 0 or h == 0:
             self.menu_trigger_zone_px["is_valid"] = False
             return image

        # Define position and appearance (could be configurable)
        radius_px = 30
        center_x_px = w - 50 # Position near right edge
        center_y_px = h // 2 # Position vertically centered
        center_px = (center_x_px, center_y_px)

        # Dynamic color based on hover state (use internal state for display)
        # Pulsating effect for base color
        intensity = 150 + int(50 * np.sin(time.time() * 5))
        base_color = (255, intensity, 0) # Blue/Orange pulsating
        draw_color = (0, 255, 0) if self.menu_trigger_is_active_display else base_color # Green when active

        # Draw filled circle and outline
        cv.circle(image, center_px, radius_px, draw_color, -1) # Filled
        cv.circle(image, center_px, radius_px, (255, 255, 255), 1) # White outline

        # Draw label
        cv.putText(image, "Menu", (center_x_px - 25, center_y_px + radius_px + 15),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

        # --- IMPORTANT: Update the pixel zone info ---
        self.menu_trigger_zone_px["cx"] = center_x_px
        self.menu_trigger_zone_px["cy"] = center_y_px
        self.menu_trigger_zone_px["radius_sq"] = radius_px**2
        self.menu_trigger_zone_px["is_valid"] = True

        return image

    def _draw_calibration_overlay(self, image, calib_state, hand_pos_display_px):
         """Draws calibration instructions and target markers."""
         h, w = image.shape[:2]
         if not calib_state or not calib_state.get('active'): return image

         overlay_color = (255, 255, 255)
         bg_color = (0, 0, 0)
         accent_color = (0, 255, 255) # Yellow/Cyan
         error_color = (0, 0, 255) # Red
         inactive_target_color = (100, 100, 100) # Grey
         active_target_color = error_color # Red target

         font = cv.FONT_HERSHEY_SIMPLEX
         font_scale_med = 0.5
         font_scale_sml = 0.4
         thickness_normal = 1
         thickness_bold = 2

         # Semi-transparent background for text
         bg = image.copy()
         cv.rectangle(bg, (0, 0), (w, 100), bg_color, -1) # Top banner
         alpha = 0.7
         image = cv.addWeighted(bg, alpha, image, 1 - alpha, 0)

         # Instructions
         step = calib_state.get('step', 0)
         total_steps = calib_state.get('total_steps', 4)
         corner_name = calib_state.get('corner_name', 'Unknown')
         cv.putText(image, f"CALIBRATION ({step + 1}/{total_steps}): RIGHT Hand", (10, 25), font, font_scale_med, overlay_color, thickness_normal)
         cv.putText(image, f"Point index to [{corner_name.upper()}] corner", (10, 50), font, font_scale_med, overlay_color, thickness_normal)
         cv.putText(image, "Press SPACEBAR to confirm", (10, 75), font, font_scale_med, accent_color, thickness_bold)
         cv.putText(image, "(ESC to cancel)", (w - 150, 20), font, font_scale_sml, overlay_color, thickness_normal)

         # Draw target circles in corners
         radius = 15
         corners_px = [(radius, radius), (w - radius, radius), (w - radius, h - radius), (radius, h - radius)]
         for i, p in enumerate(corners_px):
             color = active_target_color if step == i else inactive_target_color
             cv.circle(image, p, radius, color, -1) # Filled target
             cv.circle(image, p, radius, overlay_color, 1) # Outline

         # Draw current hand position (if available)
         if hand_pos_display_px:
             cv.circle(image, hand_pos_display_px, 10, (0, 255, 0), -1) # Green circle for hand

         return image

    def _draw_debug_overlay(self, image, debug_data):
        """Draws the detailed debug information panel."""
        h, w = image.shape[:2]

        overlay_color = (255, 255, 255) # White
        bg_color = (0, 0, 0) # Black
        accent_color = (0, 255, 255) # Yellow/Cyan
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        line_type = cv.LINE_AA
        thickness = 1
        line_height = 18 # Pixels per line of text

        panel_height = 110 # Height of the debug panel background
        start_y = h - panel_height

        # Semi-transparent background
        bg = image.copy()
        cv.rectangle(bg, (0, start_y), (w, h), bg_color, -1)
        alpha = 0.7
        image = cv.addWeighted(bg, alpha, image, 1 - alpha, 0)

        # --- Extract Debug Data ---
        # Fallback gracefully if keys are missing
        fps = debug_data.get('fps', 0.0)
        q_size = debug_data.get('q_size', 0)
        last_activity_time = debug_data.get('last_activity', time.time())
        map_info = debug_data.get('map_info', {})
        pose_info = debug_data.get('pose_info', {})
        gesture_info = debug_data.get('gesture_info', {})
        menu_info = debug_data.get('menu_info', {})
        cursor_hist = debug_data.get('cursor_history', [])
        calib_config = self.config.get('calibration', {}) # Get from config

        # --- Draw Text Lines ---
        y = start_y + 15 # Starting y-coordinate for text

        # Line 1: Poses, Gesture State, Menu State
        pose_l = pose_info.get('L', 'U')
        pose_r = pose_info.get('R', 'U')
        gest_state = gesture_info.get('state', 'Idle')
        menu_state = menu_info.get('check', 'N/A')
        text1 = f"L:{pose_l} R:{pose_r} Gest:{gest_state} Menu:{menu_state}"
        cv.putText(image, text1, (10, y), font, font_scale, overlay_color, thickness, line_type); y += line_height

        # Line 2: Mapping Details
        map_raw = map_info.get('raw', '-')
        map_cal = map_info.get('cal', '-')
        map_px = map_info.get('px', '-')
        map_smooth = map_info.get('smooth', '-')
        text2 = f"Map: Raw({map_raw}) Cal({map_cal}) -> Px({map_px}) -> Smooth({map_smooth})"
        cv.putText(image, text2, (10, y), font, font_scale, overlay_color, thickness, line_type); y += line_height

        # Line 3: Calibration Settings
        cx_min = calib_config.get('x_min', 0.0)
        cx_max = calib_config.get('x_max', 1.0)
        cy_min = calib_config.get('y_min', 0.0)
        cy_max = calib_config.get('y_max', 1.0)
        cal_en = calib_config.get('enabled', False)
        text3 = f"Calib X[{cx_min:.2f}-{cx_max:.2f}] Y[{cy_min:.2f}-{cy_max:.2f}] En:{cal_en}"
        cv.putText(image, text3, (10, y), font, font_scale, overlay_color, thickness, line_type); y += line_height

        # Line 4: Timings and Queue
        time_since_last_act = time.time() - last_activity_time
        text4 = f"LastAct:{time_since_last_act:.1f}s ago | QSize:{q_size}"
        cv.putText(image, text4, (10, y), font, font_scale, overlay_color, thickness, line_type); y += line_height

        # Line 5: Menu Debug (Distances)
        # menu_dist = menu_info.get('dist_sq', -1.0)
        # menu_zone_rsq = menu_info.get('zone_r_sq', -1.0)
        # menu_dist_str = f"{menu_dist:.0f}" if menu_dist >= 0 else "N/A"
        # menu_zone_str = f"{menu_zone_rsq:.0f}" if menu_zone_rsq >= 0 else "N/A"
        # text5 = f"MenuChk: DistSq={menu_dist_str} ZoneRSq={menu_zone_str}"
        # cv.putText(image, text5, (10, y), font, font_scale, overlay_color, thickness, line_type); y += line_height


        # --- Draw Cursor History Path ---
        if len(cursor_hist) > 1:
            # Cursor history is in SCREEN coordinates. Need to map to FRAME coordinates.
            screen_w, screen_h = self.config.get('screen_width', 1920), self.config.get('screen_height', 1080) # Get actual screen size
            frame_w, frame_h = w, h

            if screen_w > 0 and screen_h > 0 and frame_w > 0 and frame_h > 0:
                 # Convert list of tuples to numpy array
                 pts_screen = np.array(list(cursor_hist), dtype=np.int32)
                 # Scale points from screen coords to frame coords
                 pts_frame = pts_screen.copy()
                 pts_frame[:, 0] = pts_frame[:, 0] * frame_w // screen_w
                 pts_frame[:, 1] = pts_frame[:, 1] * frame_h // screen_h
                 # Clip to frame boundaries just in case
                 pts_frame = np.clip(pts_frame, [0, 0], [frame_w - 1, frame_h - 1])
                 # Draw the polyline
                 cv.polylines(image, [pts_frame], isClosed=False, color=accent_color, thickness=1)

        return image


    def update(self, frame, processed_data, calib_state, debug_mode=False, fps=0.0, queue_size=0):
        """
        Updates the display with the latest frame and overlays.
        Args:
            frame (np.ndarray): The current camera frame.
            processed_data (dict): Output from HandInputProcessor.
            calib_state (dict): Output from CalibrationManager.get_state().
            debug_mode (bool): Whether to show the debug overlay.
            fps (float): Current main loop FPS for display.
            queue_size (int): Current size of the detection queue.
        Returns:
            tuple: (display_frame, current_display_dims, menu_trigger_zone_px)
                   Returns the frame ready for display, its dimensions, and the trigger zone info.
                   Returns (None, (0,0), default_zone) on error or no window.
        """
        if not self.window_created:
            logger.error("Cannot update UI: Window not created.")
            # Return structure consistent with success case but with None frame
            return None, (0,0), self.menu_trigger_zone_px

        if frame is None:
             logger.warning("Cannot update UI: Received None frame.")
             # What to display? Last frame? Black screen? For now, skip display.
             # Return last known dimensions if available
             return None, self.last_display_dims, self.menu_trigger_zone_px


        # --- Get Data from Processor Output ---
        landmarks_l = processed_data.get('landmarks_l')
        landmarks_r = processed_data.get('landmarks_r')
         # Get menu trigger hover state for visual feedback
        proc_debug_info = processed_data.get('debug_info', {})
        menu_check_state = proc_debug_info.get("menu_check", "Off")
        
        

        proc_debug_info = processed_data.get('debug_info', {})
        # Get hand position ON THE DISPLAY FRAME (calculated by input processor)
        hand_pos_display_px = proc_debug_info.get("hand_disp_px")
        # Get menu trigger hover state for visual feedback
        menu_check_state = proc_debug_info.get("menu_check", "Off")
        self.menu_trigger_is_active_display = not menu_check_state.startswith("Off")

        # --- Prepare Frame Copy for Drawing ---
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        self.last_display_dims = (w, h) # Update last known dimensions

        # --- Draw Overlays ---
        # 1. Landmarks (optional, could be toggled)
        # Make a list of landmarks to draw
        all_landmarks = [lm for lm in [landmarks_l, landmarks_r] if lm is not None]
        display_frame = self._draw_landmarks(display_frame, all_landmarks)

        # 2. Menu Trigger Circle (Always drawn, updates self.menu_trigger_zone_px)
        display_frame = self._draw_menu_trigger_circle(display_frame)

        # 3. Calibration Overlay (if active)
        if calib_state and calib_state.get('active'):
            display_frame = self._draw_calibration_overlay(display_frame, calib_state, hand_pos_display_px)
        # 4. Normal/Debug Overlay (if not calibrating)
        else:
            # Basic Info Line
            info_text = "C:Calibrate | D:Debug | Q:Exit | M:Menu | Hover circle->Menu"
            cv.putText(display_frame, info_text, (10, h - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)

            # FPS Counter
            cv.putText(display_frame, f"FPS: {fps:.1f}", (w - 80, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

            # Debug Panel (if enabled)
            if debug_mode:
                # Update debug cache
                self.debug_values_cache['fps'] = fps
                self.debug_values_cache['q_size'] = queue_size
                if processed_data.get('click_event') or processed_data.get('scroll_amount'):
                     self.debug_values_cache['last_activity'] = time.time()
                if 'cursor_smooth_px' in processed_data and processed_data['cursor_smooth_px']:
                     hist = self.debug_values_cache['cursor_history']
                     hist.append(processed_data['cursor_smooth_px'])
                     if len(hist) > self.MAX_CURSOR_HISTORY: hist.pop(0)

                # Populate debug data from processor and other sources
                self.debug_values_cache['map_info'] = proc_debug_info # Contains map stages
                self.debug_values_cache['pose_info'] = {"L": proc_debug_info.get("pose_L","U"), "R": proc_debug_info.get("pose_R","U")}
                self.debug_values_cache['gesture_info'] = {"state": proc_debug_info.get("gest_state","Idle"), "output": proc_debug_info.get("gest_output")}
                self.debug_values_cache['menu_info'] = {
                    "check": proc_debug_info.get("menu_check","N/A"),
                    "dist_sq": proc_debug_info.get("menu_dist_sq",-1.0),
                    "zone_r_sq": proc_debug_info.get("menu_zone_r_sq", -1.0) # Get from internal state
                }

                display_frame = self._draw_debug_overlay(display_frame, self.debug_values_cache)


        # --- Resize if Necessary ---
        final_frame = display_frame
        target_w = self.config.get('display_width')
        target_h = self.config.get('display_height')
        if target_w and target_h and (w != target_w or h != target_h):
            try:
                final_frame = cv.resize(display_frame, (target_w, target_h), interpolation=cv.INTER_LINEAR)
                self.last_display_dims = (target_w, target_h) # Update dims after resize
            except cv.error as e:
                logger.error(f"Failed to resize frame to {target_w}x{target_h}: {e}")
                # Keep original size frame if resize fails
                final_frame = display_frame
                self.last_display_dims = (w, h)

        # --- Show Frame ---
        try:
             cv.imshow(self.WINDOW_NAME, final_frame)
        except cv.error as e:
            # Handle potential errors if window was closed unexpectedly etc.
            logger.error(f"cv.imshow error: {e}")
            if "NULL window" in str(e) or "Invalid window handle" in str(e):
                 logger.warning("OpenCV window seems to be closed. Attempting to recreate.")
                 self.window_created = False # Mark as not created
                 self.create_window()      # Try to recreate it
                 # Might need to skip showing this frame if recreate fails immediately
                 return None, (0,0), self.menu_trigger_zone_px


        # Return the potentially resized frame, its dims, and the trigger zone info
        return final_frame, self.last_display_dims, self.menu_trigger_zone_px


    def show_menu(self):
        if self.floating_menu:
            self.floating_menu.show()
        else:
            print("Floating menu not available.")

    def hide_menu(self):
        if self.floating_menu:
            self.floating_menu.hide()

    def toggle_menu(self):
        if self.floating_menu:
            self.floating_menu.toggle()

    def destroy(self):
        """Cleans up UI resources."""
        logger.info("Destroying UI Manager...")
        if self.floating_menu:
             self.floating_menu.destroy()
             self.floating_menu = None
        if self.window_created:
             try:
                  cv.destroyWindow(self.WINDOW_NAME)
                  logger.debug(f"OpenCV window '{self.WINDOW_NAME}' destroyed.")
             except Exception as e:
                  logger.error(f"Error destroying OpenCV window: {e}")
        self.window_created = False
        # Call destroyAllWindows() maybe once at the very end in main controller?
        # cv.destroyAllWindows() # Destroy any other potential CV windows