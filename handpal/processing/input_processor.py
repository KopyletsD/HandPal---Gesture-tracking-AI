import time
import logging
import numpy as np
from pynput.mouse import Button, Controller # For gesture return types

# Import other processing components
from .smoothing import MotionSmoother
from .gestures import GestureRecognizer

logger = logging.getLogger(__name__)
mouse = Controller()
class HandInputProcessor:
    def __init__(self, config, smoother: MotionSmoother, gesture_recognizer: GestureRecognizer):
        self.config = config
        self.smoother = smoother
        self.gesture_recognizer = gesture_recognizer

        # Screen dimensions needed for mapping (updated externally)
        self.screen_width = 1920 # Default, should be updated
        self.screen_height = 1080

        # Internal state for processing
        self.last_right_hand_lm_norm = None # Store raw normalized landmark [0..1] of right index tip
        self._last_proc_dims = (0, 0) # Dimensions of the frame results were processed on
        self.last_cursor_target_px = None # Last calculated target (x, y) before smoothing

        # Menu trigger state (internal logic, result passed up)
        self.menu_trigger_hovering = False
        self._menu_trigger_enter_time = 0
        self.MENU_HOVER_DELAY = 0.3 # Seconds to hover before activation

        # Debugging info
        self.debug_info = {
             "map_raw": "-", "map_cal": "-", "map_px": "-", "map_smooth": "-",
             "pose_L": "U", "pose_R": "U",
             "gest_state": "Idle", "gest_output": None,
             "menu_check": "N/A", "menu_dist_sq": -1.0, "menu_zone_r_sq": -1.0,
             "hand_norm": None, "hand_disp_px": None
        }


    def update_screen_dimensions(self, width, height):
        """Update screen dimensions used for mapping."""
        if width > 0 and height > 0:
            if self.screen_width != width or self.screen_height != height:
                logger.info(f"Screen dimensions updated to {width}x{height}")
                self.screen_width = width
                self.screen_height = height
                self.smoother.reset() # Reset smoother if screen size changes
        else:
            logger.warning(f"Attempted to set invalid screen dimensions: {width}x{height}")

    def _map_to_screen(self, norm_x, norm_y):
        """Maps normalized hand coordinates (0-1) to screen pixels."""
        if self.screen_width <= 0 or self.screen_height <= 0:
             return None # Cannot map without valid screen size

        sw, sh = self.screen_width, self.screen_height
        self.debug_info["map_raw"] = f"{norm_x:.3f},{norm_y:.3f}" # Log raw input

        # --- Apply Calibration ---
        calibrated_x, calibrated_y = norm_x, norm_y
        if self.config.get("calibration.enabled", False):
            x_min = self.config.get("calibration.x_min", 0.0)
            x_max = self.config.get("calibration.x_max", 1.0)
            y_min = self.config.get("calibration.y_min", 0.0)
            y_max = self.config.get("calibration.y_max", 1.0)

            x_range = x_max - x_min
            y_range = y_max - y_min

            # Avoid division by zero and ensure valid range
            if x_range > 0.01 and y_range > 0.01:
                # Clamp input normalized coords to the calibration bounds
                clamped_x = max(x_min, min(norm_x, x_max))
                clamped_y = max(y_min, min(norm_y, y_max))
                # Remap the clamped value to 0-1 within the calibration range
                calibrated_x = (clamped_x - x_min) / x_range
                calibrated_y = (clamped_y - y_min) / y_range
            else:
                 # Use raw if calibration range is invalid
                 calibrated_x, calibrated_y = norm_x, norm_y

        self.debug_info["map_cal"] = f"{calibrated_x:.3f},{calibrated_y:.3f}" # Log calibrated

        # --- Expand to Screen with Margin ---
        # Apply screen margin (expand the 0-1 range)
        margin = self.config.get("calibration.screen_margin", 0.1)
        expanded_x = calibrated_x * (1.0 + 2.0 * margin) - margin
        expanded_y = calibrated_y * (1.0 + 2.0 * margin) - margin

        # Convert expanded normalized coords to screen pixels
        screen_x = int(expanded_x * sw)
        screen_y = int(expanded_y * sh)

        # Clamp final pixel coordinates to screen boundaries
        screen_x = max(0, min(screen_x, sw - 1))
        screen_y = max(0, min(screen_y, sh - 1))

        self.debug_info["map_px"] = f"{screen_x},{screen_y}" # Log mapped pixels
        return screen_x, screen_y

    def _get_right_hand_pos_in_display_pixels(self, display_w, display_h):
        """
        Calculates the estimated pixel coordinates of the right hand index tip
        on the *display window* (passed in dims). Returns (x, y) or None.
        Uses self.last_right_hand_lm_norm (normalized on process frame).
        """
        if self.last_right_hand_lm_norm is None: return None
        if display_w <= 0 or display_h <= 0: return None # Need valid display dims
        proc_w, proc_h = self._last_proc_dims
        if proc_w <= 0 or proc_h <= 0: return None # Need valid process dims

        norm_x_proc, norm_y_proc = self.last_right_hand_lm_norm

        # Convert normalized coords (relative to process frame) to pixels on process frame
        pixel_x_proc = norm_x_proc * proc_w
        pixel_y_proc = norm_y_proc * proc_h

        # Calculate scaling factors from process frame to display frame
        scale_x = display_w / proc_w
        scale_y = display_h / proc_h

        # Scale pixel coordinates to the display frame size
        display_x = int(pixel_x_proc * scale_x)
        display_y = int(pixel_y_proc * scale_y)

        # Clamp to display bounds
        display_x = max(0, min(display_x, display_w - 1))
        display_y = max(0, min(display_y, display_h - 1))

        self.debug_info["hand_disp_px"] = (display_x, display_y)
        return (display_x, display_y)

    def _check_menu_trigger(self, menu_zone_px, display_dims):
        """
        Checks if right hand index (in display pixels) is inside the trigger zone.
        Args:
            menu_zone_px (dict): Dict with 'cx', 'cy', 'radius_sq', 'is_valid' in DISPLAY pixels.
            display_dims (tuple): (width, height) of the display window.
        Returns:
            bool: True if the menu should be activated THIS frame, False otherwise.
        """
        if not menu_zone_px or not menu_zone_px.get("is_valid", False):
            if self.menu_trigger_hovering: logger.debug("Menu trigger deactivated (Zone Invalid).")
            self.menu_trigger_hovering = False; self._menu_trigger_enter_time = 0
            self.debug_info["menu_check"] = "Off (Zone Invalid)"
            return False

        display_w, display_h = display_dims
        hand_pos_display_px = self._get_right_hand_pos_in_display_pixels(display_w, display_h)

        if hand_pos_display_px is None:
            if self.menu_trigger_hovering: logger.debug("Menu trigger deactivated (Hand Lost).")
            self.menu_trigger_hovering = False; self._menu_trigger_enter_time = 0
            self.debug_info["menu_check"] = "Off (No Hand)"
            return False

        # Compare hand position (display pixels) with zone (display pixels)
        mx_px, my_px = mouse.position
        hx_px, hy_px = hand_pos_display_px
        cx_px = menu_zone_px["cx"]
        cy_px = menu_zone_px["cy"]
        radius_sq_px = menu_zone_px["radius_sq"]
        self.debug_info["menu_zone_r_sq"] = radius_sq_px # Log for debug

        dist_sq_px = (hx_px - cx_px)**2 + (hy_px - cy_px)**2
        is_inside = dist_sq_px < radius_sq_px
        self.debug_info["menu_dist_sq"] = dist_sq_px # Log for debug
        activate_now = False
        """
        if is_inside:
            if not Hovering: # Just entered the zone
                Hovering = True
                EnterTime = now
                logger.debug(f"Menu Trigger Zone Entered. HandPx={hand_pos_display_px}")
                self.debug_info["menu_check"] = "Entered"
            else: # Already inside, check hover time
                hover_time = now - EnterTime
                print(f"Hovering: {hover_time:.2f}s")
                if hover_time >= self.MENU_HOVER_DELAY:
                    # Hover time met, signal activation
                    # Only activate ONCE per entry+hover period
                    #if self.debug_info["menu_check"] != "ACTIVATE!":
                    activate_now = True
                    logger.debug(f"Menu Trigger HOVER MET ({hover_time:.2f}s). Activating.")
                    self.debug_info["menu_check"] = "ACTIVATE!"
                else:
                    # Still hovering, update debug info
                    self.debug_info["menu_check"] = f"Hover {hover_time:.1f}s"
        else: # Not inside the zone
            if Hovering:
                logger.debug("Menu Trigger Zone Exited.")
                Hovering = False # Reset hover state
            self._menu_trigger_enter_time = 0
            self.debug_info["menu_check"] = "Off (Outside)"
        """
        if is_inside:
            activate_now = True
        return activate_now

    def process(self, detection_results, proc_dims, display_dims, menu_trigger_zone_px):
        """
        Processes raw detection results to produce actionable outputs.
        Args:
            detection_results: MediaPipe Hands result object.
            proc_dims (tuple): (width, height) of the frame used for detection.
            display_dims (tuple): (width, height) of the OpenCV display window.
            menu_trigger_zone_px (dict): Location/size of menu trigger in display pixels.
        Returns:
            dict: A dictionary containing processed information:
                'landmarks_l': Left hand landmarks or None
                'landmarks_r': Right hand landmarks or None
                'pose_l': Detected pose string for left hand
                'pose_r': Detected pose string for right hand
                'cursor_target_px': Target (x, y) pixel coords before smoothing, or None
                'cursor_smooth_px': Smoothed (x, y) pixel coords, or None
                'click_event': 'click' or 'double_click' or None
                'scroll_amount': Vertical scroll amount (float) or None
                'activate_menu': Boolean, True if menu should be shown this frame
                'calib_norm_pos': Normalized (0-1) right hand pos for calibration, or None
                'debug_info': Dictionary with internal processing values for overlay
        """
        self._last_proc_dims = proc_dims # Store dimensions used for this result set
        display_w, display_h = display_dims # For calculating hand pos on display

        # Reset results for this frame
        output = {
            'landmarks_l': None, 'landmarks_r': None,
            'pose_l': 'U', 'pose_r': 'U',
            'cursor_target_px': None, 'cursor_smooth_px': None,
            'click_event': None, 'scroll_amount': None,
            'activate_menu': False,
            'calib_norm_pos': None, # Normalized pos for calibration step processing
            'debug_info': self.debug_info # Pass internal debug state
        }
        # Reset per-frame debug info that needs clearing
        self.debug_info["gest_output"] = None
        self.debug_info["hand_norm"] = None
        self.debug_info["hand_disp_px"] = None

        # --- Extract Landmarks ---
        self.last_right_hand_lm_norm = None # Reset for this frame
        if detection_results and detection_results.multi_hand_landmarks and detection_results.multi_handedness:
            for i, hand_lm in enumerate(detection_results.multi_hand_landmarks):
                try:
                    classification = detection_results.multi_handedness[i].classification[0]
                    label = classification.label
                    # score = classification.score # Could check score threshold here if needed
                except (IndexError, AttributeError, TypeError):
                    logger.warning(f"Could not get handedness label for hand {i}.")
                    continue

                if label == "Right":
                    output['landmarks_r'] = hand_lm
                    # Get normalized coordinates of index finger tip (landmark 8)
                    try:
                        index_tip = hand_lm.landmark[8]
                        self.last_right_hand_lm_norm = (index_tip.x, index_tip.y)
                        output['calib_norm_pos'] = self.last_right_hand_lm_norm # Use raw norm for calib
                        self.debug_info["hand_norm"] = self.last_right_hand_lm_norm
                    except (IndexError, TypeError):
                        logger.warning("Right hand detected, but index tip landmark missing.")
                        self.last_right_hand_lm_norm = None

                elif label == "Left":
                    output['landmarks_l'] = hand_lm
                    # No specific landmark needed globally for left hand here

        # --- Right Hand Processing (Cursor, Menu Trigger) ---
        if output['landmarks_r'] and self.last_right_hand_lm_norm:
            norm_x, norm_y = self.last_right_hand_lm_norm

            # 1. Map to Screen Coordinates (Target Position)
            target_px = self._map_to_screen(norm_x, norm_y)
            if target_px:
                output['cursor_target_px'] = target_px
                self.last_cursor_target_px = target_px # Store for potential use

                # 2. Apply Motion Smoothing
                smooth_px = self.smoother.update(target_px[0], target_px[1], self.screen_width, self.screen_height)
                output['cursor_smooth_px'] = smooth_px
                self.debug_info["map_smooth"] = f"{smooth_px[0]},{smooth_px[1]}" # Log smoothed

            # 3. Check Menu Trigger (using hand pos on display window)
            output['activate_menu'] = self._check_menu_trigger(menu_trigger_zone_px, display_dims)

        else: # No right hand detected or landmark missing
            self.smoother.reset() # Reset smoother if hand is lost
            output['cursor_target_px'] = None
            output['cursor_smooth_px'] = self.smoother.last_smooth_pos # Return last known pos? Or None?
            # Reset menu trigger if hand lost
            if self.menu_trigger_hovering: logger.debug("Menu trigger reset (Right Hand Lost).")
            self.menu_trigger_hovering = False; self._menu_trigger_enter_time = 0
            self.debug_info["menu_check"] = "Off (No R Hand)"
            output['calib_norm_pos'] = None # Cannot calibrate without right hand


        # --- Left Hand Processing (Gestures) ---
        if output['landmarks_l']:
            # 1. Check for Scroll Gesture
            scroll = self.gesture_recognizer.check_scroll_gesture(output['landmarks_l'], "Left")
            if scroll is not None:
                output['scroll_amount'] = scroll
                self.debug_info["gest_output"] = f"Scroll: {scroll:.1f}"

            # 2. Check for Click Gesture (only if not scrolling)
            if output['scroll_amount'] is None:
                click = self.gesture_recognizer.check_thumb_index_click(output['landmarks_l'], "Left")
                if click: # 'click' or 'double_click'
                    output['click_event'] = click
                    self.debug_info["gest_output"] = click.capitalize()

        # --- Pose Detection (Both Hands) ---
        output['pose_l'] = self.gesture_recognizer.detect_hand_pose(output['landmarks_l'], "Left")
        output['pose_r'] = self.gesture_recognizer.detect_hand_pose(output['landmarks_r'], "Right")
        self.debug_info["pose_L"] = output['pose_l']
        self.debug_info["pose_R"] = output['pose_r']

        # Update general gesture state for debug
        internal_gest_state = self.gesture_recognizer.gesture_state.get("active_gesture")
        self.debug_info["gest_state"] = str(internal_gest_state) if internal_gest_state else "Idle"

        return output