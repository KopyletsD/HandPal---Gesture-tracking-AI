import numpy as np
from collections import deque
import time
import logging
from pynput.mouse import Button # Keep Button here for state tracking

logger = logging.getLogger(__name__)

class GestureRecognizer:
    def __init__(self, config):
        self.config = config # Use injected config
        self.last_positions = {}
        self.gesture_state = {
            "scroll_active": False,
            "last_click_time": 0,
            "last_click_button": None,
            "scroll_history": deque(maxlen=5),
            "active_gesture": None, # Internal state: 'click', 'scroll', None
            "last_pose": {"Left": "U", "Right": "U"}, # Keep track per hand
            "pose_stable_count": {"Left": 0, "Right": 0}
        }
        self.POSE_STABILITY_THRESHOLD = 3 # How many frames pose must be stable

    def _dist(self, p1, p2):
        # Simple Euclidean distance for landmarks
        if None in [p1, p2] or not all(hasattr(p, 'x') for p in [p1,p2]):
            return float('inf')
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def check_thumb_index_click(self, hand_lm, handedness):
        """Checks for thumb-index tip touch on the LEFT hand. Returns 'click' or 'double_click'."""
        if handedness != "Left" or hand_lm is None:
            return None # Only left hand clicks
        # If scrolling is active, disable clicks
        if self.gesture_state["scroll_active"]:
             return None

        try:
            thumb_tip = hand_lm.landmark[4]
            index_tip = hand_lm.landmark[8]
        except (IndexError, TypeError):
            logger.warning("Missing thumb/index landmarks for click check.")
            return None # Landmarks not available

        dist = self._dist(thumb_tip, index_tip)
        now = time.time()
        click_threshold = self.config.get("gesture_sensitivity", 0.03) # Get from config
        gesture = None

        if dist < click_threshold:
            # If currently not clicking, check cooldown and double click potential
            if self.gesture_state["active_gesture"] != "click":
                cooldown = self.config.get("click_cooldown", 0.5)
                if (now - self.gesture_state["last_click_time"]) > cooldown:
                    double_click_window = self.config.get("gesture_settings.double_click_time", 0.35)
                    # Check for double click: same button, within time window
                    if (now - self.gesture_state["last_click_time"]) < double_click_window and \
                       self.gesture_state["last_click_button"] == Button.left:
                        gesture = "double_click"
                        logger.debug("Double click detected")
                    else:
                        gesture = "click"
                        logger.debug("Single click detected")

                    # Update state ONLY if a gesture was detected
                    self.gesture_state["last_click_time"] = now
                    self.gesture_state["last_click_button"] = Button.left # Assuming left click for now
                    self.gesture_state["active_gesture"] = "click" # Mark click active
                # else: still in cooldown
        else:
            # Finger distance increased, reset active click state if it was active
            if self.gesture_state["active_gesture"] == "click":
                self.gesture_state["active_gesture"] = None
                logger.debug("Click gesture ended (fingers separated)")

        return gesture # Return the detected gesture type or None

    def check_scroll_gesture(self, hand_landmarks, handedness):
         """Detects scroll gesture (index/middle up) on LEFT hand. Returns scroll amount."""
         if handedness != "Left":
             return None # Only left hand scrolls
         # Prevent scrolling if clicking
         if self.gesture_state["active_gesture"] == "click":
             return None

         try:
             index_tip = hand_landmarks.landmark[8]
             middle_tip = hand_landmarks.landmark[12]
             index_mcp = hand_landmarks.landmark[5]
             middle_mcp = hand_landmarks.landmark[9]
             ring_tip = hand_landmarks.landmark[16]
             pinky_tip = hand_landmarks.landmark[20]
         except (IndexError, TypeError):
              logger.warning("Missing landmarks for scroll check.")
              return None

         # Conditions for scroll pose
         index_extended = index_tip.y < index_mcp.y - 0.02 # Finger tips higher than base (y decreases upwards)
         middle_extended = middle_tip.y < middle_mcp.y - 0.02
         fingers_close = abs(index_tip.x - middle_tip.x) < 0.08 # Index and middle tips close horizontally
         ring_pinky_folded = (ring_tip.y > middle_mcp.y) and (pinky_tip.y > index_mcp.y) # Ring/pinky tips lower than middle base

         is_scroll_pose = index_extended and middle_extended and fingers_close and ring_pinky_folded
         scroll_amount = None

         if is_scroll_pose:
             # Pose detected, activate scroll mode if not already active
             if not self.gesture_state["scroll_active"]:
                 self.gesture_state["scroll_active"] = True
                 self.gesture_state["scroll_history"].clear()
                 self.gesture_state["active_gesture"] = "scroll" # Mark scroll active
                 self.last_positions['scroll_y'] = index_tip.y # Store initial Y pos
                 logger.debug("Scroll gesture started.")
             else:
                 # Scroll is active, calculate delta Y
                 prev_y = self.last_positions.get('scroll_y', index_tip.y)
                 curr_y = index_tip.y
                 # Positive delta_y means hand moved down (scroll up content typically)
                 # Negative delta_y means hand moved up (scroll down content typically)
                 delta_y = curr_y - prev_y
                 sensitivity = self.config.get("gesture_settings.scroll_sensitivity", 5)

                 # Add to history for smoothing
                 self.gesture_state["scroll_history"].append(delta_y)
                 if len(self.gesture_state["scroll_history"]) > 0:
                     smooth_delta = sum(self.gesture_state["scroll_history"]) / len(self.gesture_state["scroll_history"])
                     # Only trigger scroll if movement is significant enough
                     if abs(smooth_delta) > 0.0005: # Tune this threshold
                         scroll_amount = smooth_delta * sensitivity * -100 # Scale and invert direction
                         # logger.debug(f"Scroll delta: {smooth_delta:.4f}, Amount: {scroll_amount:.1f}") # DEBUG

                 # Update last position for next frame
                 self.last_positions['scroll_y'] = curr_y
         else:
             # Pose not detected, deactivate scroll mode if it was active
             if self.gesture_state["scroll_active"]:
                 self.gesture_state["scroll_active"] = False
                 self.gesture_state["scroll_history"].clear()
                 if 'scroll_y' in self.last_positions: del self.last_positions['scroll_y']
                 if self.gesture_state["active_gesture"] == "scroll":
                     self.gesture_state["active_gesture"] = None # Reset active state
                 logger.debug("Scroll gesture ended.")

         return scroll_amount # Return calculated scroll amount or None

    def detect_hand_pose(self, hand_lm, handedness):
        """Detects basic static hand poses based on finger extension."""
        # Returns a string like "Point", "Open", "Fist", "Two", "Scroll", or "U" (Unknown)
        # Appends "?" if pose is not stable yet.
        if hand_lm is None or handedness not in ["Left", "Right"]: return "U"

        # Check for active scroll gesture first (overrides other poses for left hand)
        if handedness == "Left" and self.gesture_state["scroll_active"]:
            pose = "Scroll"
        else:
            try:
                # Get landmarks needed for pose detection
                lm = hand_lm.landmark
                wrist = lm[0]
                thumb_tip = lm[4]
                index_tip, index_pip = lm[8], lm[6]
                middle_tip, middle_pip = lm[12], lm[10]
                ring_tip, ring_pip = lm[16], lm[14]
                pinky_tip, pinky_pip = lm[20], lm[18]
            except (IndexError, TypeError):
                logger.warning(f"Missing landmarks for pose detection ({handedness}).")
                return "U" # Unknown if landmarks missing

            # Check finger extensions (tip further 'up' than pip joint, adjusted for wrist)
            # A small y_extension_threshold helps account for noise/slight curl
            y_extension_threshold = 0.02 # Relative to wrist-pip distance maybe? Or fixed? Let's try fixed.
            thumb_extended = thumb_tip.y < wrist.y - y_extension_threshold # Thumb up relative to wrist
            index_extended = index_tip.y < index_pip.y - y_extension_threshold
            middle_extended = middle_tip.y < middle_pip.y - y_extension_threshold
            ring_extended = ring_tip.y < ring_pip.y - y_extension_threshold
            pinky_extended = pinky_tip.y < pinky_pip.y - y_extension_threshold

            num_extended_fingers = sum([index_extended, middle_extended, ring_extended, pinky_extended])
            pose = "U" # Default

            if index_extended and num_extended_fingers == 1:
                pose = "Point"
            elif index_extended and middle_extended and num_extended_fingers == 2:
                 pose = "Two" # Or Peace
            elif num_extended_fingers >= 4: # Allow for thumb not being perfectly extended
                pose = "Open"
            elif num_extended_fingers == 0 and not thumb_extended: # All fingers curled, thumb down/in
                 pose = "Fist"
            # Add other poses here if needed (e.g., "Three", "OK")

        # --- Pose Stability Check ---
        last_pose = self.gesture_state["last_pose"].get(handedness, "U")
        stable_count = self.gesture_state["pose_stable_count"].get(handedness, 0)

        if pose == last_pose and pose != "U":
            stable_count += 1
        else: # Pose changed or is Unknown
            stable_count = 0

        # Update state cache
        self.gesture_state["last_pose"][handedness] = pose
        self.gesture_state["pose_stable_count"][handedness] = stable_count

        # Return stable pose, or pose + "?" if not stable yet
        if stable_count >= self.POSE_STABILITY_THRESHOLD or pose == "Scroll": # Scroll is immediate
             return pose
        elif pose != "U": # It's a recognized pose, just not stable
             return pose + "?"
        else: # It's Unknown
             return "U" # Return "U" instead of "U?"