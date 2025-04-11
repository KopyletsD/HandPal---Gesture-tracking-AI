import logging

logger = logging.getLogger(__name__)

class MotionSmoother:
    def __init__(self, config):
        self.config = config # Use injected config
        self.last_smooth_pos = None # Stores last smoothed (x, y) in SCREEN pixels
        self._update_settings()

    def _update_settings(self):
        """Update smoothing parameters from config."""
        factor = max(0.01, min(0.99, self.config.get("smoothing_factor", 0.7)))
        self.alpha = 1.0 - factor # Alpha for new position (lower alpha = more smoothing)
        # Inactive zone threshold (squared, in normalized screen coordinates)
        inactive_zone_norm = self.config.get("inactivity_zone", 0.015)
        self.inactive_zone_sq_norm = inactive_zone_norm**2
        logger.debug(f"Smoother updated: alpha={self.alpha:.2f}, inactive_zone={inactive_zone_norm:.3f}")

    def update(self, target_x_px, target_y_px, screen_w, screen_h):
        """
        Smooths the target position (in screen pixels).
        Returns the smoothed (x, y) in screen pixels.
        """
        target_px = (target_x_px, target_y_px)

        if self.last_smooth_pos is None:
            # First update, or after reset
            self.last_smooth_pos = target_px
            return target_px

        if screen_w <= 0 or screen_h <= 0:
             logger.warning("Invalid screen dimensions for smoother.")
             return self.last_smooth_pos # Return last known good position

        # --- Inactivity Zone Check (using normalized coordinates) ---
        # Convert current target and last smoothed pos to normalized screen coords
        last_norm_x = self.last_smooth_pos[0] / screen_w
        last_norm_y = self.last_smooth_pos[1] / screen_h
        target_norm_x = target_x_px / screen_w
        target_norm_y = target_y_px / screen_h

        # Calculate squared distance in normalized space
        dist_sq_norm = (target_norm_x - last_norm_x)**2 + (target_norm_y - last_norm_y)**2

        # If movement is smaller than the threshold, return the last position
        if dist_sq_norm < self.inactive_zone_sq_norm:
            # logger.debug("Movement within inactivity zone, returning last pos.") # DEBUG
            return self.last_smooth_pos

        # --- Apply Exponential Smoothing ---
        self._update_settings() # Re-read config in case it changed dynamically

        # Apply smoothing formula in PIXEL space
        smooth_x = int(self.alpha * target_x_px + (1.0 - self.alpha) * self.last_smooth_pos[0])
        smooth_y = int(self.alpha * target_y_px + (1.0 - self.alpha) * self.last_smooth_pos[1])

        # Clamp to screen bounds (safety check)
        smooth_x = max(0, min(smooth_x, screen_w - 1))
        smooth_y = max(0, min(smooth_y, screen_h - 1))

        self.last_smooth_pos = (smooth_x, smooth_y)
        return self.last_smooth_pos

    def reset(self):
        """Resets the smoother state."""
        self.last_smooth_pos = None
        logger.debug("MotionSmoother reset.")

    def set_config(self, config):
         """Allow updating the config dynamically."""
         self.config = config
         self._update_settings()