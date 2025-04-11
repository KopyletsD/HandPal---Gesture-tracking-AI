import logging
import time

logger = logging.getLogger(__name__)

class CalibrationManager:
    def __init__(self, config):
        self.config = config # Injected config for saving results
        self.is_active = False
        self.points_normalized = [] # Store collected points (normalized 0-1)
        self.current_step = 0
        self.corners = ["top-left", "top-right", "bottom-right", "bottom-left"]
        self.start_time = 0

    def start(self):
        """Initiates the calibration process."""
        if self.is_active:
            logger.warning("Calibration already active.")
            return False

        logger.info("Starting calibration sequence...")
        print("\n--- CALIBRATION START ---") # User feedback
        self.is_active = True
        self.config.set("calibration.active", True) # Set flag in shared config
        self.points_normalized = []
        self.current_step = 0
        self.start_time = time.time()
        # The UI Manager will use self.is_active and self.current_step to display prompts
        # Announce first step
        self._print_step_instructions()
        return True

    def cancel(self):
        """Cancels the ongoing calibration."""
        if not self.is_active:
            return False

        logger.info("Calibration cancelled by user.")
        print("\n--- CALIBRATION CANCELLED ---") # User feedback
        self.is_active = False
        self.config.set("calibration.active", False) # Reset flag
        self.points_normalized = []
        self.current_step = 0
        return True

    def process_step(self, hand_pos_normalized):
        """
        Processes a calibration step, recording the provided hand position.
        Args:
            hand_pos_normalized (tuple): (x, y) normalized hand coordinates (0-1).
        Returns:
            bool: True if calibration completed, False otherwise.
        """
        if not self.is_active:
            logger.warning("process_step called while calibration is not active.")
            return False

        if hand_pos_normalized is None or len(hand_pos_normalized) != 2:
            logger.warning("Invalid hand position received for calibration step.")
            print("(!) Right hand position not detected or invalid. Please try again.")
            return False

        # Record the normalized point
        x_norm, y_norm = hand_pos_normalized
        self.points_normalized.append((x_norm, y_norm))
        corner_name = self.corners[self.current_step]

        logger.info(f"Calibration point {self.current_step + 1}/{len(self.corners)} ({corner_name}) captured: ({x_norm:.3f}, {y_norm:.3f})")
        print(f"-> Point {self.current_step + 1}/{len(self.corners)} ({corner_name}) captured.")

        self.current_step += 1

        if self.current_step >= len(self.corners):
            return self.complete() # All points collected, try to complete
        else:
            # Print instructions for the next step
            self._print_step_instructions()
            return False # Calibration ongoing

    def complete(self):
        """Finalizes calibration using the collected points."""
        if not self.is_active:
            logger.error("complete() called but calibration not active.")
            return False
        if len(self.points_normalized) != len(self.corners):
            logger.error(f"Calibration completion failed: Expected {len(self.corners)} points, got {len(self.points_normalized)}.")
            print(f"(!) ERROR: Incorrect number of points ({len(self.points_normalized)}/{len(self.corners)}). Calibration cancelled.")
            self.cancel()
            return False

        logger.info("Completing calibration...")
        try:
            # Extract min/max from normalized points
            xs = [p[0] for p in self.points_normalized]
            ys = [p[1] for p in self.points_normalized]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # Basic validation of the calibrated area size
            if (x_max - x_min < 0.05) or (y_max - y_min < 0.05):
                logger.warning("Calibration area is very small. Results might be inaccurate.")
                print("(!) WARNING: Calibration area seems small. Tracking might be less accurate.")

            # Update the config object (this triggers validation within Config class)
            # Set enabled=True explicitly
            self.config.set("calibration.enabled", True)
            self.config.set("calibration.x_min", x_min)
            self.config.set("calibration.x_max", x_max)
            self.config.set("calibration.y_min", y_min)
            self.config.set("calibration.y_max", y_max)

            # Save the updated configuration
            self.config.save()

            elapsed = time.time() - self.start_time
            logger.info(f"Calibration completed and saved successfully in {elapsed:.1f}s.")
            logger.info(f"New bounds: X=[{x_min:.3f}-{x_max:.3f}], Y=[{y_min:.3f}-{y_max:.3f}]")
            print("\n--- CALIBRATION SAVED ---") # User feedback

        except Exception as e:
            logger.exception("Error during calibration completion or saving.")
            print("(!) ERROR: Failed to save calibration data.")
            # Disable calibration in config if save failed
            self.config.set("calibration.enabled", False)
            self.cancel() # Go back to non-calibrated state
            return False
        finally:
            # Ensure state is reset regardless of success/failure after attempt
            self.is_active = False
            self.config.set("calibration.active", False)
            self.points_normalized = []
            self.current_step = 0

        return True # Completed successfully

    def get_state(self):
         """Returns the current calibration state for UI display."""
         return {
             "active": self.is_active,
             "step": self.current_step,
             "total_steps": len(self.corners),
             "corner_name": self.corners[self.current_step] if self.is_active and self.current_step < len(self.corners) else None
         }

    def _print_step_instructions(self):
         """Prints instructions for the current calibration step to the console."""
         if self.is_active and self.current_step < len(self.corners):
              corner_name = self.corners[self.current_step]
              print(f"\nStep {self.current_step + 1}/{len(self.corners)}: Point RIGHT index finger towards the [{corner_name.upper()}] corner.")
              print("   Press [SPACEBAR] to capture the point.")
              print("   Press [ESC] to cancel calibration.")