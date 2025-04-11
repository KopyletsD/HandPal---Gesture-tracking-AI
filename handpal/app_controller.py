import time
import os
import queue
import logging
import threading
import tkinter as tk
import cv2 as cv # For key handling
from pynput.mouse import Controller as MouseController, Button as MouseButton

# Import HandPal components
from .config import Config
from .detection.detector import DetectionThread
from .processing.input_processor import HandInputProcessor
from .processing.smoothing import MotionSmoother # Needed? InputProcessor owns it
from .processing.gestures import GestureRecognizer # Needed? InputProcessor owns it
from .ui.ui_manager import UIManager
from .calibration.manager import CalibrationManager
from .apps.manager import AppManager

logger = logging.getLogger(__name__)

class ApplicationController:
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        self.stop_event = threading.Event()
        self.debug_mode = False # Controlled by key press/args

        # --- Initialize Core Components ---
        logger.info("Initializing application components...")
        self.mouse_controller = MouseController()

        # Processing components (owned by InputProcessor)
        self.smoother = MotionSmoother(config)
        self.gesture_recognizer = GestureRecognizer(config)
        self.input_processor = HandInputProcessor(config, self.smoother, self.gesture_recognizer)

        # App Manager
        self.app_manager = AppManager(config)

        # Calibration Manager
        self.calibration_manager = CalibrationManager(config)

        # Detection Thread and Data Queue
        self.data_queue = queue.Queue(maxsize=2) # Queue for Frame/Results/Dims from detector
        self.detection_thread = None # Thread object, created in start()

        # UI Components
        self._initialize_tkinter() # Sets up self.tk_root
        self.ui_manager = UIManager(config, self.tk_root, self.app_manager)

        # --- State Variables ---
        self.last_valid_frame = None # Keep last good frame for display continuity
        self.current_display_dims = (0, 0) # w, h - Updated by UI Manager
        self.current_menu_trigger_zone = {} # Updated by UI Manager
        self.screen_size = self._get_screen_size()
        self.input_processor.update_screen_dimensions(self.screen_size[0], self.screen_size[1])

        # FPS calculation
        self.fps_stats = []
        self.fps_update_interval = 1.0 # seconds
        self.last_fps_update_time = time.time()
        self.current_fps = 0.0

        logger.info("ApplicationController initialized.")
        logger.info(f"Screen dimensions: {self.screen_size}")

    def _initialize_tkinter(self):
        """Initializes the hidden Tkinter root window for the menu."""
        logger.debug("Initializing Tkinter root...")
        try:
            self.tk_root = tk.Tk()
            self.tk_root.withdraw() # Hide the main window
            logger.info("Tkinter root initialized successfully.")
        except tk.TclError as e:
            logger.error(f"Failed to initialize Tkinter: {e}. Floating menu will be disabled.")
            self.tk_root = None
        except Exception as e:
             logger.exception(f"Unexpected error initializing Tkinter: {e}")
             self.tk_root = None


    def _get_screen_size(self):
         """Gets primary monitor screen dimensions."""
         # Attempt using Tkinter first if available
         if self.tk_root:
              try:
                   # Ensure tk_root is updated to get accurate screen info
                   self.tk_root.update_idletasks()
                   width = self.tk_root.winfo_screenwidth()
                   height = self.tk_root.winfo_screenheight()
                   if width > 0 and height > 0:
                        return (width, height)
                   else:
                        logger.warning("Tkinter reported invalid screen dimensions (<=0).")
              except Exception as e:
                   logger.warning(f"Could not get screen size via Tkinter: {e}")

         # Fallback methods (Platform specific - consider adding libraries like 'screeninfo')
         logger.warning("Falling back to less reliable screen size detection.")
         try:
              # Example fallback using ctypes on Windows (less robust)
              if os.name == 'nt':
                   import ctypes
                   user32 = ctypes.windll.user32
                   return (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
         except Exception as e:
              logger.error(f"Error getting screen size via fallback: {e}")

         # Absolute fallback default
         logger.error("Could not determine screen size. Using default 1920x1080.")
         return (1920, 1080)


    def start(self, start_calibration=False, initial_debug_mode=False):
        """Starts the application, camera, and detection thread."""
        if self.running:
            logger.warning("Start called but application is already running.")
            return True

        logger.info("Starting HandPal Application Controller...")
        self.stop_event.clear()
        self.debug_mode = initial_debug_mode # Set initial debug state

        # 1. Create UI Window
        if not self.ui_manager.create_window():
             logger.error("Failed to create UI window. Aborting start.")
             return False

        # 2. Start Detection Thread
        logger.info("Starting detection thread...")
        self.detection_thread = DetectionThread(self.config, self.data_queue, self.stop_event)
        self.detection_thread.start()

        # Give the thread a moment to initialize camera etc.
        time.sleep(1.0) # Adjust as needed
        if not self.detection_thread.is_alive():
             logger.error("Detection thread failed to start or exited prematurely.")
             self.ui_manager.destroy()
             return False
        # Add check if detection thread had init errors? (Needs communication back)

        # 3. Initial Calibration (if requested)
        if start_calibration:
             logger.info("Auto-starting calibration...")
             self.calibration_manager.start()
             # UI manager will show prompts based on calibration state

        self.running = True
        logger.info("HandPal started successfully. Entering main loop...")

        # 4. Enter Main Loop
        self.main_loop()

        # 5. Post-Loop Cleanup (happens after main_loop exits)
        logger.info("Main loop exited. Performing final cleanup.")
        self._cleanup() # Ensure cleanup runs even if loop exits unexpectedly

        return True # Indicate successful run completion (or graceful stop)


    def stop(self):
        """Signals the application to stop."""
        if not self.running and not self.stop_event.is_set():
             logger.info("Stop called but application not running. Ensuring stop event is set.")
             self.stop_event.set() # Make sure event is set for any waiting threads
             self._cleanup() # Perform cleanup just in case
             return

        if not self.running:
             logger.info("Stop called but application already stopped.")
             return

        logger.info("Stop signal received. Initiating shutdown sequence...")
        self.running = False # Signal main loop to exit
        self.stop_event.set() # Signal detection thread to stop


    def _cleanup(self):
         """Internal cleanup routine called after main loop exits or on error."""
         logger.info("Performing application cleanup...")

         # Ensure detection thread is joined
         if self.detection_thread and self.detection_thread.is_alive():
              logger.debug("Waiting for DetectionThread to join...")
              self.detection_thread.join(timeout=2.0)
              if self.detection_thread.is_alive():
                   logger.warning("DetectionThread did not join within timeout.")
              else:
                   logger.debug("DetectionThread joined successfully.")
         self.detection_thread = None

         # Destroy UI elements
         if self.ui_manager:
              self.ui_manager.destroy()
              self.ui_manager = None

         # Destroy Tkinter root (if managed here)
         if self.tk_root:
              try:
                   logger.debug("Destroying Tkinter root...")
                   self.tk_root.quit()
                   self.tk_root.destroy()
                   logger.debug("Tkinter root destroyed.")
              except Exception as e:
                   logger.error(f"Error destroying Tkinter root: {e}")
              self.tk_root = None

         # Optional: Clear queue?
         while not self.data_queue.empty():
             try: self.data_queue.get_nowait()
             except queue.Empty: break
         logger.debug("Data queue cleared.")

         # Maybe restore default cursor if custom was set (handled in main.py usually)

         logger.info("Application cleanup finished.")


    def _handle_input(self):
        """Processes keyboard input from OpenCV window."""
        key = cv.waitKey(1) & 0xFF # Non-blocking check

        if key == 255: # No key pressed
             return

        logger.debug(f"Key pressed: {key} (ord='{chr(key)}' if 32<=key<=126 else 'N/A')")

        # --- Global Keys ---
        if key == ord('q'):
            logger.info("'q' pressed. Stopping application.")
            self.stop()
        elif key == ord('d'):
             self.debug_mode = not self.debug_mode
             logger.info(f"Debug mode toggled {'ON' if self.debug_mode else 'OFF'}.")
             if self.debug_mode and self.ui_manager: # Clear history when turning on
                  self.ui_manager.debug_values_cache['cursor_history'] = []
        elif key == ord('m'):
             if self.ui_manager:
                  logger.debug("'m' pressed, toggling menu.")
                  self.ui_manager.toggle_menu()

        # --- Calibration Keys ---
        elif key == ord('c'):
             if not self.calibration_manager.is_active:
                  # Cannot start calibration if menu is open? Decide policy.
                  if self.ui_manager and self.ui_manager.floating_menu and self.ui_manager.floating_menu.visible:
                       logger.warning("Cannot start calibration while menu is open.")
                       print("(!) Close the menu before starting calibration.")
                  else:
                       self.calibration_manager.start()
                       if self.ui_manager: self.ui_manager.hide_menu() # Ensure menu is hidden
             else:
                  logger.warning("'c' pressed but calibration already active.")
        elif key == 32: # Spacebar
             if self.calibration_manager.is_active:
                  logger.debug("Spacebar pressed during calibration.")
                  # Need to get current hand position for the calibration manager
                  norm_pos = self.input_processor.debug_info.get("hand_norm") # Get last processed normalized pos
                  if norm_pos:
                       completed = self.calibration_manager.process_step(norm_pos)
                       if completed:
                            logger.info("Calibration completed via spacebar.")
                            # Loop will naturally transition out of calib mode
                  else:
                       logger.warning("Spacebar pressed for calibration, but no hand position available.")
                       print("(!) Cannot capture point: Right hand position not found.")
             else:
                  logger.debug("Spacebar pressed outside calibration - ignored.")

        # --- Escape Key ---
        elif key == 27: # ESC
             logger.debug("ESC key pressed.")
             if self.calibration_manager.is_active:
                  logger.info("ESC pressed during calibration - cancelling.")
                  self.calibration_manager.cancel()
             elif self.ui_manager and self.ui_manager.floating_menu and self.ui_manager.floating_menu.visible:
                  logger.info("ESC pressed while menu open - closing menu.")
                  self.ui_manager.hide_menu()
             else:
                  # Potentially add other ESC actions? Close app?
                   logger.debug("ESC pressed with no active action - ignored.")
                   pass


    def _process_frame_data(self, frame_data, results, proc_dims):
        """Processes results from the detection thread."""
        if frame_data is None: return # Skip if no frame

        # Process landmarks, gestures, mapping, etc.
        processed_output = self.input_processor.process(
            results,
            proc_dims,
            self.current_display_dims, # Pass current display size
            self.current_menu_trigger_zone # Pass current trigger zone info
        )

        # --- Act on Processed Output ---
        is_calibrating = self.calibration_manager.is_active

        # 1. Handle Cursor Movement (if not calibrating)
        if not is_calibrating:
             smooth_px = processed_output.get('cursor_smooth_px')
             if smooth_px:
                  # Check if position actually changed to avoid redundant calls
                  last_mouse_pos = getattr(self, '_last_mouse_pos', None)
                  if smooth_px != last_mouse_pos:
                       try:
                           self.mouse_controller.position = smooth_px
                           self._last_mouse_pos = smooth_px # Store last moved position
                           # logger.debug(f"Mouse moved to: {smooth_px}") # Very spammy
                       except Exception as e:
                            logger.error(f"Failed to set mouse position to {smooth_px}: {e}")

        # 2. Handle Click Events (if not calibrating)
        if not is_calibrating:
            click_type = processed_output.get('click_event')
            if click_type:
                 button = MouseButton.left # Assuming left click
                 count = 2 if click_type == 'double_click' else 1
                 try:
                      logger.info(f"Performing {click_type} ({button}, count={count})")
                      self.mouse_controller.click(button, count)
                 except Exception as e:
                      logger.error(f"Mouse click error ({click_type}): {e}")

        # 3. Handle Scroll Events (if not calibrating)
        if not is_calibrating:
            scroll_amount = processed_output.get('scroll_amount')
            if scroll_amount is not None and abs(scroll_amount) > 0.1: # Add threshold?
                 # pynput scroll: (dx, dy) - dy is vertical
                 # scroll_amount positive = hand down = scroll content up (negative dy)
                 # scroll_amount negative = hand up = scroll content down (positive dy)
                 scroll_dy = int(scroll_amount * -1) # Invert and convert to int clicks
                 if scroll_dy != 0:
                      try:
                           # logger.debug(f"Scrolling dy: {scroll_dy} (from amount: {scroll_amount:.2f})")
                           self.mouse_controller.scroll(0, scroll_dy)
                      except Exception as e:
                           logger.error(f"Mouse scroll error (dy={scroll_dy}): {e}")

        # 4. Handle Menu Activation
        if processed_output.get('activate_menu'):
             if not is_calibrating: # Don't show menu during calibration
                  logger.info("Menu activation triggered.")
                  self.ui_manager.show_menu()
                  # Maybe add a small delay before allowing gestures again?
             else:
                  logger.debug("Menu activation trigger ignored during calibration.")

        # --- Update UI ---
        if self.ui_manager:
             display_frame, new_dims, new_zone = self.ui_manager.update(
                  frame_data,
                  processed_output,
                  self.calibration_manager.get_state(), # Pass current calib state
                  self.debug_mode,
                  self.current_fps,
                  self.data_queue.qsize()
             )
             # Update controller's knowledge of display/zone from UI manager
             self.current_display_dims = new_dims
             self.current_menu_trigger_zone = new_zone

             # Update last valid frame if we got one back
             if display_frame is not None:
                  self.last_valid_frame = display_frame


    def _update_tkinter(self):
         """Handles Tkinter event loop processing."""
         if self.tk_root:
              try:
                   # Process Tkinter events without blocking
                   self.tk_root.update_idletasks()
                   self.tk_root.update()
              except tk.TclError as e:
                   if "application has been destroyed" in str(e).lower():
                        logger.warning("Tkinter root was destroyed externally. Stopping.")
                        self.stop() # Trigger graceful shutdown
                   else:
                        # Log other TclErrors but allow loop to continue if possible
                        logger.error(f"Tkinter update error: {e}")
              except Exception as e:
                   logger.exception(f"Unexpected error during Tkinter update: {e}")


    def _calculate_fps(self):
        """Calculates and updates FPS based on loop timings."""
        now = time.perf_counter()
        # Remove timings older than the update interval
        self.fps_stats = [t for t in self.fps_stats if now - t < self.fps_update_interval]
        # Add current frame time
        self.fps_stats.append(now)
        # Calculate FPS
        if len(self.fps_stats) > 1:
             duration = self.fps_stats[-1] - self.fps_stats[0]
             if duration > 0:
                  # FPS is count / duration. Count is len - 1 intervals.
                  self.current_fps = (len(self.fps_stats) - 1) / duration
             else: self.current_fps = 0.0 # Avoid division by zero
        else: self.current_fps = 0.0

        # Optional: Log FPS periodically
        # if now - self.last_fps_update_time > self.fps_update_interval:
        #      logger.debug(f"Main Loop FPS: {self.current_fps:.1f}")
        #      self.last_fps_update_time = now


    def main_loop(self):
        """The main application loop."""
        while self.running:
            loop_start_time = time.perf_counter()

            # 1. Update Tkinter (process GUI events)
            self._update_tkinter()
            # Check if Tkinter update caused a stop signal
            if not self.running: break

            # 2. Get data from Detection Thread
            frame_data, results, proc_dims = None, None, None
            try:
                # Use timeout to prevent blocking indefinitely if detector stalls
                frame_data, results, proc_dims = self.data_queue.get(block=True, timeout=0.02)
                # If we got data, update the last valid frame
                if frame_data is not None:
                     self.last_valid_frame = frame_data
            except queue.Empty:
                # No new data, use the last valid frame for display continuity
                frame_data = self.last_valid_frame
                results = None # Don't re-process old results
                proc_dims = self.input_processor._last_proc_dims # Use last known dims
                # logger.debug("No new frame data from detector.") # Can be spammy
            except Exception as e:
                 logger.exception(f"Error getting data from queue: {e}")
                 time.sleep(0.05) # Avoid tight loop on queue error
                 continue # Skip rest of loop iteration

            # 3. Process Data and Update UI
            try:
                 self._process_frame_data(frame_data, results, proc_dims)
            except Exception as e:
                 logger.exception("Error during frame processing or UI update.")
                 # Maybe draw error message on last_valid_frame?


            # 4. Handle Keyboard Input
            self._handle_input()
            # Check if input handling caused a stop signal
            if not self.running: break

            # 5. Calculate FPS
            self._calculate_fps()

            # Optional: Ensure minimum loop time / sleep?
            # elapsed = time.perf_counter() - loop_start_time
            # sleep_time = (1.0 / MAX_MAIN_LOOP_FPS) - elapsed
            # if sleep_time > 0: time.sleep(sleep_time)

        # --- End of Loop ---
        logger.info("Exiting main loop.")