import threading
import time
import queue
import cv2 as cv
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

class DetectionThread(threading.Thread):
    def __init__(self, config, data_q, stop_evt):
        super().__init__(daemon=True, name="DetectionThread")
        self.config = config
        self.data_q = data_q       # Queue to send results TO main thread
        self.stop_evt = stop_evt   # Event to signal thread termination
        self.cap = None
        self.hands = None
        self.mp_hands = mp.solutions.hands

        # Get processing dimensions and flip status from config
        self.proc_w = self.config.get("process_width", 640)
        self.proc_h = self.config.get("process_height", 360)
        self.flip = self.config.get("flip_camera", True)
        self.device_id = self.config.get("device", 0)
        self.cam_width = self.config.get("width", 1280)
        self.cam_height = self.config.get("height", 720)
        self.cam_fps = self.config.get("max_fps", 30) # Use max_fps config

        logger.info(f"DetectionThread configured: Device={self.device_id}, CamRes={self.cam_width}x{self.cam_height}@{self.cam_fps}fps, ProcRes={self.proc_w}x{self.proc_h}, Flip={self.flip}")

    def _initialize_resources(self):
        """Initializes webcam and MediaPipe Hands model."""
        logger.info("Initializing detection resources...")
        try:
            # Try DirectShow first on Windows
            self.cap = cv.VideoCapture(self.device_id, cv.CAP_DSHOW)
            if not self.cap.isOpened():
                logger.warning("CAP_DSHOW failed, trying default backend...")
                self.cap = cv.VideoCapture(self.device_id)

            if not self.cap.isOpened():
                raise IOError(f"Cannot open webcam {self.device_id}")

            # Set camera parameters
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cam_width)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cam_height)
            self.cap.set(cv.CAP_PROP_FPS, self.cam_fps)
            # Try setting a small buffer size to reduce latency
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

            # Verify settings (actual values might differ)
            actual_w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv.CAP_PROP_FPS)
            logger.info(f"Webcam {self.device_id} opened. Actual settings: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS (Buffer: {self.cap.get(cv.CAP_PROP_BUFFERSIZE)})")

            # Initialize MediaPipe Hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=self.config.get("use_static_image_mode", False),
                max_num_hands=2,
                min_detection_confidence=self.config.get("min_detection_confidence", 0.6),
                min_tracking_confidence=self.config.get("min_tracking_confidence", 0.5)
            )
            logger.info("MediaPipe Hands initialized.")
            return True
        except Exception as e:
            logger.exception(f"Failed to initialize detection resources: {e}")
            self._release_resources() # Clean up if init failed partially
            return False

    def _release_resources(self):
        """Releases webcam and MediaPipe resources."""
        logger.info("Releasing detection resources...")
        if hasattr(self.hands, 'close') and self.hands:
             try: self.hands.close(); logger.debug("MediaPipe Hands closed.")
             except Exception as e: logger.error(f"Error closing MediaPipe Hands: {e}")
             self.hands = None
        if self.cap and self.cap.isOpened():
            try: self.cap.release(); logger.debug("Webcam released.")
            except Exception as e: logger.error(f"Error releasing webcam: {e}")
            self.cap = None

    def run(self):
        """The main loop for the detection thread."""
        if not self._initialize_resources():
            logger.error("DetectionThread stopping due to initialization failure.")
            return # Don't start loop if init failed

        logger.info("DetectionThread run loop starting...")
        frame_count = 0
        start_time = time.perf_counter()

        while not self.stop_evt.is_set():
            if self.cap is None or not self.cap.isOpened():
                logger.error("Webcam became unavailable. Attempting to reconnect...")
                time.sleep(1.0)
                self._release_resources() # Clean up before trying again
                if not self._initialize_resources():
                    logger.error("Reconnection failed. Stopping thread.")
                    break # Exit loop if reconnection fails
                else:
                    logger.info("Reconnection successful.")
                continue # Try reading again

            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    # logger.warning("Failed to grab frame.") # Can be spammy
                    time.sleep(0.01) # Wait briefly before retrying
                    continue

                # --- Preprocessing ---
                if self.flip:
                    frame = cv.flip(frame, 1) # Flip horizontally if configured

                # Resize for processing (performance)
                process_frame = cv.resize(frame, (self.proc_w, self.proc_h), interpolation=cv.INTER_LINEAR)

                # Convert color BGR -> RGB for MediaPipe
                rgb_frame = cv.cvtColor(process_frame, cv.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False # Optimize: pass as read-only

                # --- Hand Detection ---
                results = self.hands.process(rgb_frame)
                rgb_frame.flags.writeable = True # Make writable again if needed later

                # --- Send Results ---
                # Package original frame, results, and processing dims used
                data_package = (frame.copy(), results, (self.proc_w, self.proc_h))
                try:
                    # Put data onto the queue for the main thread
                    # Use a small timeout to avoid blocking indefinitely if queue is full
                    self.data_q.put(data_package, block=True, timeout=0.05)
                    frame_count += 1
                except queue.Full:
                    # If queue is full, discard oldest item and try again (non-blocking)
                    # This prioritizes newer frames
                    try:
                        self.data_q.get_nowait() # Discard oldest
                        self.data_q.put_nowait(data_package) # Try putting again
                        frame_count += 1
                        # logger.warning("Detection queue was full, dropped oldest frame.")
                    except queue.Empty:
                         pass # Queue became empty between check and get
                    except queue.Full:
                         logger.warning("Detection queue full even after dropping frame. Skipping.")
                         pass # Still full, skip this frame

            except cv.error as cv_err:
                 logger.error(f"OpenCV Error in detection loop: {cv_err}. Attempting recovery.")
                 time.sleep(0.5) # Wait a bit before potentially trying to reconnect
                 # Consider triggering reconnect logic here if error seems persistent
            except Exception as e:
                logger.exception(f"Unexpected error in detection loop: {e}")
                time.sleep(0.1) # Prevent rapid error loops

        # --- Cleanup ---
        elapsed_time = time.perf_counter() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        logger.info(f"DetectionThread run loop finished. Processed {frame_count} frames in {elapsed_time:.2f}s (Avg Detector FPS: {avg_fps:.1f})")
        self._release_resources()