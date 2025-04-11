import sys
import os
import argparse
import logging
import time

# Add project root to Python path to allow absolute imports like 'from handpal.config import Config'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import HandPal components using absolute paths
from handpal.config import Config
from handpal.app_controller import ApplicationController
from handpal.ui.cursor import set_custom_cursor, restore_default_cursor

# --- Logging Setup ---
# Configure root logger BEFORE any component logs are created
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file_handler = logging.FileHandler("handpal.log", mode='a') # Append mode
log_file_handler.setFormatter(log_formatter)
log_console_handler = logging.StreamHandler(sys.stdout)
log_console_handler.setFormatter(log_formatter)

# Get the root logger and add handlers
root_logger = logging.getLogger() # Get root logger
root_logger.addHandler(log_file_handler)
root_logger.addHandler(log_console_handler)
root_logger.setLevel(logging.INFO) # Default level

# Get specific logger for this entry point
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
def parse_arguments():
    # Use defaults directly from Config class for single source of truth
    default_cfg_obj = Config() # Create temporary default config to get paths etc.

    parser = argparse.ArgumentParser(
        description="HandPal - Mouse control with gestures & menu.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Core Settings ---
    parser.add_argument('--device', type=int, default=None,
                        help=f'Webcam device ID.')
    parser.add_argument('--width', type=int, default=None,
                        help='Webcam capture width.')
    parser.add_argument('--height', type=int, default=None,
                        help='Webcam capture height.')
    parser.add_argument('--display_width', type=int, default=None, # Use dot notation for clarity
                        help='Preview window width (uses camera res if None).')
    parser.add_argument('--display_height', type=int, default=None,
                        help='Preview window height (uses camera res if None).')

    # --- Behavior ---
    parser.add_argument('--flip_camera', action=argparse.BooleanOptionalAction, default=None,
                         help='Flip webcam horizontally.')
    parser.add_argument('--calibrate', action='store_true', default=False,
                        help='Start calibration sequence immediately on launch.')

    # --- Debugging & Config ---
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug logging and overlay.')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level.')
    parser.add_argument('--reset_config', action='store_true', default=False,
                        help=f'Delete user config file ({Config.CONFIG_FILENAME}) and exit.')
    parser.add_argument('--config_file', type=str, default=Config.CONFIG_FILENAME,
                        help='Path to user configuration JSON file.') # Keep for specifying *which* file to load/reset

    # --- UI / App Files ---
    parser.add_argument('--cursor_file', type=str, default=None,
                        help='Path to custom cursor (.cur) file (Windows only). Overrides config.')
    parser.add_argument('--apps_file', type=str, default=None,
                        help='Path to applications CSV file. Overrides config.')


    args = parser.parse_args()
    return args

# -----------------------------------------------------------------------------
# Main Execution Block
# -----------------------------------------------------------------------------
def main():
    args = parse_arguments()
    app_instance = None
    custom_cursor_was_set = False

    try:
        # --- Initial Setup ---
        # 1. Set Log Level based on args BEFORE loading config (which might log)
        log_level_name = args.log_level.upper()
        if args.debug and log_level_name == 'INFO': # If --debug is set, default to DEBUG unless explicitly higher
             log_level_name = 'DEBUG'
        log_level = getattr(logging, log_level_name, logging.INFO)
        root_logger.setLevel(log_level)
        logger.info(f"Logging level set to: {log_level_name}")
        logger.debug(f"Parsed arguments: {args}")

        # 2. Handle Config Reset
        if args.reset_config:
            config_path_to_reset = args.config_file # Use the specified path
            logger.info(f"Attempting to reset config file: {config_path_to_reset}")
            if os.path.exists(config_path_to_reset):
                try:
                    os.remove(config_path_to_reset)
                    logger.info(f"Successfully removed config file: {config_path_to_reset}")
                    print(f"Config file '{config_path_to_reset}' removed.")
                    return 0 # Exit successfully after reset
                except OSError as e:
                    logger.error(f"Error removing config file '{config_path_to_reset}': {e}")
                    print(f"ERROR: Could not remove config file '{config_path_to_reset}': {e}")
                    return 1 # Exit with error
            else:
                logger.info("Config file not found, nothing to reset.")
                print("Config file not found, nothing to reset.")
                return 0 # Exit successfully

        # 3. Load Configuration (passing CLI args for overrides)
        logger.info("Loading configuration...")
        config = Config(args=args)
        logger.info("Configuration loaded.")

        # 4. Set Custom Cursor (using path from final config)
        if os.name == 'nt':
            cursor_path_from_config = config.get('custom_cursor_path')
            if cursor_path_from_config:
                logger.info(f"Attempting to set custom cursor from config: {cursor_path_from_config}")
                # Config class should store absolute path now
                if os.path.exists(cursor_path_from_config):
                    custom_cursor_was_set = set_custom_cursor(cursor_path_from_config)
                else:
                    logger.error(f"Cursor file specified in config not found: {cursor_path_from_config}")
            else:
                logger.info("No custom cursor path specified in config.")


        # --- Application Initialization ---
        logger.info("Initializing HandPal Application Controller...")
        app_instance = ApplicationController(config)
        logger.info("Application Controller initialized.")


        # --- Start Application ---
        logger.info("Starting application main loop...")
        # Pass initial state flags from args to the start method
        app_instance.start(
            start_calibration=args.calibrate,
            initial_debug_mode=args.debug
        )
        # The start method now blocks until the main loop finishes or is stopped

        logger.info("Application has finished running.")
        return 0 # Exit successfully

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt (Ctrl+C) received. Stopping...")
        # If app was running, signal it to stop gracefully
        if app_instance and app_instance.running:
            app_instance.stop()
        # Cleanup happens in the finally block
        return 130 # Standard exit code for Ctrl+C

    except Exception as e:
        logger.exception("An unhandled exception occurred in the main execution block.")
        print(f"\nFATAL ERROR: {e}\nCheck 'handpal.log' for detailed traceback.", file=sys.stderr)
        # Attempt cleanup even on unhandled errors
        if app_instance and app_instance.running:
            try: app_instance.stop()
            except Exception as cleanup_err: logger.error(f"Error during stop on exception: {cleanup_err}")
        return 1 # Exit with error code

    finally:
        # --- Final Cleanup ---
        logger.info("Executing main finally block...")

        # Ensure app cleanup method is called if instance exists
        # (AppController's start/stop should handle internal cleanup now)
        # if app_instance:
        #     logger.debug("Ensuring AppController cleanup is called...")
        #     # Calling stop again here might be redundant if it was already called
        #     # It should be safe if stop() handles being called multiple times
        #     app_instance.stop() # This should trigger _cleanup if not already done


        # Restore default cursor if we set a custom one
        if custom_cursor_was_set:
            logger.info("Restoring default system cursor...")
            restore_default_cursor()

        logger.info("HandPal application finished.")
        print("\nHandPal terminated.")
        # Allow logs to flush
        logging.shutdown()
        time.sleep(0.1)


# -----------------------------------------------------------------------------
# Entry Point Guard
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure the HandPalProject root is on the path when run directly
    if PROJECT_ROOT not in sys.path:
         sys.path.insert(0, PROJECT_ROOT)
         logger.debug(f"Added {PROJECT_ROOT} to sys.path")

    exit_code = main()
    sys.exit(exit_code)