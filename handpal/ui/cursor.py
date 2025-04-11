import os
import ctypes
import logging

logger = logging.getLogger(__name__)

# --- Windows Specific Cursor Functions ---
_original_cursors = {} # Store original handles if needed for perfect restore
_custom_cursor_handle = None

def set_custom_cursor(cursor_path):
    """Sets the system's normal cursor to a custom .cur file (Windows only)."""
    global _custom_cursor_handle
    if os.name != 'nt':
        logger.warning("Custom cursor setting is only supported on Windows.")
        return False
    try:
        if not cursor_path or not os.path.exists(cursor_path):
            raise FileNotFoundError(f"Cursor file not found or path is invalid: {cursor_path}")

        user32 = ctypes.windll.user32
        # Load the custom cursor file
        # Use LoadCursorFromFileW for Unicode path support
        hCursor = user32.LoadCursorFromFileW(str(cursor_path))
        if hCursor == 0:
            error_code = ctypes.get_last_error()
            raise OSError(f"Failed to load cursor file (Error {error_code}). Path: {cursor_path}")

        # Define system cursor identifiers (more robust than hardcoded numbers)
        OCR_NORMAL = 32512 # Standard arrow
        # You could add others here if needed (e.g., OCR_IBEAM = 32513)

        # --- Optional: Store the original cursor handle for perfect restore ---
        # if OCR_NORMAL not in _original_cursors:
        #     hOriginal = user32.CopyIcon(user32.GetCursor()) # Get current cursor handle
        #     if hOriginal: _original_cursors[OCR_NORMAL] = hOriginal

        # Set the system cursor
        if not user32.SetSystemCursor(hCursor, OCR_NORMAL):
            error_code = ctypes.get_last_error()
            # Important: If SetSystemCursor succeeds, the system takes ownership
            # of the hCursor. If it fails, we *might* need to destroy it.
            # However, LoadCursorFromFile docs are murky on ownership if Set fails.
            # Let's assume failure means it wasn't taken.
            # user32.DestroyCursor(hCursor) # Maybe needed? Test carefully.
            raise OSError(f"Failed to set system cursor (Error {error_code}).")

        # If successful, store the handle *only for potential later destruction*
        # We don't need to destroy it if SetSystemCursor worked, but maybe on restore?
        _custom_cursor_handle = hCursor # Store the handle used

        logger.info(f"Custom system cursor set to '{os.path.basename(cursor_path)}'")
        return True

    except FileNotFoundError as e:
        logger.error(f"Error setting custom cursor: {e}")
        return False
    except OSError as e:
         logger.error(f"System error setting custom cursor: {e}")
         return False
    except Exception as e:
        logger.exception(f"Unexpected error setting custom cursor: {e}")
        return False

def restore_default_cursor():
    """Restores the default system cursors (Windows only)."""
    if os.name != 'nt':
        return
    try:
        user32 = ctypes.windll.user32
        SPI_SETCURSORS = 0x0057 # SystemParametersInfo action to restore cursors
        SPIF_UPDATEINIFILE = 0x01 # Optional: Update user profile
        SPIF_SENDCHANGE = 0x02    # Optional: Broadcast change message

        # Call SystemParametersInfo to restore all system cursors to default
        if not user32.SystemParametersInfoW(SPI_SETCURSORS, 0, None, SPIF_UPDATEINIFILE | SPIF_SENDCHANGE):
            error_code = ctypes.get_last_error()
            raise OSError(f"SystemParametersInfo failed to restore default cursors (Error {error_code}).")

        logger.info("Default system cursors restored via SystemParametersInfo.")

        # --- Optional: Clean up loaded custom cursor handle if we stored one ---
        # global _custom_cursor_handle
        # if _custom_cursor_handle:
        #     # Does SPI_SETCURSORS invalidate the handle? Maybe.
        #     # Destroying it might be needed, or might be harmful. TEST CAREFULLY.
        #     # if user32.DestroyCursor(_custom_cursor_handle):
        #     #     logger.debug("Destroyed custom cursor handle.")
        #     # else:
        #     #     logger.warning("Failed to destroy custom cursor handle.")
        #     _custom_cursor_handle = None

        # --- Alternative Restore (If SPI fails or for specific cursors) ---
        # if OCR_NORMAL in _original_cursors:
        #     if user32.SetSystemCursor(_original_cursors[OCR_NORMAL], OCR_NORMAL):
        #         logger.info("Restored original OCR_NORMAL cursor.")
        #     else:
        #         logger.error("Failed to restore original OCR_NORMAL cursor.")
        #     # We might need to DestroyIcon on the original handle if CopyIcon requires it
        #     # user32.DestroyIcon(_original_cursors[OCR_NORMAL])
        #     del _original_cursors[OCR_NORMAL]

    except OSError as e:
         logger.error(f"System error restoring default cursors: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error restoring default cursors: {e}")