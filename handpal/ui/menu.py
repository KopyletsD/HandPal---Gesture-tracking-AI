# handpal/ui/menu.py
import tkinter as tk
from tkinter import Toplevel, Frame, Label, Button as TkButton, BOTH, X
import logging

# Make sure AppManager can be imported if type hinting needed (optional)
# from ..apps.manager import AppManager
# from ..config import Config

logger = logging.getLogger(__name__)

class FloatingMenu:
    def __init__(self, root: tk.Tk, app_manager, config):
        """
        Initializes the floating menu.
        Args:
            root (tk.Tk): The main Tkinter root window.
            app_manager (AppManager): Instance to get apps and launch them.
            config (Config): Application configuration.
        """
        if root is None:
             raise ValueError("FloatingMenu requires a Tkinter root window.")
        self.root = root
        self.app_manager = app_manager
        self.config = config

        self.window = Toplevel(self.root)
        self.window.title("HandPal Menu")
        self.window.attributes("-topmost", True)
        self.window.overrideredirect(True) # No window decorations
        self.window.geometry("280x400+50+50") # Initial size/pos
        self.window.configure(bg='#222831') # Background

        # Store the button container frame and button widgets
        self.button_container = None
        self.buttons = []

        # Create structural elements first
        self._create_base_elements()
        # Populate with initial application buttons
        self._create_app_buttons()

        self.window.withdraw() # Start hidden
        self.visible = False

        # Bind events for dragging the window
        self.window.bind("<ButtonPress-1>", self._start_move)
        self.window.bind("<ButtonRelease-1>", self._stop_move)
        self.window.bind("<B1-Motion>", self._do_move)
        self._offset_x = 0
        self._offset_y = 0
        logger.info("FloatingMenu initialized.")

    def _create_base_elements(self):
        """Creates the non-application parts of the menu (title, container, close btn)."""
        # Title Frame
        title_frame = Frame(self.window, bg='#222831', pady=15)
        title_frame.pack(fill=X)
        Label(title_frame, text="HANDPAL MENU", font=("Helvetica", 14, "bold"), bg='#222831', fg='#EEEEEE').pack()
        Label(title_frame, text="Launch Application", font=("Helvetica", 10), bg='#222831', fg='#00ADB5').pack(pady=(0, 10))

        # Button Container Frame (Store reference to add/remove buttons later)
        self.button_container = Frame(self.window, bg='#222831', padx=20)
        self.button_container.pack(fill=BOTH, expand=True) # Fill available space

        # Bottom Frame for Close Button (Pack last to stay at bottom)
        bottom_frame = Frame(self.window, bg='#222831', pady=15)
        bottom_frame.pack(fill=X, side=tk.BOTTOM)

        TkButton(
            bottom_frame, text="✖ Close Menu", bg='#393E46', fg='#EEEEEE',
            font=("Helvetica", 10), relief=tk.FLAT, borderwidth=0,
            padx=10, pady=5, width=15, command=self.hide
        ).pack(pady=5) # Center the close button (or adjust as needed)

    def _create_app_buttons(self):
        """Clears and recreates application buttons in the container."""
        if not self.button_container:
             logger.error("Button container not initialized in FloatingMenu.")
             return

        # Clear existing buttons first
        for widget in self.button_container.winfo_children():
            widget.destroy()
        self.buttons.clear() # Clear the list of button widgets

        # Get the latest list of applications from the AppManager
        self.apps = self.app_manager.get_applications()
        logger.debug(f"Recreating menu buttons for {len(self.apps)} apps.")

        # Display message if no apps found
        if not self.apps:
             Label(self.button_container, text="No applications found.",
                   font=("Helvetica", 10), bg='#222831', fg='#AAAAAA').pack(pady=20)
             return # Don't proceed if no apps

        # Create buttons for each application
        for app_info in self.apps:
            # Create a frame for each button row to control padding
            app_frame = Frame(self.button_container, bg='#222831', pady=5)
            app_frame.pack(fill=X)

            # Safely get app details with defaults
            icon = app_info.get('icon', ' ')
            label = app_info.get('label', '?')
            color = app_info.get('color', '#555555')
            path = app_info.get('path') # Path is crucial for the command

            # Create the button, ensuring lambda captures the correct path
            btn = TkButton(
                app_frame,
                text=f"{icon} {label}",
                bg=color,
                fg="white",
                font=("Helvetica", 11),
                relief=tk.FLAT,
                borderwidth=0,
                padx=10,
                pady=8,
                width=20,    # Fixed width helps alignment
                anchor='w',  # Align text to the left (west)
                command=lambda p=path: self._launch_and_hide(p) # Launch via AppManager
            )
            btn.pack(fill=X) # Make button fill its frame horizontally
            self.buttons.append(btn) # Store reference if needed

    def _launch_and_hide(self, path):
         """Helper to launch app via AppManager and then hide the menu."""
         if path:
             self.app_manager.launch(path)
             self.hide()
         else:
              logger.warning("Menu button clicked but associated path is empty.")

    def show(self):
        """Makes the menu window visible."""
        if not self.visible:
            try:
                self.window.deiconify() # Show the window
                self.window.lift()      # Bring it to the front
                self.visible = True
                logger.debug("FloatingMenu shown.")
            except tk.TclError as e:
                 logger.error(f"Tkinter error showing menu: {e}")

    def hide(self):
        """Hides the menu window."""
        if self.visible:
            try:
                self.window.withdraw() # Hide the window
                self.visible = False
                logger.debug("FloatingMenu hidden.")
            except tk.TclError as e:
                 logger.error(f"Tkinter error hiding menu: {e}")

    def toggle(self):
        """Toggles the visibility of the menu."""
        if self.visible:
            self.hide()
        else:
            self.show()

    # --- Window Dragging Methods ---
    def _start_move(self, event):
        self._offset_x = event.x
        self._offset_y = event.y

    def _stop_move(self, event):
        self._offset_x = 0
        self._offset_y = 0

    def _do_move(self, event):
        new_x = self.window.winfo_x() + event.x - self._offset_x
        new_y = self.window.winfo_y() + event.y - self._offset_y
        self.window.geometry(f"+{new_x}+{new_y}")

    # --- ADDED METHOD ---
    def update_apps(self):
        """Reloads apps from AppManager and rebuilds the menu buttons."""
        logger.info("Updating FloatingMenu applications display...")
        # Assumes AppManager has already been told to reload its internal list.
        # We just need to recreate the button widgets based on the new list.
        try:
             self._create_app_buttons()
             logger.info("FloatingMenu buttons recreated.")
        except Exception as e:
             logger.exception("Error updating floating menu apps display.")
    # --- END OF ADDED METHOD ---

    def destroy(self):
         """Destroys the menu window."""
         try:
              if self.window:
                   self.window.destroy()
                   self.window = None # Prevent further calls
              logger.info("FloatingMenu destroyed.")
         except tk.TclError:
              logger.debug("FloatingMenu window already destroyed.")
         except Exception as e:
              logger.error(f"Error destroying FloatingMenu: {e}")