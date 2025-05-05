import tkinter as tk
from tkinter import Toplevel, Frame, Label, Button as TkButton, BOTH, X
import logging
# Removed CSV functions - they are now in apps.manager

logger = logging.getLogger(__name__)

class FloatingMenu:
    def __init__(self, root, app_manager, config):
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
        self.app_manager = app_manager # Store the app manager
        self.config = config # Store config if needed for styling etc.

        self.window = Toplevel(self.root)
        self.window.title("HandPal Menu")
        # Make window floating and topmost
        self.window.attributes("-topmost", True)
        # Remove window decorations (title bar, borders)
        self.window.overrideredirect(True)
        # Initial size and position (can be configured later)
        self.window.geometry("280x400+50+50") # Width x Height + X + Y
        # Set background colors (consider getting from config)
        self.window.configure(bg='#222831')

        # Load applications using the AppManager
        self.apps = self.app_manager.get_applications()

        self._create_elements()
        self.window.withdraw() # Start hidden
        self.visible = False

        # Bind events for dragging the window
        self.window.bind("<ButtonPress-1>", self._start_move)
        self.window.bind("<ButtonRelease-1>", self._stop_move)
        self.window.bind("<B1-Motion>", self._do_move)
        self._offset_x = 0
        self._offset_y = 0
        logger.info("FloatingMenu initialized.")

    def _create_elements(self):
        """Creates the Tkinter widgets inside the menu window."""
        # Title Frame
        title_frame = Frame(self.window, bg='#222831', pady=15)
        title_frame.pack(fill=X)
        Label(title_frame, text="HANDPAL MENU", font=("Helvetica", 14, "bold"), bg='#222831', fg='#EEEEEE').pack()
        Label(title_frame, text="Launch Application", font=("Helvetica", 10), bg='#222831', fg='#00ADB5').pack(pady=(0, 10))

        # Scrollable area frame
        scroll_area = Frame(self.window, bg='#222831')
        scroll_area.pack(fill=BOTH, expand=True, padx=(30, 10), pady=(0, 5))  # Add horizontal padding

        canvas = tk.Canvas(scroll_area, bg='#222831', highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=BOTH, expand=True)

        scrollbar = tk.Scrollbar(scroll_area, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill='y')

        # Container inside canvas
        button_container = Frame(canvas, bg='#222831')
        canvas.create_window((0, 0), window=button_container, anchor="nw")

        # Configure scrolling
        canvas.configure(yscrollcommand=scrollbar.set)
        button_container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.buttons = []

        for app_info in self.apps:
            app_frame = Frame(button_container, bg='#222831', pady=8)
            app_frame.pack(fill=X)

            icon = app_info.get('icon', ' ')
            label = app_info.get('label', '?')
            color = app_info.get('color', '#555555')
            path = app_info.get('path')

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
                width=20,
                anchor='w',
                command=lambda p=path: self._launch_and_hide(p)
            )
            btn.pack(fill=X)
            self.buttons.append(btn)

        # Bottom frame for Close button
        bottom_frame = Frame(self.window, bg='#222831')
        bottom_frame.pack(fill=X, side=tk.BOTTOM, pady=(5, 5))

        TkButton(
            bottom_frame,
            text="✖ Close Menu",
            bg='#393E46', fg='#EEEEEE',
            font=("Helvetica", 10),
            relief=tk.FLAT, borderwidth=0,
            padx=10, pady=6,
            width=20,
            command=self.hide
        ).pack()



    def _launch_and_hide(self, path):
         """Helper to launch app and then hide menu."""
         if path:
             self.app_manager.launch(path) # Use AppManager to launch
             self.hide() # Hide menu after launching
         else:
              logger.warning("Attempted to launch app with no path from menu.")


    def show(self):
        """Makes the menu window visible."""
        if not self.visible:
            self.window.deiconify() # Show the window
            self.window.lift()      # Bring it to the front
            self.visible = True
            logger.debug("FloatingMenu shown.")

    def hide(self):
        """Hides the menu window."""
        if self.visible:
            self.window.withdraw() # Hide the window
            self.visible = False
            logger.debug("FloatingMenu hidden.")

    def toggle(self):
        """Toggles the visibility of the menu."""
        if self.visible:
            self.hide()
        else:
            self.show()

    # --- Window Dragging Methods ---
    def _start_move(self, event):
        """Records the initial mouse offset when dragging starts."""
        self._offset_x = event.x
        self._offset_y = event.y

    def _stop_move(self, event):
        """Resets the offset when dragging stops."""
        self._offset_x = 0
        self._offset_y = 0

    def _do_move(self, event):
        """Moves the window based on mouse movement during drag."""
        # Calculate new window position
        new_x = self.window.winfo_x() + event.x - self._offset_x
        new_y = self.window.winfo_y() + event.y - self._offset_y
        self.window.geometry(f"+{new_x}+{new_y}") # Set new position

    def update_apps(self):
         """Reloads apps from AppManager and rebuilds the menu."""
         logger.info("Updating menu applications...")
         self.apps = self.app_manager.get_applications()
         # Clear existing widgets in button container (or destroy/recreate window?)
         for widget in self.window.winfo_children():
              widget.destroy()
         # Recreate all elements
         self._create_elements()
         logger.info("FloatingMenu elements recreated.")
         # Ensure it remains hidden/visible as it was
         if not self.visible:
              self.window.withdraw()
         else:
              self.window.deiconify()
              self.window.lift()


    def destroy(self):
         """Destroys the menu window."""
         try:
              self.window.destroy()
              logger.info("FloatingMenu destroyed.")
         except tk.TclError:
              logger.debug("FloatingMenu window already destroyed.") # Ignore if already gone
         except Exception as e:
              logger.error(f"Error destroying FloatingMenu: {e}")