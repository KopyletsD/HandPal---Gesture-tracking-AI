import os
import json
import logging
import sys # Needed for finding script dir

logger = logging.getLogger(__name__) # Use module-specific logger

class Config:
    # Define path relative to this config file's location
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR) # Go one level up to HandPalProject/
    _DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
    _ASSETS_DIR = os.path.join(_PROJECT_ROOT, 'resources')

    DEFAULT_CURSOR_PATH = os.path.join(_ASSETS_DIR, "cursor.cur") # Default relative path
    DEFAULT_APPS_CSV_PATH = os.path.join(_DATA_DIR, "applications.csv")

    DEFAULT_CONFIG = {
        "device": 0,
        "width": 1280,
        "height": 720,
        "process_width": 640,
        "process_height": 360,
        "flip_camera": True,
        "display_width": None, # Keep None for auto-size initially
        "display_height": None,
        "min_detection_confidence": 0.6,
        "min_tracking_confidence": 0.5,
        "use_static_image_mode": False,
        "smoothing_factor": 0.7,
        "inactivity_zone": 0.015,
        "click_cooldown": 0.4,
        "gesture_sensitivity": 0.02,
        "gesture_settings": {"scroll_sensitivity": 4, "double_click_time": 0.35},
        "calibration": {
            "enabled": True,
            "screen_margin": 0.1,
            "x_min": 0.15, "x_max": 0.85, "y_min": 0.15, "y_max": 0.85,
            "active": False # Internal state, not saved/loaded
        },
        "max_fps": 60,
        "custom_cursor_path": DEFAULT_CURSOR_PATH, # Use defined default
        "apps_csv_path": DEFAULT_APPS_CSV_PATH # Use defined default
    }

    CONFIG_FILENAME = os.path.expanduser("~/.handpal_config.json")

    def _deep_update(self, target, source):
        # (Keep unchanged)
        for k, v in source.items():
            if isinstance(v, dict) and k in target and isinstance(target[k], dict): self._deep_update(target[k], v)
            else: target[k] = v

    def __init__(self, args=None):
        self.config = json.loads(json.dumps(self.DEFAULT_CONFIG)) # Deep copy
        self.load_from_file() # Load from file first
        if args: self._apply_cli_args(args) # Apply CLI overrides
        self.config["calibration"]["active"] = False # Ensure internal state is reset
        self._validate_calibration(); self._validate_display_dims()
        logger.debug(f"Final config: {json.dumps(self.config, indent=2)}")

    def load_from_file(self):
        if os.path.exists(self.CONFIG_FILENAME):
            try:
                with open(self.CONFIG_FILENAME, 'r') as f:
                    loaded_config = json.load(f)
                    # Ensure loaded paths are absolute if they aren't already
                    cursor_path = loaded_config.get('custom_cursor_path')
                    if cursor_path and not os.path.isabs(cursor_path):
                         loaded_config['custom_cursor_path'] = os.path.abspath(os.path.join(self._PROJECT_ROOT, cursor_path))

                    apps_path = loaded_config.get('apps_csv_path')
                    if apps_path and not os.path.isabs(apps_path):
                         loaded_config['apps_csv_path'] = os.path.abspath(os.path.join(self._PROJECT_ROOT, apps_path))

                    self._deep_update(self.config, loaded_config)
                logger.info(f"Config loaded from {self.CONFIG_FILENAME}")
            except Exception as e:
                logger.error(f"Error loading config file '{self.CONFIG_FILENAME}': {e}. Using defaults.")
        else:
             logger.info(f"Config file '{self.CONFIG_FILENAME}' not found. Using defaults.")

    def _apply_cli_args(self, args):
        # (Keep mostly unchanged, adjust key names if needed)
        cli_args = vars(args)
        for key, value in cli_args.items():
            if value is not None:
                config_key = key.replace('_', '.') # Allow display.width style keys
                if key == 'flip_camera' and value is False: self.config['flip_camera'] = False
                elif key == 'cursor_file': self.config['custom_cursor_path'] = os.path.abspath(value) if value else None # Store absolute path
                elif key == 'apps_file': self.config['apps_csv_path'] = os.path.abspath(value) if value else None # Store absolute path
                # Handle potentially nested keys simpler
                elif config_key in self.config:
                    if not isinstance(self.config[config_key], dict):
                         try: # Try to convert type
                             self.config[config_key] = type(self.config[config_key])(value)
                         except (ValueError, TypeError): # Fallback to direct assignment
                             self.config[config_key] = value
                    else:
                         logger.warning(f"Cannot override nested config '{config_key}' via CLI.")
                elif '.' in config_key: # Basic support for direct nested keys like display.width
                     parts = config_key.split('.')
                     d = self.config
                     try:
                         for p in parts[:-1]: d = d[p]
                         last_key = parts[-1]
                         if last_key in d and not isinstance(d[last_key], dict):
                             try: d[last_key] = type(d[last_key])(value)
                             except (ValueError, TypeError): d[last_key] = value
                         else:
                             logger.warning(f"Cannot set nested CLI arg '{config_key}'.")
                     except KeyError:
                          logger.warning(f"Invalid nested CLI arg key '{config_key}'.")


    def _validate_calibration(self):
        # (Keep unchanged)
        calib = self.config["calibration"]; default = self.DEFAULT_CONFIG["calibration"]; rx = ry = False
        nums = ["x_min", "x_max", "y_min", "y_max"]
        if not all(isinstance(calib.get(k), (int, float)) for k in nums): rx=ry=True; logger.warning("Non-numeric calibration values, reset.")
        if not (0<=calib.get("x_min",-1)<calib.get("x_max",-1)<=1 and abs(calib.get("x_max",0)-calib.get("x_min",0))>=0.05): rx=True; logger.warning("Invalid X calibration, reset.")
        if not (0<=calib.get("y_min",-1)<calib.get("y_max",-1)<=1 and abs(calib.get("y_max",0)-calib.get("y_min",0))>=0.05): ry=True; logger.warning("Invalid Y calibration, reset.")
        if rx: calib["x_min"], calib["x_max"] = default["x_min"], default["x_max"]
        if ry: calib["y_min"], calib["y_max"] = default["y_min"], default["y_max"]


    def _validate_display_dims(self):
        # (Keep unchanged)
        w, h = self.get("display_width"), self.get("display_height")
        if not ((w is None) or (isinstance(w, int) and w > 0)): self.set("display_width", None); logger.warning("Invalid display_width.")
        if not ((h is None) or (isinstance(h, int) and h > 0)): self.set("display_height", None); logger.warning("Invalid display_height.")

    def get(self, key, default=None):
        # (Keep unchanged)
        try: val = self.config; keys = key.split('.'); [val := val[k] for k in keys]; return val
        except Exception: return default

    def set(self, key, value):
        # (Keep unchanged)
        try: d = self.config; keys = key.split('.'); [d := d[k] for k in keys[:-1]]; d[keys[-1]] = value
        except Exception: logger.error(f"Failed to set config key '{key}'."); return False
        if key.startswith("calibration."): self._validate_calibration()
        if key.startswith("display_"): self._validate_display_dims(); return True # Return True on success


    def save(self):
        # (Keep unchanged)
        try:
            save_cfg = json.loads(json.dumps(self.config)); save_cfg.get("calibration", {}).pop("active", None)
            # Store relative paths if they are within the project structure
            cursor_path = save_cfg.get('custom_cursor_path')
            if cursor_path and cursor_path.startswith(self._PROJECT_ROOT):
                save_cfg['custom_cursor_path'] = os.path.relpath(cursor_path, self._PROJECT_ROOT)

            apps_path = save_cfg.get('apps_csv_path')
            if apps_path and apps_path.startswith(self._PROJECT_ROOT):
                save_cfg['apps_csv_path'] = os.path.relpath(apps_path, self._PROJECT_ROOT)

            with open(self.CONFIG_FILENAME, 'w') as f: json.dump(save_cfg, f, indent=2)
            logger.info(f"Config saved to {self.CONFIG_FILENAME}")
        except Exception as e: logger.error(f"Error saving config: {e}")

    def __getitem__(self, key): return self.get(key)
    def __setitem__(self, key, value): self.set(key, value)