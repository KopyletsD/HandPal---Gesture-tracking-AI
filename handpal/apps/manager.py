import os
import subprocess
import webbrowser
import logging
import csv # Moved CSV handling here

logger = logging.getLogger(__name__)

class AppManager:
    def __init__(self, config):
        self.config = config
        self.csv_path = self.config.get('apps_csv_path') # Get path from config
        self.applications = self._load_applications()

    def _ensure_csv_exists(self):
        """Creates a default apps CSV if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            logger.info(f"Applications file not found. Creating default at: {self.csv_path}")
            try:
                os.makedirs(os.path.dirname(self.csv_path), exist_ok=True) # Ensure directory exists
                with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Add header row
                    writer.writerow(['label', 'path', 'color', 'icon'])
                    # Add some default examples
                    writer.writerow(['Calculator', 'calc.exe' if os.name == 'nt' else 'gnome-calculator', '#0078D7', '🧮']) # OS specific example
                    writer.writerow(['Browser', 'https://duckduckgo.com', '#DE5833', '🌐'])
                    writer.writerow(['Notepad', 'notepad.exe' if os.name == 'nt' else 'gedit', '#FFDA63', '📝']) # OS specific example
                logger.info(f"Created default applications file: {self.csv_path}")
            except Exception as e:
                logger.error(f"Failed to create default applications file '{self.csv_path}': {e}")
                return False
        return True

    def _load_applications(self):
        """Reads application data from the CSV file."""
        if not self._ensure_csv_exists():
             return [{'label':'Error Loading','path':'','color':'#F00','icon':'⚠'}] # Return error indicator

        apps = []
        default_color = '#555555'
        default_icon = '🚀'
        try:
            with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
                # Use DictReader for easier access by column name
                reader = csv.DictReader(f)
                if not reader.fieldnames or 'label' not in reader.fieldnames or 'path' not in reader.fieldnames:
                     logger.error(f"CSV file '{self.csv_path}' missing required columns ('label', 'path').")
                     return [{'label':'Bad CSV Format','path':'','color':'#F00','icon':'⚠'}]

                for i, row in enumerate(reader):
                    label = row.get('label', f'App {i+1}').strip()
                    path = row.get('path', '').strip()

                    if not path:
                        logger.warning(f"Skipping row {i+1} in '{self.csv_path}' (missing path).")
                        continue

                    color = row.get('color', default_color).strip()
                    # Basic color validation (starts with #, 7 chars long hex)
                    if not (color.startswith('#') and len(color) == 7):
                        try: int(color[1:], 16) # Check if hex parsable
                        except ValueError: color = default_color # Fallback if invalid hex
                    icon = row.get('icon', default_icon).strip()

                    apps.append({'label': label, 'path': path, 'color': color, 'icon': icon})

            logger.info(f"Loaded {len(apps)} applications from {self.csv_path}")
        except FileNotFoundError:
             logger.error(f"Applications file not found during load: {self.csv_path}")
             apps = [{'label':'File Not Found','path':'','color':'#F00','icon':'⚠'}]
        except Exception as e:
            logger.error(f"Error reading applications file '{self.csv_path}': {e}")
            apps = [{'label':'Error Reading','path':'','color':'#F00','icon':'⚠'}]
        return apps

    def get_applications(self):
        """Returns the list of loaded application dictionaries."""
        return self.applications

    def reload_applications(self):
         """Forces a reload of applications from the CSV file."""
         logger.info("Reloading applications from CSV...")
         self.applications = self._load_applications()
         # Potentially signal UI to update here if needed

    def launch(self, app_label_or_path):
        """Launches an application by its label or direct path."""
        path_to_launch = None
        # Check if it's a label from our list first
        for app in self.applications:
            if app['label'] == app_label_or_path:
                path_to_launch = app['path']
                break

        # If not found by label, assume it's a direct path
        if path_to_launch is None:
            path_to_launch = app_label_or_path

        if not path_to_launch:
            logger.warning("Launch attempt with empty path/label.")
            return

        logger.info(f"Attempting to launch: {path_to_launch}")
        try:
            if path_to_launch.startswith(('http://', 'https://')):
                webbrowser.open(path_to_launch)
                logger.info(f"Opened URL in browser: {path_to_launch}")
            elif os.name == 'nt':
                # os.startfile is generally preferred on Windows for opening files/apps
                # It behaves like double-clicking the file in Explorer.
                os.startfile(path_to_launch)
                logger.info(f"Launched via os.startfile: {path_to_launch}")
            else:
                # For Linux/macOS, subprocess.Popen is more common.
                # Use shell=True cautiously, ensure path isn't user-controlled without sanitization
                # Or, split into command and args if needed.
                subprocess.Popen([path_to_launch], shell=False) # shell=False is safer
                logger.info(f"Launched via subprocess.Popen: {path_to_launch}")

        except FileNotFoundError:
            logger.error(f"Launch failed: File or application not found at '{path_to_launch}'")
        except OSError as e:
             # Handles cases like permission errors or if path is a directory
             logger.error(f"Launch failed for '{path_to_launch}': Operating system error - {e}")
        except Exception as e:
            logger.exception(f"Unexpected error launching '{path_to_launch}': {e}")