# handpal/apps/manager.py
import os
import subprocess
import webbrowser
import logging
import csv

logger = logging.getLogger(__name__)

class AppManager:
    def __init__(self, config):
        self.config = config
        # Get path from config, ensuring it's absolute for reliability
        csv_path_from_config = self.config.get('apps_csv_path')
        if csv_path_from_config and not os.path.isabs(csv_path_from_config):
             # If relative, assume it's relative to project root (where config likely resolved it)
             # Or, use a known base directory if config doesn't guarantee absolute paths
             project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
             self.csv_path = os.path.abspath(os.path.join(project_root, csv_path_from_config))
             logger.warning(f"Relative apps_csv_path found in config, resolved to: {self.csv_path}")
        elif csv_path_from_config: # It's absolute
             self.csv_path = csv_path_from_config
        else: # Path missing in config, use default logic (which should also be absolute now)
             self.csv_path = config.DEFAULT_APPS_CSV_PATH # Assuming Config class defines this absolutely

        logger.info(f"AppManager using CSV path: {self.csv_path}")
        self.applications = self._load_applications() # Initial load

    def _ensure_csv_exists(self):
        """Creates a default apps CSV if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            logger.info(f"Applications file not found. Creating default at: {self.csv_path}")
            try:
                os.makedirs(os.path.dirname(self.csv_path), exist_ok=True) # Ensure directory exists
                with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['label', 'path', 'color', 'icon']) # Header
                    # Defaults
                    writer.writerow(['Calculator', 'calc.exe' if os.name == 'nt' else 'gnome-calculator', '#0078D7', '🧮'])
                    writer.writerow(['Browser', 'https://duckduckgo.com', '#DE5833', '🌐'])
                    writer.writerow(['Notepad', 'notepad.exe' if os.name == 'nt' else 'gedit', '#FFDA63', '📝'])
                logger.info(f"Created default applications file: {self.csv_path}")
            except Exception as e:
                logger.error(f"Failed to create default applications file '{self.csv_path}': {e}")
                return False
        return True

    def _load_applications(self):
        """Reads application data from the CSV file."""
        if not self._ensure_csv_exists():
             logger.error("Cannot load applications, failed to ensure CSV exists.")
             return [{'label':'Error Loading','path':'','color':'#F00','icon':'⚠'}]

        apps = []
        default_color = '#555555'
        default_icon = '🚀'
        try:
            with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
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
                    if not (color.startswith('#') and len(color) == 7):
                        try: int(color[1:], 16) # Check hex validity
                        except ValueError: color = default_color
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

    # --- ADDED METHOD ---
    def reload_applications(self):
         """Forces a reload of applications from the CSV file."""
         logger.info(f"Reloading applications from CSV: {self.csv_path}")
         # Re-call the internal load method to refresh self.applications
         self.applications = self._load_applications()
         logger.info(f"Reload complete. Found {len(self.applications)} applications.")
    # --- END OF ADDED METHOD ---

    def launch(self, app_label_or_path):
        """Launches an application by its label or direct path."""
        path_to_launch = None
        # Try matching label first
        for app in self.applications:
            if app['label'] == app_label_or_path:
                path_to_launch = app['path']
                break

        # If no label match, assume it's a direct path
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
                os.startfile(path_to_launch)
                logger.info(f"Launched via os.startfile: {path_to_launch}")
            else:
                # Consider using shlex.split if path might contain spaces and needs args
                subprocess.Popen([path_to_launch], shell=False) # Safer default
                logger.info(f"Launched via subprocess.Popen: {path_to_launch}")

        except FileNotFoundError:
            logger.error(f"Launch failed: File or application not found at '{path_to_launch}'")
        except OSError as e:
             logger.error(f"Launch failed for '{path_to_launch}': OS error - {e}")
        except Exception as e:
            logger.exception(f"Unexpected error launching '{path_to_launch}': {e}")