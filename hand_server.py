import asyncio
import websockets
import json
import logging
import os
import sys
import time
import ctypes
import webbrowser
import subprocess
from pynput.mouse import Controller, Button as PynputButton

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("handpal_server.log"), logging.StreamHandler()]
)
logger = logging.getLogger("HandPalServer")

# --- Global Mouse Controller ---
mouse = Controller()
screen_size = (1920, 1080) 
try:
    if os.name == 'nt':
        user32 = ctypes.windll.user32
        screen_size = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    logger.info(f"Server screen size detected/defaulted to: {screen_size}")
except Exception as e:
    logger.warning(f"Could not detect screen size, using default {screen_size}: {e}")


# -----------------------------------------------------------------------------
# Config Class (Simplified for Server)
# -----------------------------------------------------------------------------
class Config:
    DEFAULT_CONFIG = {
        "custom_cursor_path": "red_cursor.cur", 
        "server_port": 8765,
        "server_host": "localhost"
    }
    CONFIG_FILENAME = os.path.expanduser("~/.handpal_config.json") 

    def _deep_update(self, target, source):
        for k, v in source.items():
            if isinstance(v, dict) and k in target and isinstance(target[k], dict):
                self._deep_update(target[k], v)
            else:
                target[k] = v

    def __init__(self, args=None):
        self.config = json.loads(json.dumps(self.DEFAULT_CONFIG)) 
        if os.path.exists(self.CONFIG_FILENAME):
            try:
                with open(self.CONFIG_FILENAME, 'r') as f:
                    loaded_config = json.load(f)
                    relevant_loaded_config = {k: v for k, v in loaded_config.items() if k in self.config}
                    self._deep_update(self.config, relevant_loaded_config)
                logger.info(f"Server-relevant config loaded from {self.CONFIG_FILENAME}")
            except Exception as e:
                logger.error(f"Error loading server config: {e}")
        
        if args: 
            if getattr(args, 'port', None) is not None:
                self.config['server_port'] = args.port
            if getattr(args, 'host', None) is not None:
                self.config['server_host'] = args.host
            if getattr(args, 'cursor_file', None) is not None:
                self.config['custom_cursor_path'] = args.cursor_file
        logger.debug(f"Final server config: {self.config}")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value): 
        self.config[key] = value
        return True

    def save(self): 
        try:
            full_config_to_save = {}
            if os.path.exists(self.CONFIG_FILENAME):
                with open(self.CONFIG_FILENAME, 'r') as f:
                    full_config_to_save = json.load(f)
            
            for k, v in self.config.items():
                full_config_to_save[k] = v
            
            with open(self.CONFIG_FILENAME, 'w') as f:
                json.dump(full_config_to_save, f, indent=2)
            logger.info(f"Config (with server updates) saved to {self.CONFIG_FILENAME}")
        except Exception as e:
            logger.error(f"Error saving server config: {e}")

# --- Windows Cursor Functions (Server-Side) ---
def set_custom_cursor(cursor_path_config_val):
    if os.name != 'nt': return False
    try:
        cursor_path = cursor_path_config_val
        if not os.path.isabs(cursor_path) and not os.path.exists(cursor_path):
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError: 
                script_dir = os.getcwd()
            potential_path = os.path.join(script_dir, cursor_path)
            if os.path.exists(potential_path):
                cursor_path = potential_path
            else:
                logger.warning(f"Cursor file '{cursor_path_config_val}' not found directly or relative to script '{script_dir}'.")
                return False

        if not os.path.exists(cursor_path):
            raise FileNotFoundError(f"Cursor file not found: {cursor_path}")
        
        user32 = ctypes.windll.user32
        hCursor = user32.LoadCursorFromFileW(cursor_path)
        if hCursor == 0:
            raise Exception(f"Failed to load cursor (Error {ctypes.get_last_error()}).")
        
        OCR_NORMAL = 32512 
        if not user32.SetSystemCursor(hCursor, OCR_NORMAL):
            raise Exception(f"Failed to set system cursor (Error {ctypes.get_last_error()}).")
        
        logger.info(f"Custom cursor '{os.path.basename(cursor_path)}' set on server.")
        return True
    except Exception as e:
        logger.error(f"Error setting custom cursor on server: {e}")
        return False

def restore_default_cursor():
    if os.name != 'nt': return
    try:
        user32 = ctypes.windll.user32
        SPI_SETCURSORS = 0x57
        if not user32.SystemParametersInfoW(SPI_SETCURSORS, 0, None, 3): 
            raise Exception(f"Failed to restore default cursors (Error {ctypes.get_last_error()}).")
        logger.info("Default system cursors restored on server.")
    except Exception as e:
        logger.error(f"Error restoring default cursors on server: {e}")

# --- Action Handler ---
class ActionHandler:
    def __init__(self, current_config, websocket_connection=None):
        self.config = current_config
        self.websocket = websocket_connection

    async def handle_message(self, message_str):
        try:
            data = json.loads(message_str)
            action_type = data.get("type")

            if self.config is None:
                logger.error("ActionHandler's config is None. Cannot process message.")
                return

            if action_type == "move":
                norm_x, norm_y = data.get("x"), data.get("y")
                if norm_x is not None and norm_y is not None:
                    screen_w, screen_h = screen_size
                    target_x = int(norm_x * screen_w)
                    target_y = int(norm_y * screen_h)
                    target_x = max(0, min(target_x, screen_w - 1))
                    target_y = max(0, min(target_y, screen_h - 1))
                    mouse.position = (target_x, target_y)
            
            elif action_type == "click":
                button_str = data.get("button", "left")
                count = data.get("count", 1)
                button = PynputButton.left if button_str == "left" else PynputButton.right
                mouse.click(button, count)
                logger.info(f"Executed click: {button_str} x{count}")

            elif action_type == "scroll":
                dy = data.get("dy", 0) 
                if dy != 0:
                    mouse.scroll(0, int(dy)) 
                    logger.debug(f"Executed scroll: dy={dy}")
            
            elif action_type == "drag":
                drag_action = data.get("action")
                if drag_action == "start":
                    mouse.press(PynputButton.left)
                    logger.info("Drag started")
                elif drag_action == "end":
                    mouse.release(PynputButton.left)
                    logger.info("Drag ended")

            elif action_type == "launch":
                path = data.get("path")
                if path == "@tutorial":
                    logger.info("Client requested tutorial. Sending command to client.")
                    if self.websocket:
                        await self.websocket.send(json.dumps({"type": "command", "command": "start_tutorial"}))
                elif path:
                    self.launch_application(path)
            
            elif action_type == "set_config": 
                key = data.get("key")
                value = data.get("value")
                if key and value is not None:
                    if key == "custom_cursor_path": 
                        if self.config.set(key, value):
                            set_custom_cursor(value) 
                            self.config.save() 
                    else:
                        logger.warning(f"Client tried to set unhandled server config key: {key}")
            
            elif action_type == "request_initial_cursor": 
                cursor_path = self.config.get('custom_cursor_path')
                if cursor_path:
                    set_custom_cursor(cursor_path)

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {message_str}")
        except Exception as e:
            logger.error(f"Error handling action: {e}, message: {message_str}")

    def launch_application(self, path):
        if not path:
            logger.warning("Launch attempt with empty path on server.")
            return
        logger.info(f"Server launching: {path}")
        try:
            if path.startswith(('http://', 'https://')):
                webbrowser.open(path)
            elif os.name == 'nt':
                try:
                    subprocess.Popen(path, shell=True) 
                except FileNotFoundError:
                     os.startfile(path) 
            else: 
                subprocess.Popen(path)
        except FileNotFoundError:
            logger.error(f"Launch failed on server: File not found '{path}'")
        except Exception as e:
            logger.error(f"Launch failed on server '{path}': {e}")


connected_clients = set()
server_config_instance = None 

# FIXED: Updated handler function signature to match newer websockets library
async def handler(websocket):
    """WebSocket connection handler - updated for newer websockets library versions"""
    try:
        global server_config_instance 
        
        client_address = websocket.remote_address
        logger.info(f"Client {client_address} connected")
        connected_clients.add(websocket)
        
        if server_config_instance is None:
            logger.error(f"CRITICAL: server_config_instance is None when handler for {client_address} started. This should not happen.")
            await websocket.close(code=1011, reason="Server configuration error")
            return

        action_handler_instance = ActionHandler(server_config_instance, websocket)

        async for message in websocket:
            await action_handler_instance.handle_message(message)

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Client {client_address} disconnected normally.")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"Client {client_address} connection closed with error: {e}")
    except Exception as e:
        logger.exception(f"UNEXPECTED ERROR in WebSocket handler for {client_address}: {e}")
        try:
            await websocket.close(code=1011, reason="Internal server error during handling")
        except:
            pass
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.info(f"Handler for {client_address} finished processing.")

async def main_server(host, port):
    """Main server function - updated for newer websockets library"""
    try:
        # Check websockets library version and use appropriate method
        import websockets
        websockets_version = getattr(websockets, '__version__', '0.0')
        logger.info(f"Using websockets library version: {websockets_version}")
        
        # For newer versions (11.0+), use the new API
        if hasattr(websockets, 'serve'):
            async with websockets.serve(handler, host, port):
                logger.info(f"HandPal WebSocket Server started on ws://{host}:{port}")
                await asyncio.Future()  # Run forever
        else:
            # Fallback for older versions
            start_server = websockets.serve(handler, host, port)
            logger.info(f"HandPal WebSocket Server started on ws://{host}:{port}")
            await start_server
            await asyncio.Future()  # Run forever
            
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

def parse_server_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="HandPal Server")
    parser.add_argument('--port', type=int, help='Port for the WebSocket server.')
    parser.add_argument('--host', type=str, help='Host address for the WebSocket server (e.g., 0.0.0.0).')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging for server.')
    parser.add_argument('--cursor-file', type=str, help='Path to custom .cur file (Windows only) for server to use.')
    parser.add_argument('--reset-config', action='store_true', help='Delete relevant server parts from config and exit.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_server_arguments()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers:
            h.setLevel(logging.DEBUG)
        logger.info("Server debug logging enabled.")

    server_config_instance = Config(args)
    
    if args.reset_config:
        logger.info("Resetting server-specific keys in config to default (if they exist).")
        default_server_conf = Config.DEFAULT_CONFIG
        existing_conf_path = Config.CONFIG_FILENAME
        current_full_config = {}
        if os.path.exists(existing_conf_path):
            try:
                with open(existing_conf_path, 'r') as f:
                    current_full_config = json.load(f)
            except Exception as e:
                logger.error(f"Could not load existing config for reset: {e}")
        for key in default_server_conf:
            current_full_config[key] = default_server_conf[key] 
        try:
            with open(existing_conf_path, 'w') as f:
                json.dump(current_full_config, f, indent=2)
            logger.info(f"Server-specific config keys reset/set to default in {existing_conf_path}")
        except Exception as e:
            logger.error(f"Failed to save reset config: {e}")
        sys.exit(0)

    initial_cursor_path = server_config_instance.get('custom_cursor_path')
    cursor_set_on_startup = False
    if initial_cursor_path:
        cursor_set_on_startup = set_custom_cursor(initial_cursor_path)

    try:
        asyncio.run(main_server(
            server_config_instance.get('server_host'),
            server_config_instance.get('server_port')
        ))
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    finally:
        if cursor_set_on_startup: 
            logger.info("Restoring default system cursor on server exit...")
            restore_default_cursor()
        logger.info("HandPal Server stopped.")