"""
Project: openSMT
File: main.py
Description: Entry point with OS-specific config loading.
License: MIT
"""

import sys
import os
import platform
from PySide6.QtWidgets import QApplication
from config_manager import ConfigManager
from main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 1. Detect OS
    current_os = platform.system().lower() # 'darwin' for macOS, 'linux' for Linux
    os_folder = "macos" if current_os == "darwin" else "linux"
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Path becomes: config/macos/ or config/linux/
    config_path = os.path.join(base_dir, "config", os_folder)
    
    if not os.path.exists(config_path):
        os.makedirs(config_path)
        print(f"Created missing config directory: {config_path}")

    # 2. Load Config
    cfg_manager = ConfigManager(config_path)
    config_data = cfg_manager.load_main_config("main.json")
    
    # Fallback if file is missing
    if not config_data:
        config_data = {
            "device_name": f"openSMT ({current_os})",
            "drivers": {"camera": [], "gcode": []}
        }

    # 3. Launch
    window = MainWindow(config_data)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()