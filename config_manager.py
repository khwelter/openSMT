"""
Project: openSMT
File: config_manager.py
Description: Handles JSON configuration loading, nesting, and hot-reloading.
License: MIT
"""

import os
import json
import logging
from PySide6.QtCore import QObject, Signal, QFileSystemWatcher

class ConfigManager(QObject):
    # Signal emitted when a config file is modified externally
    config_changed = Signal(str)

    def __init__(self, config_dir="config"):
        super().__init__()
        self.config_dir = config_dir
        self.data = {}
        self.watcher = QFileSystemWatcher()
        self.watcher.fileChanged.connect(self._on_file_changed)
        
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)

    def load_main_config(self, filename="main.json"):
        """Loads the master config and recursively loads referenced units."""
        path = os.path.join(self.config_dir, filename)
        self.data = self._load_json(path)
        self._watch_file(path)
        
        # In a real scenario, we would parse 'units' or 'drivers' 
        # references here to load sub-files.
        return self.data

    def _load_json(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Error loading {path}: {e}")
            return {}

    def _watch_file(self, path):
        if path not in self.watcher.files():
            self.watcher.addPath(path)

    def _on_file_changed(self, path):
        logging.info(f"Config file changed: {path}")
        # Reload the specific data (simplified for now)
        self.config_changed.emit(path)

    def save_config(self, filename, data):
        path = os.path.join(self.config_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)