"""
Project: openSMT
File: main_window.py
Description: Main UI with Scrollable Cameras and Docked G-Code Terminals.
License: MIT
"""

import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QApplication, QDockWidget, QScrollArea)
from PySide6.QtCore import Qt, Signal, Slot, QThread

from drivers.camera_driver import CameraWorker
from drivers.gcode_handler import GCodeSerialThread
from widgets.gcode_terminal import GCodeTerminalWidget

class CameraWidget(QWidget): # Wrapped in QWidget for better layout control
    def __init__(self, name):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.label = QLabel(f"Connecting {name}...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background: black; color: #0f0; border: 2px solid #222;")
        self.label.setMinimumSize(400, 300)
        self.layout.addWidget(self.label)

    @Slot(np.ndarray)
    def update_frame(self, frame):
        h, w, ch = frame.shape
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

# Note: Using the imports from your existing files for CameraWorker and GCodeSerialThread
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel

class MainWindow(QMainWindow):
    def __init__(self, config_data):
        super().__init__()
        self.config = config_data
        self.gcode_threads = {}
        self.camera_workers = []

        self.setWindowTitle(f"openSMT - {self.config.get('device_name', 'System')}")
        self.resize(1400, 900)

        # --- Central Area: Camera Feeds ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.central_widget = QWidget()
        self.camera_layout = QHBoxLayout(self.central_widget)
        self.scroll_area.setWidget(self.central_widget)
        self.setCentralWidget(self.scroll_area)

        self.setup_hardware()

    def setup_hardware(self):
        drivers = self.config.get("drivers", {})

        # 1. Setup G-Code Terminals (Docks)
        for g_cfg in drivers.get("gcode", []):
            name = g_cfg.get("name", "G-Code")
            port = g_cfg.get("port", "/dev/tty.usb")
            baud = g_cfg.get("baudrate", 115200)

            terminal = GCodeTerminalWidget(name)
            dock = QDockWidget(f"Terminal: {name}", self)
            dock.setWidget(terminal)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

            thread = GCodeSerialThread(port, baud)
            terminal.command_submitted.connect(thread.send_command)
            thread.line_received.connect(lambda line, t=terminal: t.append_log(f"RX: {line}"))
            thread.status_msg.connect(lambda msg, t=terminal: t.append_log(f"SYS: {msg}"))
            
            self.gcode_threads[name] = thread
            thread.start()

        # 2. Setup Cameras (Restored)
        for c_cfg in drivers.get("camera", []):
            name = c_cfg.get("name", "Cam")
            idx = c_cfg.get("index", 0)

            cam_widget = CameraWidget(name)
            self.camera_layout.addWidget(cam_widget)

            worker = CameraWorker(idx, name)
            worker.image_ready.connect(cam_widget.update_frame)
            worker.start()
            self.camera_workers.append(worker)

    def closeEvent(self, event):
        for thread in self.gcode_threads.values(): thread.stop()
        for worker in self.camera_workers: worker.stop()
        event.accept()