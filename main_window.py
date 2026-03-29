"""
Project: openSMT
File: main_window.py
Description: Main GUI with proper QThread/QObject lifecycle management.
License: MIT
"""

import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QApplication, QDockWidget, QScrollArea, QLabel)
from PySide6.QtCore import Qt, Signal, Slot, QThread
from PySide6.QtGui import QImage, QPixmap

from drivers.camera_driver import CameraWorker
from drivers.gcode_handler import GCodeSerialThread
from widgets.gcode_terminal import GCodeTerminalWidget

class CameraDisplayWidget(QWidget):
    def __init__(self, name):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.label = QLabel(f"Connecting {name}...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background: black; color: #0f0; border: 1px solid #333;")
        self.label.setMinimumSize(400, 300)
        self.layout.addWidget(self.label)

    @Slot(np.ndarray)
    def update_frame(self, frame):
        h, w, ch = frame.shape
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))

class MainWindow(QMainWindow):
    def __init__(self, config_data):
        super().__init__()
        self.config = config_data
        self.gcode_threads = {}
        self.camera_threads = [] # Keep track of the threads
        self.camera_workers = [] # Keep track of the workers

        self.setWindowTitle(f"openSMT - {self.config.get('device_name', 'System')}")
        self.resize(1400, 900)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.container_widget = QWidget()
        self.camera_layout = QHBoxLayout(self.container_widget)
        self.camera_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.scroll_area.setWidget(self.container_widget)
        self.setCentralWidget(self.scroll_area)

        self.setup_hardware()

    def setup_hardware(self):
        drivers = self.config.get("drivers", {})

        # 1. G-Code Terminals
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

        # 2. Camera Workers (The Correct QThread Pattern)
        for c_cfg in drivers.get("camera", []):
            name = c_cfg.get("name", "Cam")
            idx = c_cfg.get("index", 0)
            dpi_map = c_cfg.get("dpi_settings", [])

            display = CameraDisplayWidget(name)
            self.camera_layout.addWidget(display)

            # Create Thread and Worker
            thread = QThread()
            worker = CameraWorker(idx, name, dpi_map)
            worker.moveToThread(thread)

            # Connect Signals
            thread.started.connect(worker.process)
            worker.image_ready.connect(display.update_frame)
            
            # Save references
            self.camera_threads.append(thread)
            self.camera_workers.append(worker)
            
            thread.start()

    def closeEvent(self, event):
        for thread in self.gcode_threads.values():
            thread.stop()
        
        for worker, thread in zip(self.camera_workers, self.camera_threads):
            worker.stop()
            thread.quit()
            thread.wait()
        event.accept()