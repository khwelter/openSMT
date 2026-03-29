"""
Project: openSMT
File: drivers/camera_driver.py
Description: OpenCV camera handler with DPI interpolation and threading.
License: MIT
"""

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, QThread, QTimer

class CameraWorker(QObject):
    image_ready = Signal(np.ndarray)

    def __init__(self, camera_index, name, dpi_map):
        super().__init__()
        self.camera_index = camera_index
        self.name = name
        self.dpi_map = dpi_map # Use this later for interpolation
        self.keep_running = True

    def get_dots_per_mm(self, current_height):
        """Interpolates dots-per-mm based on Z-height."""
        heights = [item['height'] for item in self.dpi_map]
        dpis = [item['dpi'] for item in self.dpi_map]
        # Linear interpolation
        return np.interp(current_height, heights, dpis)

    @Slot()
    def process(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        while self.keep_running:
            ret, frame = self.cap.read()
            if ret:
                # Here we would apply vision pipelines
                self.image_ready.emit(frame)
            QThread.msleep(33)  # ~30 FPS

    def stop(self):
        self.keep_running = False
        if self.cap:
            self.cap.release()