"""
Project: openSMT
File: drivers/gcode_handler.py
Description: Threaded serial handler for G-Code devices.
License: MIT
"""

import serial
import time
from PySide6.QtCore import QThread, Signal, Slot

class GCodeSerialThread(QThread):
    """
    Handles the actual serial port I/O in a background thread.
    """
    line_received = Signal(str) # Data from device
    status_msg = Signal(str)    # Internal status updates

    def __init__(self, port, baudrate, timeout=0.1):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.keep_running = True
        self.serial_conn = None
        self.command_queue = []

    def run(self):
        try:
            self.serial_conn = serial.Serial(
                self.port, self.baudrate, timeout=self.timeout
            )
            self.status_msg.emit(f"Connected to {self.port}")
        except Exception as e:
            self.status_msg.emit(f"Connection Error: {e}")
            return

        while self.keep_running:
            # 1. Check for incoming data
            if self.serial_conn.in_waiting > 0:
                line = self.serial_conn.readline().decode('utf-8', errors='replace').strip()
                if line:
                    self.line_received.emit(line)

            # 2. Check for outgoing commands
            if self.command_queue:
                cmd = self.command_queue.pop(0)
                self.serial_conn.write(f"{cmd}\n".encode('utf-8'))

            self.msleep(10) # Prevent CPU spinning

        if self.serial_conn:
            self.serial_conn.close()

    @Slot(str)
    def send_command(self, cmd):
        self.command_queue.append(cmd)

    def stop(self):
        self.keep_running = False
        self.wait()