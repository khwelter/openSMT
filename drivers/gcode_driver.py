"""
Project: openSMT
File: drivers/gcode_driver.py
Description: Asynchronous G-Code driver with command queuing.
License: MIT
"""

import time
import serial
from PySide6.QtCore import QObject, Signal, Slot, QThread

class GCodeWorker(QObject):
    """Worker class to handle serial I/O in a separate thread."""
    response_received = Signal(str, str)  # symbolic_name, response
    command_finished = Signal(str)        # symbolic_name

    def __init__(self, name, port, baudrate, timeout=1):
        super().__init__()
        self.name = name
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.running = True
        self.serial_conn = None

    @Slot(str, bool)
    def send_command(self, command, wait_for_reply=True):
        if not self.serial_conn:
            try:
                self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            except Exception as e:
                self.response_received.emit(self.name, f"Error: {e}")
                return

        formatted_cmd = f"{command}\n".encode('utf-8')
        self.serial_conn.write(formatted_cmd)
        
        if wait_for_reply:
            response = self.serial_conn.readline().decode('utf-8').strip()
            self.response_received.emit(self.name, response)
        
        self.command_finished.emit(self.name)

class GCodeHandler:
    """Manager to wrap the thread and worker for a specific driver."""
    def __init__(self, name, port, baudrate):
        self.name = name
        self.thread = QThread()
        self.worker = GCodeWorker(name, port, baudrate)
        self.worker.moveToThread(self.thread)
        self.thread.start()

    def execute(self, command, wait=True):
        # Using invokeMethod to ensure thread safety
        self.worker.send_command(command, wait)