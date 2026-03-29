"""
Project: openSMT
File: widgets/gcode_terminal.py
Description: UI Widget for manual G-Code entry and logging.
License: MIT
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLineEdit, QLabel
from PySide6.QtCore import Qt, Signal

class GCodeTerminalWidget(QWidget):
    command_submitted = Signal(str)

    def __init__(self, device_name):
        super().__init__()
        self.device_name = device_name
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title/Label
        self.label = QLabel(f"Device: {self.device_name}")
        self.label.setStyleSheet("font-weight: bold; color: #AAA;")
        layout.addWidget(self.label)

        # Output Log (History)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("""
            background-color: #1a1a1a; 
            color: #00d4ff; 
            font-family: 'Menlo', 'Courier';
            font-size: 11px;
        """)
        layout.addWidget(self.log)

        # Input Line
        self.input = QLineEdit()
        self.input.setPlaceholderText("Type G-Code (e.g. G0 X10) and press Enter...")
        self.input.setStyleSheet("""
            background-color: #333; 
            color: #fff; 
            border: 1px solid #555;
            padding: 5px;
        """)
        self.input.returnPressed.connect(self.on_submit)
        layout.addWidget(self.input)

    def on_submit(self):
        cmd = self.input.text().strip()
        if cmd:
            self.append_log(f"TX: {cmd}", is_tx=True)
            self.command_submitted.emit(cmd)
            self.input.clear()

    def append_log(self, text, is_tx=False):
        color = "#ffffff" if is_tx else "#00d4ff"
        self.log.append(f'<span style="color:{color};">{text}</span>')
        # Auto-scroll to bottom
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())