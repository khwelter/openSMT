from __future__ import annotations

import asyncio
from concurrent.futures import Future
from pathlib import Path
import threading

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from opensmt.messaging import BusNode, SCPIMessage


class MonitorBackend(QObject):
    text_received = Signal(str)
    status_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._node: BusNode | None = None

    def start_loop(self) -> None:
        if self._thread:
            return

        def runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.run_forever()

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._loop:
            if self._node:
                future = asyncio.run_coroutine_threadsafe(self._node.close(), self._loop)
                future.result(timeout=3)
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread = None
        self._loop = None
        self._node = None

    def connect_bus(self, host: str, port: int, name: str) -> None:
        self.start_loop()
        assert self._loop is not None

        async def _connect() -> None:
            self._node = BusNode(name=name, host=host, port=port)
            await self._node.connect()

            async def on_text(packet: dict, msg: SCPIMessage) -> None:
                self.text_received.emit(
                    f"[{packet.get('source', '?')} -> {packet.get('target', '*')}] {msg.raw}"
                )

            async def on_binary(packet: dict, payload: bytes) -> None:
                self.text_received.emit(
                    f"[{packet.get('source', '?')} -> {packet.get('target', '*')}] "
                    f"BINARY {packet.get('topic', 'BINARY')} {payload.hex()}"
                )

            self._node.on_text(on_text)
            self._node.on_binary(on_binary)
            self.status_changed.emit("Verbunden")

        fut: Future[None] = asyncio.run_coroutine_threadsafe(_connect(), self._loop)
        fut.result(timeout=5)

    def send_text(self, text: str) -> None:
        if not self._loop or not self._node:
            return
        asyncio.run_coroutine_threadsafe(self._node.send_text(text), self._loop)

    def play_file(self, path: str) -> None:
        if not self._loop or not self._node:
            return

        async def _play() -> None:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                item = line.strip()
                if not item or item.startswith("#"):
                    continue
                if item.startswith("SLEEP "):
                    await asyncio.sleep(float(item.split(maxsplit=1)[1]))
                    continue
                if item.startswith("BIN "):
                    payload = bytes.fromhex(item.split(maxsplit=1)[1].replace(" ", ""))
                    await self._node.send_binary(payload, topic="PLAYBACK")
                    continue
                await self._node.send_text(item)

        asyncio.run_coroutine_threadsafe(_play(), self._loop)


class MonitorWindow(QMainWindow):
    def __init__(self, host: str, port: int, name: str) -> None:
        super().__init__()
        self.setWindowTitle("openSMT Monitor")
        self.resize(900, 540)

        self.backend = MonitorBackend()

        root = QWidget(self)
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)

        top = QHBoxLayout()
        self.host_edit = QLineEdit(host)
        self.port_edit = QLineEdit(str(port))
        self.name_edit = QLineEdit(name)
        self.connect_button = QPushButton("Verbinden")
        self.status = QLabel("Getrennt")

        top.addWidget(QLabel("Host"))
        top.addWidget(self.host_edit)
        top.addWidget(QLabel("Port"))
        top.addWidget(self.port_edit)
        top.addWidget(QLabel("Name"))
        top.addWidget(self.name_edit)
        top.addWidget(self.connect_button)
        top.addWidget(self.status)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        bottom = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.send_button = QPushButton("Senden")
        self.play_button = QPushButton("Datei abspielen")

        bottom.addWidget(self.input_edit)
        bottom.addWidget(self.send_button)
        bottom.addWidget(self.play_button)

        outer.addLayout(top)
        outer.addWidget(self.log_view)
        outer.addLayout(bottom)

        self.connect_button.clicked.connect(self._handle_connect)
        self.send_button.clicked.connect(self._handle_send)
        self.play_button.clicked.connect(self._handle_play)
        self.backend.text_received.connect(self._append_log)
        self.backend.status_changed.connect(self.status.setText)

        self._keepalive_timer = QTimer(self)
        self._keepalive_timer.start(1000)

    def closeEvent(self, event) -> None:
        self.backend.stop()
        super().closeEvent(event)

    def _handle_connect(self) -> None:
        host = self.host_edit.text().strip() or "127.0.0.1"
        port = int(self.port_edit.text().strip() or "8765")
        name = self.name_edit.text().strip() or "MONITOR_QT"

        try:
            self.backend.connect_bus(host, port, name)
        except Exception as exc:
            self._append_log(f"Verbindungsfehler: {exc}")

    def _handle_send(self) -> None:
        text = self.input_edit.text().strip()
        if not text:
            return
        self.backend.send_text(text)
        self.input_edit.clear()

    def _handle_play(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Playback Datei waehlen", "", "Text files (*.txt *.log);;All files (*.*)")
        if not path:
            return
        self.backend.play_file(path)

    def _append_log(self, line: str) -> None:
        self.log_view.append(line)


def run_qt_monitor(host: str, port: int, name: str) -> None:
    app = QApplication.instance() or QApplication([])
    win = MonitorWindow(host=host, port=port, name=name)
    win.show()
    app.exec()
