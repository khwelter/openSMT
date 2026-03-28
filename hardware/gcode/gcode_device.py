# =============================================================================
# hardware/gcode/gcode_device.py
#
# Individual G-Code device controller.
# Manages serial communication with a single G-Code interpreter, including
# a command queue with response handling.
# =============================================================================

import threading
import queue
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

import serial
from PyQt6.QtCore import QObject, pyqtSignal


class DeviceState(Enum):
    """Enumeration of possible device connection states."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    BUSY = auto()
    ERROR = auto()


@dataclass
class GCodeCommand:
    """
    Represents a G-Code command to be sent to the device.

    Attributes:
        command: The G-Code command string (e.g., "G28", "G1 X10 Y20").
        callback: Optional function to call when response is received.
                  Signature: callback(success: bool, response: str)
        timeout: Maximum time in seconds to wait for response.
        timestamp: Time when command was queued (auto-set).
    """
    command: str
    callback: Optional[Callable[[bool, str], None]] = None
    timeout: float = 10.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SerialConfig:
    """
    Serial port configuration for G-Code device.

    Attributes:
        port: Serial port path (e.g., "/dev/ttyUSB0", "COM3").
        baudrate: Communication speed (default: 115200).
        bytesize: Number of data bits (default: 8).
        parity: Parity checking ('N', 'E', 'O', 'M', 'S').
        stopbits: Number of stop bits (1, 1.5, 2).
        timeout: Read timeout in seconds.
    """
    port: str
    baudrate: int = 115200
    bytesize: int = 8
    parity: str = "N"
    stopbits: int = 1
    timeout: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SerialConfig":
        """
        Create SerialConfig from a configuration dictionary.

        Args:
            data: Dictionary containing serial configuration values.

        Returns:
            A new SerialConfig instance.
        """
        return cls(
            port=data.get("port", "/dev/ttyUSB0"),
            baudrate=data.get("baudrate", 115200),
            bytesize=data.get("bytesize", 8),
            parity=data.get("parity", "N"),
            stopbits=data.get("stopbits", 1),
            timeout=data.get("timeout", 1.0),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert SerialConfig to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "port": self.port,
            "baudrate": self.baudrate,
            "bytesize": self.bytesize,
            "parity": self.parity,
            "stopbits": self.stopbits,
            "timeout": self.timeout,
        }


class GCodeDevice(QObject):
    """
    Manages communication with a single G-Code interpreter over serial.

    This class provides:
    - Asynchronous command queue with per-command callbacks
    - Thread-safe command submission
    - Automatic response parsing and matching
    - Connection state management with Qt signals

    The device runs a dedicated worker thread that processes the command
    queue and handles serial I/O. Commands are processed in FIFO order.

    Signals:
        state_changed: Emitted when device state changes.
                       Carries (device_id: str, new_state: DeviceState).
        response_received: Emitted for every response line received.
                           Carries (device_id: str, response: str).
        error_occurred: Emitted when an error occurs.
                        Carries (device_id: str, error_message: str).
        command_completed: Emitted when a command finishes (success or fail).
                           Carries (device_id: str, command: str, success: bool).

    Example:
        device = GCodeDevice("main_controller", serial_config)
        device.state_changed.connect(on_state_change)
        device.connect_device()
        device.send_command("G28", callback=on_home_complete)
    """

    state_changed = pyqtSignal(str, DeviceState)
    response_received = pyqtSignal(str, str)
    error_occurred = pyqtSignal(str, str)
    command_completed = pyqtSignal(str, str, bool)

    # Common G-Code response indicators
    RESPONSE_OK = "ok"
    RESPONSE_ERROR_PREFIXES = ("error", "!!", "error:")

    def __init__(
        self,
        device_id: str,
        config: SerialConfig,
        name: str = "",
        parent: Optional[QObject] = None,
    ) -> None:
        """
        Initialize a G-Code device.

        Args:
            device_id: Unique identifier for this device.
            config: Serial port configuration.
            name: Human-readable name for the device.
            parent: Optional Qt parent object.
        """
        super().__init__(parent)

        self._device_id = device_id
        self._config = config
        self._name = name or device_id
        self._enabled = True

        self._state = DeviceState.DISCONNECTED
        self._serial: Optional[serial.Serial] = None

        # Command queue and worker thread
        self._command_queue: queue.Queue[Optional[GCodeCommand]] = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Lock for thread-safe state access
        self._state_lock = threading.Lock()

    @property
    def device_id(self) -> str:
        """Get the unique device identifier."""
        return self._device_id

    @property
    def name(self) -> str:
        """Get the human-readable device name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the human-readable device name."""
        self._name = value

    @property
    def config(self) -> SerialConfig:
        """Get the serial configuration."""
        return self._config

    @config.setter
    def config(self, value: SerialConfig) -> None:
        """
        Set a new serial configuration.

        Note: If currently connected, you must disconnect and reconnect
        for changes to take effect.
        """
        self._config = value

    @property
    def state(self) -> DeviceState:
        """Get the current device state (thread-safe)."""
        with self._state_lock:
            return self._state

    @property
    def is_connected(self) -> bool:
        """Check if the device is currently connected."""
        return self.state in (DeviceState.CONNECTED, DeviceState.BUSY)

    @property
    def enabled(self) -> bool:
        """Check if the device is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the device."""
        self._enabled = value

    @property
    def port(self) -> str:
        """Get the serial port path."""
        return self._config.port

    @property
    def queue_size(self) -> int:
        """Get the number of commands waiting in the queue."""
        return self._command_queue.qsize()

    def _set_state(self, new_state: DeviceState) -> None:
        """
        Update the device state and emit signal (thread-safe).

        Args:
            new_state: The new state to set.
        """
        with self._state_lock:
            if self._state != new_state:
                self._state = new_state
                self.state_changed.emit(self._device_id, new_state)

    def connect_device(self) -> bool:
        """
        Open the serial connection and start the worker thread.

        Returns:
            True if connection initiated successfully, False otherwise.
        """
        if self.is_connected:
            return True

        if not self._enabled:
            self.error_occurred.emit(self._device_id, "Device is disabled")
            return False

        self._set_state(DeviceState.CONNECTING)

        try:
            # Configure parity
            parity_map = {
                "N": serial.PARITY_NONE,
                "E": serial.PARITY_EVEN,
                "O": serial.PARITY_ODD,
                "M": serial.PARITY_MARK,
                "S": serial.PARITY_SPACE,
            }
            parity = parity_map.get(self._config.parity.upper(), serial.PARITY_NONE)

            # Configure stop bits
            stopbits_map = {
                1: serial.STOPBITS_ONE,
                1.5: serial.STOPBITS_ONE_POINT_FIVE,
                2: serial.STOPBITS_TWO,
            }
            stopbits = stopbits_map.get(self._config.stopbits, serial.STOPBITS_ONE)

            # Open serial connection
            self._serial = serial.Serial(
                port=self._config.port,
                baudrate=self._config.baudrate,
                bytesize=self._config.bytesize,
                parity=parity,
                stopbits=stopbits,
                timeout=self._config.timeout,
            )

            # Clear any pending data
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()

            # Start worker thread
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"GCode-{self._device_id}",
                daemon=True,
            )
            self._worker_thread.start()

            self._set_state(DeviceState.CONNECTED)
            return True

        except serial.SerialException as e:
            self._set_state(DeviceState.ERROR)
            self.error_occurred.emit(self._device_id, f"Connection failed: {e}")
            return False

    def disconnect_device(self) -> None:
        """
        Close the serial connection and stop the worker thread.

        Any pending commands in the queue will be discarded.
        """
        # Signal worker thread to stop
        self._stop_event.set()

        # Put None to unblock the queue
        self._command_queue.put(None)

        # Wait for worker thread to finish
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None

        # Close serial connection
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

        # Clear the command queue
        while not self._command_queue.empty():
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                break

        self._set_state(DeviceState.DISCONNECTED)

    def send_command(
        self,
        command: str,
        callback: Optional[Callable[[bool, str], None]] = None,
        timeout: float = 10.0,
    ) -> bool:
        """
        Queue a G-Code command for execution.

        Commands are processed asynchronously in FIFO order. The optional
        callback is invoked when the command completes.

        Args:
            command: G-Code command string (newline added automatically).
            callback: Optional callback function with signature:
                      callback(success: bool, response: str)
            timeout: Maximum seconds to wait for response.

        Returns:
            True if command was queued, False if device not connected.
        """
        if not self.is_connected:
            if callback:
                callback(False, "Device not connected")
            return False

        cmd = GCodeCommand(
            command=command.strip(),
            callback=callback,
            timeout=timeout,
        )
        self._command_queue.put(cmd)
        return True

    def send_command_sync(self, command: str, timeout: float = 10.0) -> tuple[bool, str]:
        """
        Send a command and wait for the response synchronously.

        Warning: This blocks the calling thread. Do not call from the GUI thread.

        Args:
            command: G-Code command string.
            timeout: Maximum seconds to wait for response.

        Returns:
            Tuple of (success: bool, response: str).
        """
        result_event = threading.Event()
        result: list[tuple[bool, str]] = []

        def sync_callback(success: bool, response: str) -> None:
            result.append((success, response))
            result_event.set()

        if not self.send_command(command, callback=sync_callback, timeout=timeout):
            return False, "Device not connected"

        result_event.wait(timeout=timeout + 1.0)

        if result:
            return result[0]
        return False, "Timeout waiting for response"

    def clear_queue(self) -> int:
        """
        Clear all pending commands from the queue.

        Commands that are already being processed are not affected.

        Returns:
            Number of commands that were cleared.
        """
        count = 0
        while not self._command_queue.empty():
            try:
                cmd = self._command_queue.get_nowait()
                if cmd is not None and cmd.callback:
                    cmd.callback(False, "Command cancelled")
                count += 1
            except queue.Empty:
                break
        return count

    def _worker_loop(self) -> None:
        """
        Worker thread main loop.

        Processes commands from the queue and handles serial I/O.
        """
        while not self._stop_event.is_set():
            try:
                # Wait for a command with timeout to allow checking stop_event
                try:
                    cmd = self._command_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # None is the sentinel to stop the thread
                if cmd is None:
                    break

                # Process the command
                self._process_command(cmd)

            except Exception as e:
                self.error_occurred.emit(self._device_id, f"Worker error: {e}")

    def _process_command(self, cmd: GCodeCommand) -> None:
        """
        Send a command and wait for its response.

        Args:
            cmd: The GCodeCommand to process.
        """
        if self._serial is None or not self._serial.is_open:
            if cmd.callback:
                cmd.callback(False, "Serial port not open")
            self.command_completed.emit(self._device_id, cmd.command, False)
            return

        self._set_state(DeviceState.BUSY)

        try:
            # Send the command
            command_line = cmd.command + "\n"
            self._serial.write(command_line.encode("utf-8"))
            self._serial.flush()

            # Collect response lines until we get 'ok' or error
            response_lines: list[str] = []
            start_time = time.time()
            success = False

            while (time.time() - start_time) < cmd.timeout:
                if self._stop_event.is_set():
                    break

                if self._serial.in_waiting > 0:
                    line = self._serial.readline().decode("utf-8", errors="replace").strip()

                    if line:
                        response_lines.append(line)
                        self.response_received.emit(self._device_id, line)

                        # Check for completion
                        line_lower = line.lower()
                        if line_lower == self.RESPONSE_OK:
                            success = True
                            break
                        elif line_lower.startswith(self.RESPONSE_ERROR_PREFIXES):
                            success = False
                            break
                else:
                    time.sleep(0.01)

            # Compile full response
            full_response = "\n".join(response_lines)

            # Invoke callback
            if cmd.callback:
                cmd.callback(success, full_response)

            self.command_completed.emit(self._device_id, cmd.command, success)

        except serial.SerialException as e:
            error_msg = f"Serial error: {e}"
            self.error_occurred.emit(self._device_id, error_msg)
            if cmd.callback:
                cmd.callback(False, error_msg)
            self.command_completed.emit(self._device_id, cmd.command, False)
            self._set_state(DeviceState.ERROR)
            return

        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            self.error_occurred.emit(self._device_id, error_msg)
            if cmd.callback:
                cmd.callback(False, error_msg)
            self.command_completed.emit(self._device_id, cmd.command, False)

        # Return to connected state if no error
        if self.state != DeviceState.ERROR:
            self._set_state(DeviceState.CONNECTED)

    def __repr__(self) -> str:
        """Return string representation of the device."""
        return (
            f"GCodeDevice(id={self._device_id!r}, name={self._name!r}, "
            f"port={self._config.port!r}, state={self.state.name})"
        )

    def __del__(self) -> None:
        """Ensure clean disconnection when object is destroyed."""
        self.disconnect_device()
