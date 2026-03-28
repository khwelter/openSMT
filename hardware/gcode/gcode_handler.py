# =============================================================================
# hardware/gcode/gcode_handler.py
#
# Central manager for all G-Code devices.
# Instantiates and manages GCodeDevice instances based on configuration.
# Provides a unified interface for device operations.
# =============================================================================

from typing import Any, Callable, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from config.config_manager import ConfigManager
from hardware.gcode.gcode_device import GCodeDevice, SerialConfig, DeviceState


class GCodeHandler(QObject):
    """
    Central manager for G-Code interpreter devices.

    The GCodeHandler:
    - Creates GCodeDevice instances from configuration
    - Provides device lookup by ID or name
    - Forwards device signals to a unified interface
    - Handles configuration reloads

    Signals:
        device_added: Emitted when a new device is created.
                      Carries device_id: str.
        device_removed: Emitted when a device is removed.
                        Carries device_id: str.
        device_state_changed: Forwarded from individual devices.
                              Carries (device_id: str, state: DeviceState).
        device_error: Forwarded from individual devices.
                      Carries (device_id: str, error: str).
        all_devices_disconnected: Emitted when all devices have been disconnected.

    Example:
        handler = GCodeHandler(config_manager)
        handler.device_state_changed.connect(on_state_change)
        handler.initialize_devices()
        handler.connect_all()
        
        device = handler.get_device("main_controller")
        device.send_command("G28")
    """

    device_added = pyqtSignal(str)
    device_removed = pyqtSignal(str)
    device_state_changed = pyqtSignal(str, DeviceState)
    device_error = pyqtSignal(str, str)
    all_devices_disconnected = pyqtSignal()

    def __init__(
        self,
        config_manager: ConfigManager,
        parent: Optional[QObject] = None,
    ) -> None:
        """
        Initialize the G-Code handler.

        Args:
            config_manager: The application configuration manager.
            parent: Optional Qt parent object.
        """
        super().__init__(parent)

        self._config_manager = config_manager
        self._devices: dict[str, GCodeDevice] = {}

        # Connect to config reload signal
        self._config_manager.config_reloaded.connect(self._on_config_reloaded)

    @property
    def devices(self) -> dict[str, GCodeDevice]:
        """Get a copy of the devices dictionary."""
        return self._devices.copy()

    @property
    def device_ids(self) -> list[str]:
        """Get a list of all device IDs."""
        return list(self._devices.keys())

    @property
    def device_count(self) -> int:
        """Get the number of registered devices."""
        return len(self._devices)

    def initialize_devices(self) -> int:
        """
        Create device instances from the current configuration.

        Any existing devices are disconnected and removed first.

        Returns:
            Number of devices created.
        """
        # Clean up existing devices
        self.disconnect_all()
        self._remove_all_devices()

        # Create devices from config
        device_configs = self._config_manager.gcode_devices

        for device_config in device_configs:
            self._create_device_from_config(device_config)

        return len(self._devices)

    def _create_device_from_config(self, config: dict[str, Any]) -> Optional[GCodeDevice]:
        """
        Create a single GCodeDevice from a configuration dictionary.

        Args:
            config: Device configuration from the config file.

        Returns:
            The created device, or None if creation failed.
        """
        device_id = config.get("id")
        if not device_id:
            return None

        # Skip if device already exists
        if device_id in self._devices:
            return self._devices[device_id]

        # Create serial config
        serial_config = SerialConfig.from_dict(config)

        # Create device
        device = GCodeDevice(
            device_id=device_id,
            config=serial_config,
            name=config.get("name", device_id),
            parent=self,
        )
        device.enabled = config.get("enabled", True)

        # Connect device signals to our forwarding signals
        device.state_changed.connect(self._on_device_state_changed)
        device.error_occurred.connect(self._on_device_error)

        # Store device
        self._devices[device_id] = device
        self.device_added.emit(device_id)

        return device

    def get_device(self, device_id: str) -> Optional[GCodeDevice]:
        """
        Get a device by its ID.

        Args:
            device_id: The unique device identifier.

        Returns:
            The device, or None if not found.
        """
        return self._devices.get(device_id)

    def get_device_by_name(self, name: str) -> Optional[GCodeDevice]:
        """
        Get a device by its human-readable name.

        Args:
            name: The device name to search for.

        Returns:
            The first device matching the name, or None if not found.
        """
        for device in self._devices.values():
            if device.name == name:
                return device
        return None

    def get_device_by_port(self, port: str) -> Optional[GCodeDevice]:
        """
        Get a device by its serial port.

        Args:
            port: The serial port path.

        Returns:
            The device using that port, or None if not found.
        """
        for device in self._devices.values():
            if device.port == port:
                return device
        return None

    def connect_device(self, device_id: str) -> bool:
        """
        Connect a specific device.

        Args:
            device_id: The device to connect.

        Returns:
            True if connection initiated, False if device not found.
        """
        device = self.get_device(device_id)
        if device:
            return device.connect_device()
        return False

    def disconnect_device(self, device_id: str) -> bool:
        """
        Disconnect a specific device.

        Args:
            device_id: The device to disconnect.

        Returns:
            True if device was found, False otherwise.
        """
        device = self.get_device(device_id)
        if device:
            device.disconnect_device()
            return True
        return False

    def connect_all(self, enabled_only: bool = True) -> dict[str, bool]:
        """
        Connect all devices.

        Args:
            enabled_only: If True, only connect devices marked as enabled.

        Returns:
            Dictionary mapping device_id to connection success status.
        """
        results = {}
        for device_id, device in self._devices.items():
            if enabled_only and not device.enabled:
                results[device_id] = False
                continue
            results[device_id] = device.connect_device()
        return results

    def disconnect_all(self) -> None:
        """Disconnect all devices."""
        for device in self._devices.values():
            device.disconnect_device()
        self.all_devices_disconnected.emit()

    def send_command_to_device(
        self,
        device_id: str,
        command: str,
        callback: Optional[Callable[[bool, str], None]] = None,
        timeout: float = 10.0,
    ) -> bool:
        """
        Send a G-Code command to a specific device.

        Args:
            device_id: Target device ID.
            command: G-Code command string.
            callback: Optional completion callback.
            timeout: Command timeout in seconds.

        Returns:
            True if command was queued, False if device not found or not connected.
        """
        device = self.get_device(device_id)
        if device:
            return device.send_command(command, callback, timeout)
        return False

    def send_command_to_all(
        self,
        command: str,
        callback: Optional[Callable[[str, bool, str], None]] = None,
        timeout: float = 10.0,
        connected_only: bool = True,
    ) -> dict[str, bool]:
        """
        Send a G-Code command to all devices (parallel execution).

        Args:
            command: G-Code command string.
            callback: Optional callback with signature:
                      callback(device_id: str, success: bool, response: str)
            timeout: Command timeout in seconds.
            connected_only: If True, only send to connected devices.

        Returns:
            Dictionary mapping device_id to queue success status.
        """
        results = {}

        for device_id, device in self._devices.items():
            if connected_only and not device.is_connected:
                results[device_id] = False
                continue

            # Wrap callback to include device_id
            if callback:
                def make_callback(did: str) -> Callable[[bool, str], None]:
                    def wrapped(success: bool, response: str) -> None:
                        callback(did, success, response)
                    return wrapped
                device_callback = make_callback(device_id)
            else:
                device_callback = None

            results[device_id] = device.send_command(command, device_callback, timeout)

        return results

    def update_device_name(self, device_id: str, new_name: str) -> bool:
        """
        Update the name of a device.

        Also updates the configuration so the change persists.

        Args:
            device_id: The device to rename.
            new_name: The new human-readable name.

        Returns:
            True if device was found and renamed, False otherwise.
        """
        device = self.get_device(device_id)
        if device:
            device.name = new_name
            # Update config
            self._config_manager.update_gcode_device(device_id, {"name": new_name})
            return True
        return False

    def update_device_port(self, device_id: str, new_port: str) -> bool:
        """
        Update the serial port of a device.

        The device should be disconnected before changing the port.
        Also updates the configuration so the change persists.

        Args:
            device_id: The device to update.
            new_port: The new serial port path.

        Returns:
            True if device was found and updated, False otherwise.
        """
        device = self.get_device(device_id)
        if device:
            # Update the device config
            new_config = SerialConfig(
                port=new_port,
                baudrate=device.config.baudrate,
                bytesize=device.config.bytesize,
                parity=device.config.parity,
                stopbits=device.config.stopbits,
                timeout=device.config.timeout,
            )
            device.config = new_config
            # Update config file
            self._config_manager.update_gcode_device(device_id, {"port": new_port})
            return True
        return False

    def get_connected_devices(self) -> list[GCodeDevice]:
        """
        Get a list of all currently connected devices.

        Returns:
            List of connected GCodeDevice instances.
        """
        return [d for d in self._devices.values() if d.is_connected]

    def get_device_states(self) -> dict[str, DeviceState]:
        """
        Get the current state of all devices.

        Returns:
            Dictionary mapping device_id to DeviceState.
        """
        return {did: d.state for did, d in self._devices.items()}

    def _remove_all_devices(self) -> None:
        """Remove all devices from the handler."""
        for device_id in list(self._devices.keys()):
            self._remove_device(device_id)

    def _remove_device(self, device_id: str) -> bool:
        """
        Remove a device from the handler.

        Args:
            device_id: The device to remove.

        Returns:
            True if device was found and removed, False otherwise.
        """
        device = self._devices.pop(device_id, None)
        if device:
            device.disconnect_device()
            # Disconnect signals
            try:
                device.state_changed.disconnect(self._on_device_state_changed)
                device.error_occurred.disconnect(self._on_device_error)
            except TypeError:
                pass  # Already disconnected
            self.device_removed.emit(device_id)
            return True
        return False

    def _on_device_state_changed(self, device_id: str, state: DeviceState) -> None:
        """
        Forward device state change signal.

        Args:
            device_id: The device that changed state.
            state: The new state.
        """
        self.device_state_changed.emit(device_id, state)

    def _on_device_error(self, device_id: str, error: str) -> None:
        """
        Forward device error signal.

        Args:
            device_id: The device that had an error.
            error: The error message.
        """
        self.device_error.emit(device_id, error)

    def _on_config_reloaded(self, new_config: dict[str, Any]) -> None:
        """
        Handle configuration reload.

        This method is called when the user confirms a config reload after
        external changes. Devices are not automatically reconnected.

        Args:
            new_config: The new configuration dictionary.
        """
        # Note: We don't automatically reinitialize here because the user
        # should be prompted. The GUI should call initialize_devices()
        # after the user confirms.
        pass

    def prepare_for_config_reload(self) -> dict[str, bool]:
        """
        Prepare for a configuration reload by disconnecting all devices.

        Call this before prompting the user to reload configuration.

        Returns:
            Dictionary of device_id to was_connected status, so connections
            can be restored after reload if desired.
        """
        connection_states = {
            did: d.is_connected for did, d in self._devices.items()
        }
        self.disconnect_all()
        return connection_
