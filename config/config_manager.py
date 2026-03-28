# =============================================================================
# config/config_manager.py
#
# Central configuration management for openSMT.
# Handles loading, saving, and hot-reloading of JSON configuration files.
# Manages separate vision pipeline files in a dedicated directory.
# Emits Qt signals when configuration changes are detected from external edits.
# =============================================================================

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from utils.file_watcher import FileWatcher


# Path to the default configuration shipped with the application
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.json"
_DEFAULT_PIPELINES_PATH = Path(__file__).parent.parent / "pipelines"


class ConfigManager(QObject):
    """
    Manages application configuration with hot-reload support.

    The ConfigManager handles:
    - Loading configuration from JSON files
    - Saving configuration changes
    - Monitoring the config file for external changes
    - Loading and managing vision pipeline files from a directory
    - Emitting signals when the user should be prompted to reload

    Signals:
        config_changed_externally: Emitted when the config file is modified
                                   outside the application. The connected slot
                                   should prompt the user to reload.
        config_reloaded: Emitted after the configuration has been successfully
                         reloaded. Carries the new configuration dictionary.
        config_error: Emitted when a configuration error occurs.
                      Carries an error message string.
        pipeline_changed_externally: Emitted when a pipeline file is modified
                                     outside the application. Carries pipeline_id.
        pipeline_reloaded: Emitted after a pipeline has been reloaded.
                           Carries (pipeline_id, pipeline_dict).

    Example:
        config_mgr = ConfigManager()
        config_mgr.config_changed_externally.connect(prompt_user_to_reload)
        config_mgr.config_reloaded.connect(apply_new_config)
        config_mgr.load("/path/to/config.json")
    """

    config_changed_externally = pyqtSignal()
    config_reloaded = pyqtSignal(dict)
    config_error = pyqtSignal(str)
    pipeline_changed_externally = pyqtSignal(str)
    pipeline_reloaded = pyqtSignal(str, dict)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        """
        Initialize the ConfigManager.

        Args:
            parent: Optional Qt parent object for memory management.
        """
        super().__init__(parent)

        self._config: dict[str, Any] = {}
        self._config_path: Optional[Path] = None
        self._pipelines: dict[str, dict[str, Any]] = {}
        self._pipeline_directory: Optional[Path] = None

        # File watcher for main config
        self._file_watcher = FileWatcher(self)
        self._suppress_next_change = False

        # File watchers for pipeline files (one per file)
        self._pipeline_watchers: dict[str, FileWatcher] = {}
        self._suppress_pipeline_change: dict[str, bool] = {}

        # Connect the file watcher to our handler
        self._file_watcher.file_changed.connect(self._on_file_changed)

    def load(self, config_path: str | Path) -> bool:
        """
        Load configuration from a JSON file.

        If the file doesn't exist, it will be created from the default
        configuration template. Also loads all pipeline files from the
        configured pipeline directory.

        Args:
            config_path: Path to the configuration file.

        Returns:
            True if configuration was loaded successfully, False otherwise.
        """
        path = Path(config_path).resolve()

        # Create config file from default if it doesn't exist
        if not path.exists():
            if not self._create_default_config(path):
                return False

        # Read and parse the configuration
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._config = json.load(f)
            self._config_path = path

            # Start watching for external changes
            self._file_watcher.start(path)

            # Load pipelines
            self._load_pipelines()

            return True

        except json.JSONDecodeError as e:
            self.config_error.emit(f"Invalid JSON in config file: {e}")
            return False
        except OSError as e:
            self.config_error.emit(f"Failed to read config file: {e}")
            return False

    def reload(self) -> bool:
        """
        Reload configuration from the current file.

        Call this after the user confirms they want to reload after an
        external change. Also reloads all pipeline files.

        Returns:
            True if reload was successful, False otherwise.
        """
        if self._config_path is None:
            self.config_error.emit("No configuration file loaded")
            return False

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._config = json.load(f)

            # Reload pipelines
            self._load_pipelines()

            self.config_reloaded.emit(self._config.copy())
            return True

        except json.JSONDecodeError as e:
            self.config_error.emit(f"Invalid JSON in config file: {e}")
            return False
        except OSError as e:
            self.config_error.emit(f"Failed to reload config file: {e}")
            return False

    def save(self) -> bool:
        """
        Save the current configuration to the file.

        Returns:
            True if save was successful, False otherwise.
        """
        if self._config_path is None:
            self.config_error.emit("No configuration file path set")
            return False

        try:
            # Suppress the file watcher for our own save
            self._suppress_next_change = True

            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)

            return True

        except OSError as e:
            self._suppress_next_change = False
            self.config_error.emit(f"Failed to save config file: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a top-level configuration value.

        Args:
            key: The configuration key to retrieve.
            default: Value to return if the key doesn't exist.

        Returns:
            The configuration value, or the default if not found.
        """
        return self._config.get(key, default)

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """
        Get a nested configuration value using a sequence of keys.

        Args:
            *keys: Sequence of keys forming the path to the value.
            default: Value to return if any key in the path doesn't exist.

        Returns:
            The configuration value, or the default if not found.

        Example:
            # Get config["application"]["log_level"]
            log_level = config_mgr.get_nested("application", "log_level")
        """
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def set(self, key: str, value: Any) -> None:
        """
        Set a top-level configuration value.

        Note: Changes are not persisted until save() is called.

        Args:
            key: The configuration key to set.
            value: The value to assign.
        """
        self._config[key] = value

    def set_nested(self, *keys: str, value: Any) -> bool:
        """
        Set a nested configuration value using a sequence of keys.

        Creates intermediate dictionaries if they don't exist.

        Args:
            *keys: Sequence of keys forming the path to the value.
            value: The value to assign at the nested path.

        Returns:
            True if the value was set successfully, False if the path is invalid.

        Example:
            # Set config["application"]["log_level"] = "DEBUG"
            config_mgr.set_nested("application", "log_level", value="DEBUG")
        """
        if len(keys) < 1:
            return False

        current = self._config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                return False
            current = current[key]

        current[keys[-1]] = value
        return True

    @property
    def config(self) -> dict[str, Any]:
        """
        Get a copy of the full configuration dictionary.

        Returns a copy to prevent untracked modifications.
        """
        return self._config.copy()

    @property
    def config_path(self) -> Optional[Path]:
        """Get the path to the current configuration file."""
        return self._config_path

    @property
    def gcode_devices(self) -> list[dict[str, Any]]:
        """Get the list of G-Code device configurations."""
        return self._config.get("gcode_devices", [])

    @property
    def cameras(self) -> list[dict[str, Any]]:
        """Get the list of camera configurations."""
        return self._config.get("cameras", [])

    @property
    def pipeline_directory(self) -> Optional[Path]:
        """Get the path to the pipeline directory."""
        return self._pipeline_directory

    @property
    def vision_pipelines(self) -> dict[str, dict[str, Any]]:
        """Get a copy of all loaded vision pipeline configurations."""
        return self._pipelines.copy()

    @property
    def pipeline_ids(self) -> list[str]:
        """Get a list of all loaded pipeline IDs."""
        return list(self._pipelines.keys())

    def get_vision_pipeline(self, pipeline_id: str) -> Optional[dict[str, Any]]:
        """
        Get a specific vision pipeline configuration by ID.

        Args:
            pipeline_id: The unique identifier of the pipeline.

        Returns:
            A copy of the pipeline configuration dictionary, or None if not found.
        """
        pipeline = self._pipelines.get(pipeline_id)
        return pipeline.copy() if pipeline else None

    def save_vision_pipeline(self, pipeline_id: str, pipeline_data: dict[str, Any]) -> bool:
        """
        Save a vision pipeline to its JSON file.

        Creates a new file if the pipeline doesn't exist.

        Args:
            pipeline_id: The unique identifier for the pipeline.
            pipeline_data: The pipeline configuration dictionary.

        Returns:
            True if save was successful, False otherwise.
        """
        if self._pipeline_directory is None:
            self.config_error.emit("Pipeline directory not configured")
            return False

        # Ensure pipeline has the correct ID
        pipeline_data["id"] = pipeline_id

        file_path = self._pipeline_directory / f"{pipeline_id}.json"

        try:
            # Suppress file watcher for our own save
            self._suppress_pipeline_change[pipeline_id] = True

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(pipeline_data, f, indent=4, ensure_ascii=False)

            # Update in-memory cache
            self._pipelines[pipeline_id] = pipeline_data.copy()

            # Start watching this file if not already
            if pipeline_id not in self._pipeline_watchers:
                self._setup_pipeline_watcher(pipeline_id, file_path)

            return True

        except OSError as e:
            self._suppress_pipeline_change[pipeline_id] = False
            self.config_error.emit(f"Failed to save pipeline '{pipeline_id}': {e}")
            return False

    def delete_vision_pipeline(self, pipeline_id: str) -> bool:
        """
        Delete a vision pipeline file.

        Args:
            pipeline_id: The pipeline to delete.

        Returns:
            True if deleted successfully, False otherwise.
        """
        if self._pipeline_directory is None:
            return False

        # Don't allow deleting the default pipeline
        if pipeline_id == "default":
            self.config_error.emit("Cannot delete the default pipeline")
            return False

        file_path = self._pipeline_directory / f"{pipeline_id}.json"

        try:
            # Stop watching this file
            if pipeline_id in self._pipeline_watchers:
                self._pipeline_watchers[pipeline_id].stop()
                del self._pipeline_watchers[pipeline_id]

            # Delete file
            if file_path.exists():
                file_path.unlink()

            # Remove from cache
            self._pipelines.pop(pipeline_id, None)
            self._suppress_pipeline_change.pop(pipeline_id, None)

            return True

        except OSError as e:
            self.config_error.emit(f"Failed to delete pipeline '{pipeline_id}': {e}")
            return False

    def reload_pipeline(self, pipeline_id: str) -> bool:
        """
        Reload a specific pipeline from its file.

        Args:
            pipeline_id: The pipeline to reload.

        Returns:
            True if reload was successful, False otherwise.
        """
        if self._pipeline_directory is None:
            return False

        file_path = self._pipeline_directory / f"{pipeline_id}.json"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                pipeline_data = json.load(f)

            self._pipelines[pipeline_id] = pipeline_data
            self.pipeline_reloaded.emit(pipeline_id, pipeline_data.copy())
            return True

        except (json.JSONDecodeError, OSError) as e:
            self.config_error.emit(f"Failed to reload pipeline '{pipeline_id}': {e}")
            return False

    def update_gcode_device(self, device_id: str, updates: dict[str, Any]) -> bool:
        """
        Update a G-Code device configuration.

        Args:
            device_id: The unique ID of the device to update.
            updates: Dictionary of fields to update.

        Returns:
            True if the device was found and updated, False otherwise.
        """
        for device in self._config.get("gcode_devices", []):
            if device.get("id") == device_id:
                device.update(updates)
                return True
        return False

    def update_camera(self, camera_id: str, updates: dict[str, Any]) -> bool:
        """
        Update a camera configuration.

        Args:
            camera_id: The unique ID of the camera to update.
            updates: Dictionary of fields to update.

        Returns:
            True if the camera was found and updated, False otherwise.
        """
        for camera in self._config.get("cameras", []):
            if camera.get("id") == camera_id:
                camera.update(updates)
                return True
        return False

    def _load_pipelines(self) -> None:
        """
        Load all pipeline files from the configured pipeline directory.

        Creates the directory and default pipeline if they don't exist.
        """
        # Stop existing pipeline watchers
        for watcher in self._pipeline_watchers.values():
            watcher.stop()
        self._pipeline_watchers.clear()
        self._suppress_pipeline_change.clear()
        self._pipelines.clear()

        # Determine pipeline directory
        pipeline_dir_name = self._config.get("pipeline_directory", "pipelines")

        if self._config_path:
            # Relative to config file location
            self._pipeline_directory = self._config_path.parent / pipeline_dir_name
        else:
            # Fallback to current directory
            self._pipeline_directory = Path(pipeline_dir_name).resolve()

        # Create directory if needed
        if not self._pipeline_directory.exists():
            self._create_default_pipelines()

        # Load all .json files in the pipeline directory
        for file_path in self._pipeline_directory.glob("*.json"):
            self._load_pipeline_file(file_path)

    def _load_pipeline_file(self, file_path: Path) -> bool:
        """
        Load a single pipeline file.

        Args:
            file_path: Path to the pipeline JSON file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                pipeline_data = json.load(f)

            # Use filename (without extension) as ID if not specified
            pipeline_id = pipeline_data.get("id", file_path.stem)
            pipeline_data["id"] = pipeline_id

            self._pipelines[pipeline_id] = pipeline_data

            # Set up file watcher for this pipeline
            self._setup_pipeline_watcher(pipeline_id, file_path)

            return True

        except (json.JSONDecodeError, OSError) as e:
            self.config_error.emit(f"Failed to load pipeline '{file_path.name}': {e}")
            return False

    def _setup_pipeline_watcher(self, pipeline_id: str, file_path: Path) -> None:
        """
        Set up a file watcher for a pipeline file.

        Args:
            pipeline_id: The pipeline identifier.
            file_path: Path to the pipeline file.
        """
        watcher = FileWatcher(self)

        # Create a closure to capture pipeline_id
        def on_changed(path: str, pid: str = pipeline_id) -> None:
            self._on_pipeline_file_changed(pid)

        watcher.file_changed.connect(on_changed)
        watcher.start(file_path)

        self._pipeline_watchers[pipeline_id] = watcher
        self._suppress_pipeline_change[pipeline_id] = False

    def _on_pipeline_file_changed(self, pipeline_id: str) -> None:
        """
        Handle pipeline file change notification.

        Args:
            pipeline_id: The pipeline that was modified.
        """
        # Check if we should suppress this notification (our own save)
        if self._suppress_pipeline_change.get(pipeline_id, False):
            self._suppress_pipeline_change[pipeline_id] = False
            return

        # Emit signal to notify about external change
        self.pipeline_changed_externally.emit(pipeline_id)

    def _create_default_config(self, target_path: Path) -> bool:
        """
        Create a new configuration file from the default template.

        Args:
            target_path: Path where the new config file should be created.

        Returns:
            True if the file was created successfully, False otherwise.
        """
        try:
            # Ensure parent directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy default config to target location
            shutil.copy(_DEFAULT_CONFIG_PATH, target_path)
            return True

        except OSError as e:
            self.config_error.emit(f"Failed to create config file: {e}")
            return False

    def _create_default_pipelines(self) -> None:
        """
        Create the pipeline directory with default pipeline files.
        """
        if self._pipeline_directory is None:
            return

        try:
            self._pipeline_directory.mkdir(parents=True, exist_ok=True)

            # Copy default pipelines if they exist in the package
            if _DEFAULT_PIPELINES_PATH.exists():
                for src_file in _DEFAULT_PIPELINES_PATH.glob("*.json"):
                    dst_file = self._pipeline_directory / src_file.name
                    if not dst_file.exists():
                        shutil.copy(src_file, dst_file)
            else:
                # Create minimal default pipeline
                default_pipeline = {
                    "id": "default",
                    "name": "Default Pipeline",
                    "description": "Passthrough pipeline with no processing",
                    "version": "1.0.0",
                    "steps": []
                }
                default_path = self._pipeline_directory / "default.json"
                with open(default_path, "w", encoding="utf-8") as f:
                    json.dump(default_pipeline, f, indent=4)

        except OSError as e:
            self.config_error.emit(f"Failed to create pipeline directory: {e}")

    def _on_file_changed(self, path: str) -> None:
        """
        Handle file change notifications from the file watcher.

        Args:
            path: Path to the changed file.
        """
        # Check if we should suppress this notification (our own save)
        if self._suppress_next_change:
            self._suppress_next_change = False
            return

        # Emit signal to prompt user for reload
        self.config_changed_externally.emit()

    def stop_watching(self) -> None:
        """Stop watching the configuration file and all pipeline files."""
        self._file_watcher.stop()
        for watcher in self._pipeline_watchers.values():
            watcher.stop()
        self._pipeline_watchers.clear()

    def __del__(self) -> None:
        """Ensure file watching is stopped when the object is destroyed."""
        self.stop_watching()
