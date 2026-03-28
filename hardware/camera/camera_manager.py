# =============================================================================
# hardware/camera/camera_manager.py
#
# Central manager for all camera devices.
# Instantiates and manages CameraDevice instances based on configuration.
# Provides a unified interface for camera operations and pipeline management.
# =============================================================================

from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from config.config_manager import ConfigManager
from hardware.camera.camera_device import CameraDevice, CameraConfig, CameraState
from hardware.camera.vision_pipeline import VisionPipeline


class CameraManager(QObject):
    """
    Central manager for camera devices.

    The CameraManager:
    - Creates CameraDevice instances from configuration
    - Manages vision pipelines and assigns them to cameras
    - Provides device lookup by ID or name
    - Forwards device signals to a unified interface
    - Handles configuration reloads

    Signals:
        camera_added: Emitted when a new camera is created.
                      Carries camera_id: str.
        camera_removed: Emitted when a camera is removed.
                        Carries camera_id: str.
        camera_state_changed: Forwarded from individual cameras.
                              Carries (camera_id: str, state: CameraState).
        camera_error: Forwarded from individual cameras.
                      Carries (camera_id: str, error: str).
        frame_ready: Forwarded from individual cameras.
                     Carries (camera_id: str, frame: np.ndarray).
        all_cameras_stopped: Emitted when all cameras have been stopped.

    Example:
        manager = CameraManager(config_manager)
        manager.camera_state_changed.connect(on_state_change)
        manager.frame_ready.connect(display_frame)
        manager.initialize_cameras()
        manager.start_all()
    """

    camera_added = pyqtSignal(str)
    camera_removed = pyqtSignal(str)
    camera_state_changed = pyqtSignal(str, CameraState)
    camera_error = pyqtSignal(str, str)
    frame_ready = pyqtSignal(str, np.ndarray)
    fps_updated = pyqtSignal(str, float)
    all_cameras_stopped = pyqtSignal()

    def __init__(
        self,
        config_manager: ConfigManager,
        parent: Optional[QObject] = None,
    ) -> None:
        """
        Initialize the camera manager.

        Args:
            config_manager: The application configuration manager.
            parent: Optional Qt parent object.
        """
        super().__init__(parent)

        self._config_manager = config_manager
        self._cameras: dict[str, CameraDevice] = {}
        self._pipelines: dict[str, VisionPipeline] = {}

        # Connect to config signals
        self._config_manager.config_reloaded.connect(self._on_config_reloaded)
        self._config_manager.pipeline_reloaded.connect(self._on_pipeline_reloaded)

    @property
    def cameras(self) -> dict[str, CameraDevice]:
        """Get a copy of the cameras dictionary."""
        return self._cameras.copy()

    @property
    def camera_ids(self) -> list[str]:
        """Get a list of all camera IDs."""
        return list(self._cameras.keys())

    @property
    def camera_count(self) -> int:
        """Get the number of registered cameras."""
        return len(self._cameras)

    @property
    def pipelines(self) -> dict[str, VisionPipeline]:
        """Get a copy of the loaded pipelines dictionary."""
        return self._pipelines.copy()

    @property
    def pipeline_ids(self) -> list[str]:
        """Get a list of all loaded pipeline IDs."""
        return list(self._pipelines.keys())

    def initialize_cameras(self) -> int:
        """
        Create camera instances from the current configuration.

        Also loads all vision pipelines. Any existing cameras are
        stopped and removed first.

        Returns:
            Number of cameras created.
        """
        # Clean up existing cameras
        self.stop_all()
        self._remove_all_cameras()

        # Load pipelines first
        self._load_pipelines()

        # Create cameras from config
        camera_configs = self._config_manager.cameras

        for camera_config in camera_configs:
            self._create_camera_from_config(camera_config)

        return len(self._cameras)

    def _load_pipelines(self) -> None:
        """
        Load all vision pipelines from configuration.
        """
        self._pipelines.clear()

        pipeline_configs = self._config_manager.vision_pipelines

        for pipeline_id, pipeline_data in pipeline_configs.items():
            pipeline = VisionPipeline.from_dict(pipeline_data)
            self._pipelines[pipeline_id] = pipeline

    def _create_camera_from_config(
        self,
        config: dict[str, Any],
    ) -> Optional[CameraDevice]:
        """
        Create a single CameraDevice from a configuration dictionary.

        Args:
            config: Camera configuration from the config file.

        Returns:
            The created camera, or None if creation failed.
        """
        camera_id = config.get("id")
        if not camera_id:
            return None

        # Skip if camera already exists
        if camera_id in self._cameras:
            return self._cameras[camera_id]

        # Create camera config
        camera_config = CameraConfig.from_dict(config)

        # Create camera
        camera = CameraDevice(
            camera_id=camera_id,
            config=camera_config,
            name=config.get("name", camera_id),
            parent=self,
        )
        camera.enabled = config.get("enabled", True)

        # Assign pipeline if configured
        active_pipeline_id = camera_config.active_pipeline
        if active_pipeline_id in self._pipelines:
            camera.set_pipeline(self._pipelines[active_pipeline_id])

        # Connect camera signals to our forwarding signals
        camera.state_changed.connect(self._on_camera_state_changed)
        camera.error_occurred.connect(self._on_camera_error)
        camera.frame_ready.connect(self._on_frame_ready)
        camera.fps_updated.connect(self._on_fps_updated)

        # Store camera
        self._cameras[camera_id] = camera
        self.camera_added.emit(camera_id)

        return camera

    def get_camera(self, camera_id: str) -> Optional[CameraDevice]:
        """
        Get a camera by its ID.

        Args:
            camera_id: The unique camera identifier.

        Returns:
            The camera, or None if not found.
        """
        return self._cameras.get(camera_id)

    def get_camera_by_name(self, name: str) -> Optional[CameraDevice]:
        """
        Get a camera by its human-readable name.

        Args:
            name: The camera name to search for.

        Returns:
            The first camera matching the name, or None if not found.
        """
        for camera in self._cameras.values():
            if camera.name == name:
                return camera
        return None

    def get_camera_by_index(self, device_index: int) -> Optional[CameraDevice]:
        """
        Get a camera by its device index.

        Args:
            device_index: The OpenCV device index.

        Returns:
            The camera using that index, or None if not found.
        """
        for camera in self._cameras.values():
            if camera.device_index == device_index:
                return camera
        return None

    def get_pipeline(self, pipeline_id: str) -> Optional[VisionPipeline]:
        """
        Get a vision pipeline by its ID.

        Args:
            pipeline_id: The unique pipeline identifier.

        Returns:
            The pipeline, or None if not found.
        """
        return self._pipelines.get(pipeline_id)

    def start_camera(self, camera_id: str) -> bool:
        """
        Start a specific camera.

        Args:
            camera_id: The camera to start.

        Returns:
            True if started successfully, False if camera not found.
        """
        camera = self.get_camera(camera_id)
        if camera:
            return camera.start_capture()
        return False

    def stop_camera(self, camera_id: str) -> bool:
        """
        Stop a specific camera.

        Args:
            camera_id: The camera to stop.

        Returns:
            True if camera was found, False otherwise.
        """
        camera = self.get_camera(camera_id)
        if camera:
            camera.stop_capture()
            return True
        return False

    def start_all(self, enabled_only: bool = True) -> dict[str, bool]:
        """
        Start all cameras.

        Args:
            enabled_only: If True, only start cameras marked as enabled.

        Returns:
            Dictionary mapping camera_id to start success status.
        """
        results = {}
        for camera_id, camera in self._cameras.items():
            if enabled_only and not camera.enabled:
                results[camera_id] = False
                continue
            results[camera_id] = camera.start_capture()
        return results

    def stop_all(self) -> None:
        """Stop all cameras."""
        for camera in self._cameras.values():
            camera.stop_capture()
        self.all_cameras_stopped.emit()

    def pause_camera(self, camera_id: str) -> bool:
        """
        Pause a specific camera.

        Args:
            camera_id: The camera to pause.

        Returns:
            True if camera was found, False otherwise.
        """
        camera = self.get_camera(camera_id)
        if camera:
            camera.pause_capture()
            return True
        return False

    def resume_camera(self, camera_id: str) -> bool:
        """
        Resume a paused camera.

        Args:
            camera_id: The camera to resume.

        Returns:
            True if camera was found, False otherwise.
        """
        camera = self.get_camera(camera_id)
        if camera:
            camera.resume_capture()
            return True
        return False

    def set_camera_pipeline(
        self,
        camera_id: str,
        pipeline_id: str,
    ) -> bool:
        """
        Set the vision pipeline for a camera.

        Args:
            camera_id: The target camera ID.
            pipeline_id: The pipeline ID to assign.

        Returns:
            True if both camera and pipeline were found, False otherwise.
        """
        camera = self.get_camera(camera_id)
        pipeline = self.get_pipeline(pipeline_id)

        if camera is None:
            return False

        # Allow setting to None (no pipeline)
        camera.set_pipeline(pipeline)

        # Update config
        self._config_manager.update_camera(camera_id, {"active_pipeline": pipeline_id})

        return True

    def update_camera_name(self, camera_id: str, new_name: str) -> bool:
        """
        Update the name of a camera.

        Also updates the configuration so the change persists.

        Args:
            camera_id: The camera to rename.
            new_name: The new human-readable name.

        Returns:
            True if camera was found and renamed, False otherwise.
        """
        camera = self.get_camera(camera_id)
        if camera:
            camera.name = new_name
            self._config_manager.update_camera(camera_id, {"name": new_name})
            return True
        return False

    def update_camera_device_index(
        self,
        camera_id: str,
        new_index: int,
    ) -> bool:
        """
        Update the device index of a camera.

        The camera should be stopped before changing the index.
        Also updates the configuration so the change persists.

        Args:
            camera_id: The camera to update.
            new_index: The new OpenCV device index.

        Returns:
            True if camera was found and updated, False otherwise.
        """
        camera = self.get_camera(camera_id)
        if camera:
            new_config = CameraConfig(
                device_index=new_index,
                width=camera.config.width,
                height=camera.config.height,
                fps=camera.config.fps,
                dpi_profiles=camera.config.dpi_profiles,
                active_pipeline=camera.config.active_pipeline,
            )
            camera.config = new_config
            self._config_manager.update_camera(camera_id, {"device_index": new_index})
            return True
        return False

    def reload_pipeline(self, pipeline_id: str) -> bool:
        """
        Reload a specific pipeline from configuration.

        Updates all cameras using this pipeline.

        Args:
            pipeline_id: The pipeline to reload.

        Returns:
            True if pipeline was reloaded, False otherwise.
        """
        pipeline_data = self._config_manager.get_vision_pipeline(pipeline_id)
        if pipeline_data is None:
            return False

        # Create new pipeline instance
        pipeline = VisionPipeline.from_dict(pipeline_data)
        self._pipelines[pipeline_id] = pipeline

        # Update all cameras using this pipeline
        for camera in self._cameras.values():
            if camera.active_pipeline_id == pipeline_id:
                camera.set_pipeline(pipeline)

        return True

    def reload_all_pipelines(self) -> int:
        """
        Reload all pipelines from configuration.

        Updates all cameras with their new pipelines.

        Returns:
            Number of pipelines reloaded.
        """
        self._load_pipelines()

        # Reassign pipelines to cameras
        for camera in self._cameras.values():
            pipeline_id = camera.active_pipeline_id
            if pipeline_id in self._pipelines:
                camera.set_pipeline(self._pipelines[pipeline_id])
            else:
                camera.set_pipeline(None)

        return len(self._pipelines)

    def get_streaming_cameras(self) -> list[CameraDevice]:
        """
        Get a list of all currently streaming cameras.

        Returns:
            List of streaming CameraDevice instances.
        """
        return [c for c in self._cameras.values() if c.is_streaming]

    def get_camera_states(self) -> dict[str, CameraState]:
        """
        Get the current state of all cameras.

        Returns:
            Dictionary mapping camera_id to CameraState.
        """
        return {cid: c.state for cid, c in self._cameras.items()}

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Get the current frame from a specific camera.

        Args:
            camera_id: The camera to get the frame from.

        Returns:
            The current frame, or None if not available.
        """
        camera = self.get_camera(camera_id)
        if camera:
            return camera.get_current_frame()
        return None

    def _remove_all_cameras(self) -> None:
        """Remove all cameras from the manager."""
        for camera_id in list(self._cameras.keys()):
            self._remove_camera(camera_id)

    def _remove_camera(self, camera_id: str) -> bool:
        """
        Remove a camera from the manager.

        Args:
            camera_id: The camera to remove.

        Returns:
            True if camera was found and removed, False otherwise.
        """
        camera = self._cameras.pop(camera_id, None)
        if camera:
            camera.stop_capture()
            # Disconnect signals
            try:
                camera.state_changed.disconnect(self._on_camera_state_changed)
                camera.error_occurred.disconnect(self._on_camera_error)
                camera.frame_ready.disconnect(self._on_frame_ready)
                camera.fps_updated.disconnect(self._on_fps_updated)
            except TypeError:
                pass  # Already disconnected
            self.camera_removed.emit(camera_id)
            return True
        return False

    def _on_camera_state_changed(
        self,
        camera_id: str,
        state: CameraState,
    ) -> None:
        """
        Forward camera state change signal.

        Args:
            camera_id: The camera that changed state.
            state: The new state.
        """
        self.camera_state_changed.emit(camera_id, state)

    def _on_camera_error(self, camera_id: str, error: str) -> None:
        """
        Forward camera error signal.

        Args:
            camera_id: The camera that had an error.
            error: The error message.
        """
        self.camera_error.emit(camera_id, error)

    def _on_frame_ready(self, camera_id: str, frame: np.ndarray) -> None:
        """
        Forward frame ready signal.

        Args:
            camera_id: The camera that produced the frame.
            frame: The processed frame.
        """
        self.frame_ready.emit(camera_id, frame)

    def _on_fps_updated(self, camera_id: str, fps: float) -> None:
        """
        Forward FPS update signal.

        Args:
            camera_id: The camera reporting FPS.
            fps: The current FPS value.
        """
        self.fps_updated.emit(camera_id, fps)

    def _on_config_reloaded(self, new_config: dict[str, Any]) -> None:
        """
        Handle configuration reload.

        This method is called when the user confirms a config reload after
        external changes. Cameras are not automatically restarted.

        Args:
            new_config: The new configuration dictionary.
        """
        # Note: We don't automatically reinitialize here because the user
        # should be prompted. The GUI should call initialize_cameras()
        # after the user confirms.
        pass

    def _on_pipeline_reloaded(
        self,
        pipeline_id: str,
        pipeline_data: dict[str, Any],
    ) -> None:
        """
        Handle single pipeline reload.

        Args:
            pipeline_id: The pipeline that was reloaded.
            pipeline_data: The new pipeline data.
        """
        # Reload this specific pipeline
        self.reload_pipeline(pipeline_id)

    def prepare_for_config_reload(self) -> dict[str, bool]:
        """
        Prepare for a configuration reload by stopping all cameras.

        Call this before prompting the user to reload configuration.

        Returns:
            Dictionary of camera_id to was_streaming status, so streams
            can be restored after reload if desired.
        """
        streaming_states = {
            cid: c.is_streaming for cid, c in self._cameras.items()
        }
        self.stop_all()
        return streaming_states

    def __repr__(self) -> str:
        """Return string representation of the manager."""
        streaming = len(self.get_streaming_cameras())
        return (
            f"CameraManager(cameras={self.camera_count}, streaming={streaming}, "
            f"pipelines={len(self._pipelines)})"
        )

    def __del__(self) -> None:
        """Ensure clean shutdown when object is destroyed."""
        self.stop_all()
