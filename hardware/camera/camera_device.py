# =============================================================================
# hardware/camera/camera_device.py
#
# Individual camera device controller.
# Manages OpenCV video capture in a dedicated thread, applies vision
# pipelines, and emits frames via Qt signals.
# =============================================================================

import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, QMutex, QMutexLocker


class CameraState(Enum):
    """Enumeration of possible camera states."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    STREAMING = auto()
    PAUSED = auto()
    ERROR = auto()


@dataclass
class DpiProfile:
    """
    DPI configuration for a specific camera height.

    Attributes:
        height_mm: Camera height in millimeters.
        dpi_x: Horizontal DPI at this height.
        dpi_y: Vertical DPI at this height.
    """
    height_mm: float
    dpi_x: float
    dpi_y: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DpiProfile":
        """Create DpiProfile from a configuration dictionary."""
        return cls(
            height_mm=data.get("height_mm", 0.0),
            dpi_x=data.get("dpi_x", 96.0),
            dpi_y=data.get("dpi_y", 96.0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "height_mm": self.height_mm,
            "dpi_x": self.dpi_x,
            "dpi_y": self.dpi_y,
        }


@dataclass
class CameraConfig:
    """
    Configuration for a camera device.

    Attributes:
        device_index: OpenCV device index or path.
        width: Capture width in pixels.
        height: Capture height in pixels.
        fps: Target frames per second (0 for maximum).
        dpi_profiles: List of DPI profiles for different heights.
        active_pipeline: ID of the vision pipeline to apply.
    """
    device_index: int | str = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    dpi_profiles: list[DpiProfile] = field(default_factory=list)
    active_pipeline: str = "default"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraConfig":
        """Create CameraConfig from a configuration dictionary."""
        resolution = data.get("resolution", {})
        dpi_profiles = [
            DpiProfile.from_dict(p)
            for p in data.get("dpi_profiles", [])
        ]
        return cls(
            device_index=data.get("device_index", 0),
            width=resolution.get("width", 1920),
            height=resolution.get("height", 1080),
            fps=data.get("fps", 30),
            dpi_profiles=dpi_profiles,
            active_pipeline=data.get("active_pipeline", "default"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device_index": self.device_index,
            "resolution": {
                "width": self.width,
                "height": self.height,
            },
            "fps": self.fps,
            "dpi_profiles": [p.to_dict() for p in self.dpi_profiles],
            "active_pipeline": self.active_pipeline,
        }


class CameraDevice(QObject):
    """
    Manages a single camera with threaded capture and vision pipeline support.

    This class provides:
    - Asynchronous frame capture in a dedicated thread
    - Vision pipeline processing on captured frames
    - Thread-safe frame access
    - DPI profile management for coordinate conversion
    - Qt signals for frame updates and state changes

    The camera runs continuously once started, emitting processed frames
    at the configured rate. Frames can be retrieved via the frame_ready
    signal or by calling get_current_frame().

    Signals:
        frame_ready: Emitted when a new processed frame is available.
                     Carries (camera_id: str, frame: np.ndarray).
        state_changed: Emitted when camera state changes.
                       Carries (camera_id: str, state: CameraState).
        error_occurred: Emitted when an error occurs.
                        Carries (camera_id: str, error_message: str).
        fps_updated: Emitted periodically with current FPS.
                     Carries (camera_id: str, fps: float).

    Example:
        camera = CameraDevice("top_camera", config)
        camera.frame_ready.connect(display_frame)
        camera.set_pipeline(my_pipeline)
        camera.start_capture()
    """

    frame_ready = pyqtSignal(str, np.ndarray)
    state_changed = pyqtSignal(str, CameraState)
    error_occurred = pyqtSignal(str, str)
    fps_updated = pyqtSignal(str, float)

    def __init__(
        self,
        camera_id: str,
        config: CameraConfig,
        name: str = "",
        parent: Optional[QObject] = None,
    ) -> None:
        """
        Initialize a camera device.

        Args:
            camera_id: Unique identifier for this camera.
            config: Camera configuration.
            name: Human-readable name for the camera.
            parent: Optional Qt parent object.
        """
        super().__init__(parent)

        self._camera_id = camera_id
        self._config = config
        self._name = name or camera_id
        self._enabled = True

        self._state = CameraState.DISCONNECTED
        self._capture: Optional[cv2.VideoCapture] = None

        # Current frame storage (thread-safe)
        self._current_frame: Optional[np.ndarray] = None
        self._frame_mutex = QMutex()

        # Vision pipeline (can be swapped at runtime)
        self._pipeline: Optional[Any] = None  # VisionPipeline
        self._pipeline_mutex = QMutex()

        # Capture thread control
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

        # FPS tracking
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0

        # State lock
        self._state_lock = threading.Lock()

    @property
    def camera_id(self) -> str:
        """Get the unique camera identifier."""
        return self._camera_id

    @property
    def name(self) -> str:
        """Get the human-readable camera name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the human-readable camera name."""
        self._name = value

    @property
    def config(self) -> CameraConfig:
        """Get the camera configuration."""
        return self._config

    @config.setter
    def config(self, value: CameraConfig) -> None:
        """
        Set a new camera configuration.

        Note: If currently streaming, you must stop and restart
        for changes to take effect.
        """
        self._config = value

    @property
    def state(self) -> CameraState:
        """Get the current camera state (thread-safe)."""
        with self._state_lock:
            return self._state

    @property
    def is_streaming(self) -> bool:
        """Check if the camera is currently streaming."""
        return self.state == CameraState.STREAMING

    @property
    def is_paused(self) -> bool:
        """Check if the camera is paused."""
        return self.state == CameraState.PAUSED

    @property
    def enabled(self) -> bool:
        """Check if the camera is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the camera."""
        self._enabled = value

    @property
    def device_index(self) -> int | str:
        """Get the OpenCV device index."""
        return self._config.device_index

    @property
    def resolution(self) -> tuple[int, int]:
        """Get the configured resolution (width, height)."""
        return (self._config.width, self._config.height)

    @property
    def current_fps(self) -> float:
        """Get the current measured FPS."""
        return self._current_fps

    @property
    def active_pipeline_id(self) -> str:
        """Get the ID of the active vision pipeline."""
        return self._config.active_pipeline

    def _set_state(self, new_state: CameraState) -> None:
        """
        Update the camera state and emit signal (thread-safe).

        Args:
            new_state: The new state to set.
        """
        with self._state_lock:
            if self._state != new_state:
                self._state = new_state
                self.state_changed.emit(self._camera_id, new_state)

    def set_pipeline(self, pipeline: Any) -> None:
        """
        Set the vision pipeline to apply to captured frames.

        Thread-safe. Can be called while streaming.

        Args:
            pipeline: VisionPipeline instance, or None to disable processing.
        """
        with QMutexLocker(self._pipeline_mutex):
            self._pipeline = pipeline
            if pipeline:
                self._config.active_pipeline = pipeline.pipeline_id

    def get_pipeline(self) -> Optional[Any]:
        """
        Get the current vision pipeline.

        Returns:
            The current VisionPipeline, or None.
        """
        with QMutexLocker(self._pipeline_mutex):
            return self._pipeline

    def start_capture(self) -> bool:
        """
        Open the camera and start capturing frames.

        Returns:
            True if capture started successfully, False otherwise.
        """
        if self.is_streaming or self.is_paused:
            return True

        if not self._enabled:
            self.error_occurred.emit(self._camera_id, "Camera is disabled")
            return False

        self._set_state(CameraState.CONNECTING)

        try:
            # Open capture device
            self._capture = cv2.VideoCapture(self._config.device_index)

            if not self._capture.isOpened():
                self._set_state(CameraState.ERROR)
                self.error_occurred.emit(
                    self._camera_id,
                    f"Failed to open camera at index {self._config.device_index}"
                )
                return False

            # Configure capture properties
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
            if self._config.fps > 0:
                self._capture.set(cv2.CAP_PROP_FPS, self._config.fps)

            # Reset FPS tracking
            self._frame_count = 0
            self._fps_start_time = time.time()

            # Start capture thread
            self._stop_event.clear()
            self._pause_event.set()
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name=f"Camera-{self._camera_id}",
                daemon=True,
            )
            self._capture_thread.start()

            self._set_state(CameraState.STREAMING)
            return True

        except Exception as e:
            self._set_state(CameraState.ERROR)
            self.error_occurred.emit(self._camera_id, f"Failed to start capture: {e}")
            return False

    def stop_capture(self) -> None:
        """
        Stop capturing and release the camera.
        """
        # Signal thread to stop
        self._stop_event.set()
        self._pause_event.set()  # Unpause if paused to allow thread to exit

        # Wait for thread to finish
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        self._capture_thread = None

        # Release capture device
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

        # Clear current frame
        with QMutexLocker(self._frame_mutex):
            self._current_frame = None

        self._set_state(CameraState.DISCONNECTED)

    def pause_capture(self) -> None:
        """
        Pause frame capture without releasing the camera.

        The camera remains open but frames are not processed.
        """
        if self.is_streaming:
            self._pause_event.clear()
            self._set_state(CameraState.PAUSED)

    def resume_capture(self) -> None:
        """
        Resume frame capture after pausing.
        """
        if self.is_paused:
            self._pause_event.set()
            self._set_state(CameraState.STREAMING)

    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get a copy of the most recent processed frame.

        Thread-safe.

        Returns:
            Copy of the current frame, or None if no frame available.
        """
        with QMutexLocker(self._frame_mutex):
            if self._current_frame is not None:
                return self._current_frame.copy()
            return None

    def get_raw_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame without pipeline processing.

        Useful for calibration or testing. Only works when streaming.

        Returns:
            Raw frame from the camera, or None if not available.
        """
        if self._capture is None or not self._capture.isOpened():
            return None

        ret, frame = self._capture.read()
        if ret:
            return frame
        return None

    def get_dpi_at_height(self, height_mm: float) -> tuple[float, float]:
        """
        Get interpolated DPI values for a given camera height.

        Uses linear interpolation between defined DPI profiles.

        Args:
            height_mm: Camera height in millimeters.

        Returns:
            Tuple of (dpi_x, dpi_y).
        """
        profiles = self._config.dpi_profiles
        if not profiles:
            return (96.0, 96.0)

        # Sort profiles by height
        sorted_profiles = sorted(profiles, key=lambda p: p.height_mm)

        # Find surrounding profiles for interpolation
        lower = None
        upper = None
        for profile in sorted_profiles:
            if profile.height_mm <= height_mm:
                lower = profile
            elif upper is None:
                upper = profile

        # Edge cases
        if lower is None:
            return (sorted_profiles[0].dpi_x, sorted_profiles[0].dpi_y)
        if upper is None:
            return (lower.dpi_x, lower.dpi_y)

        # Linear interpolation
        ratio = (height_mm - lower.height_mm) / (upper.height_mm - lower.height_mm)
        dpi_x = lower.dpi_x + ratio * (upper.dpi_x - lower.dpi_x)
        dpi_y = lower.dpi_y + ratio * (upper.dpi_y - lower.dpi_y)

        return (dpi_x, dpi_y)

    def pixels_to_mm(
        self,
        pixels_x: float,
        pixels_y: float,
        height_mm: float,
    ) -> tuple[float, float]:
        """
        Convert pixel coordinates to millimeters.

        Args:
            pixels_x: X coordinate in pixels.
            pixels_y: Y coordinate in pixels.
            height_mm: Current camera height in millimeters.

        Returns:
            Tuple of (mm_x, mm_y).
        """
        dpi_x, dpi_y = self.get_dpi_at_height(height_mm)
        mm_x = (pixels_x / dpi_x) * 25.4
        mm_y = (pixels_y / dpi_y) * 25.4
        return (mm_x, mm_y)

    def mm_to_pixels(
        self,
        mm_x: float,
        mm_y: float,
        height_mm: float,
    ) -> tuple[float, float]:
        """
        Convert millimeter coordinates to pixels.

        Args:
            mm_x: X coordinate in millimeters.
            mm_y: Y coordinate in millimeters.
            height_mm: Current camera height in millimeters.

        Returns:
            Tuple of (pixels_x, pixels_y).
        """
        dpi_x, dpi_y = self.get_dpi_at_height(height_mm)
        pixels_x = (mm_x / 25.4) * dpi_x
        pixels_y = (mm_y / 25.4) * dpi_y
        return (pixels_x, pixels_y)

    def _capture_loop(self) -> None:
        """
        Main capture thread loop.

        Continuously reads frames, applies the vision pipeline,
        and emits the processed frames.
        """
        while not self._stop_event.is_set():
            # Wait if paused
            self._pause_event.wait(timeout=0.1)
            if self._stop_event.is_set():
                break

            if not self._pause_event.is_set():
                continue

            # Capture frame
            if self._capture is None or not self._capture.isOpened():
                time.sleep(0.01)
                continue

            ret, frame = self._capture.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Apply pipeline
            processed_frame = self._apply_pipeline(frame)

            # Store frame (thread-safe)
            with QMutexLocker(self._frame_mutex):
                self._current_frame = processed_frame

            # Emit signal
            self.frame_ready.emit(self._camera_id, processed_frame)

            # Update FPS counter
            self._update_fps()

            # Throttle to target FPS if set
            if self._config.fps > 0:
                time.sleep(1.0 / self._config.fps)

    def _apply_pipeline(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply the vision pipeline to a frame.

        Args:
            frame: Input frame.

        Returns:
            Processed frame.
        """
        with QMutexLocker(self._pipeline_mutex):
            if self._pipeline is not None:
                try:
                    return self._pipeline.process(frame)
                except Exception as e:
                    self.error_occurred.emit(
                        self._camera_id, f"Pipeline error: {e}"
                    )
        return frame

    def _update_fps(self) -> None:
        """Update the FPS counter and emit signal periodically."""
        self._frame_count += 1
        elapsed = time.time() - self._fps_start_time

        # Update FPS every second
        if elapsed >= 1.0:
            self._current_fps = self._frame_count / elapsed
            self.fps_updated.emit(self._camera_id, self._current_fps)
            self._frame_count = 0
            self._fps_start_time = time.time()

    def __repr__(self) -> str:
        """Return string representation of the camera."""
        return (
            f"CameraDevice(id={self._camera_id!r}, name={self._name!r}, "
            f"index={self._config.device_index}, state={self.state.name})"
        )

    def __del__(self) -> None:
        """Ensure clean shutdown when object is destroyed."""
        self.stop_capture()
