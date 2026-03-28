# =============================================================================
# hardware/camera/__init__.py
#
# Camera management submodule.
# Provides OpenCV-based camera capture with threaded operation and
# configurable vision pipelines.
# =============================================================================

from hardware.camera.camera_device import CameraDevice
from hardware.camera.camera_manager import CameraManager
from hardware.camera.vision_pipeline import VisionPipeline, PipelineStep

__all__ = ["CameraDevice", "CameraManager", "VisionPipeline", "PipelineStep"]
