# =============================================================================
# hardware/camera/vision_pipeline.py
#
# Composable vision pipeline framework for OpenCV operations.
# Pipelines are defined in JSON and executed as a sequence of steps,
# each step being an OpenCV operation with configurable parameters.
# =============================================================================

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import cv2
import numpy as np


@dataclass
class PipelineStep:
    """
    Represents a single step in a vision pipeline.

    Attributes:
        operation: Name of the OpenCV operation (e.g., "cvtColor", "threshold").
        params: Dictionary of parameters to pass to the operation.
        enabled: Whether this step is active.
    """
    operation: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineStep":
        """
        Create a PipelineStep from a configuration dictionary.

        Args:
            data: Dictionary containing step configuration.

        Returns:
            A new PipelineStep instance.
        """
        return cls(
            operation=data.get("operation", ""),
            params=data.get("params", {}),
            enabled=data.get("enabled", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the step to a dictionary.

        Returns:
            Dictionary representation of the step.
        """
        return {
            "operation": self.operation,
            "params": self.params,
            "enabled": self.enabled,
        }


class VisionPipeline:
    """
    Executes a sequence of OpenCV operations on image frames.

    The pipeline is configured from JSON and supports common OpenCV
    operations like color conversion, thresholding, blurring, edge
    detection, and morphological operations.

    Supported operations:
        - cvtColor: Color space conversion
        - threshold: Binary thresholding
        - adaptiveThreshold: Adaptive thresholding
        - GaussianBlur: Gaussian blur
        - medianBlur: Median blur
        - bilateralFilter: Bilateral filter
        - Canny: Canny edge detection
        - Sobel: Sobel derivatives
        - Laplacian: Laplacian operator
        - dilate: Morphological dilation
        - erode: Morphological erosion
        - morphologyEx: Advanced morphological operations
        - resize: Image resizing
        - flip: Image flipping
        - rotate: Image rotation
        - convertScaleAbs: Scale and convert to 8-bit
        - normalize: Normalize pixel values
        - equalizeHist: Histogram equalization
        - inRange: Color range thresholding
        - bitwise_not: Bitwise NOT operation
        - bitwise_and: Bitwise AND (with mask support)
        - findContours: Contour detection (draws contours)
        - HoughCircles: Circle detection (draws circles)
        - HoughLinesP: Line detection (draws lines)

    Example:
        pipeline = VisionPipeline.from_dict(pipeline_config)
        processed_frame = pipeline.process(frame)
    """

    # Mapping of OpenCV color conversion codes
    COLOR_CODES = {
        "COLOR_BGR2GRAY": cv2.COLOR_BGR2GRAY,
        "COLOR_BGR2RGB": cv2.COLOR_BGR2RGB,
        "COLOR_BGR2HSV": cv2.COLOR_BGR2HSV,
        "COLOR_BGR2LAB": cv2.COLOR_BGR2LAB,
        "COLOR_RGB2BGR": cv2.COLOR_RGB2BGR,
        "COLOR_RGB2GRAY": cv2.COLOR_RGB2GRAY,
        "COLOR_GRAY2BGR": cv2.COLOR_GRAY2BGR,
        "COLOR_GRAY2RGB": cv2.COLOR_GRAY2RGB,
        "COLOR_HSV2BGR": cv2.COLOR_HSV2BGR,
        "COLOR_LAB2BGR": cv2.COLOR_LAB2BGR,
    }

    # Mapping of threshold types
    THRESHOLD_TYPES = {
        "THRESH_BINARY": cv2.THRESH_BINARY,
        "THRESH_BINARY_INV": cv2.THRESH_BINARY_INV,
        "THRESH_TRUNC": cv2.THRESH_TRUNC,
        "THRESH_TOZERO": cv2.THRESH_TOZERO,
        "THRESH_TOZERO_INV": cv2.THRESH_TOZERO_INV,
        "THRESH_OTSU": cv2.THRESH_OTSU,
    }

    # Mapping of adaptive threshold types
    ADAPTIVE_METHODS = {
        "ADAPTIVE_THRESH_MEAN_C": cv2.ADAPTIVE_THRESH_MEAN_C,
        "ADAPTIVE_THRESH_GAUSSIAN_C": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    }

    # Mapping of morphological operations
    MORPH_OPS = {
        "MORPH_ERODE": cv2.MORPH_ERODE,
        "MORPH_DILATE": cv2.MORPH_DILATE,
        "MORPH_OPEN": cv2.MORPH_OPEN,
        "MORPH_CLOSE": cv2.MORPH_CLOSE,
        "MORPH_GRADIENT": cv2.MORPH_GRADIENT,
        "MORPH_TOPHAT": cv2.MORPH_TOPHAT,
        "MORPH_BLACKHAT": cv2.MORPH_BLACKHAT,
    }

    # Mapping of border types
    BORDER_TYPES = {
        "BORDER_CONSTANT": cv2.BORDER_CONSTANT,
        "BORDER_REPLICATE": cv2.BORDER_REPLICATE,
        "BORDER_REFLECT": cv2.BORDER_REFLECT,
        "BORDER_WRAP": cv2.BORDER_WRAP,
        "BORDER_DEFAULT": cv2.BORDER_DEFAULT,
    }

    # Mapping of interpolation methods
    INTERPOLATION = {
        "INTER_NEAREST": cv2.INTER_NEAREST,
        "INTER_LINEAR": cv2.INTER_LINEAR,
        "INTER_AREA": cv2.INTER_AREA,
        "INTER_CUBIC": cv2.INTER_CUBIC,
        "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
    }

    # Mapping of rotation codes
    ROTATE_CODES = {
        "ROTATE_90_CLOCKWISE": cv2.ROTATE_90_CLOCKWISE,
        "ROTATE_180": cv2.ROTATE_180,
        "ROTATE_90_COUNTERCLOCKWISE": cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    def __init__(
        self,
        pipeline_id: str = "",
        name: str = "",
        description: str = "",
        version: str = "1.0.0",
    ) -> None:
        """
        Initialize an empty vision pipeline.

        Args:
            pipeline_id: Unique identifier for this pipeline.
            name: Human-readable name.
            description: Description of what the pipeline does.
            version: Version string.
        """
        self._pipeline_id = pipeline_id
        self._name = name
        self._description = description
        self._version = version
        self._steps: list[PipelineStep] = []
        self._last_error: Optional[str] = None

        # Build operation dispatch table
        self._operations: dict[str, Callable[[np.ndarray, dict], np.ndarray]] = {
            "cvtColor": self._op_cvt_color,
            "threshold": self._op_threshold,
            "adaptiveThreshold": self._op_adaptive_threshold,
            "GaussianBlur": self._op_gaussian_blur,
            "medianBlur": self._op_median_blur,
            "bilateralFilter": self._op_bilateral_filter,
            "Canny": self._op_canny,
            "Sobel": self._op_sobel,
            "Laplacian": self._op_laplacian,
            "dilate": self._op_dilate,
            "erode": self._op_erode,
            "morphologyEx": self._op_morphology_ex,
            "resize": self._op_resize,
            "flip": self._op_flip,
            "rotate": self._op_rotate,
            "convertScaleAbs": self._op_convert_scale_abs,
            "normalize": self._op_normalize,
            "equalizeHist": self._op_equalize_hist,
            "inRange": self._op_in_range,
            "bitwise_not": self._op_bitwise_not,
            "bitwise_and": self._op_bitwise_and,
            "findContours": self._op_find_contours,
            "HoughCircles": self._op_hough_circles,
            "HoughLinesP": self._op_hough_lines_p,
        }

    @property
    def pipeline_id(self) -> str:
        """Get the pipeline identifier."""
        return self._pipeline_id

    @property
    def name(self) -> str:
        """Get the pipeline name."""
        return self._name

    @property
    def description(self) -> str:
        """Get the pipeline description."""
        return self._description

    @property
    def version(self) -> str:
        """Get the pipeline version."""
        return self._version

    @property
    def steps(self) -> list[PipelineStep]:
        """Get a copy of the pipeline steps."""
        return self._steps.copy()

    @property
    def step_count(self) -> int:
        """Get the number of steps in the pipeline."""
        return len(self._steps)

    @property
    def last_error(self) -> Optional[str]:
        """Get the last error message, if any."""
        return self._last_error

    @property
    def supported_operations(self) -> list[str]:
        """Get a list of supported operation names."""
        return list(self._operations.keys())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisionPipeline":
        """
        Create a VisionPipeline from a configuration dictionary.

        Args:
            data: Pipeline configuration dictionary.

        Returns:
            A new VisionPipeline instance.
        """
        pipeline = cls(
            pipeline_id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
        )

        for step_data in data.get("steps", []):
            step = PipelineStep.from_dict(step_data)
            pipeline.add_step(step)

        return pipeline

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the pipeline to a dictionary.

        Returns:
            Dictionary representation of the pipeline.
        """
        return {
            "id": self._pipeline_id,
            "name": self._name,
            "description": self._description,
            "version": self._version,
            "steps": [step.to_dict() for step in self._steps],
        }

    def add_step(self, step: PipelineStep) -> None:
        """
        Add a step to the pipeline.

        Args:
            step: The pipeline step to add.
        """
        self._steps.append(step)

    def remove_step(self, index: int) -> bool:
        """
        Remove a step from the pipeline by index.

        Args:
            index: Index of the step to remove.

        Returns:
            True if removed, False if index out of range.
        """
        if 0 <= index < len(self._steps):
            del self._steps[index]
            return True
        return False

    def clear_steps(self) -> None:
        """Remove all steps from the pipeline."""
        self._steps.clear()

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame through all pipeline steps.

        Args:
            frame: Input image as numpy array (BGR format expected).

        Returns:
            Processed image. Returns original frame if pipeline is empty
            or if an error occurs.
        """
        self._last_error = None

        if frame is None or frame.size == 0:
            self._last_error = "Invalid input frame"
            return frame

        result = frame.copy()

        for i, step in enumerate(self._steps):
            if not step.enabled:
                continue

            if step.operation not in self._operations:
                self._last_error = f"Step {i}: Unknown operation '{step.operation}'"
                continue

            try:
                result = self._operations[step.operation](result, step.params)
            except Exception as e:
                self._last_error = f"Step {i} ({step.operation}): {e}"
                # Continue with the current result

        return result

    # -------------------------------------------------------------------------
    # Operation implementations
    # -------------------------------------------------------------------------

    def _op_cvt_color(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Convert color space."""
        code_name = params.get("code", "COLOR_BGR2GRAY")
        code = self.COLOR_CODES.get(code_name, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(img, code)

    def _op_threshold(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Apply binary threshold."""
        thresh = params.get("thresh", 127)
        maxval = params.get("maxval", 255)
        type_name = params.get("type", "THRESH_BINARY")
        thresh_type = self.THRESHOLD_TYPES.get(type_name, cv2.THRESH_BINARY)

        # Handle Otsu's method
        if "THRESH_OTSU" in type_name:
            thresh_type |= cv2.THRESH_OTSU

        # Ensure grayscale for thresholding
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, result = cv2.threshold(img, thresh, maxval, thresh_type)
        return result

    def _op_adaptive_threshold(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Apply adaptive threshold."""
        maxval = params.get("maxval", 255)
        method_name = params.get("method", "ADAPTIVE_THRESH_GAUSSIAN_C")
        method = self.ADAPTIVE_METHODS.get(method_name, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        type_name = params.get("type", "THRESH_BINARY")
        thresh_type = self.THRESHOLD_TYPES.get(type_name, cv2.THRESH_BINARY)
        block_size = params.get("blockSize", 11)
        c = params.get("C", 2)

        # Ensure grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Block size must be odd
        if block_size % 2 == 0:
            block_size += 1

        return cv2.adaptiveThreshold(img, maxval, method, thresh_type, block_size, c)

    def _op_gaussian_blur(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Apply Gaussian blur."""
        ksize = params.get("ksize", [5, 5])
        if isinstance(ksize, list):
            ksize = tuple(ksize)
        sigma_x = params.get("sigmaX", 0)
        sigma_y = params.get("sigmaY", 0)
        return cv2.GaussianBlur(img, ksize, sigma_x, sigmaY=sigma_y)

    def _op_median_blur(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Apply median blur."""
        ksize = params.get("ksize", 5)
        # Kernel size must be odd
        if ksize % 2 == 0:
            ksize += 1
        return cv2.medianBlur(img, ksize)

    def _op_bilateral_filter(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Apply bilateral filter."""
        d = params.get("d", 9)
        sigma_color = params.get("sigmaColor", 75)
        sigma_space = params.get("sigmaSpace", 75)
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    def _op_canny(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Apply Canny edge detection."""
        threshold1 = params.get("threshold1", 100)
        threshold2 = params.get("threshold2", 200)
        aperture_size = params.get("apertureSize", 3)
        l2_gradient = params.get("L2gradient", False)

        # Ensure grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return cv2.Canny(img, threshold1, threshold2, apertureSize=aperture_size,
                         L2gradient=l2_gradient)

    def _op_sobel(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Apply Sobel derivatives."""
        dx = params.get("dx", 1)
        dy = params.get("dy", 0)
        ksize = params.get("ksize", 3)
        scale = params.get("scale", 1)
        delta = params.get("delta", 0)

        # Ensure grayscale for Sobel
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        result = cv2.Sobel(img, cv2.CV_64F, dx, dy, ksize=ksize, scale=scale, delta=delta)
        return cv2.convertScaleAbs(result)

    def _op_laplacian(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Apply Laplacian operator."""
        ksize = params.get("ksize", 3)
        scale = params.get("scale", 1)
        delta = params.get("delta", 0)

        # Ensure grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        result = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize, scale=scale, delta=delta)
        return cv2.convertScaleAbs(result)

    def _op_dilate(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Apply morphological dilation."""
        ksize = params.get("ksize", [3, 3])
        if isinstance(ksize, list):
            ksize = tuple(ksize)
        iterations = params.get("iterations", 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        return cv2.dilate(img, kernel, iterations=iterations)

    def _op_erode(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Apply morphological erosion."""
        ksize = params.get("ksize", [3, 3])
        if isinstance(ksize, list):
            ksize = tuple(ksize)
        iterations = params.get("iterations", 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        return cv2.erode(img, kernel, iterations=iterations)

    def _op_morphology_ex(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Apply advanced morphological operation."""
        op_name = params.get("op", "MORPH_OPEN")
        op = self.MORPH_OPS.get(op_name, cv2.MORPH_OPEN)
        ksize = params.get("ksize", [3, 3])
        if isinstance(ksize, list):
            ksize = tuple(ksize)
        iterations = params.get("iterations", 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        return cv2.morphologyEx(img, op, kernel, iterations=iterations)

    def _op_resize(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Resize image."""
        width = params.get("width")
        height = params.get("height")
        fx = params.get("fx", 0)
        fy = params.get("fy", 0)
        interp_name = params.get("interpolation", "INTER_LINEAR")
        interp = self.INTERPOLATION.get(interp_name, cv2.INTER_LINEAR)

        if width and height:
            return cv2.resize(img, (width, height), interpolation=interp)
        else:
            return cv2.resize(img, None, fx=fx, fy=fy, interpolation=interp)

    def _op_flip(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Flip image."""
        # 0 = vertical, 1 = horizontal, -1 = both
        flip_code = params.get("flipCode", 1)
        return cv2.flip(img, flip_code)

    def _op_rotate(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Rotate image by 90-degree increments."""
        code_name = params.get("rotateCode", "ROTATE_90_CLOCKWISE")
        code = self.ROTATE_CODES.get(code_name, cv2.ROTATE_90_CLOCKWISE)
        return cv2.rotate(img, code)

    def _op_convert_scale_abs(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Scale and convert to 8-bit."""
        alpha = params.get("alpha", 1.0)
        beta = params.get("beta", 0)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def _op_normalize(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Normalize pixel values."""
        alpha = params.get("alpha", 0)
        beta = params.get("beta", 255)
        norm_type = params.get("normType", cv2.NORM_MINMAX)
        result = np.zeros_like(img)
        cv2.normalize(img, result, alpha, beta, norm_type)
        return result

    def _op_equalize_hist(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Apply histogram equalization."""
        # Ensure grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(img)

    def _op_in_range(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Apply color range threshold."""
        lower = params.get("lower", [0, 0, 0])
        upper = params.get("upper", [255, 255, 255])
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        return cv2.inRange(img, lower, upper)

    def _op_bitwise_not(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Apply bitwise NOT."""
        return cv2.bitwise_not(img)

    def _op_bitwise_and(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Apply bitwise AND (typically with a mask)."""
        # This operates on the image with itself (useful after inRange)
        return cv2.bitwise_and(img, img)

    def _op_find_contours(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Find and draw contours."""
        mode = params.get("mode", cv2.RETR_EXTERNAL)
        method = params.get("method", cv2.CHAIN_APPROX_SIMPLE)
        color = params.get("color", [0, 255, 0])
        thickness = params.get("thickness", 2)
        min_area = params.get("minArea", 0)

        # Need grayscale/binary for contour detection
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        contours, _ = cv2.findContours(gray, mode, method)

        # Convert back to BGR for drawing if needed
        if len(img.shape) == 2:
            result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            result = img.copy()

        # Filter by area and draw
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(result, [contour], -1, color, thickness)

        return result

    def _op_hough_circles(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Detect and draw circles using Hough transform."""
        dp = params.get("dp", 1)
        min_dist = params.get("minDist", 20)
        param1 = params.get("param1", 50)
        param2 = params.get("param2", 30)
        min_radius = params.get("minRadius", 0)
        max_radius = params.get("maxRadius", 0)
        color = params.get("color", [0, 255, 0])
        thickness = params.get("thickness", 2)

        # Need grayscale for Hough circles
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = img.copy()
        else:
            gray = img
            result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp, min_dist,
            param1=param1, param2=param2,
            minRadius=min_radius, maxRadius=max_radius
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(result, (i[0], i[1]), i[2], color, thickness)
                cv2.circle(result, (i[0], i[1]), 2, [0, 0, 255], 3)

        return result

    def _op_hough_lines_p(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Detect and draw lines using probabilistic Hough transform."""
        rho = params.get("rho", 1)
        theta = params.get("theta", np.pi / 180)
        threshold = params.get("threshold", 50)
        min_line_length = params.get("minLineLength", 50)
        max_line_gap = params.get("maxLineGap", 10)
        color = params.get("color", [0, 255, 0])
        thickness = params.get("thickness", 2)

        # Need grayscale/edges for Hough lines
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = img.copy()
        else:
            gray = img
            result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        lines = cv2.HoughLinesP(
            gray, rho, theta, threshold,
            minLineLength=min_line_length, maxLineGap=max_line_gap
        )

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), color, thickness)

        return result

    def __repr__(self) -> str:
        """Return string representation of the pipeline."""
        return (
            f"VisionPipeline(id={self._pipeline_id!r}, name={self._name!r}, "
            f"steps={self.step_count})"
        )
