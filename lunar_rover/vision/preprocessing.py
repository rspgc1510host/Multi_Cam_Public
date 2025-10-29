"""Preprocessing pipeline for lunar imagery.

This module provides a comprehensive preprocessing pipeline for both real and
synthetic lunar imagery before detection tasks. It supports various image
sources including local files, video frames, and arrays, with configurable
preprocessing steps and feature extraction utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import yaml
except ImportError:
    yaml = None


ArrayLike = np.ndarray
ConfigDict = Dict[str, Any]


__all__ = (
    "load_image",
    "load_video_frame",
    "normalize",
    "denoise",
    "enhance_contrast",
    "resize_image",
    "extract_edges",
    "extract_texture_descriptors",
    "PreprocessingPipeline",
)


def load_image(
    image_path: Union[str, Path],
    grayscale: bool = False,
) -> np.ndarray:
    """Load an image from a local file path.

    Example:
        >>> img = load_image("lunar_surface.png")
        >>> img.shape
        (480, 640, 3)

    Args:
        image_path: Path to the image file.
        grayscale: If True, load image as grayscale. Otherwise, load as BGR.

    Returns:
        Image as a NumPy array with shape (H, W, C) for color or (H, W) for
        grayscale.

    Raises:
        ImportError: If OpenCV is not installed.
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be loaded.
    """
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for image loading.")

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(image_path), flag)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return image


def load_video_frame(
    video_path: Union[str, Path],
    frame_index: int = 0,
    grayscale: bool = False,
) -> np.ndarray:
    """Load a specific frame from a video file.

    Example:
        >>> frame = load_video_frame("lunar_traverse.mp4", frame_index=100)
        >>> frame.shape
        (720, 1280, 3)

    Args:
        video_path: Path to the video file.
        frame_index: Zero-based index of the frame to extract.
        grayscale: If True, convert frame to grayscale.

    Returns:
        Video frame as a NumPy array.

    Raises:
        ImportError: If OpenCV is not installed.
        FileNotFoundError: If the video file does not exist.
        ValueError: If the frame cannot be loaded or frame_index is invalid.
    """
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for video frame loading.")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index < 0 or frame_index >= total_frames:
            raise ValueError(
                f"Invalid frame_index {frame_index}. "
                f"Video has {total_frames} frames."
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret or frame is None:
            raise ValueError(f"Failed to read frame {frame_index} from {video_path}")

        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame
    finally:
        cap.release()


def normalize(
    image: np.ndarray,
    method: str = "minmax",
    target_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Normalize image intensity values.

    Example:
        >>> img = np.array([[0, 128, 255]], dtype=np.uint8)
        >>> normalized = normalize(img, method="minmax")
        >>> normalized.min(), normalized.max()
        (0.0, 1.0)

    Args:
        image: Input image as NumPy array.
        method: Normalization method. Options: "minmax", "zscore", "clip".
        target_range: Target range for minmax normalization (min, max).

    Returns:
        Normalized image as float32 array.

    Raises:
        ValueError: If method is not recognized.
    """
    image_float = image.astype(np.float32)

    if method == "minmax":
        min_val = image_float.min()
        max_val = image_float.max()

        if max_val - min_val < 1e-7:
            return np.full_like(image_float, target_range[0])

        normalized = (image_float - min_val) / (max_val - min_val)
        normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
        return normalized

    elif method == "zscore":
        mean = image_float.mean()
        std = image_float.std()

        if std < 1e-7:
            return np.zeros_like(image_float)

        return (image_float - mean) / std

    elif method == "clip":
        return np.clip(image_float, target_range[0], target_range[1])

    else:
        raise ValueError(
            f"Unknown normalization method: {method}. "
            f"Choose from 'minmax', 'zscore', or 'clip'."
        )


def denoise(
    image: np.ndarray,
    method: str = "gaussian",
    kernel_size: int = 5,
    sigma: float = 1.0,
    h: float = 10.0,
) -> np.ndarray:
    """Apply denoising to an image.

    Example:
        >>> noisy = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> denoised = denoise(noisy, method="gaussian")
        >>> denoised.shape
        (100, 100)

    Args:
        image: Input image as NumPy array.
        method: Denoising method. Options: "gaussian", "bilateral", "nlmeans".
        kernel_size: Kernel size for gaussian/bilateral filters.
        sigma: Standard deviation for Gaussian filter.
        h: Filter strength for non-local means denoising.

    Returns:
        Denoised image.

    Raises:
        ImportError: If OpenCV is not installed.
        ValueError: If method is not recognized.
    """
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for denoising.")

    if method == "gaussian":
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    elif method == "bilateral":
        return cv2.bilateralFilter(image, kernel_size, sigma * 2, sigma * 2)

    elif method == "nlmeans":
        if len(image.shape) == 2:
            return cv2.fastNlMeansDenoising(image, None, h)
        else:
            return cv2.fastNlMeansDenoisingColored(image, None, h, h)

    else:
        raise ValueError(
            f"Unknown denoising method: {method}. "
            f"Choose from 'gaussian', 'bilateral', or 'nlmeans'."
        )


def enhance_contrast(
    image: np.ndarray,
    method: str = "clahe",
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    alpha: float = 1.5,
    beta: float = 0.0,
) -> np.ndarray:
    """Enhance image contrast.

    Example:
        >>> low_contrast = np.full((100, 100), 128, dtype=np.uint8)
        >>> enhanced = enhance_contrast(low_contrast, method="clahe")
        >>> enhanced.shape
        (100, 100)

    Args:
        image: Input image as NumPy array.
        method: Contrast enhancement method. Options: "clahe", "histogram_eq",
            "linear".
        clip_limit: Clip limit for CLAHE.
        tile_grid_size: Grid size for CLAHE.
        alpha: Gain factor for linear contrast adjustment.
        beta: Bias offset for linear contrast adjustment.

    Returns:
        Contrast-enhanced image.

    Raises:
        ImportError: If OpenCV is not installed.
        ValueError: If method is not recognized.
    """
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for contrast enhancement.")

    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        if len(image.shape) == 2:
            return clahe.apply(image)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    elif method == "histogram_eq":
        if len(image.shape) == 2:
            return cv2.equalizeHist(image)
        else:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    elif method == "linear":
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced

    else:
        raise ValueError(
            f"Unknown contrast enhancement method: {method}. "
            f"Choose from 'clahe', 'histogram_eq', or 'linear'."
        )


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    interpolation: str = "linear",
) -> np.ndarray:
    """Resize an image to the specified dimensions.

    Example:
        >>> img = np.zeros((100, 200, 3), dtype=np.uint8)
        >>> resized = resize_image(img, (50, 100))
        >>> resized.shape
        (50, 100, 3)

    Args:
        image: Input image as NumPy array.
        size: Target size as (height, width).
        interpolation: Interpolation method. Options: "nearest", "linear",
            "cubic", "area", "lanczos".

    Returns:
        Resized image.

    Raises:
        ImportError: If OpenCV is not installed.
        ValueError: If interpolation method is not recognized.
    """
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for resizing.")

    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    if interpolation not in interpolation_map:
        raise ValueError(
            f"Unknown interpolation method: {interpolation}. "
            f"Choose from {list(interpolation_map.keys())}."
        )

    height, width = size
    return cv2.resize(image, (width, height), interpolation=interpolation_map[interpolation])


def extract_edges(
    image: np.ndarray,
    method: str = "canny",
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
    aperture_size: int = 3,
    sobel_ksize: int = 3,
) -> np.ndarray:
    """Extract edges from an image.

    This is a feature extraction stub that can be used by downstream detectors
    for obstacle and crater detection.

    Example:
        >>> img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> edges = extract_edges(img, method="canny")
        >>> edges.shape
        (100, 100)

    Args:
        image: Input image as NumPy array. If color, will be converted to
            grayscale.
        method: Edge detection method. Options: "canny", "sobel", "laplacian".
        low_threshold: Lower threshold for Canny edge detector.
        high_threshold: Upper threshold for Canny edge detector.
        aperture_size: Aperture size for Canny edge detector.
        sobel_ksize: Kernel size for Sobel operator.

    Returns:
        Edge map as uint8 array.

    Raises:
        ImportError: If OpenCV is not installed.
        ValueError: If method is not recognized.
    """
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for edge extraction.")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if method == "canny":
        edges = cv2.Canny(
            gray,
            low_threshold,
            high_threshold,
            apertureSize=aperture_size,
        )
        return edges

    elif method == "sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
        return magnitude

    elif method == "laplacian":
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.abs(laplacian)
        laplacian = np.clip(laplacian, 0, 255).astype(np.uint8)
        return laplacian

    else:
        raise ValueError(
            f"Unknown edge detection method: {method}. "
            f"Choose from 'canny', 'sobel', or 'laplacian'."
        )


def extract_texture_descriptors(
    image: np.ndarray,
    method: str = "lbp",
    num_points: int = 8,
    radius: int = 1,
) -> Dict[str, Any]:
    """Extract texture descriptors from an image.

    This is a feature extraction stub that can be used by downstream detectors
    for terrain classification and surface analysis.

    Example:
        >>> img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> descriptors = extract_texture_descriptors(img, method="lbp")
        >>> "histogram" in descriptors
        True

    Args:
        image: Input image as NumPy array. If color, will be converted to
            grayscale.
        method: Texture descriptor method. Options: "lbp" (Local Binary Patterns),
            "glcm" (Gray-Level Co-occurrence Matrix), "stats".
        num_points: Number of points for LBP.
        radius: Radius for LBP.

    Returns:
        Dictionary containing texture descriptors and metadata.

    Raises:
        ValueError: If method is not recognized.
    """
    if len(image.shape) == 3:
        if cv2 is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray = image

    if method == "lbp":
        lbp_image = _compute_lbp(gray, num_points, radius)
        histogram, _ = np.histogram(lbp_image, bins=256, range=(0, 256))
        histogram = histogram.astype(np.float32)
        histogram /= histogram.sum() + 1e-7

        return {
            "method": "lbp",
            "histogram": histogram,
            "lbp_image": lbp_image,
            "num_points": num_points,
            "radius": radius,
        }

    elif method == "glcm":
        glcm_features = _compute_glcm_features(gray)
        return {
            "method": "glcm",
            **glcm_features,
        }

    elif method == "stats":
        stats = {
            "method": "stats",
            "mean": float(gray.mean()),
            "std": float(gray.std()),
            "min": float(gray.min()),
            "max": float(gray.max()),
            "entropy": _compute_entropy(gray),
        }
        return stats

    else:
        raise ValueError(
            f"Unknown texture descriptor method: {method}. "
            f"Choose from 'lbp', 'glcm', or 'stats'."
        )


def _compute_lbp(
    image: np.ndarray,
    num_points: int,
    radius: int,
) -> np.ndarray:
    """Compute Local Binary Pattern for the image."""
    height, width = image.shape
    lbp = np.zeros((height, width), dtype=np.uint8)

    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            center = image[i, j]
            pattern = 0

            for p in range(num_points):
                angle = 2 * np.pi * p / num_points
                x = i + radius * np.cos(angle)
                y = j - radius * np.sin(angle)

                x1, y1 = int(np.floor(x)), int(np.floor(y))
                x2, y2 = min(x1 + 1, height - 1), min(y1 + 1, width - 1)

                fx, fy = x - x1, y - y1
                neighbor = (
                    (1 - fx) * (1 - fy) * image[x1, y1]
                    + fx * (1 - fy) * image[x2, y1]
                    + (1 - fx) * fy * image[x1, y2]
                    + fx * fy * image[x2, y2]
                )

                if neighbor >= center:
                    pattern |= (1 << p)

            lbp[i, j] = pattern

    return lbp


def _compute_glcm_features(image: np.ndarray) -> Dict[str, float]:
    """Compute simplified GLCM-based texture features."""
    normalized = (image / image.max() * 15).astype(np.uint8) if image.max() > 0 else image
    glcm = np.zeros((16, 16), dtype=np.float32)

    height, width = normalized.shape
    for i in range(height - 1):
        for j in range(width - 1):
            val1 = normalized[i, j]
            val2 = normalized[i, j + 1]
            glcm[val1, val2] += 1

    glcm /= glcm.sum() + 1e-7

    contrast = 0.0
    homogeneity = 0.0
    energy = 0.0

    for i in range(16):
        for j in range(16):
            contrast += (i - j) ** 2 * glcm[i, j]
            homogeneity += glcm[i, j] / (1 + abs(i - j))
            energy += glcm[i, j] ** 2

    return {
        "contrast": float(contrast),
        "homogeneity": float(homogeneity),
        "energy": float(energy),
    }


def _compute_entropy(image: np.ndarray) -> float:
    """Compute entropy of the image."""
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    histogram = histogram.astype(np.float32)
    histogram /= histogram.sum() + 1e-7
    histogram = histogram[histogram > 0]
    entropy = -np.sum(histogram * np.log2(histogram))
    return float(entropy)


class PreprocessingPipeline:
    """Configurable preprocessing pipeline for lunar imagery.

    This class chains multiple preprocessing steps and can be configured via
    dictionaries, JSON, or YAML configuration files.

    Example:
        >>> config = {
        ...     "resize": {"size": (480, 640)},
        ...     "normalize": {"method": "minmax"},
        ...     "denoise": {"method": "gaussian"},
        ...     "enhance_contrast": {"method": "clahe"},
        ... }
        >>> pipeline = PreprocessingPipeline(config)
        >>> image = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        >>> processed = pipeline.process(image)
        >>> processed.shape
        (480, 640, 3)
    """

    def __init__(
        self,
        config: Optional[Union[ConfigDict, str, Path]] = None,
    ):
        """Initialize preprocessing pipeline with configuration.

        Args:
            config: Configuration as a dictionary, or path to JSON/YAML file.
                If None, an empty pipeline is created.
        """
        if config is None:
            self.config = {}
        elif isinstance(config, (str, Path)):
            self.config = self._load_config_file(config)
        else:
            self.config = dict(config)

        self.steps: List[Tuple[str, Callable, Dict[str, Any]]] = []
        self._build_pipeline()

    def _load_config_file(self, config_path: Union[str, Path]) -> ConfigDict:
        """Load configuration from JSON or YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            if config_path.suffix in [".yaml", ".yml"]:
                if yaml is None:
                    raise ImportError(
                        "PyYAML is required for YAML config files. "
                        "Install with: pip install pyyaml"
                    )
                return yaml.safe_load(f)
            elif config_path.suffix == ".json":
                return json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path.suffix}. "
                    f"Use .json, .yaml, or .yml."
                )

    def _build_pipeline(self) -> None:
        """Build the processing pipeline from configuration."""
        step_map = {
            "resize": (resize_image, ["size"]),
            "normalize": (normalize, []),
            "denoise": (denoise, []),
            "enhance_contrast": (enhance_contrast, []),
        }

        for step_name, (func, required_params) in step_map.items():
            if step_name in self.config:
                params = self.config[step_name]
                if not isinstance(params, dict):
                    raise ValueError(
                        f"Configuration for '{step_name}' must be a dictionary."
                    )

                for param in required_params:
                    if param not in params:
                        raise ValueError(
                            f"Required parameter '{param}' missing for step '{step_name}'."
                        )

                self.steps.append((step_name, func, params))

    def process(
        self,
        image: Union[np.ndarray, str, Path],
        extract_features: bool = False,
        edge_method: Optional[str] = None,
        texture_method: Optional[str] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """Process an image through the configured pipeline.

        Args:
            image: Input image as NumPy array or file path.
            extract_features: If True, also extract edges and texture features.
            edge_method: Edge detection method to use if extract_features is True.
            texture_method: Texture descriptor method if extract_features is True.

        Returns:
            Processed image, or tuple of (processed_image, features_dict) if
            extract_features is True.
        """
        if isinstance(image, (str, Path)):
            result = load_image(image)
        elif isinstance(image, np.ndarray):
            result = image.copy()
        else:
            raise TypeError(
                "Input to process must be a numpy array or file path string."
            )

        for step_name, func, params in self.steps:
            result = func(result, **params)

        if extract_features:
            features = {}

            if edge_method:
                features["edges"] = extract_edges(result, method=edge_method)

            if texture_method:
                features["texture"] = extract_texture_descriptors(
                    result, method=texture_method
                )

            return result, features

        return result

    def add_step(
        self,
        step_name: str,
        func: Callable,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a custom preprocessing step to the pipeline.

        Args:
            step_name: Name identifier for the step.
            func: Function that takes an image and keyword arguments.
            params: Parameters to pass to the function.
        """
        if params is None:
            params = {}
        self.steps.append((step_name, func, params))

    def save_config(self, output_path: Union[str, Path]) -> None:
        """Save the current configuration to a file.

        Args:
            output_path: Path to save the configuration (JSON or YAML).
        """
        output_path = Path(output_path)

        with open(output_path, "w") as f:
            if output_path.suffix in [".yaml", ".yml"]:
                if yaml is None:
                    raise ImportError(
                        "PyYAML is required for YAML output. "
                        "Install with: pip install pyyaml"
                    )
                yaml.dump(self.config, f, default_flow_style=False)
            elif output_path.suffix == ".json":
                json.dump(self.config, f, indent=2)
            else:
                raise ValueError(
                    f"Unsupported output format: {output_path.suffix}. "
                    f"Use .json, .yaml, or .yml."
                )

    def __repr__(self) -> str:
        """Return string representation of the pipeline."""
        step_names = [name for name, _, _ in self.steps]
        return f"PreprocessingPipeline(steps={step_names})"
