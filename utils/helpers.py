"""Reusable helper utilities for the Multi_Cam_Public inference pipeline.

The utilities in this module encapsulate common pre- and post-processing
operations needed across detection, re-identification, and tracking stages.
Each function validates its inputs, handles edge cases defensively, and
provides concise docstrings with usage samples for quick adoption.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:  # pragma: no cover - optional dependency
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

ArrayLike = Union[np.ndarray, "torch.Tensor"]
BoundingBoxType = Sequence[float]
ColorType = Tuple[int, int, int]

DEFAULT_COLOR_PALETTE: Tuple[ColorType, ...] = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
    (0, 128, 255),
    (255, 128, 0),
    (102, 255, 102),
)

__all__ = (
    "crop_person",
    "compute_cosine_similarity",
    "draw_annotations",
    "track_feature",
    "get_default_palette",
    "color_for_id",
)


def crop_person(frame: ArrayLike, bbox: BoundingBoxType) -> ArrayLike:
    """Crop a person region from a frame given a bounding box.

    The bounding box is clipped to the spatial extent of the frame. When the
    box falls completely outside the frame, an empty crop with zero height and
    width is returned, preserving the input data type.

    Example:
        >>> frame = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> crop = crop_person(frame, (10, 10, 50, 50))
        >>> crop.shape
        (40, 40, 3)

    Args:
        frame: Image-like array of shape ``(H, W, C)`` as ``np.ndarray`` or
            ``torch.Tensor``.
        bbox: Bounding box ``(x1, y1, x2, y2)`` with coordinates measured in
            pixels. Floats are accepted and will be rounded to the nearest
            integer pixels.

    Returns:
        Crop of the region defined by the bounding box, matching the input
        array type.
    """

    assert isinstance(bbox, Sequence) and len(bbox) == 4, (
        "Bounding box must be a sequence of four values."
    )
    frame_np, to_original = _ensure_frame(frame)

    height, width = frame_np.shape[0], frame_np.shape[1]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1, y1, x2, y2 = _clip_bbox(x1, y1, x2, y2, width, height)

    if x2 <= x1 or y2 <= y1:
        empty_crop = np.empty((0, 0, frame_np.shape[2]), dtype=frame_np.dtype)
        return to_original(empty_crop)

    x1_i = int(np.floor(x1))
    y1_i = int(np.floor(y1))
    x2_i = int(np.ceil(x2))
    y2_i = int(np.ceil(y2))

    x1_i = max(x1_i, 0)
    y1_i = max(y1_i, 0)
    x2_i = min(x2_i, width)
    y2_i = min(y2_i, height)

    cropped = frame_np[y1_i:y2_i, x1_i:x2_i].copy()
    return to_original(cropped)


def compute_cosine_similarity(
    vec1: Union[ArrayLike, Sequence[float]],
    vec2: Union[ArrayLike, Sequence[float]],
) -> float:
    """Compute the cosine similarity between two embedding vectors.

    Example:
        >>> compute_cosine_similarity([1, 0], [0, 1])
        0.0
        >>> compute_cosine_similarity([1, 1], [1, 1])
        1.0

    Args:
        vec1: First vector as a NumPy array, torch tensor, or flat sequence.
        vec2: Second vector with the same dimensionality as ``vec1``.

    Returns:
        Cosine similarity as a float in the range ``[-1, 1]``. Zero is returned
        when either vector has near-zero magnitude.
    """

    arr1 = _ensure_vector(vec1)
    arr2 = _ensure_vector(vec2)
    assert arr1.shape == arr2.shape, "Vectors must share the same shape."

    denom = float(np.linalg.norm(arr1) * np.linalg.norm(arr2))
    if denom < 1e-12:
        return 0.0

    similarity = float(np.dot(arr1, arr2) / denom)
    return max(min(similarity, 1.0), -1.0)


def draw_annotations(
    frame: ArrayLike,
    bbox: BoundingBoxType,
    gid: Union[int, str],
    color: ColorType,
    info: Optional[Mapping[str, Union[int, float, str]]] = None,
) -> ArrayLike:
    """Draw a bounding box and label annotations onto a frame.

    Example:
        >>> frame = np.zeros((50, 50, 3), dtype=np.uint8)
        >>> annotated = draw_annotations(
        ...     frame,
        ...     (5, 5, 25, 25),
        ...     1,
        ...     (255, 0, 0),
        ... )
        >>> annotated.shape
        (50, 50, 3)

    Args:
        frame: Image-like array of shape ``(H, W, C)`` as ``np.ndarray`` or
            ``torch.Tensor``.
        bbox: Bounding box ``(x1, y1, x2, y2)`` in pixel coordinates.
        gid: Identifier shown in the rendered label (e.g., track id).
        color: BGR tuple used for the box and text following OpenCV convention.
        info: Optional mapping of additional key-value pairs appended to the
            label string.

    Returns:
        Annotated frame with drawings rendered using OpenCV, preserving the
        input array type.
    """

    assert len(color) == 3, "Color must be a 3-tuple."
    frame_np, to_original = _ensure_frame(frame)

    if cv2 is None:
        raise ImportError(
            "OpenCV (cv2) is required for draw_annotations but is not installed."
        )

    height, width = frame_np.shape[0], frame_np.shape[1]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1, y1, x2, y2 = _clip_bbox(x1, y1, x2, y2, width, height)

    annotated = frame_np.copy()
    if x2 > x1 and y2 > y1:
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(annotated, pt1, pt2, color, thickness=2)

        label = str(gid)
        if info:
            formatted = " | ".join(f"{k}:{v}" for k, v in info.items())
            label = f"{label} | {formatted}"

        text_position = (pt1[0], max(pt1[1] - 5, 0))
        cv2.putText(
            annotated,
            label,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return to_original(annotated)


def track_feature(
    track_history: Sequence[Union[ArrayLike, Sequence[float]]]
) -> np.ndarray:
    """Average and L2-normalize a sequence of embedding vectors.

    Example:
        >>> embeddings = [np.ones(4), np.zeros(4)]
        >>> feature = track_feature(embeddings)
        >>> np.isclose(np.linalg.norm(feature), 1.0)
        True

    Args:
        track_history: Iterable of embeddings (NumPy arrays, torch tensors, or
            flat sequences). ``None`` values are ignored.

    Returns:
        L2-normalized mean embedding as a NumPy array of ``float32``.
    """

    assert isinstance(track_history, Sequence), (
        "Track history must be a sequence of embeddings."
    )

    valid_vectors = []
    for vec in track_history:
        if vec is None:
            continue
        arr = _ensure_vector(vec).astype(np.float32, copy=False)
        valid_vectors.append(arr)

    if not valid_vectors:
        return np.zeros(0, dtype=np.float32)

    stacked = np.stack(valid_vectors, axis=0)
    mean_vec = stacked.mean(axis=0)
    return _l2_normalize(mean_vec)


def get_default_palette() -> Tuple[ColorType, ...]:
    """Return the default color palette for consistent track coloring.

    Example:
        >>> palette = get_default_palette()
        >>> isinstance(palette[0], tuple)
        True
    """

    return DEFAULT_COLOR_PALETTE


def color_for_id(
    identifier: Union[int, str],
    palette: Optional[Sequence[ColorType]] = None,
) -> ColorType:
    """Map a track identifier to a reproducible color.

    Example:
        >>> color_for_id(1) in get_default_palette()
        True
    """

    palette = tuple(palette) if palette is not None else DEFAULT_COLOR_PALETTE
    assert palette, "Palette must contain at least one color."

    index = hash(identifier) % len(palette)
    color = palette[index]
    return tuple(int(c) for c in color)  # type: ignore[return-value]


def _ensure_frame(frame: ArrayLike) -> Tuple[np.ndarray, Callable[[np.ndarray], ArrayLike]]:
    """Validate frame inputs and create a converter to the original type."""

    if torch is not None and isinstance(frame, torch.Tensor):
        assert frame.ndim == 3, "Frame tensor must have shape (H, W, C)."

        frame_np = np.ascontiguousarray(frame.detach().cpu().numpy())
        dtype = frame.dtype
        device = frame.device

        def to_tensor(array: np.ndarray) -> "torch.Tensor":
            tensor = torch.from_numpy(np.ascontiguousarray(array))
            return tensor.to(device=device, dtype=dtype)

        return frame_np, to_tensor

    assert isinstance(frame, np.ndarray), (
        "Frame must be provided as a NumPy array or torch.Tensor."
    )
    assert frame.ndim == 3, "Frame array must have shape (H, W, C)."

    dtype = frame.dtype

    def to_numpy(array: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(array).astype(dtype, copy=False)

    return np.ascontiguousarray(frame), to_numpy


def _clip_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """Clip bounding box coordinates to the frame dimensions."""

    x1_clipped = float(np.clip(x1, 0.0, float(width)))
    y1_clipped = float(np.clip(y1, 0.0, float(height)))
    x2_clipped = float(np.clip(x2, 0.0, float(width)))
    y2_clipped = float(np.clip(y2, 0.0, float(height)))
    return x1_clipped, y1_clipped, x2_clipped, y2_clipped


def _ensure_vector(vector: Union[ArrayLike, Sequence[float]]) -> np.ndarray:
    """Convert input to a flattened NumPy vector with validation."""

    if torch is not None and isinstance(vector, torch.Tensor):
        arr = vector.detach().cpu().numpy()
    else:
        arr = np.asarray(vector)

    arr = np.ascontiguousarray(arr).astype(np.float32, copy=False)
    assert arr.ndim == 1, "Embedding vectors must be one-dimensional."
    return arr


def _l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return the L2-normalized version of ``vector``."""

    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32, copy=False)
