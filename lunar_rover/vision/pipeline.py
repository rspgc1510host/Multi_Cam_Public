"""Reusable computer vision pipelines.

The :class:`VisionPipeline` class orchestrates pre-processing, inference, and
post-processing stages for rover-mounted sensors. Each stage is composed of
callables, enabling quick experimentation with different detector models or
filtering steps while keeping business logic isolated from orchestration code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence

import numpy as np

Frame = np.ndarray
DetectionBatch = Sequence[np.ndarray]


@dataclass
class VisionPipeline:
    """Coordinate a sequence of image processing steps.

    Args:
        preprocessors: Callables executed before inference. Each callable
            receives and returns an ``np.ndarray`` representing the current
            frame state.
        inference: Callable that produces detections from the transformed
            frame.
        postprocessors: Callables applied to the inference output.
    """

    preprocessors: Sequence[Callable[[Frame], Frame]] = field(default_factory=tuple)
    inference: Callable[[Frame], DetectionBatch] | None = None
    postprocessors: Sequence[Callable[[DetectionBatch], DetectionBatch]] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        self._validate_steps(self.preprocessors, "preprocessor")
        self._validate_steps(self.postprocessors, "postprocessor")
        if self.inference is None:
            raise ValueError("VisionPipeline requires an inference callable.")

    @staticmethod
    def _validate_steps(
        steps: Sequence[Callable[..., object]],
        label: str,
    ) -> None:
        for index, step in enumerate(steps):
            if not callable(step):
                raise TypeError(
                    f"{label.title()} step at position {index} must be callable, "
                    f"received {type(step)!r}."
                )

    def run(self, frame: Frame) -> DetectionBatch:
        """Execute the pipeline on a single frame.

        The input frame is forwarded through all preprocessors, then through the
        inference callable, and finally through the post-processing steps. The
        final output is returned verbatim to allow downstream consumers to
        choose their own data structures.
        """

        transformed = frame
        for preprocessor in self.preprocessors:
            transformed = preprocessor(transformed)

        detections = self.inference(transformed)
        for postprocessor in self.postprocessors:
            detections = postprocessor(detections)

        return detections

    def add_preprocessor(self, *steps: Callable[[Frame], Frame]) -> None:
        """Append one or more preprocessing steps to the pipeline."""

        self.preprocessors = tuple((*self.preprocessors, *steps))
        self._validate_steps(self.preprocessors, "preprocessor")

    def add_postprocessor(
        self, *steps: Callable[[DetectionBatch], DetectionBatch]
    ) -> None:
        """Append one or more post-processing steps to the pipeline."""

        self.postprocessors = tuple((*self.postprocessors, *steps))
        self._validate_steps(self.postprocessors, "postprocessor")
