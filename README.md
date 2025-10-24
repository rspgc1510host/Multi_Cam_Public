# Multi_Cam_Public

This repository contains modular utilities that support camera-based person
tracking and re-identification workflows. The helpers are designed to slot into
an inference loop, handling the most common data-preparation and visualization
steps that occur between model invocations.

## Helper utilities

The module [`utils/helpers.py`](utils/helpers.py) exports reusable functions
that can be imported by detection, embedding, or tracking scripts:

- `crop_person(frame, bbox)`: Clips a bounding box to frame bounds and returns a
  correctly typed crop for model ingestion.
- `compute_cosine_similarity(vec1, vec2)`: Produces a numerically stable cosine
  similarity score between two embeddings.
- `draw_annotations(frame, bbox, gid, color, info=None)`: Renders a bounding box
  and descriptive label onto a frame using OpenCV primitives.
- `track_feature(track_history)`: Aggregates recent embeddings, applying mean
  pooling and L2 normalisation for downstream matching.
- `color_for_id(...)` / `get_default_palette()`: Supplies deterministic colours
  for track identifiers to ensure consistent visualisations.

### Example integration

```python
import cv2
import numpy as np

from utils.helpers import (
    color_for_id,
    compute_cosine_similarity,
    crop_person,
    draw_annotations,
    track_feature,
)

# Simulated model outputs
frame = np.zeros((720, 1280, 3), dtype=np.uint8)
bbox = (100.5, 200.0, 260.2, 540.7)
embeddings = [np.random.rand(512).astype("float32") for _ in range(5)]

crop = crop_person(frame, bbox)
feature = track_feature(embeddings)
similarity = compute_cosine_similarity(feature, feature)
color = color_for_id(42)
annotated = draw_annotations(
    frame,
    bbox,
    gid=42,
    color=color,
    info={"cos": similarity},
)

cv2.imshow("Annotated", annotated)
cv2.waitKey(1)
```

> **Note:** `draw_annotations` requires the optional `opencv-python` package.
When PyTorch tensors are provided for frames or embeddings, the helpers preserve
PyTorch output types and devices for seamless integration with GPU pipelines.
