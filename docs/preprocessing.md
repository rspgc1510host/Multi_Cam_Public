# Lunar Vision Preprocessing Module

## Overview

The `lunar_rover.vision.preprocessing` module provides a comprehensive preprocessing pipeline for both real and synthetic lunar imagery before detection tasks. It supports various image sources including local files, video frames, and arrays, with configurable preprocessing steps and feature extraction utilities.

## Installation

The preprocessing module requires OpenCV for most operations:

```bash
pip install opencv-python numpy
```

For YAML configuration support:

```bash
pip install pyyaml
```

## Basic Usage

### Loading Images

Load images from local files:

```python
from lunar_rover.vision.preprocessing import load_image

# Load color image
image = load_image("lunar_surface.png")

# Load grayscale image
grayscale_image = load_image("lunar_surface.png", grayscale=True)
```

### Loading Video Frames

Extract specific frames from video files:

```python
from lunar_rover.vision.preprocessing import load_video_frame

# Load frame 100 from video
frame = load_video_frame("lunar_traverse.mp4", frame_index=100)

# Load as grayscale
grayscale_frame = load_video_frame("lunar_traverse.mp4", frame_index=100, grayscale=True)
```

### Image Normalization

Normalize image intensity values using different methods:

```python
from lunar_rover.vision.preprocessing import normalize
import numpy as np

image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

# Min-max normalization (default)
normalized = normalize(image, method="minmax", target_range=(0.0, 1.0))

# Z-score normalization
zscore_normalized = normalize(image, method="zscore")

# Clipping
clipped = normalize(image, method="clip", target_range=(50.0, 200.0))
```

### Denoising

Remove noise from images using various filters:

```python
from lunar_rover.vision.preprocessing import denoise

# Gaussian blur
denoised = denoise(image, method="gaussian", kernel_size=5, sigma=1.0)

# Bilateral filter (edge-preserving)
bilateral = denoise(image, method="bilateral", kernel_size=5, sigma=75.0)

# Non-local means denoising
nlmeans = denoise(image, method="nlmeans", h=10.0)
```

### Contrast Enhancement

Enhance image contrast for better feature visibility:

```python
from lunar_rover.vision.preprocessing import enhance_contrast

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
enhanced = enhance_contrast(image, method="clahe", clip_limit=2.0, tile_grid_size=(8, 8))

# Histogram equalization
hist_eq = enhance_contrast(image, method="histogram_eq")

# Linear contrast adjustment
linear = enhance_contrast(image, method="linear", alpha=1.5, beta=10.0)
```

### Resizing

Resize images to target dimensions:

```python
from lunar_rover.vision.preprocessing import resize_image

# Resize to 480x640
resized = resize_image(image, size=(480, 640))

# With different interpolation methods
resized_cubic = resize_image(image, size=(480, 640), interpolation="cubic")
resized_lanczos = resize_image(image, size=(480, 640), interpolation="lanczos")
```

## Feature Extraction

### Edge Detection

Extract edges for obstacle and crater detection:

```python
from lunar_rover.vision.preprocessing import extract_edges

# Canny edge detection
edges = extract_edges(image, method="canny", low_threshold=50, high_threshold=150)

# Sobel edge detection
sobel_edges = extract_edges(image, method="sobel", sobel_ksize=3)

# Laplacian edge detection
laplacian_edges = extract_edges(image, method="laplacian")
```

### Texture Descriptors

Extract texture features for terrain classification:

```python
from lunar_rover.vision.preprocessing import extract_texture_descriptors

# Local Binary Patterns (LBP)
lbp_features = extract_texture_descriptors(image, method="lbp", num_points=8, radius=1)
print(lbp_features["histogram"])  # Normalized histogram
print(lbp_features["lbp_image"])  # LBP feature map

# Gray-Level Co-occurrence Matrix (GLCM)
glcm_features = extract_texture_descriptors(image, method="glcm")
print(glcm_features["contrast"])    # Contrast measure
print(glcm_features["homogeneity"])  # Homogeneity measure
print(glcm_features["energy"])      # Energy measure

# Statistical features
stats = extract_texture_descriptors(image, method="stats")
print(stats["mean"], stats["std"], stats["entropy"])
```

## Preprocessing Pipeline

The `PreprocessingPipeline` class allows you to chain multiple preprocessing steps with configuration management.

### Creating a Pipeline

```python
from lunar_rover.vision.preprocessing import PreprocessingPipeline

# Define configuration
config = {
    "resize": {"size": (480, 640)},
    "normalize": {"method": "minmax"},
    "denoise": {"method": "gaussian", "kernel_size": 5},
    "enhance_contrast": {"method": "clahe", "clip_limit": 2.0},
}

# Create pipeline
pipeline = PreprocessingPipeline(config)

# Process image
processed = pipeline.process(image)
```

### Processing with Feature Extraction

```python
# Process and extract features in one step
processed, features = pipeline.process(
    image,
    extract_features=True,
    edge_method="canny",
    texture_method="lbp"
)

# Access features
edges = features["edges"]
texture_histogram = features["texture"]["histogram"]
```

### Configuration via Files

Save and load pipeline configurations:

```python
# Save configuration to JSON
pipeline.save_config("config.json")

# Save configuration to YAML (requires pyyaml)
pipeline.save_config("config.yaml")

# Load from file
pipeline = PreprocessingPipeline("config.json")
```

Example JSON configuration:

```json
{
  "resize": {
    "size": [480, 640],
    "interpolation": "linear"
  },
  "normalize": {
    "method": "minmax",
    "target_range": [0.0, 1.0]
  },
  "denoise": {
    "method": "gaussian",
    "kernel_size": 5,
    "sigma": 1.0
  },
  "enhance_contrast": {
    "method": "clahe",
    "clip_limit": 2.0,
    "tile_grid_size": [8, 8]
  }
}
```

Example YAML configuration:

```yaml
resize:
  size: [480, 640]
  interpolation: linear

normalize:
  method: minmax
  target_range: [0.0, 1.0]

denoise:
  method: gaussian
  kernel_size: 5
  sigma: 1.0

enhance_contrast:
  method: clahe
  clip_limit: 2.0
  tile_grid_size: [8, 8]
```

### Adding Custom Steps

Extend the pipeline with custom preprocessing functions:

```python
def custom_filter(image, threshold=128):
    """Custom binary threshold filter."""
    return (image > threshold).astype(np.uint8) * 255

pipeline = PreprocessingPipeline()
pipeline.add_step("custom_filter", custom_filter, {"threshold": 100})

processed = pipeline.process(image)
```

## Complete Example

Here's a complete example preprocessing lunar rover images:

```python
import numpy as np
from lunar_rover.vision.preprocessing import (
    PreprocessingPipeline,
    load_image,
    extract_edges,
)

# Load lunar surface image
image = load_image("lunar_surface.jpg")

# Create preprocessing pipeline
config = {
    "resize": {"size": (512, 512)},
    "denoise": {"method": "bilateral", "kernel_size": 5},
    "enhance_contrast": {"method": "clahe", "clip_limit": 3.0},
    "normalize": {"method": "minmax"},
}

pipeline = PreprocessingPipeline(config)

# Process image with feature extraction
processed_image, features = pipeline.process(
    image,
    extract_features=True,
    edge_method="canny",
    texture_method="lbp"
)

# Use extracted features for downstream tasks
edges = features["edges"]
texture_histogram = features["texture"]["histogram"]

print(f"Processed image shape: {processed_image.shape}")
print(f"Edge map shape: {edges.shape}")
print(f"Texture histogram length: {len(texture_histogram)}")

# Save pipeline configuration for reuse
pipeline.save_config("lunar_preprocessing_config.json")
```

## Extending the Pipeline

The preprocessing module is designed to be extensible. You can:

1. **Add custom preprocessing functions**: Create functions that accept an image and parameters, then add them to the pipeline using `add_step()`.

2. **Implement custom feature extractors**: Write feature extraction functions following the interface of `extract_edges()` and `extract_texture_descriptors()`.

3. **Create domain-specific configurations**: Save specialized configurations for different scenarios (e.g., crater detection, obstacle detection, navigation).

## API Reference

### Image Loading Functions

- `load_image(image_path, grayscale=False)` - Load image from file
- `load_video_frame(video_path, frame_index=0, grayscale=False)` - Extract video frame

### Preprocessing Functions

- `normalize(image, method="minmax", target_range=(0.0, 1.0))` - Normalize intensities
- `denoise(image, method="gaussian", ...)` - Apply denoising filters
- `enhance_contrast(image, method="clahe", ...)` - Enhance contrast
- `resize_image(image, size, interpolation="linear")` - Resize image

### Feature Extraction Functions

- `extract_edges(image, method="canny", ...)` - Extract edge features
- `extract_texture_descriptors(image, method="lbp", ...)` - Extract texture features

### Pipeline Class

- `PreprocessingPipeline(config=None)` - Create configurable pipeline
  - `process(image, extract_features=False, ...)` - Process image
  - `add_step(step_name, func, params)` - Add custom step
  - `save_config(output_path)` - Save configuration to file

## Performance Considerations

- **Batch Processing**: For processing multiple images, reuse the same pipeline instance to avoid overhead.
- **Memory Management**: Large images consume significant memory. Consider downsampling first using `resize_image()`.
- **GPU Acceleration**: OpenCV may use GPU acceleration when available. For heavy workloads, consider using GPU-optimized libraries.

## Best Practices

1. **Standardize Input**: Always resize images to a consistent resolution for model training.
2. **Preserve Information**: Use edge-preserving filters like bilateral filtering when denoising.
3. **Document Configurations**: Save pipeline configurations with descriptive names for reproducibility.
4. **Validate Outputs**: Check processed image statistics (mean, std) to ensure preprocessing doesn't introduce artifacts.
5. **Profile Performance**: Time each preprocessing step to identify bottlenecks in your pipeline.
