import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from lunar_rover.vision.preprocessing import (
    PreprocessingPipeline,
    denoise,
    enhance_contrast,
    extract_edges,
    extract_texture_descriptors,
    load_image,
    load_video_frame,
    normalize,
    resize_image,
)


def test_load_image(tmp_path):
    image_path = tmp_path / "test_image.png"
    image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

    cv2.imwrite(str(image_path), image)

    loaded_image = load_image(image_path)
    assert loaded_image.shape == image.shape


def test_load_video_frame(tmp_path):
    video_path = tmp_path / "test_video.avi"
    height, width = 64, 64
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(str(video_path), fourcc, 1, (width, height))

    for _ in range(5):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        video_writer.write(frame)

    video_writer.release()

    frame = load_video_frame(video_path, frame_index=2)
    assert frame.shape == (height, width, 3)


def test_normalize():
    image = np.array([[0, 128, 255], [64, 192, 32]], dtype=np.uint8)
    normalized = normalize(image, method="minmax")
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0


def test_denoise():
    image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    denoised = denoise(image, method="gaussian", kernel_size=5)
    assert denoised.shape == image.shape


def test_enhance_contrast():
    image = np.clip(np.random.normal(128, 20, (64, 64)).astype(np.uint8), 0, 255)
    enhanced = enhance_contrast(image, method="clahe")
    assert enhanced.shape == image.shape


def test_resize_image():
    image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
    resized = resize_image(image, (50, 100))
    assert resized.shape == (50, 100, 3)


def test_extract_edges():
    image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    edges = extract_edges(image, method="canny")
    assert edges.shape == image.shape


def test_extract_texture_descriptors():
    image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    descriptors = extract_texture_descriptors(image, method="lbp")
    assert "histogram" in descriptors


def test_preprocessing_pipeline(tmp_path):
    config = {
        "resize": {"size": (32, 32)},
        "normalize": {"method": "minmax"},
        "denoise": {"method": "gaussian", "kernel_size": 3},
        "enhance_contrast": {"method": "clahe"},
    }

    pipeline = PreprocessingPipeline(config)
    image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    processed_image = pipeline.process(image)

    assert processed_image.shape == (32, 32, 3)


def test_preprocessing_pipeline_with_features(tmp_path):
    config = {
        "resize": {"size": (32, 32)},
    }
    pipeline = PreprocessingPipeline(config)
    image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    processed_image, features = pipeline.process(
        image, extract_features=True, edge_method="canny", texture_method="lbp"
    )

    assert processed_image.shape == (32, 32, 3)
    assert "edges" in features
    assert "texture" in features


def test_pipeline_save_and_load(tmp_path):
    config = {
        "resize": {"size": (32, 32)},
        "normalize": {"method": "minmax"},
    }
    pipeline = PreprocessingPipeline(config)

    json_path = tmp_path / "config.json"
    pipeline.save_config(json_path)

    new_pipeline = PreprocessingPipeline(json_path)
    assert len(new_pipeline.steps) == len(pipeline.steps)
