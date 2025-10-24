# Multi-Camera Tracking System

A real-time multi-camera tracking system that combines **YOLOv8** for object detection, **DeepSORT** for tracking, and **OSNet** for person re-identification across multiple camera views.

## Overview

This project provides an end-to-end solution for tracking individuals across multiple camera feeds using state-of-the-art deep learning models. The system is designed to:

- **Detect** people in video streams using YOLOv8
- **Track** individuals within each camera view using DeepSORT
- **Re-identify** and associate tracked individuals across different camera angles using OSNet-based feature extraction

## Features (In Development)

- 🎥 Multi-camera video processing and synchronization
- 🔍 YOLOv8-based person detection
- 🎯 DeepSORT tracking with Kalman filtering
- 🧠 OSNet deep re-identification for cross-camera matching
- 💾 Output tracking results with visualization
- ⚡ GPU acceleration support (CUDA)

## System Requirements

### Hardware Requirements

- **GPU**: NVIDIA RTX 3050 or higher (4GB+ VRAM recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for models and outputs

### Software Requirements

- **Operating System**: Windows 10/11 (with WSL support) or Linux
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **CUDA Toolkit**: 11.7 or higher (for GPU acceleration)
- **cuDNN**: Compatible with your CUDA version

## Installation

### Step 1: Verify GPU and CUDA

Ensure your NVIDIA GPU drivers are up to date and CUDA is properly installed:

```bash
nvidia-smi
```

This should display your GPU information and CUDA version.

### Step 2: Clone the Repository

```bash
git clone https://github.com/yourusername/Multi_Cam_Public.git
cd Multi_Cam_Public
```

### Step 3: Create Virtual Environment

#### Using venv (recommended for Windows)

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

#### Using conda

```bash
conda create -n multi_cam_tracking python=3.10
conda activate multi_cam_tracking
```

### Step 4: Install Dependencies

#### For CUDA-enabled GPU (Recommended)

First, install PyTorch with CUDA support:

```bash
# For CUDA 11.8 (adjust based on your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

#### For CPU-only (Not Recommended for Production)

```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output should show PyTorch version and `CUDA Available: True`.

## Project Structure

```
Multi_Cam_Public/
├── multi_cam_tracking/          # Main package directory
│   ├── __init__.py
│   ├── data/                    # Input data and datasets
│   │   └── .gitkeep
│   ├── models/                  # Pre-trained model weights
│   │   └── .gitkeep
│   ├── scripts/                 # Executable scripts
│   │   └── __init__.py
│   ├── utils/                   # Utility functions and helpers
│   │   └── __init__.py
│   └── outputs/                 # Results, logs, and visualizations
│       └── .gitkeep
├── requirements.txt             # Project dependencies
├── README.md                    # This file
└── LICENSE                      # MIT License
```

## Usage (Coming Soon)

### Placeholder Commands

The full implementation is in progress. Once complete, you'll be able to run:

```bash
# Single camera tracking
python -m multi_cam_tracking.scripts.single_cam_track --video path/to/video.mp4

# Multi-camera tracking
python -m multi_cam_tracking.scripts.multi_cam_track --config config.yaml

# Training custom re-identification model
python -m multi_cam_tracking.scripts.train_reid --dataset path/to/dataset
```

## Core Components

### 1. YOLOv8 Detection

YOLOv8 (You Only Look Once v8) is used for real-time person detection in video frames. It provides:
- High accuracy and speed
- Pre-trained weights on COCO dataset
- Easy integration with ultralytics library

### 2. DeepSORT Tracking

DeepSORT (Deep Simple Online and Realtime Tracking) handles:
- Kalman filtering for motion prediction
- Hungarian algorithm for data association
- Deep appearance features for robust tracking

### 3. OSNet Re-identification

OSNet (Omni-Scale Network) enables cross-camera person re-identification by:
- Extracting discriminative appearance features
- Learning robust representations at multiple scales
- Matching identities across different camera views

## Configuration

Configuration files will be located in the `configs/` directory (to be created) and will support:
- Camera setup parameters
- Model selection and hyperparameters
- Tracking thresholds and matching criteria
- Output format preferences

## Development Roadmap

- [x] Project structure initialization
- [ ] YOLOv8 detection module
- [ ] DeepSORT tracking integration
- [ ] OSNet re-identification model
- [ ] Multi-camera synchronization
- [ ] Cross-camera matching algorithm
- [ ] Visualization and output utilities
- [ ] Configuration system
- [ ] Documentation and examples
- [ ] Performance optimization
- [ ] Unit tests and CI/CD

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors:
- Reduce batch size in configuration
- Process videos at lower resolution
- Use a GPU with more VRAM

### Import Errors

Ensure your virtual environment is activated:
```bash
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### Slow Performance on CPU

This system is designed for GPU acceleration. CPU-only execution will be significantly slower. Consider:
- Using a machine with NVIDIA GPU
- Using cloud GPU services (AWS, Google Cloud, etc.)
- Reducing input resolution

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [DeepSORT](https://github.com/nwojke/deep_sort) for tracking algorithms
- [Torchreid](https://github.com/KaiyangZhou/deep-person-reid) for re-identification models
- The open-source computer vision community

## Contact

For questions or issues, please open an issue on GitHub.

---

**Status**: 🚧 Project skeleton initialized - Implementation in progress
