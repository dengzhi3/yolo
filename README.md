# YOLO-PMB: Position-Aware Three-Stage Collaborative Detection for Low-Light Scenes

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2](https://img.shields.io/badge/pytorch-2.2-red.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository contains the official implementation of **YOLO-PMB**, a lightweight object detection model designed for low-light environments. Our method introduces a **position-aware three-stage collaborative design strategy** that selectively deploys three complementary modules (PSConv, C2BLRA, and MAN-Faster) at specific network stages to counteract distinct physical degradation mechanisms—edge attenuation, occlusion aliasing, and artifact interference—encountered in low-light images.

> **Paper**: *SYOLO-PMB: Object detection model in low-light scenes based on improved YOLO11* (under review)

---

## 🚀 Key Features

- **Position-Aware Deployment**: Modules are deployed only where they are most effective—PSConv in shallow layers (P1–P3), C2BLRA at the top layer and detection output (P5 + Head), and MAN-Faster in the mesoscale neck (P3/P4).
- **High Accuracy with Low Cost**: Achieves **69.4% mAP50** on ExDark (+1.8% over YOLO11n) with only **2.62M parameters** and **6.5 GFLOPs**.
- **Generalization**: Also validated on DarkFace (46.5% mAP50) for face detection under extreme low light.
- **Real-Time Inference**: Runs at **85+ FPS** on an RTX 4060 GPU, exceeding the 30 FPS real-time threshold.

---

## 📦 Repository Structure
.
├── train.py # Training script
├── val.py # Validation script
├── test_env.py # Environment check
├── test_yaml.py # YAML config tester
├── track.py # Tracking utility
├── extra_modules/ # Core module implementations
│ ├── psconv.py # Pinwheel-shaped Convolution
│ ├── c2blra.py # C2BLRA (Lightweight Bi-Level Routing Attention)
│ └── man_faster.py # MAN-Faster (Multi-scale fusion + channel recalibration)
├── yolo11-PSConv+MAN-Faster+C2BRA.yaml # Full YOLO-PMB configuration
├── yolo11-PSConv+MAN-Faster.yaml # Ablation: PSConv + MAN-Faster
├── yolo11-PSConv+c3k2_ETB.yaml # Ablation variant
├── yolo11-PSConv+c3k2_FDT.yaml # Ablation variant
├── yolo11n.pt # YOLO11n baseline weights (download separately)
├── CITATION.cff # Citation metadata
└── README.md # This file


---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/dengzhi3/yolo.git
cd yolo
```

### 2. Create a virtual environment (optional but recommended)
conda create -n yolopmb python=3.8
conda activate yolopmb

### 3. Install dependencies
# Install PyTorch with CUDA support (adjust version according to your system)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# Install Ultralytics and other dependencies
pip install ultralytics==8.3.0
pip install opencv-python numpy tqdm matplotlib pandas seaborn
