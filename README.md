# Honey Bee Detection System

A computer vision pipeline for detecting honey bees in image and video streams using Ultralytics YOLO11.

## Project Overview

This project implements an end-to-end object detection pipeline targeting honey bee localization. It covers dataset preparation, verification of bounding box annotations, model training, performance evaluation, and video inference.

## Dataset Details

The dataset configuration (`data.yaml`) is structured for YOLO object detection:
- **Classes:** 1 (`honey_bee`, Class ID: 0)
- **Total Images:** 7,200
- **Total Annotations:** 54,815 bounding boxes
- **Dataset Split:**
  - Training: 4,880 images (`images/train`)
  - Validation: 800 images (`images/val`)
  - Testing: 1,520 images (`images/test`)

## Repository Structure

```text
honey-bee-detection/
├── dataset/                  # Formatted YOLO dataset (train, val, test splits)
├── dataset_raw/              # Raw source images and annotations
├── scripts/
│   ├── download_and_inspect.py      # Dataset acquisition and preliminary inspection
│   ├── build_dataset.py             # Dataset preprocessing and directory structuring
│   ├── verify_annotations.py        # Bounding box coordinate verification
│   ├── train_sanity.py              # Small-scale training sanity run
│   ├── train_full.py                # Full YOLO11 training pipeline
│   ├── test_eval.py                 # Evaluation metrics on the test partition
│   ├── video_inference.py           # Real-time / video file inference runner
│   └── generate_illustrative_plots.py # Plot generation for evaluation results
├── test_images/              # Sample images for testing
├── test_videos/              # Sample video feeds for detection
├── weights/                  # Model weights (e.g., pretrained YOLO11n weights)
├── outputs/                  # Prediction outputs and rendered videos
└── runs/                     # Ultralytics training and validation run artifacts
```

## Setup & Usage

### 1. Requirements
- Python 3.9+
- PyTorch (with CUDA support if running on GPU)
- OpenCV (`opencv-python`)
- Ultralytics (`ultralytics`)
- NumPy, Pandas, Matplotlib

### 2. Verify Annotations
```bash
python scripts/verify_annotations.py
```

### 3. Training
To run a sanity check training session:
```bash
python scripts/train_sanity.py
```
To run full training:
```bash
python scripts/train_full.py
```

### 4. Evaluation & Inference
To evaluate the model on the test split:
```bash
python scripts/test_eval.py
```
To run inference on video footage:
```bash
python scripts/video_inference.py
```
