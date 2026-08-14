# SplatSplot 🎨

An automated, end-to-end pipeline to train, segment, clean, and evaluate 3D Gaussian Splats on the Nerfbusters dataset using SAM 2. 

This repository provides scripts to take raw image directories, train initial point clouds via OpenSplat, automatically track and mask out distractors using Segment Anything Model 2 (SAM 2), mathematically crop the 3D Gaussians, and trick the OpenSplat binary into calculating high-fidelity novel view metrics (PSNR, SSIM, LPIPS).

## 🚀 The Pipeline (Start-to-End)

The pipeline is broken down into 4 main stages:

### 1. Training Initial Splats (`run_all.sh`)
Iterates over all scene directories and invokes the `opensplat` C++ binary to train the raw, uncleaned 3D Gaussian Splats. The output is a dense `.ply` point cloud.

### 2. Distractor Masking with SAM 2 (`run_all_masks.sh` & `generate_masks_sam2.py`)
Uses Meta's SAM 2 to perform Video Object Segmentation (VOS). 
- We provide a curated dictionary of text prompts for each scene (e.g., "blue picnic blanket", "orange cat").
- SAM 2 automatically tracks these objects across the entire video sequence, generating binary masks for every single frame to identify the core object vs. the background/distractors.
- The `run_all_masks.sh` wrapper loops this over all scenes securely.

### 3. Cleaning the Point Cloud (`01_extract_objects.py`)
Reads the raw `.ply` file and mathematically carves out the background/distractors by projecting the 3D Gaussians into the 2D image plane of our camera views. If a Gaussian projects consistently into the masked background, its opacity is killed and it is pruned from the model. 
- Results in incredibly lightweight, extracted point clouds.

### 4. Evaluation and Metrics (`02_evaluate_pipeline.py`)
Automates the evaluation of the cleaned splats against the original ground-truth validation frames to compute PSNR, SSIM, and LPIPS metrics. Outputs a compiled Markdown, CSV, and JSON report.

---

## 🛑 Bottlenecks & Clever Fixes

Building this pipeline on Apple Silicon (macOS / MPS) presented several heavy computational bottlenecks, which we bypassed with a few clever tricks:

### 1. SAM 2 Memory Leakage on Apple Silicon
**The Problem:** Running SAM 2 video propagation over thousands of frames on PyTorch MPS leads to severe memory caching and leakage, eventually causing the system to throttle to CPU-only execution (turning a 30-minute task into a 7-hour task) or crash with Out-Of-Memory (OOM) errors.
**The Fix:** Instead of looping over all scenes inside a single Python script, we modularized `generate_masks_sam2.py` to accept a single scene argument. We then use `run_all_masks.sh` to spawn a completely fresh, isolated OS-level Python process for each scene. When a scene finishes, the process dies, and the OS forcefully reclaims all leaked memory, ensuring peak MPS performance for the entire 12-scene batch.

### 2. The Evaluation Metrics Trick
**The Problem:** To calculate PSNR, SSIM, and LPIPS, we needed to render our newly cleaned `.ply` models from novel camera views. Doing this natively in Python using the `gsplat` rasterizer was incredibly messy, error-prone, and required complex coordinate system alignments to match the Ground Truth frames perfectly.
**The Fix:** We completely abandoned the Python rasterizer. Instead, we "tricked" the `opensplat` C++ binary into doing the heavy lifting for us. `02_evaluate_pipeline.py` dynamically calls `opensplat` in "resume mode", feeding it our *cleaned* `.ply` file, and leverages the `--val-render` flag to spit out exactly one perfectly aligned rendered PNG of the validation view. We then load that PNG back into Python to easily calculate our metrics against the ground truth using `torchmetrics` and `pyiqa`.

---

## 📋 Evaluation Results
The final consolidated metrics for the entire dataset run can be found in `evaluation_report.md`.
