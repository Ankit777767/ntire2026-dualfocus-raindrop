# NTIRE 2026 Dual-Focus Raindrop Removal (Day & Night)

This repository contains our solution for the **NTIRE 2026 Challenge on Day and Night Raindrop Removal for Dual-Focused Images**.  
The goal is to restore a clean image from raindrop-degraded inputs under varying focus (drop-focused / background-focused) and illumination conditions (day / night).

Our approach uses a **dual-focus aware transformer-based restoration network**, trained with metric-aligned losses and designed to generalize to **single-image mixed-focus inference**, as required by the Codabench evaluation.

---

## 📌 Key Features

- Dual-focus training using **drop-focused + background-focused image pairs**
- Robust single-image inference for Codabench validation
- Transformer-based backbone (Restormer-style)
- Metric-aligned optimization:
  - PSNR (Y)
  - SSIM (Y)
  - LPIPS
- Fully cross-platform (Windows / Linux / Colab)
- Clean, reproducible research codebase

---

## 📂 Dataset Structure

### Training / Validation Dataset

data/
├── train/
│ ├── daytime/
│ │ ├── drop/00001/.png
│ │ ├── blur/00001/.png
│ │ └── clear/00001/*.png
│ └── nighttime/
│ └── same structure
│
├── val/
│ └── same structure as train
│
└── codabench/
├── 0001.png
├── 0002.png
└── ...


- `drop`  : raindrop-focused images  
- `blur`  : background-focused images  
- `clear` : ground-truth clean images  

---

## 🧠 Method Overview

- **Training**: Dual-focus image pairs are concatenated channel-wise and processed jointly.
- **Inference (Codabench)**: A single image is duplicated to simulate dual-focus input.
- **Losses** are aligned with the NTIRE evaluation metric to improve ranking stability.

---

## ⚙️ Installation

```bash
pip install torch torchvision lpips
