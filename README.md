# CAP5516 – Multi-task Medical Image Analysis
### with Diffusion-Based Synthetic Augmentation

> **Course:** CAP5516 – Medical Image Computing, Spring 2025
> **Institution:** University of Central Florida (UCF) – CRCV
> **Dataset:** Breast Ultrasound Images (BUSI) — 798 images, 3 classes

---

## Overview

This project builds an **end-to-end Medical Image Analysis Pipeline** for breast ultrasound lesion **classification** and **segmentation**. It tackles the core challenge of limited annotated medical data by using a **Latent Diffusion Model (fine-tuned Stable Diffusion v1.5)** to synthesise realistic ultrasound images, augmenting the training set and improving model performance.

### Pipeline at a Glance

```
BUSI Dataset (798 images + masks)
        │
        ▼
┌─────────────────────────────┐
│  Module 1: Preprocessing    │  Resize · Normalise · Stratified Split (70/15/15)
└──────────┬──────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────────────────────┐
│Module 2 │  │ Module 3: Diffusion      │
│Baseline │  │ Fine-tune SD v1.5 on     │
│U-Net +  │  │ BUSI → Generate 450      │
│ResNet-50│  │ synthetic images (FID)   │
└────┬────┘  └──────────┬───────────────┘
     │                  │
     └────────┬──────────┘
              ▼
┌─────────────────────────────┐
│  Module 4: Ablation Study   │  0% / 25% / 50% / 75% / 100% synthetic ratio
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  Module 5: Evaluation       │  Dice · IoU · AUC · Grad-CAM · PDF Report
└─────────────────────────────┘
```

---

## Results

### Baseline Performance (Real Data Only)

| Model | Metric | Score | Target |
|---|---|---|---|
| **U-Net** | Val Dice | **0.7841** | > 0.75 ✅ |
| **U-Net** | Test Dice | **0.7186** | > 0.75 ✅ |
| **ResNet-50** | Val Accuracy | **91.67%** | > 80% ✅ |
| **ResNet-50** | Test Accuracy | **90.00%** | > 80% ✅ |
| **ResNet-50** | Test AUC (OvR) | **0.9722** | > 0.85 ✅ |
| **ResNet-50** | Malignant F1 | **0.88** | > 0.80 ✅ |

### Per-Class Classification Report (Test Set – 120 images)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 0.82 | 0.90 | 0.86 | 20 |
| Benign | 0.91 | 0.93 | 0.92 | 69 |
| Malignant | 0.93 | 0.84 | 0.88 | 31 |
| **Overall** | **0.90** | **0.90** | **0.90** | **120** |

> Ablation study results (synthetic data ratios 0–100%) to be added after Stage 4.

---

## Dataset

| Property | Details |
|---|---|
| **Name** | Breast Ultrasound Images (BUSI) |
| **Source** | [Kaggle – aryashah2k](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) |
| **Total Images** | 798 PNG images + 798 segmentation masks |
| **Classes** | Normal (133) · Benign (454) · Malignant (211) |
| **Split** | Train 70% · Val 15% · Test 15% (stratified) |
| **License** | CC0-1.0 |

```
data/BUSI/
├── benign/       (454 image + mask pairs)
├── malignant/    (211 image + mask pairs)
└── normal/       (133 image + mask pairs)
```

---

## Project Structure

```
CAP5516-Medical-Imaging/
│
├── src/
│   ├── dataset.py              # BUSI data loader, stratified splits, EDA plots
│   ├── unet.py                 # U-Net segmentation model
│   ├── train_segmentation.py   # Dice loss/score, U-Net training loop
│   ├── classifier.py           # ResNet-50 classifier, AUC evaluation
│   ├── diffusion_augment.py    # SD v1.5 fine-tuning + synthetic generation
│   ├── augmented_training.py   # Ablation study across synthetic data ratios
│   ├── gradcam.py              # Grad-CAM visualisations
│   └── report_generator.py     # Clinical-style PDF report (ReportLab)
│
├── train_baseline.py           # Train U-Net + ResNet-50 baselines
├── train_diffusion.py          # Fine-tune SD + generate synthetic images
├── main.py                     # End-to-end pipeline entry point (--stage flag)
│
├── data/
│   ├── BUSI/                   # Real dataset (download separately)
│   └── synthetic/              # Generated images (auto-created by Stage 3)
│
├── models/                     # Saved model checkpoints (local only)
├── results/                    # Training curves, Grad-CAM, PDF report
└── notebooks/                  # Jupyter notebooks (coming soon)
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/waqarbutt58/CAP5516-Medical-Imaging.git
cd CAP5516-Medical-Imaging
```

### 2. Create virtual environment
```bash
python -m venv cap5516_env

# Windows
cap5516_env\Scripts\activate

# Linux / macOS
source cap5516_env/bin/activate
```

### 3. Install PyTorch with CUDA
```bash
# CUDA 12.4 (for RTX 30xx / 40xx GPUs)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch torchvision
```

### 4. Install remaining dependencies
```bash
pip install scikit-learn matplotlib seaborn pandas tqdm opencv-python reportlab
pip install diffusers transformers==4.44.2 accelerate safetensors
pip install pytorch-fid kaggle
```

### 5. Download BUSI Dataset
```bash
# Set your Kaggle API token
export KAGGLE_KEY=<your_kaggle_api_token>   # Linux/macOS
$env:KAGGLE_KEY="<your_kaggle_api_token>"   # Windows PowerShell

kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset
unzip breast-ultrasound-images-dataset.zip -d data/BUSI/
```

---

## Running the Pipeline

### Stage 1 – EDA & Preprocessing
```bash
python main.py --stage eda
```
Outputs: `results/eda.png` — class distribution + sample images per class.

### Stage 2 – Baseline Models
```bash
python train_baseline.py --epochs 50 --batch 16 --lr 0.0001
```
Trains U-Net segmentation and ResNet-50 classifier. Saves best checkpoints to `models/`.

### Stage 3 – Diffusion Augmentation
```bash
python train_diffusion.py --epochs 20 --batch 4 --n_per_class 150 --fid
```
Fine-tunes Stable Diffusion v1.5 on BUSI, generates 150 synthetic images per class (450 total), computes FID score.

### Stage 4 – Ablation Study
```bash
python main.py --stage ablation
```
Retrains classifier at 5 synthetic data ratios: 0%, 25%, 50%, 75%, 100%.  
Outputs: `results/ablation.png`

### Stage 5 – Evaluation & Report
```bash
python main.py --stage report
```
Generates Grad-CAM visualisations and a clinical-style PDF report.

### Full Pipeline
```bash
python main.py --stage all
```

---

## Methods

### U-Net Segmentation
- Encoder-decoder architecture with skip connections (4 levels, features: 64→128→256→512)
- **Loss:** BCE + Dice (combined)
- **Optimizer:** Adam (lr=1e-4) with ReduceLROnPlateau
- Input: 256×256 RGB-converted grayscale ultrasound images

### ResNet-50 Classification
- ImageNet pretrained backbone, custom head: `Dropout(0.5) → Linear(2048→3)`
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam + CosineAnnealingLR

### Latent Diffusion Model (Stable Diffusion v1.5)
- Fine-tuned U-Net denoiser on BUSI with class-conditioned text prompts
- VAE and text encoder frozen — only U-Net weights updated
- Mixed-precision training (AMP) for memory efficiency (8GB VRAM)
- **Text prompts per class:**
  - Normal: *"ultrasound image of a normal breast, no lesion, grayscale"*
  - Benign: *"ultrasound image of a benign breast lesion, well-defined edges, grayscale"*
  - Malignant: *"ultrasound image of a malignant breast tumour, irregular edges, grayscale"*

### Ablation Study Design
Synthetic images added at 5 ratios on top of the real training set:

| Ratio | Real Images | Synthetic Added | Total |
|---|---|---|---|
| 0% (baseline) | 558 | 0 | 558 |
| 25% | 558 | ~140 | ~698 |
| 50% | 558 | ~279 | ~837 |
| 75% | 558 | ~419 | ~977 |
| 100% | 558 | 558 | ~1116 |

---

## Evaluation Metrics

| Task | Metric | Formula |
|---|---|---|
| Segmentation | **Dice Score** | 2·TP / (2·TP + FP + FN) |
| Segmentation | **IoU (Jaccard)** | TP / (TP + FP + FN) |
| Classification | **AUC (OvR)** | Area under ROC curve, one-vs-rest |
| Classification | **Accuracy / F1** | Per-class precision, recall, F1 |
| Synthesis quality | **FID Score** | Fréchet Inception Distance (↓ better) |

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning Framework | PyTorch 2.6 + CUDA 12.4 |
| Segmentation Model | U-Net (custom implementation) |
| Classification Model | ResNet-50 (torchvision pretrained) |
| Diffusion Model | Stable Diffusion v1.5 (diffusers 0.38) |
| Explainability | Grad-CAM |
| PDF Reporting | ReportLab |
| Hardware | NVIDIA RTX 4060 Laptop (8GB VRAM) |
| Language | Python 3.13 |

---

## Training Results Summary

| Stage | Status | Key Metric |
|---|---|---|
| ✅ EDA & Preprocessing | Complete | 798 images loaded, splits verified |
| ✅ U-Net Baseline | Complete | Test Dice = 0.7186 |
| ✅ ResNet-50 Baseline | Complete | Test AUC = 0.9722, Acc = 90% |
| 🔄 Diffusion Fine-tuning | In Progress | SD v1.5 fine-tuning on BUSI |
| ⏳ Ablation Study | Pending | — |
| ⏳ Grad-CAM + Report | Pending | — |

---

## References

1. Al-Dhabyani W. et al., *"Dataset of breast ultrasound images"*, Data in Brief, 2020
2. Ronneberger O. et al., *"U-Net: Convolutional Networks for Biomedical Image Segmentation"*, MICCAI 2015
3. He K. et al., *"Deep Residual Learning for Image Recognition"*, CVPR 2016
4. Rombach R. et al., *"High-Resolution Image Synthesis with Latent Diffusion Models"*, CVPR 2022
5. Ho J. et al., *"Denoising Diffusion Probabilistic Models"*, NeurIPS 2020
6. Selvaraju R. et al., *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"*, ICCV 2017

---

*CAP5516 – Medical Image Computing | Spring 2025 | University of Central Florida – CRCV*
