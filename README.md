# 🔬 Polyp Segmentation using Multi-Scale U-Net on Kvasir-SEG

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-FFD21E?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

### Automated pixel-level polyp detection in colonoscopy images

**Best Val Dice: 0.7941 · Best Val IoU: 0.6623 · Trained for 60 Epochs**

[🚀 **Live Demo**](https://huggingface.co/spaces/Writick/polyp-segmentation) · [📄 Project Report](#-project-report) · [🏗️ Architecture](#️-architecture) · [📊 Results](#-results) · [🛠️ Quick Start](#️-quick-start)

---

> **Course:** PCS 220 — Multimedia Processing Lab  
> **Institution:** Thapar Institute of Engineering & Technology, Patiala  
> **Supervisor:** Dr. Abhishek Kesarwani (Assistant Professor)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Dataset](#-dataset)
- [Architecture](#️-architecture)
- [Training Strategy](#-training-strategy)
- [Results](#-results)
- [Quick Start](#️-quick-start)
- [Project Structure](#-project-structure)
- [Deployment](#-deployment)
- [Team](#-team)
- [References](#-references)

---

## 🌟 Overview

Colorectal cancer ranks among the **top 3 most diagnosed cancers** globally. Early detection of precancerous polyps during colonoscopy can reduce mortality by up to **90%**, yet clinicians miss **6–27%** of polyps due to fatigue and morphological variability.

This project delivers a complete, end-to-end deep learning pipeline that segments polyps at the **pixel level** — identifying exactly which pixels in a colonoscopy frame belong to a polyp. It was developed as part of the Multimedia Processing Lab course at TIET and is **fully deployed** as a live web application on HuggingFace Spaces.

### What makes this different from a basic U-Net

| Component | Baseline U-Net | This Project |
|:---|:---|:---|
| **Architecture** | Plain encoder-decoder | **ResUNet++ with Squeeze-Excitation blocks** |
| **Bottleneck** | Single Conv Block | **ASPP (5 dilation rates — multi-scale context)** |
| **Skip Connections** | Plain concatenation | **Attention Gates (suppress irrelevant background)** |
| **Loss Function** | Dice + BCE | **Focal-Tversky + Boundary loss + Deep Supervision** |
| **LR Schedule** | ReduceLROnPlateau | **Cosine Annealing with 5-epoch linear warmup** |
| **Augmentation** | 3 basic transforms | **12-transform Albumentations pipeline** |
| **Precision** | float32 | **Mixed precision float16 (2× faster on GPU)** |
| **Training** | With early stopping | **Full 60 epochs — no premature stopping** |
| **Inference** | Single forward pass | **8-fold TTA flip ensemble** |

---

## 🚀 Live Demo

**Try it instantly — no setup needed:**

### 👉 [https://huggingface.co/spaces/Writick/polyp-segmentation](https://huggingface.co/spaces/Writick/polyp-segmentation)

Upload any colonoscopy image (JPEG/PNG) and the model returns:
- **Binary Mask** — white regions on black background showing predicted polyp pixels
- **Overlay** — red-tinted segmentation superimposed on the original image

The interface is fully **mobile-responsive** and runs entirely server-side on HuggingFace hardware — no GPU or Python installation required.

<table>
<tr>
<td align="center"><b>Desktop View — Multi-polyp</b></td>
<td align="center"><b>Mobile View — Single polyp</b></td>
</tr>
<tr>
<td>Input colonoscopy frame → Binary Mask + Red overlay for both polyp regions correctly localised</td>
<td>Fully responsive Gradio interface on Android — clean segmentation blob with red overlay on original frame</td>
</tr>
</table>

---

## 📁 Dataset

**Kvasir-SEG** — publicly available benchmark from Simula Research Laboratory, Norway.

| Property | Details |
|:---|:---|
| Total Images | 1,000 colonoscopy frames |
| Total Masks | 1,000 pixel-level binary segmentation masks |
| Image Format | JPEG |
| Mask Encoding | White = polyp, Black = background |
| Annotation | Expert endoscopist annotated |
| Input Size | Variable → resized to 256×256 for training |
| Extra Annotation | Bounding-box coordinates (JSON) |
| Source | [Simula Research Lab, Norway](https://datasets.simula.no/kvasir-seg/) |

### Data Split

| Split | Proportion | Samples |
|:---|:---|:---|
| Training | 75% | ~750 images |
| Validation | 15% | ~150 images |
| Test | 10% | ~100 images |

### Why Kvasir-SEG is challenging

Polyps vary enormously in **size** (a few mm to several cm), **shape** (sessile, pedunculated, flat), **colour** (pale pink to reddish-brown), and **texture**. Their boundaries are often visually indistinct even to trained endoscopists — making pixel-precise segmentation a genuinely hard problem.

---

## 🏗️ Architecture

### ResUNet++ with ASPP Bridge + Attention Gates

```
Input (256×256×3)
       │
  ┌────▼────────────────────────────────────────────────┐
  │              ENCODER (Residual Blocks)               │
  │  Block 1: ResBlock → SE → MaxPool  [32 filters]     │
  │  Block 2: ResBlock → SE → MaxPool  [64 filters]     │
  │  Block 3: ResBlock → SE → MaxPool  [128 filters]    │
  │  Block 4: ResBlock → SE → MaxPool  [256 filters]    │
  └────────────────────────────────────────────────────┘
       │
  ┌────▼────────────────────────────────────────────────┐
  │         BRIDGE — ASPP (5 dilation rates)            │
  │   d=1, d=6, d=12, d=18, d=24 → Concat → Conv      │
  └────────────────────────────────────────────────────┘
       │
  ┌────▼────────────────────────────────────────────────┐
  │     DECODER (Attention Gates on skip connections)   │
  │  Up 4: Upsample + AttGate(skip4) + ResBlock        │
  │  Up 3: Upsample + AttGate(skip3) + ResBlock        │
  │  Up 2: Upsample + AttGate(skip2) + ResBlock        │
  │  Up 1: Upsample + AttGate(skip1) + ResBlock        │
  └────────────────────────────────────────────────────┘
       │
  ┌────▼─────────────────────────────────────────────────┐
  │  OUTPUT: Conv 1×1 + Sigmoid → 256×256×1 binary mask │
  │  + 2 auxiliary deep supervision heads (training only)│
  └──────────────────────────────────────────────────────┘
```

### Key Components Explained

**Residual Blocks (Encoder/Decoder)**
Each block uses pre-activation residual connections with a 1×1 shortcut projection when filter counts change. This prevents vanishing gradients and enables the network to go deeper without degradation.

**Squeeze-and-Excitation (SE) Blocks**
After each residual block, SE modules perform channel-wise recalibration — globally averaging spatial features, then learning which channels are important for polyp detection. This adds negligible parameters but meaningfully improves representational quality.

**ASPP Bridge (Bottleneck)**
The Atrous Spatial Pyramid Pooling bridge applies parallel dilated convolutions at 5 different rates (1, 6, 12, 18, 24) at the deepest representation. This captures polyp context across vastly different scales simultaneously — critical since polyps range from a few pixels to hundreds.

**Attention Gates (Skip Connections)**
Instead of concatenating encoder features directly, each skip connection passes through an attention gate that learns to highlight polyp-relevant spatial regions and suppress background. This reduces the decoder's workload and sharpens boundary predictions.

**Deep Supervision**
During training, auxiliary segmentation heads are attached at two intermediate decoder levels with weighted losses (0.3× each). This forces the decoder to produce useful intermediate representations and provides richer gradient signal through the network.

---

## 🔧 Training Strategy

### Loss Function

```
Total Loss = Main Loss (×1.0) + Aux Loss 1 (×0.3) + Aux Loss 2 (×0.3)

where each component loss =
    0.5 × Focal-Tversky Loss + 0.5 × Boundary-Aware BCE
```

**Focal-Tversky Loss** penalises false negatives 2.3× harder than false positives — critical for small polyps where missing them has clinical consequences. The focal exponent further concentrates training on hard examples.

**Boundary Loss** adds an additional penalty specifically at mis-segmented edges (where the predicted mask disagrees with the ground truth contour). Clinicians focus most on polyp boundaries for size estimation, so this is clinically meaningful.

### Learning Rate Schedule

```
Epochs  0–5  : Linear warmup  1e-6 → 2e-4
Epochs  5–60 : Cosine annealing  2e-4 → 1e-6
```

No early stopping — the full 60 epochs allow the cosine schedule to complete its decay cycle, which consistently outperforms plateau-based stopping.

### Augmentation Pipeline (12 transforms, training only)

| Category | Transform | Probability |
|:---|:---|:---|
| Geometric | Horizontal + Vertical Flip | 50% each |
| Geometric | Random 90° Rotate | 50% |
| Geometric | Shift/Scale/Rotate (±30°) | 50% |
| Deformation | Elastic Transform | 30% |
| Deformation | Grid Distortion | 30% |
| Deformation | Optical Distortion | 30% |
| Intensity | CLAHE (contrast normalisation) | 40% |
| Intensity | Hue/Saturation/Value | 40% |
| Intensity | Random Brightness/Contrast | 40% |
| Noise | Gaussian Noise | 20% |
| Dropout | CoarseDropout (simulates occlusion) | 20% |

### Hyperparameters

| Parameter | Value |
|:---|:---|
| Input Size | 352 × 352 × 3 |
| Batch Size | 8 (reduce to 4 if OOM) |
| Epochs | 60 (no early stopping) |
| Peak LR | 2 × 10⁻⁴ |
| Minimum LR | 1 × 10⁻⁶ |
| Warmup Epochs | 5 |
| Base Filters | 32 |
| Mixed Precision | float16 |
| Optimiser | Adam |
| GPU | NVIDIA T4 (Google Colab) |

---

## 📊 Results

### Quantitative Results (Best Checkpoint — Epoch 54)

| Metric | Value |
|:---|:---|
| **Val Dice Coefficient** | **0.7941** |
| **Val IoU (Jaccard)** | **0.6623** |
| Val Total Loss | 0.7005 |
| Val Main Output Loss | 0.3842 |

### Comparison with Published Baselines on Kvasir-SEG

| Model | Dice | IoU | Reference |
|:---|:---:|:---:|:---|
| Standard U-Net | 0.818 | 0.746 | Jha et al., 2020 |
| ResU-Net | 0.813 | 0.793 | Zhang et al., 2018 |
| Double U-Net | 0.813 | 0.733 | Jha et al., 2020 |
| PraNet | 0.898 | 0.840 | Fan et al., 2020 |
| Multi-Scale U-Net *(our prototype)* | 0.567 | 0.456 | This work |
| **ResUNet++ *(this project)*** | **0.7941** | **0.6623** | **This work** |

The ResUNet++ achieves a **40% improvement in Dice** over our initial Multi-Scale U-Net prototype (0.567 → 0.794) and performs competitively with the Standard U-Net baseline (0.818) — despite training from scratch with no pretrained backbone.

### Training Curves

The model shows healthy learning behaviour across all 60 epochs:
- **Loss** decreases monotonically with training and validation closely tracking
- **Dice Coefficient** rises consistently, peaking at epoch 54 (val Dice = 0.7941)
- **IoU** follows the Dice curve, peaking simultaneously at 0.6623
- No evidence of overfitting — train/val gap remains narrow throughout

### Qualitative Predictions

Each test prediction is visualised as a 4-panel strip:

```
[ Original Image ] | [ Ground Truth Mask ] | [ Predicted Mask ] | [ Overlay ]
                                                                   GT=green, Pred=red
```

Sample results:
- **Dice 0.941** — Large sessile polyp: near-perfect boundary delineation
- **Dice 0.853** — Multi-region polyp: both regions correctly segmented
- **Dice 0.769** — Small polyp: correct region, minor edge over-segmentation
- **Dice 0.743** — Complex morphology: primary region captured, satellite missed

---

## 🛠️ Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/polyp-segmentation.git
cd polyp-segmentation
```

### Option 1 — Run on Google Colab (Recommended)

1. Open the notebook in Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/polyp-segmentation/blob/main/polyp_seg_max_accuracy.ipynb)

2. Set runtime to **GPU** (T4 or A100): `Runtime → Change runtime type → GPU`

3. Run all cells top-to-bottom. Cell 1 will prompt you to upload `kvasir-seg.zip`.

4. Download the Kvasir-SEG dataset:
   ```
   https://datasets.simula.no/kvasir-seg/
   ```

### Option 2 — Local Installation

```bash
pip install tensorflow>=2.13.0 albumentations==1.3.1 opencv-python-headless \
            scikit-learn tqdm matplotlib gradio huggingface_hub
```

### Notebook Cell Guide

| Cell | Description |
|:---:|:---|
| 1 | Upload & extract `kvasir-seg.zip` |
| 2 | Install libraries, enable mixed precision |
| 3 | Set all hyperparameters (edit here if needed) |
| 4 | Load paths, stratified train/val/test split |
| 5 | Build 12-transform Albumentations augmentation |
| 6 | Build tf.data pipeline (lazy loading + prefetch) |
| 7 | Visualise augmented training samples |
| 8 | Define ResUNet++ with ASPP + Attention Gates |
| 9 | Define losses, metrics, cosine LR; compile |
| 10 | Set up cosine warmup LR callback |
| 11 | Train for 60 epochs (saves best checkpoint) |
| 12 | Plot training curves from CSV log |
| 13 | Evaluate on test set with 8-fold TTA |
| 14 | Export model + build HuggingFace deployment zip |

### ⚠️ GPU Memory Notes

| GPU | Recommended Settings |
|:---|:---|
| T4 (15GB) | `IMG_SIZE=352, BATCH_SIZE=8` ✅ |
| T4 (low memory) | `IMG_SIZE=256, BATCH_SIZE=4` |
| A100 (40GB) | `IMG_SIZE=384, BATCH_SIZE=16` 🚀 |
| CPU only | Not recommended (training would take days) |

---

## 📁 Project Structure

```
polyp-segmentation/
│
├── 📓 polyp_seg_max_accuracy.ipynb    # Main training notebook (Google Colab)
│
├── 🚀 hf_deployment/                  # HuggingFace Spaces deployment
│   ├── app.py                         # Gradio web application
│   ├── best_model.h5                  # Trained model weights
│   ├── config.json                    # Model config (IMG_SIZE, threshold, etc.)
│   ├── requirements.txt               # Python dependencies for HF Space
│   └── README.md                      # HuggingFace Space card
│
├── 📊 outputs/                        # Generated during training
│   ├── best_model.keras               # Best checkpoint (Keras format)
│   ├── training_log.csv               # Per-epoch metrics log
│   ├── training_curves.png            # Training history plots
│   ├── sample_data.png                # Augmented sample visualisation
│   └── predictions/                   # Prediction strip images
│       ├── pred_0000_dice0.941.png
│       ├── pred_0001_dice0.853.png
│       └── ...
│
├── 📄 Polyp_Segmentation_Report.pdf   # Full project report (PCS 220)
└── 📖 README.md                       # This file
```

---

## 🤗 Deployment

The model is deployed as a **Gradio web app** on HuggingFace Spaces.

### Live URL
```
https://huggingface.co/spaces/Writick/polyp-segmentation
```

### How it works

```
User uploads colonoscopy image (JPEG/PNG)
            ↓
    Resize to 256×256
            ↓
    Normalise [0, 1]
            ↓
  Model inference (ResUNet++)
            ↓
  Threshold at 0.5
            ↓
  Return: Binary Mask + Red Overlay
```

### Deploy your own instance

```bash
# 1. Run Cell 14 in the notebook to generate hf_deployment.zip
# 2. Extract the zip
# 3. Create a new HuggingFace Space (Gradio SDK)
# 4. Upload all files from hf_deployment/ to the Space

# Or use the HuggingFace CLI:
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload YOUR_USERNAME/polyp-segmentation ./hf_deployment/ . --repo-type space
```

### Gradio Interface Features

- ✅ Upload any colonoscopy JPEG or PNG
- ✅ Real-time inference (no GPU required on client side)
- ✅ Binary mask output (downloadable)
- ✅ Red overlay for visual inspection
- ✅ Fully mobile-responsive
- ✅ No installation required for end users

---

## 👥 Team

**Thapar Institute of Engineering & Technology, Patiala**
Department of Computer Science & Engineering
PCS 220: Multimedia Processing Lab — May 2026

| Name | Roll No. | Contribution |
|:---|:---|:---|
| **Writick Parui** | 8025320111 | Architecture design, model training, Gradio interface, HuggingFace deployment |
| **Sougata Mukherjee** | 8025320095 | Architecture design, model training, deployment testing & validation |
| **Shreya Srivastava** | 8025320091 | Report documentation, research & literature review |

**Supervisor:** Dr. Abhishek Kesarwani (Assistant Professor, CSE Department, TIET)

---

## 📚 References

```bibtex
@inproceedings{jha2020kvasir,
  title     = {Kvasir-SEG: A Segmented Polyp Dataset},
  author    = {Jha, Debesh and others},
  booktitle = {MMM 2020},
  publisher = {Springer}
}

@inproceedings{ronneberger2015unet,
  title     = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author    = {Ronneberger, O. and Fischer, P. and Brox, T.},
  booktitle = {MICCAI 2015},
  publisher = {Springer}
}

@inproceedings{fan2020pranet,
  title     = {PraNet: Parallel Reverse Attention Network for Polyp Segmentation},
  author    = {Fan, Deng-Ping and others},
  booktitle = {MICCAI 2020},
  publisher = {Springer}
}

@inproceedings{jha2020doubleunet,
  title     = {DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation},
  author    = {Jha, Debesh and others},
  booktitle = {CBMS 2020},
  publisher = {IEEE}
}

@inproceedings{long2015fcn,
  title     = {Fully Convolutional Networks for Semantic Segmentation},
  author    = {Long, Jonathan and Shelhamer, Evan and Darrell, Trevor},
  booktitle = {CVPR 2015},
  publisher = {IEEE}
}
```

---

## 📄 Project Report

The full academic project report (16 pages, PCS 220: Multimedia Processing Lab) covers:
- Complete literature review and motivation
- Detailed architecture description with layer-by-layer tables
- Training configuration and loss function derivation
- Quantitative results and comparison with baselines
- Qualitative prediction visualisations
- Deployment documentation with interface screenshots

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution.

---

<div align="center">

**Made with ❤️ at Thapar Institute of Engineering & Technology, Patiala**

[🚀 Try the Live Demo](https://huggingface.co/spaces/Writick/polyp-segmentation) · [⭐ Star this repo](#) · [📄 Read the Report](#-project-report)

</div>
