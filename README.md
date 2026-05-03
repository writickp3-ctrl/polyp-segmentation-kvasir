# 🔬 Polyp Segmentation using ResUNet++ on Kvasir-SEG

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Albumentations](https://img.shields.io/badge/Albumentations-1.3.1-blueviolet?style=for-the-badge)
![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-FFD21E?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

### Automated pixel-level polyp detection in colonoscopy images

**Val Dice: 0.7695 · Val IoU: 0.6291 · TTA Test Dice: 0.7777 · TTA Test IoU: 0.6825**

[🚀 **Live Demo**](https://huggingface.co/spaces/Writick/polyp-segmentation) · [📄 Report](#-project-report) · [🏗️ Architecture](#️-architecture) · [📊 Results](#-results) · [🛠️ Quick Start](#️-quick-start)

---

> **Course:** PCS 220 — Multimedia Processing Lab
> **Institution:** Thapar Institute of Engineering & Technology, Patiala
> **Supervisor:** Dr. Abhishek Kesarwani (Assistant Professor, CSE)

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

Colorectal cancer ranks among the **top 3 most diagnosed cancers** globally. Early detection of precancerous polyps during colonoscopy can reduce mortality by up to **90%**, yet clinicians miss **6–27%** of polyps due to fatigue, limited field of view and strong morphological variability.

This project delivers a complete, end-to-end deep learning pipeline that segments polyps at the **pixel level** — identifying exactly which pixels in a colonoscopy frame belong to a polyp. The final model is **ResUNet++**: a residual encoder-decoder with squeeze-and-excitation recalibration, an ASPP bottleneck, attention-gated skip connections and deep supervision. It is trained on **Kvasir-SEG** and deployed as a live web application on HuggingFace Spaces.

### What makes this different from a basic U-Net

| Component | Baseline U-Net | This Project (ResUNet++) |
|:---|:---|:---|
| **Architecture** | Plain encoder-decoder | **Residual blocks + Squeeze-Excitation** |
| **Input Size** | 256 × 256 | **352 × 352** |
| **Bottleneck** | Single Conv Block | **ASPP (dilation rates 1, 3, 6, 12)** |
| **Skip Connections** | Plain concatenation | **Attention Gates** |
| **Loss Function** | Dice + BCE | **Focal-Tversky + Boundary Loss** |
| **Supervision** | Single output head | **3 heads: output + ds3 + ds2** |
| **LR Schedule** | ReduceLROnPlateau | **5-epoch warmup + cosine annealing** |
| **Augmentation** | 3 basic transforms | **12-transform Albumentations pipeline** |
| **Precision** | float32 | **Mixed precision float16** |
| **Training** | With early stopping | **Full 60 epochs — no early stopping** |
| **Inference** | Single forward pass | **8-fold TTA flip/rotate ensemble** |

---

## 🚀 Live Demo

**Try it instantly — no setup needed:**

### 👉 [https://huggingface.co/spaces/Writick/polyp-segmentation](https://huggingface.co/spaces/Writick/polyp-segmentation)

Upload any colonoscopy image (JPEG/PNG) and the model returns:
- **Binary Mask** — white polyp regions on a black background
- **Overlay** — red-tinted segmentation superimposed on the original image

The interface is fully **mobile-responsive** and runs entirely server-side on HuggingFace hardware — no local GPU or Python installation required.

> ⚠️ **Model weights (`best_model.h5`) are not stored in this repository due to GitHub's 100 MB file size limit.** They are hosted directly on HuggingFace Spaces and loaded automatically when you use the live demo.

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
| Input Size | Variable → resized to **352 × 352** for training |
| Source | [datasets.simula.no/kvasir-seg](https://datasets.simula.no/kvasir-seg/) |

### Data Split

| Split | Proportion | Approx. Samples | Purpose |
|:---|:---|:---|:---|
| Training | 75% | ~750 | Model optimisation + augmentation |
| Validation | 15% | ~150 | Checkpoint selection |
| Test | 10% | ~100 | Single-pass & 8-fold TTA evaluation |

Split is deterministic via `train_test_split` with `SEED = 42`.

### Why Kvasir-SEG is hard

Polyps vary enormously in **size** (a few mm to several cm), **shape** (sessile, pedunculated, flat), **colour** (pale pink to reddish-brown) and **texture**. Their boundaries are often visually indistinct even to trained endoscopists, making pixel-precise segmentation a genuinely difficult problem.

---

## 🏗️ Architecture

### ResUNet++ with ASPP Bridge · Attention Gates · Deep Supervision

```
Input (352×352×3)
       │
  ┌────▼────────────────────────────────────────────────────────────┐
  │                 STEM CONV  32 filters  352×352                  │
  └────────────────────────────────────────────────────────────────┘
       │
  ┌────▼────────────────────────────────────────────────────────────┐
  │                ENCODER  (Residual + SE blocks)                  │
  │  E1: ResBlock + SE + MaxPool   32f  →  176×176                 │
  │  E2: ResBlock + SE + MaxPool   64f  →   88×88                  │
  │  E3: ResBlock + SE + MaxPool  128f  →   44×44                  │
  │  E4: ResBlock + SE + MaxPool  256f  →   22×22                  │
  └────────────────────────────────────────────────────────────────┘
       │
  ┌────▼────────────────────────────────────────────────────────────┐
  │           BRIDGE — ASPP  512f  (rates 1, 3, 6, 12)             │
  └────────────────────────────────────────────────────────────────┘
       │
  ┌────▼────────────────────────────────────────────────────────────┐
  │       DECODER  (Transposed Conv + Attention Gate + ResBlock)    │
  │  D4: Up + AttGate(skip E4) + ResBlock  256f  →   44×44         │
  │  D3: Up + AttGate(skip E3) + ResBlock  128f  →   88×88   ←ds3  │
  │  D2: Up + AttGate(skip E2) + ResBlock   64f  →  176×176  ←ds2  │
  │  D1: Up + AttGate(skip E1) + ResBlock   32f  →  352×352         │
  └────────────────────────────────────────────────────────────────┘
       │
  ┌────▼────────────────────────────────────────────────────────────┐
  │  OUTPUT: Conv 1×1 + Sigmoid  →  352×352×1  binary mask         │
  │  (+ auxiliary heads ds3 and ds2 — training only)               │
  └────────────────────────────────────────────────────────────────┘
```

### Architecture Components Explained

**Residual Blocks**
Pre-activation residual connections with 1×1 shortcut projections. Prevents vanishing gradients and enables deeper encoding without degradation.

**Squeeze-and-Excitation (SE)**
Channel-wise recalibration after each residual block. The SE module globally averages spatial features, then learns which channels are most informative for polyp detection. Adds negligible parameters.

**ASPP Bridge**
Atrous Spatial Pyramid Pooling at the bottleneck uses parallel dilated convolutions (rates 1, 3, 6, 12) to capture context across multiple scales simultaneously — essential since polyps range from a few pixels to hundreds.

**Attention Gates**
Each skip connection passes through an attention gate before concatenation with the decoder. Gates learn to suppress background noise and highlight polyp-relevant spatial regions, sharpening boundary predictions.

**Deep Supervision**
Auxiliary segmentation heads at decoder levels D3 (ds3) and D2 (ds2) contribute to training loss, improving gradient flow to early layers. Only the main `output` head is used at inference time.

---

## 🔧 Training Strategy

### Loss Function

```
Total Loss = Main Loss (×1.0) + ds3 Loss (×0.3) + ds2 Loss (×0.3)

Each head loss = 0.8 × Focal-Tversky Loss + 0.2 × Boundary Loss
```

**Focal-Tversky Loss** — penalises false negatives more heavily than false positives, which is critical for small polyps where under-segmentation has clinical consequences. The focal exponent concentrates training on hard examples.

**Boundary Loss** — adds an edge-aware penalty at mis-segmented contours. Since clinicians rely on accurate boundaries for size estimation, this is clinically meaningful.

### Learning Rate Schedule

```
Epochs  0 – 5  :  Linear warmup   1e-6  →  2e-4
Epochs  5 – 60 :  Cosine annealing  2e-4  →  1e-6
```

No early stopping — the full 60 epochs let the cosine schedule complete its decay, providing consistently better results than plateau-based stopping.

### 12-Transform Albumentations Augmentation Pipeline

| Category | Transform | Probability |
|:---|:---|:---|
| Geometric | Horizontal Flip | 50% |
| Geometric | Vertical Flip | 50% |
| Geometric | Random 90° Rotate | 50% |
| Geometric | ShiftScaleRotate (±30°) | 50% |
| Deformation | ElasticTransform | 30% |
| Deformation | GridDistortion | 30% |
| Deformation | OpticalDistortion | 30% |
| Intensity | CLAHE | 40% |
| Intensity | HueSaturationValue | 40% |
| Intensity | RandomBrightnessContrast | 40% |
| Noise | GaussNoise | 20% |
| Occlusion | CoarseDropout | 20% |

Applied to training data only; validation and test use deterministic preprocessing.

### Hyperparameters

| Parameter | Value |
|:---|:---|
| Input Size | 352 × 352 × 3 |
| Batch Size | 8 |
| Epochs | 60 (no EarlyStopping) |
| Peak LR | 2e-4 (Adam) |
| Warmup Epochs | 5 |
| Base Filters | 32 |
| Mixed Precision | float16 |
| Checkpoint Monitor | `val_output_dice_coef` |
| Checkpoint Format | `best_model.keras` |
| Log File | `training_log.csv` |

---

## 📊 Results

### Best Validation Metrics (Epoch 58)

| Metric | Value |
|:---|:---|
| `val_output_dice_coef` | **0.7695** |
| `val_output_iou_metric` | **0.6291** |
| `val_loss` | 0.7767 |
| `val_output_loss` | 0.4291 |

### Test Set Evaluation (100 samples)

| Evaluation Mode | Dice | IoU |
|:---|:---|:---|
| Single-pass | 0.7607 ± 0.2352 | 0.6612 ± 0.2541 |
| **8-fold TTA** | **0.7777 ± 0.2302** | **0.6825 ± 0.2500** |
| TTA improvement | **+1.70% Dice** | — |

8-fold TTA averages predictions over horizontal flip, vertical flip, 90°/180°/270° rotations, and combinations before thresholding at 0.5.

### Comparison with Published Baselines (Kvasir-SEG)

| Model | Dice | IoU | Reference |
|:---|:---|:---|:---|
| Standard U-Net | 0.818 | 0.746 | Jha et al., 2020 |
| ResU-Net | 0.813 | 0.793 | Zhang et al., 2018 |
| Double U-Net | 0.813 | 0.733 | Jha et al., 2020 |
| PraNet | 0.898 | 0.840 | Fan et al., 2020 |
| **This Project (Val)** | **0.7695** | **0.6291** | This work |
| **This Project + TTA (Test)** | **0.7777** | **0.6825** | This work |

> Note: Baselines are trained/evaluated with full data and larger compute. This project uses Colab T4 (15 GB) with 1,000 training images.

---

## 🛠️ Quick Start

### Option 1 — Google Colab (Recommended)

1. Open the notebook: [`polyp_seg_max_accuracy_deployed_model.ipynb`](./polyp_seg_max_accuracy_deployed_model.ipynb)

2. Set runtime to **GPU** (T4): `Runtime → Change runtime type → GPU`

3. Download Kvasir-SEG from the official source and upload as `kvasir-seg.zip` when Cell 1 prompts:
   ```
   https://datasets.simula.no/kvasir-seg/
   ```

4. Run all cells top-to-bottom. Training takes approximately **3–4 hours** on a T4 GPU.

### Option 2 — Local Installation

```bash
# Clone the repository
git clone https://github.com/writickp3-ctrl/polyp-segmentation-kvasir.git
cd polyp-segmentation-kvasir

# Install dependencies
pip install tensorflow>=2.13.0 \
            albumentations==1.3.1 \
            opencv-python-headless \
            scikit-learn \
            tqdm \
            matplotlib \
            gradio \
            huggingface_hub
```

### Notebook Cell Guide

| Cell | Description |
|:---:|:---|
| 1 | Upload & extract `kvasir-seg.zip`; auto-detect folder structure |
| 2 | Install libraries, set random seeds, enable mixed precision |
| 3 | Configuration — IMG_SIZE=352, BATCH_SIZE, EPOCHS, LR |
| 4 | Load image/mask paths; deterministic train/val/test split |
| 5 | Define 12-transform Albumentations augmentation pipeline |
| 6 | Build tf.data datasets with OpenCV preprocessing + prefetch |
| 7 | Visualise and save augmented training samples |
| 8 | Define ResUNet++ (Res blocks, SE, ASPP, Attention Gates) |
| 9 | Define Focal-Tversky + Boundary loss, metrics; compile |
| 10 | Build and plot warmup + cosine LR schedule |
| 11 | Train 60 epochs — ModelCheckpoint + CSVLogger |
| 12 | Plot training curves from `training_log.csv` |
| 13 | Load `best_model.keras`; evaluate single-pass + 8-fold TTA |
| 14 | Export HuggingFace deployment package |

### ⚠️ GPU Memory Guide

| GPU | Recommended Settings |
|:---|:---|
| T4 (15 GB) | `IMG_SIZE=352, BATCH_SIZE=8` ✅ Default |
| T4 (low memory) | `IMG_SIZE=256, BATCH_SIZE=4` |
| A100 (40 GB) | `IMG_SIZE=384, BATCH_SIZE=16` 🚀 |
| CPU only | ❌ Not recommended — training would take days |

---

## 📁 Project Structure

```
polyp-segmentation-kvasir/
│
├── 📓 polyp_seg_max_accuracy_deployed_model.ipynb   # Main training notebook
│
├── 🚀 hf_deployment/                                # HuggingFace Spaces app
│   ├── app.py                                       # Gradio web application
│   ├── requirements.txt                             # HF Space dependencies
│   └── README.md                                    # HuggingFace Space card
│                                                    # ⚠️ best_model.h5 NOT stored here
│                                                    #    (>100 MB — hosted on HF Spaces)
│
├── 📄 Polyp_Segmentation_FinalReport.pdf            # Full academic report (PCS 220)
├── 🖼️ polyp-_result.png                             # Training history plot
├── .gitignore                                       # Ignores model weights & zip files
└── 📖 README.md                                     # This file
```

> **Note on model weights:** `best_model.h5` / `best_model.keras` exceed GitHub's 100 MB limit and are therefore not tracked in this repository. They are deployed directly to HuggingFace Spaces and can be reproduced by running the notebook end-to-end.

---

## 🤗 Deployment

The trained ResUNet++ model is deployed as a **Gradio web application** on HuggingFace Spaces.

### Live URL
```
https://huggingface.co/spaces/Writick/polyp-segmentation
```

### Inference Pipeline

```
User uploads colonoscopy image (JPEG / PNG)
                ↓
    Read with OpenCV → resize to 352×352
                ↓
        Normalise to [0, 1]
                ↓
    ResUNet++ forward pass (best_model.h5)
                ↓
    Extract main output head probability map
                ↓
         Threshold at 0.5
                ↓
  Return ① Binary Mask  +  ② Red Overlay
```

### Deploy Your Own Instance

```bash
# Step 1: Run Cell 14 in the notebook — generates hf_deployment.zip
# Step 2: Create a new HuggingFace Space (Gradio SDK)
# Step 3: Upload files from hf_deployment/ (excluding best_model.h5 if >100 MB)
#         Upload model weights separately via HF UI or Git LFS

# Or use the CLI:
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload YOUR_USERNAME/polyp-segmentation ./hf_deployment/ . --repo-type space
```

### Interface Features

- ✅ Upload any colonoscopy JPEG or PNG
- ✅ Real-time inference (server-side — no client GPU needed)
- ✅ Binary mask output (downloadable)
- ✅ Red overlay for visual inspection
- ✅ Fully mobile-responsive Gradio UI
- ✅ No local installation required for end users

---

## 👥 Team

**Thapar Institute of Engineering & Technology, Patiala**
Department of Computer Science & Engineering
PCS 220: Multimedia Processing Lab — May 2026

| Name | Roll No. | Contribution |
|:---|:---|:---|
| **Writick Parui** | 8025320111 | ResUNet++ architecture design, model training, Gradio interface development, HuggingFace deployment & hosting |
| **Sougata Mukherjee** | 8025320095 | ResUNet++ architecture design, model training, deployment testing & inference validation |
| **Shreya Srivastava** | 8025320091 | Report documentation, literature review & research |

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

@article{zhang2018resunet,
  title   = {Road Extraction by Deep Residual U-Net},
  author  = {Zhang, Zhengxin and Liu, Qingjie and Wang, Yunhong},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  volume  = {15},
  number  = {5},
  year    = {2018}
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

The full academic report (`Polyp_Segmentation_FinalReport.pdf`, 18 pages) is included in this repository and covers:

- Complete literature review and clinical motivation
- Detailed ResUNet++ architecture with layer-by-layer tables and diagram
- Focal-Tversky + Boundary loss derivation
- Warmup-cosine LR schedule explanation
- Albumentations augmentation pipeline details
- Quantitative results with baseline comparison
- 8-fold TTA evaluation methodology
- Qualitative prediction grids
- Deployment documentation with interface screenshots
- Contribution summary

---

## 📜 License

This project is released under the **MIT License** — free to use, modify, and distribute with attribution.

---

<div align="center">

**Made with ❤️ at Thapar Institute of Engineering & Technology, Patiala**

[🚀 Try the Live Demo](https://huggingface.co/spaces/Writick/polyp-segmentation) · [⭐ Star this repo](https://github.com/writickp3-ctrl/polyp-segmentation-kvasir) · [📄 Read the Report](./Polyp_Segmentation_FinalReport.pdf)

</div>
