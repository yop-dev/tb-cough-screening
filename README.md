# Lightweight TB Cough Screening using PyTorch with Res2TSM-Enhanced and Baseline CNNs

This repository provides a PyTorch-based framework for training and evaluating lightweight models for Tuberculosis (TB) screening using cough sounds. It includes advanced Res2TSM-enhanced MobileNetV4 variants and a suite of standard CNN baseline models, all implemented in PyTorch.

## 📝 Project Overview

The project aims to leverage deep learning techniques to develop accessible TB screening tools. By analyzing log-Mel spectrograms of cough sounds, the models can identify potential TB indicators. This implementation exclusively uses PyTorch for all model training and evaluation tasks.

**Key Features:**
-   **PyTorch Implementation:** All models and training pipelines are implemented in PyTorch.
-   **Advanced Architectures:** Features MobileNetV4 variants enhanced with Res2Net and Temporal Shift Modules (TSM).
-   **Baseline Models:** Includes PyTorch implementations of common CNN architectures (MobileNetV2, MobileNetV3Small, EfficientNetB0/B3, ResNet50, InceptionV3, DenseNet121) for comparative studies. These are derived from Keras application counterparts.
-   **Data Preprocessing:** Includes scripts to convert raw audio files (`.wav`) into log-Mel spectrograms (`.npy`).
-   **Flexible Training:** Supports training all included PyTorch models with customizable parameters.
-   **Reproducibility:** Employs fixed seeds for consistent results.

## 📁 Repository Structure

```
.
├── data/
│   ├── raw/                # (gitignored) Place your .wav audio files here
│   ├── specs/              # Auto-generated .npy log-Mel spectrograms
│   └── splits/             # Auto-generated train/val/test file lists and CV folds
│       └── split_data.py   # Script to create data splits
├── models/
│   ├── MobileNetV4_Conv_Blur_Medium_Enhanced/ # PyTorch MobileNetV4 models and utilities
│   │   ├── models.py       # Custom MobileNetV4 variants (Base, TSM, Res2Net, Res2TSM)
│   │   └── utils.py        # Utility modules for enhancements
│   ├── pytorch_baselines/  # PyTorch baseline CNN model definitions
│   │   └── base_models.py  # PyTorch versions of standard CNNs
│   └── keras/              # Original Keras baseline model definitions (for reference)
│       └── base_models.py
├── scripts/
│   ├── preprocess.py       # Converts .wav audio to log-Mel spectrograms
│   ├── train_pytorch.py    # Trains all PyTorch models (enhanced and baselines)
│   └── cross_k-fold        # Script for k-fold cross-validation for PyTorch models
├── outputs/                # (gitignored) Saved model checkpoints (.pth) and training histories (.json)
├── requirements.txt        # Python dependencies for the project
└── README.md
```

## 🚀 Getting Started

Follow these steps to set up the project and train your models.

### 1. Clone the Repository
```bash
git clone https://github.com/yop-dev/tb-cough-screening.git
cd tb-cough-screening
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Then, install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data
-   Place raw cough audio files (`.wav`) into `data/raw/`. Subdirectories like `data/raw/train` are supported by `scripts/preprocess.py`.
-   Modify `scripts/preprocess.py`'s `RAW_DIRS` list to point to your data locations.

### 4. Preprocess Audio Data
Convert `.wav` files into log-Mel spectrograms using `scripts/preprocess.py`.
```bash
python scripts/preprocess.py
```
Spectrograms (`.npy` files) will be saved in `data/specs/`, mirroring `RAW_DIRS` structure.

### 5. Split Data for Training
Create train/validation/test splits using `data/splits/split_data.py`. This script uses spectrograms from `data/specs/` (update `SPEC_DIR` in the script if needed).
```bash
python data/splits/split_data.py
```
Output lists are saved in `data/splits/`. **Crucially, ensure the label generation logic within this script matches your dataset's naming convention or metadata.**

### 6. Train a PyTorch Model
Use `scripts/train_pytorch.py` to train any of the available PyTorch models.

**Example (Enhanced MobileNetV4):**
```bash
python scripts/train_pytorch.py \
    --model v4_r2tsm \
    --train-files data/splits/train_files.npy \
    --train-labels data/splits/train_labels.npy \
    --val-files data/splits/val_files.npy \
    --val-labels data/splits/val_labels.npy \
    --data-dir data/specs/train \
    --epochs 15 \
    --batch-size 32 \
    --lr 1e-3 \
    --dropout-rate 0.3 \
    --output-dir outputs/models_v4_r2tsm \
    --verbose
```

**Example (PyTorch Baseline - EfficientNetB0):**
```bash
python scripts/train_pytorch.py \
    --model effb0_pt \
    --train-files data/splits/train_files.npy \
    --train-labels data/splits/train_labels.npy \
    --val-files data/splits/val_files.npy \
    --val-labels data/splits/val_labels.npy \
    --data-dir data/specs/train \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-4 \
    --dropout-rate 0.5 \
    --output-dir outputs/models_effb0_pt \
    --verbose
```

**Key Command-Line Arguments for `train_pytorch.py`:**
-   `--model`: Choose the model.
    -   Enhanced MobileNetV4: `v4_base`, `v4_tsm`, `v4_r2n`, `v4_r2tsm`, `v4_small`, `v4_hybrid`.
    -   PyTorch Baselines: `mnet2_pt`, `mnet3s_pt`, `effb0_pt`, `effb3_pt`, `res50_pt`, `incepv3_pt`, `dnet121_pt`.
-   `--train-files`, `--train-labels`, `--val-files`, `--val-labels`: Paths to data split files.
-   `--data-dir`: Base directory for spectrogram `.npy` files.
-   `--epochs`, `--batch-size`, `--lr`: Standard training parameters.
-   `--dropout-rate`: Dropout rate for the model's classification head (default: 0.5).
-   `--output-dir`: Directory for saving checkpoints and history (default: `outputs`).
-   `--verbose`: Enable detailed epoch logging.

### 7. Cross-Validation (Optional)
The `scripts/cross_k-fold` script can be used for k-fold cross-validation. It iterates through folds defined by `data/splits/split_data.py` (if `N_FOLDS` > 1).
It trains the four main MobileNetV4 variants (`v4_base`, `v4_tsm`, `v4_r2n`, `v4_r2tsm`). Modify this script if you wish to include other models in automated CV.

**Conceptual Usage (adapt as needed):**
```bash
python scripts/cross_k-fold \
    --spec-dir data/specs/train \
    --labels-csv data/labels.csv \ # Or adapt to use .npy split files from data/splits
    --n-splits 3 \                 # Should match N_FOLDS from split_data.py
    --epochs 15 \
    --batch-size 32 \
    --lr 1e-3
```
The script expects labels to be loaded from a CSV or requires adaptation to use the `.npy` files generated by `data/splits/split_data.py` for each fold.

## 📊 Outputs
-   **Model Checkpoints:** `.pth` files in the output directory.
-   **Training History:** `.json` files with metrics per epoch.

## 🔬 Model Architectures

### Enhanced MobileNetV4 Variants
Located in `models/MobileNetV4_Conv_Blur_Medium_Enhanced/`. These build upon `timm`'s `mobilenetv4_conv_blur_medium` with:
-   **Res2Net Blocks:** Multi-scale feature representation.
-   **Temporal Shift Modules (TSM):** Efficient temporal reasoning.
-   **Res2TSM Blocks:** Combined Res2Net and TSM.

### PyTorch Baseline Models
Located in `models/pytorch_baselines/`. These are standard CNN architectures implemented in PyTorch using `timm` backbones, with a classification head structure inspired by their Keras Application counterparts (original Keras definitions in `models/keras/` for reference).
-   MobileNetV2 (`mnet2_pt`)
-   MobileNetV3Small (`mnet3s_pt`)
-   EfficientNetB0 (`effb0_pt`), EfficientNetB3 (`effb3_pt`)
-   ResNet50 (`res50_pt`)
-   InceptionV3 (`incepv3_pt`) - Note: Optimal input size is 299x299. `SpectrogramDataset` resizes; adjust `img_size` in `train_pytorch.py` for best results.
-   DenseNet121 (`dnet121_pt`)

## 🔄 Reproducibility
Training scripts use fixed seeds for NumPy and PyTorch.

## Acknowledgments
-   Inspired by Res2Net and Temporal Shift Module (TSM) papers.
-   Baseline model structures adapted from Keras Applications.
-   Thanks to the CODA DREAM challenge for cough datasets.
