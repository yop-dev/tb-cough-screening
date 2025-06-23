# Lightweight TB Cough Screening using PyTorch with Res2TSM-Enhanced MobileNetV4

This repository provides a PyTorch-based framework for training and evaluating lightweight models for Tuberculosis (TB) screening using cough sounds. The core models are Res2TSM-enhanced variants of MobileNetV4, designed for efficient and accurate classification.

## 📝 Project Overview

The project aims to leverage deep learning techniques to develop accessible TB screening tools. By analyzing log-Mel spectrograms of cough sounds, the models can identify potential TB indicators. This implementation uses PyTorch for all model training and evaluation tasks.

**Key Features:**
-   **PyTorch Exclusivity:** All models and training pipelines are implemented in PyTorch.
-   **Advanced Architectures:** Utilizes MobileNetV4 variants, enhanced with Res2Net and Temporal Shift Modules (TSM) for improved feature extraction from spectrograms.
-   **Data Preprocessing:** Includes scripts to convert raw audio files (`.wav`) into log-Mel spectrograms (`.npy`) suitable for model input.
-   **Flexible Training:** Supports training various MobileNetV4 configurations with customizable parameters.
-   **Reproducibility:** Employs fixed seeds for consistent results across experiments.

## 📁 Repository Structure

```
.
├── data/
│   ├── raw/                # (gitignored) Place your .wav audio files here (e.g., data/raw/train, data/raw/val)
│   ├── specs/              # Auto-generated .npy log-Mel spectrograms
│   └── splits/             # Auto-generated train/validation/test file lists and CV folds
│       └── split_data.py   # Script to create data splits
├── models/
│   └── MobileNetV4_Conv_Blur_Medium_Enhanced/ # PyTorch MobileNetV4 models and utilities
│       ├── models.py       # Model definitions (MobileNetV4_Base, _TSM, _Res2Net, _Res2TSM)
│       └── utils.py        # Utility modules (Res2TSMBlock, Res2NetBlock, TemporalShift)
├── scripts/
│   ├── preprocess.py       # Converts .wav audio to log-Mel spectrograms
│   ├── train_pytorch.py    # Trains PyTorch models
│   └── cross_k-fold        # Script for k-fold cross-validation (if applicable)
├── outputs/                # (gitignored) Saved model checkpoints (.pth) and training histories (.json)
├── requirements.txt        # Python dependencies for the project
└── README.md
```

## 🚀 Getting Started

Follow these steps to set up the project and train your first model.

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
-   Place your raw cough audio files (in `.wav` format) into the `data/raw/` directory. You can create subdirectories like `data/raw/train`, `data/raw/val`, etc., if you have pre-split data.
-   If your dataset is not yet split, you can place all audio files in a common directory within `data/raw/`.

### 4. Preprocess Audio Data
Convert the raw `.wav` files into log-Mel spectrograms. The `preprocess.py` script handles this conversion.

**Configuration (inside `scripts/preprocess.py`):**
-   Update `RAW_DIRS`: List of directories containing your raw `.wav` files (e.g., `["data/raw/train", "data/raw/val"]`).
-   `OUT_DIR`: Directory where spectrograms (`.npy` files) will be saved (default: `data/specs`).
-   Other parameters like `SR` (sampling rate), `DURATION`, `N_MELS`, etc., can be adjusted as needed.

**Run Preprocessing:**
```bash
python scripts/preprocess.py
```
This will generate `.npy` files in the `data/specs/` directory, mirroring the structure of `RAW_DIRS`.

### 5. Split Data for Training and Evaluation
Create train, validation, and (optionally) test sets, along with k-fold cross-validation splits if needed. The `split_data.py` script uses the generated spectrograms.

**Configuration (inside `data/splits/split_data.py`):**
-   `SPEC_DIR`: Directory containing the `.npy` spectrograms (e.g., `data/specs/train` if your spectrograms are in a 'train' subfolder).
-   `OUT_DIR`: Directory where the split files (`train_files.npy`, `val_labels.npy`, etc.) will be saved (default: `data/splits`).
-   Adjust `TEST_SIZE`, `VAL_SIZE`, `RANDOM_STATE`, and `N_FOLDS` as required.
-   **Important:** Modify the `labels` array creation logic based on your filename conventions or if you have a separate metadata file for labels. The current example derives labels from filenames containing "Positive" or "Negative".

**Run Data Splitting:**
```bash
python data/splits/split_data.py
```
This will create `.npy` files in `data/splits/` containing lists of filenames and corresponding labels for each data subset.

### 6. Train a PyTorch Model
Use the `train_pytorch.py` script to train your chosen MobileNetV4 variant.

**Example Command:**
```bash
python scripts/train_pytorch.py \
    --model v4_r2tsm \
    --train-files data/splits/train_files.npy \
    --train-labels data/splits/train_labels.npy \
    --val-files data/splits/val_files.npy \
    --val-labels data/splits/val_labels.npy \
    --data-dir data/specs/train \  # Main directory where .npy files from train_files.npy are located
    --epochs 15 \
    --batch-size 32 \
    --lr 1e-3 \
    --output-dir outputs/models \ # Specify where to save model weights and history
    --verbose # Optional: for detailed epoch logging
```

**Key Command-Line Arguments for `train_pytorch.py`:**
-   `--model`: Choose the model architecture. Options:
    -   `v4_base`: Standard MobileNetV4 Conv Blur Medium.
    -   `v4_tsm`: MobileNetV4 with Temporal Shift Module.
    -   `v4_r2n`: MobileNetV4 with Res2Net block.
    -   `v4_r2tsm`: MobileNetV4 with combined Res2TSM block.
    -   `v4_small`: Timm's `mobilenetv4_conv_small`.
    -   `v4_hybrid`: Timm's `mobilenetv4_hybrid_medium`.
-   `--train-files`, `--train-labels`: Paths to the `.npy` files for training data and labels.
-   `--val-files`, `--val-labels`: Paths to the `.npy` files for validation data and labels.
-   `--data-dir`: The base directory to search for the actual spectrogram `.npy` files listed in the train/val splits.
-   `--epochs`: Number of training epochs.
-   `--batch-size`: Batch size for training.
-   `--lr`: Learning rate.
-   `--output-dir`: (Optional, defaults to `outputs`) Directory to save model checkpoints (`.pth`) and training history (`.json`).
-   `--verbose`: (Optional) Print detailed logs for each epoch.

### 7. Cross-Validation (Optional)
If you have generated k-fold splits using `data/splits/split_data.py`, you can perform cross-validation. The repository includes a script `scripts/cross_k-fold` (its usage might need to be adapted or verified based on its specific implementation, as it was not fully detailed in the initial request).

Typically, you would iterate through each fold's training and validation files, running `scripts/train_pytorch.py` for each.

**Example for `scripts/cross_k-fold` (Conceptual):**
The `README.md` previously mentioned a `cross_val.py` script with arguments like:
```bash
python scripts/cross_k-fold \ # Assuming this is the correct script
    --spec-dir data/specs/train \
    --labels-csv data/labels.csv \ # Or use generated .npy fold files
    --n-splits 3 \ # Should align with N_FOLDS in split_data.py
    --epochs 15 \
    --batch-size 32 \
    --lr 1e-3
```
You may need to adjust this script or create a new one to loop through the pre-generated fold files from `data/splits/` (e.g., `fold1_train_files.npy`, `fold1_val_files.npy`, etc.) and call `train_pytorch.py` accordingly.

## 📊 Outputs

-   **Model Checkpoints:** Trained model weights are saved as `.pth` files in the specified output directory (e.g., `outputs/models/`).
-   **Training History:** Training metrics (like AUC per epoch) are saved as `.json` files in the same output directory.

You can use these outputs for further analysis, plotting ROC curves, calculating confidence intervals, etc., using your preferred tools or custom scripts.

## 🔬 Model Architecture

The core models are based on **MobileNetV4 Conv Blur Medium**, a lightweight and efficient convolutional neural network. Enhancements include:
-   **Res2Net Blocks:** These blocks introduce multi-scale feature representation within residual blocks, allowing the network to capture features at different granularities.
-   **Temporal Shift Modules (TSM):** TSM shifts parts of channels along the temporal dimension, enabling temporal reasoning at a low computational cost, which is beneficial for sequential data like spectrograms.
-   **Res2TSM Blocks:** Combine the strengths of Res2Net and TSM for robust spatio-temporal feature learning.

The specific variants available are defined in `models/MobileNetV4_Conv_Blur_Medium_Enhanced/models.py` and can be selected using the `--model` argument in `scripts/train_pytorch.py`.

## 🔄 Reproducibility

The training scripts (`train_pytorch.py`) set fixed seeds for NumPy and PyTorch to ensure that experiments are reproducible.

## Acknowledgments

-   This work is inspired by the concepts presented in the Res2Net and Temporal Shift Module (TSM) research papers.
-   Thanks to the CODA DREAM challenge for providing access to cough datasets that can be used with this framework.
