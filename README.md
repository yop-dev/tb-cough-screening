# Res2TSM-Enhanced-MobileNetV4-for-TB-Cough-Analysis

This repository contains code for training and evaluating lightweight TB cough-screening models, including Res2TSM-enhanced MobileNetV4 variants in PyTorch and a suite of Keras baselines.

## 📁 Repository Structure

```text
.
├── data/
│   ├── raw/                # (gitignored) place your .wav files here
│   ├── specs/              # auto-generated .npy spectrograms
│   └── splits/             # train/val/test and CV fold lists
├── models/
│   ├── pytorch/            # PyTorch MobileNetV4 + variants
│   │   └── mobilenetv4_conv_blur_medium.py
│   └── keras/              # Keras baseline architectures
│       └── base_models.py
├── scripts/
│   ├── preprocess.py       # .wav → log-Mel spectrograms
│   ├── split_data.py       # create train/val/test and CV splits
│   ├── train_pytorch.py    # train any PyTorch model via `--model`
│   ├── train_keras.py      # train any Keras model via `--model`
│   └── cross_val.py        # 3-fold CV for the four PyTorch variants
├── outputs/                # saved checkpoints & history files
├── requirements.txt        # Python deps for PyTorch side
├── environment.yml         # (optional) conda env for Keras side
└── README.md




## 🚀 Quickstart

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yop-dev/tb-cough-screening.git
   cd tb-cough-screening

2. **Install Dependencies**
   
  - PyTorch side
     pip install -r requirements.txt
   
  - Keras side (if using conda)
     conda env create -f environment.yml
     conda activate tb-cough

3. **Download / place data**

   - Gather dataset (e.g from CODA)

4. **Preprocess Audio**

   - python scripts/preprocess.py

5.**Split Data**

  - python scripts/split_data.py

6. **Train Pytorch Model**

   python scripts/train_pytorch.py \
  --model v4_r2tsm \
  --train-files data/splits/train_files.npy \
  --train-labels data/splits/train_labels.npy \
  --val-files data/splits/val_files.npy \
  --val-labels data/splits/val_labels.npy \
  --data-dir data/specs/train \
  --epochs 15 \
  --batch-size 32 \
  --lr 1e-3

7. **Train Keras Model**
   python scripts/train_keras.py \
  --model mnet2 \
  --train-files data/splits/train_files.npy \
  --train-labels data/splits/train_labels.npy \
  --val-files data/splits/val_files.npy \
  --val-labels data/splits/val_labels.npy \
  --data-dirs data/specs/train \
  --epochs 15 \
  --batch-size 32 \
  --lr 1e-3

8. **Cross Validation (MobileNetV4 Conv Blur Medium)**
   python scripts/cross_val.py \
  --spec-dir data/specs/train \
  --labels-csv data/labels.csv \
  --n-splits 3 \
  --epochs 15 \
  --batch-size 32 \
  --lr 1e-3


📊 Outputs
outputs/ contains model weights (.pth or .h5) and training histories (.json).

Use your own analysis scripts or notebooks to plot ROC curves, bootstrap CIs, etc.

🔬 Reproducibility
Fixed seeds (NumPy, PyTorch, TensorFlow) ensure consistent results.

Hyperparameters (batch size, LR, epochs, model variants) are configurable via CLI flags.

A bootstrap CI script (scripts/bootstrap_ci.py, optional) can estimate uncertainty on CV AUC differences.

🙏 Acknowledgments
Inspired by the Res2Net and Temporal Shift Module (TSM) papers.

Thanks to the CODA DREAM challenge for the cough dataset.



    
