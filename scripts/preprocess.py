"""
scripts/preprocess.py

Convert raw .wav cough recordings into log-Mel spectrogram .npy files.
"""

import os
import glob
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────────
RAW_DIRS = [
    "data/raw/train",            # update these paths as needed
    "data/raw/val",
    "data/raw/test"
]
OUT_DIR    = "data/specs"        # will mirror RAW_DIRS structure here
SR         = 16000               # target sampling rate
DURATION   = 0.5                 # seconds per clip
N_MELS     = 128                 # number of Mel bands
HOP_LENGTH = int(0.01 * SR)      # 10 ms
WIN_LENGTH = int(0.025 * SR)     # 25 ms
FMIN       = 20                  # minimum freq for Mel
FMAX       = SR // 2             # Nyquist

def process_file(wav_path, out_dir):
    # 1) Load & resample
    y, sr = sf.read(wav_path)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    # 2) Crop or pad to fixed duration
    n_samples = int(DURATION * SR)
    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))
    else:
        y = y[:n_samples]

    # 3) Compute Mel spectrogram (power=2.0 for energy)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )

    # 4) Log compression
    log_mel = np.log1p(mel)

    # 5) Save as float32 .npy
    rel_path = os.path.relpath(wav_path, start=os.path.commonpath(RAW_DIRS))
    base = os.path.splitext(rel_path)[0] + ".npy"
    out_path = os.path.join(out_dir, base)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, log_mel.astype(np.float32))

if __name__ == "__main__":
    for raw_dir in RAW_DIRS:
        out_dir = os.path.join(OUT_DIR, os.path.basename(raw_dir))
        files = glob.glob(os.path.join(raw_dir, "*.wav"))
        print(f"Processing {len(files)} files from {raw_dir} → {out_dir}")
        for wav in tqdm(files, desc=os.path.basename(raw_dir)):
            process_file(wav, out_dir)
    print("Done.")
