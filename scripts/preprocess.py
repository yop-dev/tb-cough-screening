import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────────
RAW_DIR    = "data/raw/train"    # Your folder of .wav cough files
OUT_DIR    = "data/specs/train"  # Where the .npy spectrograms will go
SR         = 16000               # Target sampling rate
DURATION   = 0.5                 # Seconds per clip
N_MELS     = 128                 # Number of Mel bands
HOP_LENGTH = 160                 # 10 ms hop (160 samples)
WIN_LENGTH = 400                 # 25 ms window (400 samples)
FMIN       = 20
FMAX       = 8000

os.makedirs(OUT_DIR, exist_ok=True)

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
    # 3) Compute Mel spectrogram
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
    # 4) Log compress
    log_mel = np.log1p(mel)
    # 5) Save as .npy
    base = os.path.splitext(os.path.basename(wav_path))[0]
    np.save(os.path.join(out_dir, base + ".npy"), log_mel.astype(np.float32))

if __name__ == "__main__":
    for fname in tqdm(os.listdir(RAW_DIR), desc="Preprocessing audio"):
        if not fname.lower().endswith(".wav"):
            continue
        process_file(os.path.join(RAW_DIR, fname), OUT_DIR)
