import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

# ─── CONFIG ────────────────────────────────────────────────────────────────
SPEC_DIR     = "data/specs/train"    # where your .npy spectrograms now live
OUT_DIR      = "data/splits"         # output folder for file lists
TEST_SIZE    = 0.1
VAL_SIZE     = 0.1
RANDOM_STATE = 42
N_FOLDS      = 3                     # if you used 3-fold CV

os.makedirs(OUT_DIR, exist_ok=True)

# ─── GATHER FILES & LABELS ─────────────────────────────────────────────────
# Assumes your .npy filenames encode the class or that you have a parallel labels array:
files = np.array([f for f in os.listdir(SPEC_DIR) if f.endswith(".npy")])
# Example: labels derived from filename prefix "TB_Positive_xxx.npy" vs "TB_Negative_xxx.npy"
labels = np.array([1 if "Positive" in f else 0 for f in files])

# ─── HOLD-OUT SPLIT ─────────────────────────────────────────────────────────
train_files, test_files, train_labels, test_labels = train_test_split(
    files, labels,
    test_size=TEST_SIZE,
    stratify=labels,
    random_state=RANDOM_STATE
)
train_files, val_files, train_labels, val_labels = train_test_split(
    train_files, train_labels,
    test_size=VAL_SIZE/(1-TEST_SIZE),
    stratify=train_labels,
    random_state=RANDOM_STATE
)

# Save hold-out splits
np.save(os.path.join(OUT_DIR, "train_files.npy"), train_files)
np.save(os.path.join(OUT_DIR, "train_labels.npy"), train_labels)
np.save(os.path.join(OUT_DIR, "val_files.npy"),   val_files)
np.save(os.path.join(OUT_DIR, "val_labels.npy"),   val_labels)
np.save(os.path.join(OUT_DIR, "test_files.npy"),  test_files)
np.save(os.path.join(OUT_DIR, "test_labels.npy"), test_labels)

# ─── OPTIONAL: STRATIFIED K-FOLD ─────────────────────────────────────────────
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
for fold, (tr_idx, va_idx) in enumerate(kf.split(files, labels), start=1):
    np.save(os.path.join(OUT_DIR, f"fold{fold}_train_files.npy"), files[tr_idx])
    np.save(os.path.join(OUT_DIR, f"fold{fold}_train_labels.npy"), labels[tr_idx])
    np.save(os.path.join(OUT_DIR, f"fold{fold}_val_files.npy"),   files[va_idx])
    np.save(os.path.join(OUT_DIR, f"fold{fold}_val_labels.npy"),  labels[va_idx])

print("Splits written to", OUT_DIR)
