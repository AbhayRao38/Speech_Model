import numpy as np
from sklearn.model_selection import train_test_split
import os

# === Paths ===
X_PATH = "X_speech.npy"
y_PATH = "y_speech.npy"
SAVE_DIR = "model_speech"

# === Load full speech data ===
X = np.load(X_PATH)
y = np.load(y_PATH)

print(f"✅ Loaded X shape: {X.shape}, y shape: {y.shape}")
print("📊 Classes:", np.unique(y))

# === Generate test split ===
all_indices = np.arange(len(X))
_, test_indices, _, y_test = train_test_split(
    all_indices, y, test_size=0.2, stratify=y, random_state=42
)

# === Save test indices and labels ===
np.save(os.path.join(SAVE_DIR, "X_speech_test_idx.npy"), test_indices)
np.save(os.path.join(SAVE_DIR, "y_speech_test.npy"), y[test_indices])

print("\n✅ Saved test files:")
print("   📁 X_speech_test_idx.npy:", test_indices.shape)
print("   📁 y_speech_test.npy    :", y[test_indices].shape)
