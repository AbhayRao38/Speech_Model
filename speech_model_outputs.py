import numpy as np
from tensorflow.keras.models import load_model

# --- Load speech model ---
speech_model = load_model('speech_model.h5')

# --- Load and normalize speech input ---
X_speech = np.load('X_speech.npy')  # Replace with correct path
X_speech = X_speech / 255.0  # Normalize if required (based on training)

# --- Predict and save ---
speech_preds = speech_model.predict(X_speech)
np.save('speech_model_outputs.npy', speech_preds)

print("✅ Saved speech_model_outputs.npy with shape:", speech_preds.shape)
