import numpy as np

probs = np.load("speech_model_outputs.npy")  # replace with actual file path
print(probs.shape)
print(probs[:5])  # print first 5 predictions for preview
