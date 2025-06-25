import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# === 1. Load Saved Model and Scaler ===
model_path = r"model_speech/best_lightgbm_model.pkl"
scaler_path = r"model_speech/scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# === 2. Load Data ===
X = np.load("X_speech.npy")
y = np.load("y_speech.npy")

# === 3. Feature Engineering ===
def create_intelligent_features(X):
    features = []
    features.append(X)
    features.append(np.mean(X, axis=1, keepdims=True))
    features.append(np.std(X, axis=1, keepdims=True) + 1e-8)
    features.append(np.min(X, axis=1, keepdims=True))
    features.append(np.max(X, axis=1, keepdims=True))
    features.append(np.median(X, axis=1, keepdims=True))
    features.append(np.percentile(X, 25, axis=1, keepdims=True))
    features.append(np.percentile(X, 75, axis=1, keepdims=True))
    features.append(np.sqrt(np.mean(X**2, axis=1, keepdims=True)))
    features.append(np.sum(np.diff(np.sign(X), axis=1) != 0, axis=1, keepdims=True))
    weighted_mean = np.sum(X * np.arange(X.shape[1]), axis=1, keepdims=True) / (np.sum(np.abs(X), axis=1, keepdims=True) + 1e-8)
    features.append(weighted_mean)
    if X.shape[1] > 2:
        first_diff = np.diff(X, axis=1)
        features.append(np.mean(first_diff, axis=1, keepdims=True))
        features.append(np.std(first_diff, axis=1, keepdims=True))
        if first_diff.shape[1] > 1:
            second_diff = np.diff(first_diff, axis=1)
            features.append(np.mean(second_diff, axis=1, keepdims=True))
            features.append(np.std(second_diff, axis=1, keepdims=True))
    signal_power = np.mean(X**2, axis=1, keepdims=True)
    noise_power = np.var(X, axis=1, keepdims=True) + 1e-8
    snr = signal_power / noise_power
    features.append(snr)
    peak_vals = np.max(np.abs(X), axis=1, keepdims=True)
    avg_vals = np.mean(np.abs(X), axis=1, keepdims=True) + 1e-8
    par = peak_vals / avg_vals
    features.append(par)
    X_enhanced = np.concatenate(features, axis=1)
    X_enhanced = np.nan_to_num(X_enhanced, nan=0.0, posinf=1e6, neginf=-1e6)
    return X_enhanced

def select_best_features(X, y, n_features=60):
    selector_f = SelectKBest(score_func=f_classif, k=n_features)
    X_f = selector_f.fit_transform(X, y)
    f_scores = selector_f.scores_

    selector_mi = SelectKBest(score_func=mutual_info_classif, k=n_features)
    X_mi = selector_mi.fit_transform(X, y)
    mi_scores = selector_mi.scores_

    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X, y)
    rf_importance = rf_temp.feature_importances_

    combined_scores = 0.4 * f_scores + 0.3 * mi_scores + 0.3 * rf_importance
    top_indices = np.argsort(combined_scores)[-n_features:]
    X_selected = X[:, top_indices]
    return X_selected, top_indices

X_enhanced = create_intelligent_features(X)
X_selected, _ = select_best_features(X_enhanced, y, n_features=60)

# === 4. Split & Scale ===
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.22, stratify=y, random_state=42
)
X_test_scaled = scaler.transform(X_test)

# === 5. Predict & Report ===
y_pred = model.predict(X_test_scaled)
proba = model.predict_proba(X_test_scaled)

# === 🔍 6. Variance Analysis ===
# Calculate variance across class probabilities for each sample
sample_variances = np.var(proba, axis=1)
mean_var = np.mean(sample_variances)
std_var = np.std(sample_variances)

print(f"\n📊 Prediction Probability Variance:")
print(f"   Mean variance across samples: {mean_var:.6f}")
print(f"   Std of variance across samples: {std_var:.6f}")

# Optional: Plot histogram of variances
plt.figure(figsize=(7, 5))
sns.histplot(sample_variances, bins=30, kde=True, color="green")
plt.title("Distribution of Prediction Variance per Sample")
plt.xlabel("Variance of Predicted Probabilities")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("variance_histogram.png")
plt.show()

# === 7. Accuracy & Classification Report ===
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))

label_map = {
    0: "Neutral", 1: "Calm", 2: "Happy", 3: "Sad",
    4: "Angry", 5: "Fearful", 6: "Disgust", 7: "Surprised"
}
target_names = [label_map[i] for i in sorted(np.unique(y))]

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# === 8. Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap="Blues")
plt.title("Confusion Matrix - LightGBM")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_lgbm.png")
plt.show()
