import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import pandas as pd

# Load Data
X = np.load(r"C:\Users\praoa\OneDrive\Desktop\Projects\multimodal-emotion-mci-system\model_speech\X_speech.npy")
y = np.load(r"C:\Users\praoa\OneDrive\Desktop\Projects\multimodal-emotion-mci-system\model_speech\y_speech.npy")

print("="*60)
print("🔍 COMPREHENSIVE DATASET ANALYSIS")
print("="*60)

# === 1. BASIC DATA EXPLORATION ===
print("\n1️⃣ BASIC DATA EXPLORATION")
print("-" * 30)
print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Classes: {np.unique(y)}")

# === 2. DATA QUALITY CHECKS ===
print("\n2️⃣ DATA QUALITY CHECKS")
print("-" * 30)

# Check for missing values
nan_count = np.isnan(X).sum()
inf_count = np.isinf(X).sum()
print(f"NaN values: {nan_count}")
print(f"Infinite values: {inf_count}")

# Check for constant features
constant_features = []
for i in range(X.shape[1]):
    if np.std(X[:, i]) == 0:
        constant_features.append(i)
print(f"Constant features (std=0): {len(constant_features)} -> {constant_features}")

# Check feature ranges
print(f"Min value across all features: {X.min():.4f}")
print(f"Max value across all features: {X.max():.4f}")
print(f"Mean value across all features: {X.mean():.4f}")
print(f"Std value across all features: {X.std():.4f}")

# === 3. FEATURE STATISTICS ===
print("\n3️⃣ FEATURE STATISTICS")
print("-" * 30)

# Calculate statistics for each feature
feature_stats = []
for i in range(X.shape[1]):
    stats_dict = {
        'feature': i,
        'mean': X[:, i].mean(),
        'std': X[:, i].std(),
        'min': X[:, i].min(),
        'max': X[:, i].max(),
        'range': X[:, i].max() - X[:, i].min(),
        'skewness': stats.skew(X[:, i]),
        'kurtosis': stats.kurtosis(X[:, i])
    }
    feature_stats.append(stats_dict)

feature_df = pd.DataFrame(feature_stats)
print("Feature statistics summary:")
print(feature_df.describe())

# === 4. CLASS DISTRIBUTION ANALYSIS ===
print("\n4️⃣ CLASS DISTRIBUTION ANALYSIS")
print("-" * 30)

unique_labels, counts = np.unique(y, return_counts=True)
label_map = {
    0: "Neutral", 1: "Calm", 2: "Happy", 3: "Sad",
    4: "Angry", 5: "Fearful", 6: "Disgust", 7: "Surprised"
}

for label, count in zip(unique_labels, counts):
    percentage = (count / len(y)) * 100
    emotion = label_map.get(label, f"Class_{label}")
    print(f"{emotion:>10}: {count:>3} samples ({percentage:>5.1f}%)")

# Calculate class imbalance ratio
min_samples = counts.min()
max_samples = counts.max()
imbalance_ratio = max_samples / min_samples
print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")

# === 5. FEATURE SCALING ANALYSIS ===
print("\n5️⃣ FEATURE SCALING ANALYSIS")
print("-" * 30)

# Check if features are on similar scales
feature_ranges = X.max(axis=0) - X.min(axis=0)
print(f"Feature range variance: {np.var(feature_ranges):.4f}")
print(f"Largest feature range: {feature_ranges.max():.4f}")
print(f"Smallest feature range: {feature_ranges.min():.4f}")
print(f"Range ratio (max/min): {feature_ranges.max()/feature_ranges.min():.2f}")

# === 6. CORRELATION ANALYSIS ===
print("\n6️⃣ CORRELATION ANALYSIS")
print("-" * 30)

# Calculate correlation matrix
corr_matrix = np.corrcoef(X.T)
# Count high correlations (>0.8)
high_corr_pairs = 0
for i in range(len(corr_matrix)):
    for j in range(i+1, len(corr_matrix)):
        if abs(corr_matrix[i, j]) > 0.8:
            high_corr_pairs += 1

print(f"Highly correlated feature pairs (|r| > 0.8): {high_corr_pairs}")
print(f"Average absolute correlation: {np.abs(corr_matrix).mean():.4f}")

# === 7. BASELINE MODEL PERFORMANCE ===
print("\n7️⃣ BASELINE MODEL PERFORMANCE")
print("-" * 30)

# Quick baseline with Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Test different preprocessing
results = {}

# Raw data
rf_raw = RandomForestClassifier(n_estimators=100, random_state=42)
rf_raw.fit(X_train, y_train)
raw_score = rf_raw.score(X_test, y_test)
results['Raw Data'] = raw_score

# Standardized data
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)
rf_std = RandomForestClassifier(n_estimators=100, random_state=42)
rf_std.fit(X_train_std, y_train)
std_score = rf_std.score(X_test_std, y_test)
results['StandardScaler'] = std_score

# MinMax scaled data
scaler_mm = MinMaxScaler()
X_train_mm = scaler_mm.fit_transform(X_train)
X_test_mm = scaler_mm.transform(X_test)
rf_mm = RandomForestClassifier(n_estimators=100, random_state=42)
rf_mm.fit(X_train_mm, y_train)
mm_score = rf_mm.score(X_test_mm, y_test)
results['MinMaxScaler'] = mm_score

print("Random Forest Baseline Results:")
for method, score in results.items():
    print(f"{method:>15}: {score:.4f} ({score*100:.1f}%)")

# === 8. FEATURE IMPORTANCE ===
print("\n8️⃣ FEATURE IMPORTANCE")
print("-" * 30)

# Get feature importance from best performing baseline
best_method = max(results, key=results.get)
if best_method == 'Raw Data':
    feature_importance = rf_raw.feature_importances_
elif best_method == 'StandardScaler':
    feature_importance = rf_std.feature_importances_
else:
    feature_importance = rf_mm.feature_importances_

# Top 10 most important features
top_features = np.argsort(feature_importance)[::-1][:10]
print(f"Top 10 most important features (using {best_method}):")
for i, feat_idx in enumerate(top_features):
    print(f"  {i+1:>2}. Feature {feat_idx:>2}: {feature_importance[feat_idx]:.4f}")

# === 9. DATA SEPARABILITY ===
print("\n9️⃣ DATA SEPARABILITY ANALYSIS")
print("-" * 30)

# Use best preprocessing for PCA
if best_method == 'StandardScaler':
    X_processed = X_train_std
elif best_method == 'MinMaxScaler':
    X_processed = X_train_mm
else:
    X_processed = X_train

# PCA analysis
pca = PCA()
pca.fit(X_processed)
cumsum_variance = np.cumsum(pca.explained_variance_ratio_)

# Find components needed for 95% variance
components_95 = np.argmax(cumsum_variance >= 0.95) + 1
components_90 = np.argmax(cumsum_variance >= 0.90) + 1

print(f"Components needed for 90% variance: {components_90}")
print(f"Components needed for 95% variance: {components_95}")
print(f"First component explains: {pca.explained_variance_ratio_[0]:.3f} of variance")
print(f"First 3 components explain: {cumsum_variance[2]:.3f} of variance")

# === 10. RECOMMENDATIONS ===
print("\n" + "="*60)
print("🎯 RECOMMENDATIONS")
print("="*60)

recommendations = []

# Data quality recommendations
if nan_count > 0 or inf_count > 0:
    recommendations.append("⚠️  Clean NaN/Infinite values before training")

if len(constant_features) > 0:
    recommendations.append(f"🗑️  Remove {len(constant_features)} constant features")

# Scaling recommendations
if feature_ranges.max()/feature_ranges.min() > 10:
    best_baseline = max(results, key=results.get)
    recommendations.append(f"📏 Use {best_baseline} (improved baseline by {(results[best_baseline] - results['Raw Data'])*100:.1f}%)")

# Class imbalance recommendations
if imbalance_ratio > 2:
    recommendations.append(f"⚖️  Address class imbalance (ratio: {imbalance_ratio:.2f})")

# Model architecture recommendations
if results[best_method] > 0.75:
    recommendations.append("✅ Good feature separability - Deep learning should work well")
elif results[best_method] > 0.6:
    recommendations.append("⚠️  Moderate separability - Try feature engineering or ensemble methods")
else:
    recommendations.append("❌ Poor separability - Consider feature engineering or different features")

# Dimensionality recommendations
if components_95 < 20:
    recommendations.append(f"📉 Consider PCA: {components_95} components retain 95% variance")

# CNN specific recommendations
if X.shape[1] < 50:
    recommendations.append("🔄 For CNN: Consider data augmentation or different architecture")

print("\nKey Actions:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")

print(f"\n💡 Your baseline Random Forest achieved {results[best_method]*100:.1f}% vs your CNN's 59%")
print("   This suggests the CNN architecture needs optimization!")

# === 11. NEXT STEPS CODE ===
print("\n" + "="*60)
print("📝 NEXT STEPS - RUN THIS ANALYSIS")
print("="*60)
print("""
Based on the results above, I'll provide you with:
1. Optimized preprocessing pipeline
2. Improved CNN architecture 
3. Better training strategies
4. Alternative model approaches

Please share the output of this analysis!
""")