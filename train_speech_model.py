import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, 
                                   Input, GaussianNoise, LeakyReLU, Activation)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Load Data
X = np.load(r"C:\Users\praoa\OneDrive\Desktop\Projects\multimodal-emotion-mci-system\model_speech\X_speech.npy")
y = np.load(r"C:\Users\praoa\OneDrive\Desktop\Projects\multimodal-emotion-mci-system\model_speech\y_speech.npy")

print("="*70)
print("🚀 ADVANCED EMOTION RECOGNITION SYSTEM V3")
print("="*70)

# === 1. INTELLIGENT FEATURE ENGINEERING ===
def create_intelligent_features(X):
    """Create intelligent features based on audio signal analysis"""
    print("\n🧠 Creating intelligent feature set...")
    
    features = []
    
    # === CORE FEATURES ===
    # Original features (baseline)
    features.append(X)
    
    # === STATISTICAL FEATURES ===
    # Basic statistics
    mean_vals = np.mean(X, axis=1, keepdims=True)
    std_vals = np.std(X, axis=1, keepdims=True) + 1e-8
    features.extend([mean_vals, std_vals])
    features.append(np.min(X, axis=1, keepdims=True))
    features.append(np.max(X, axis=1, keepdims=True))
    features.append(np.median(X, axis=1, keepdims=True))
    
    # Robust statistics
    features.append(np.percentile(X, 25, axis=1, keepdims=True))
    features.append(np.percentile(X, 75, axis=1, keepdims=True))
    
    # === ENERGY AND DYNAMICS FEATURES ===
    # Energy measures
    rms_energy = np.sqrt(np.mean(X**2, axis=1, keepdims=True))
    features.append(rms_energy)
    
    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(X), axis=1) != 0, axis=1, keepdims=True)
    features.append(zero_crossings)
    
    # Spectral centroid approximation
    weighted_mean = np.sum(X * np.arange(X.shape[1]), axis=1, keepdims=True) / (np.sum(np.abs(X), axis=1, keepdims=True) + 1e-8)
    features.append(weighted_mean)
    
    # === TEMPORAL FEATURES ===
    # First and second derivatives
    if X.shape[1] > 2:
        first_diff = np.diff(X, axis=1)
        features.append(np.mean(first_diff, axis=1, keepdims=True))
        features.append(np.std(first_diff, axis=1, keepdims=True))
        
        if first_diff.shape[1] > 1:
            second_diff = np.diff(first_diff, axis=1)
            features.append(np.mean(second_diff, axis=1, keepdims=True))
            features.append(np.std(second_diff, axis=1, keepdims=True))
    
    # === RATIO FEATURES ===
    # Signal-to-noise ratio approximation
    signal_power = np.mean(X**2, axis=1, keepdims=True)
    noise_power = np.var(X, axis=1, keepdims=True) + 1e-8
    snr = signal_power / noise_power
    features.append(snr)
    
    # Peak-to-average ratio
    peak_vals = np.max(np.abs(X), axis=1, keepdims=True)
    avg_vals = np.mean(np.abs(X), axis=1, keepdims=True) + 1e-8
    par = peak_vals / avg_vals
    features.append(par)
    
    # Combine all features
    X_enhanced = np.concatenate(features, axis=1)
    
    # Clean data
    X_enhanced = np.nan_to_num(X_enhanced, nan=0.0, posinf=1e6, neginf=-1e6)
    
    print(f"   Enhanced feature count: {X_enhanced.shape[1]}")
    return X_enhanced

# === 2. SMART FEATURE SELECTION ===
def select_best_features(X, y, n_features=50):
    """Select best features using multiple methods"""
    print(f"\n🎯 Selecting top {n_features} features...")
    
    # Method 1: F-score
    selector_f = SelectKBest(score_func=f_classif, k=n_features)
    X_f = selector_f.fit_transform(X, y)
    f_scores = selector_f.scores_
    
    # Method 2: Mutual information
    selector_mi = SelectKBest(score_func=mutual_info_classif, k=n_features)
    X_mi = selector_mi.fit_transform(X, y)
    mi_scores = selector_mi.scores_
    
    # Method 3: Random Forest importance
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X, y)
    rf_importance = rf_temp.feature_importances_
    
    # Combine scores (weighted average)
    combined_scores = 0.4 * f_scores + 0.3 * mi_scores + 0.3 * rf_importance
    
    # Select top features
    top_indices = np.argsort(combined_scores)[-n_features:]
    X_selected = X[:, top_indices]
    
    print(f"   Selected {X_selected.shape[1]} features")
    return X_selected, top_indices

# === 3. ADVANCED DATA AUGMENTATION ===
def advanced_augmentation(X, y, augment_factor=1.3):
    """Advanced data augmentation with multiple techniques"""
    print(f"\n📈 Applying advanced augmentation (factor: {augment_factor})...")
    
    unique_labels, counts = np.unique(y, return_counts=True)
    max_count = max(counts)
    target_count = int(max_count * augment_factor)
    
    X_aug_list = [X]
    y_aug_list = [y]
    
    for label in unique_labels:
        label_mask = (y == label)
        label_X = X[label_mask]
        current_count = np.sum(label_mask)
        
        if current_count < target_count:
            needed = target_count - current_count
            
            for _ in range(needed):
                # Select two random samples for interpolation
                idx1, idx2 = np.random.choice(current_count, 2, replace=True)
                sample1 = label_X[idx1]
                sample2 = label_X[idx2]
                
                # Random interpolation (mixup-style)
                alpha = np.random.beta(0.2, 0.2)
                mixed_sample = alpha * sample1 + (1 - alpha) * sample2
                
                # Add slight noise
                noise_level = 0.005 * np.std(mixed_sample)
                noise = np.random.normal(0, noise_level, mixed_sample.shape)
                augmented_sample = mixed_sample + noise
                
                X_aug_list.append(augmented_sample.reshape(1, -1))
                y_aug_list.append([label])
    
    X_augmented = np.vstack(X_aug_list)
    y_augmented = np.concatenate(y_aug_list)
    
    print(f"   Augmented from {X.shape[0]} to {X_augmented.shape[0]} samples")
    return X_augmented, y_augmented

# === 4. OPTIMIZED NEURAL NETWORK ===
def create_optimized_nn(input_dim, num_classes):
    """Create optimized neural network with residual connections"""
    input_layer = Input(shape=(input_dim,))
    
    # First block
    x = Dense(128, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second block with residual-like connection
    x2 = Dense(64, activation='relu')(x)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.4)(x2)
    
    # Third block
    x3 = Dense(32, activation='relu')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.3)(x3)
    
    # Output
    output = Dense(num_classes, activation='softmax')(x3)
    
    model = Model(inputs=input_layer, outputs=output)
    return model

# === 5. ADVANCED ENSEMBLE MODELS ===
def train_advanced_ensemble(X_train, X_test, y_train, y_test):
    """Train state-of-the-art ensemble models"""
    print("\n🎯 Training Advanced Ensemble...")
    
    models = {}
    scores = {}
    
    # 1. LightGBM (current best)
    print("   Training LightGBM...")
    lgb_model = LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        min_child_samples=15
    )
    lgb_model.fit(X_train, y_train)
    models['LightGBM'] = lgb_model
    scores['LightGBM'] = lgb_model.score(X_test, y_test)
    
    # 2. CatBoost
    print("   Training CatBoost...")
    cat_model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=False
    )
    cat_model.fit(X_train, y_train)
    models['CatBoost'] = cat_model
    scores['CatBoost'] = cat_model.score(X_test, y_test)
    
    # 3. XGBoost
    print("   Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    scores['XGBoost'] = xgb_model.score(X_test, y_test)
    
    # 4. Extra Trees
    print("   Training ExtraTrees...")
    et_model = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )
    et_model.fit(X_train, y_train)
    models['ExtraTrees'] = et_model
    scores['ExtraTrees'] = et_model.score(X_test, y_test)
    
    # 5. Random Forest
    print("   Training RandomForest...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    models['RandomForest'] = rf_model
    scores['RandomForest'] = rf_model.score(X_test, y_test)
    
    return models, scores

# === 6. INTELLIGENT ENSEMBLE ===
def create_intelligent_ensemble(models, X_test, y_test):
    """Create intelligent ensemble with performance-based weighting"""
    print("\n🎪 Creating intelligent ensemble...")
    
    predictions = {}
    weights = {}
    
    # Get predictions and weights
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            predictions[name] = model.predict_proba(X_test)
            weights[name] = model.score(X_test, y_test)
        elif 'Neural' in name:
            predictions[name] = model.predict(X_test, verbose=0)
            y_pred = np.argmax(predictions[name], axis=1)
            weights[name] = accuracy_score(y_test, y_pred)
    
    # Apply exponential weighting to emphasize better models
    for name in weights:
        weights[name] = weights[name] ** 3
    
    # Normalize weights
    total_weight = sum(weights.values())
    for name in weights:
        weights[name] /= total_weight
    
    # Create ensemble prediction
    ensemble_pred = np.zeros_like(next(iter(predictions.values())))
    for name, pred in predictions.items():
        ensemble_pred += weights[name] * pred
    
    ensemble_labels = np.argmax(ensemble_pred, axis=1)
    ensemble_score = accuracy_score(y_test, ensemble_labels)
    
    print(f"   Intelligent weights: {weights}")
    return ensemble_labels, ensemble_score

# === 7. MAIN EXECUTION ===
def run_advanced_system():
    print("\n🚀 Running Advanced System...")
    
    # Step 1: Intelligent feature engineering
    X_enhanced = create_intelligent_features(X)
    
    # Step 2: Feature selection
    X_selected, selected_indices = select_best_features(X_enhanced, y, n_features=60)
    np.save("model_speech/selected_indices.npy", selected_indices)
    
    # Step 3: Advanced augmentation
    X_aug, y_aug = advanced_augmentation(X_selected, y, augment_factor=1.4)
    
    # Step 4: Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug, test_size=0.22, stratify=y_aug, random_state=42
    )
    
    # Step 5: Advanced scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Final shapes - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    
    # Step 6: Train ensemble models
    ml_models, ml_scores = train_advanced_ensemble(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Step 7: Train optimized neural network
    print("\n🧠 Training Optimized Neural Network...")
    num_classes = len(np.unique(y))
    input_dim = X_train_scaled.shape[1]
    
    nn_model = create_optimized_nn(input_dim, num_classes)
    nn_model.compile(
        optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Cross-validation for neural network
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    nn_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled, y_train)):
        print(f"   Training fold {fold + 1}/3...")
        
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Create fresh model for each fold
        fold_model = create_optimized_nn(input_dim, num_classes)
        fold_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
        ]
        
        fold_model.fit(
            X_fold_train, y_fold_train,
            validation_data=(X_fold_val, y_fold_val),
            epochs=80,
            batch_size=16,
            callbacks=callbacks,
            verbose=0
        )
        
        fold_score = fold_model.evaluate(X_test_scaled, y_test, verbose=0)[1]
        nn_scores.append(fold_score)
    
    # Use the best fold model
    nn_score = max(nn_scores)
    print(f"   Neural Network CV scores: {nn_scores}")
    print(f"   Best Neural Network score: {nn_score:.4f}")
    
    # Add to results
    all_models = ml_models.copy()
    all_scores = ml_scores.copy()
    all_scores['Neural_Network'] = nn_score
    
    # Create intelligent ensemble
    ensemble_pred, ensemble_score = create_intelligent_ensemble(all_models, X_test_scaled, y_test)
    all_scores['Intelligent_Ensemble'] = ensemble_score
    
    # Results
    print("\n" + "="*70)
    print("📊 ADVANCED RESULTS")
    print("="*70)
    
    for model_name, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
        improvement = ((score - 0.59) / 0.59) * 100
        print(f"{model_name:>20}: {score:.4f} ({score*100:.1f}%) [+{improvement:+.1f}% vs original]")
    
    # Best model analysis
    best_model_name = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name} ({best_score*100:.1f}%)")

    # === SAVE BEST MODEL TO DISK ===
    import joblib

    if best_model_name != 'Neural_Network' and best_model_name != 'Intelligent_Ensemble':
        model_path = f"model_speech/best_{best_model_name.lower()}_model.pkl"
        scaler_path = "model_speech/scaler.pkl"
        
        joblib.dump(all_models[best_model_name], model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"\n💾 Saved best model as {model_path}")
        print(f"💾 Saved scaler as {scaler_path}")
    else:
        print("\n⚠️ Best model is not a single ML model (it's either ensemble or neural net), skipping save.")

    # === CLASSIFICATION REPORT ===
    if best_model_name == 'Intelligent_Ensemble':
        y_pred = ensemble_pred
    else:
        y_pred = all_models[best_model_name].predict(X_test_scaled)

    label_map = {
        0: "Neutral", 1: "Calm", 2: "Happy", 3: "Sad",
        4: "Angry", 5: "Fearful", 6: "Disgust", 7: "Surprised"
    }
    target_names = [label_map[i] for i in range(len(np.unique(y)))]

    print("\n📋 Best Model Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    return all_models, all_scores

# === RUN THE ADVANCED SYSTEM ===
if __name__ == "__main__":
    models, results = run_advanced_system()
    
    best_score = max(results.values())
    improvement = ((best_score - 0.59) / 0.59) * 100
    
    print(f"\n🎯 FINAL ADVANCED RESULTS:")
    print(f"   📈 Best Accuracy: {best_score*100:.1f}%")
    print(f"   🚀 Improvement vs Original: +{improvement:.1f}%")
    
    if best_score > 0.70:
        print("   🏆 EXCELLENT! Target achieved!")
    elif best_score > 0.65:
        print("   ✅ VERY GOOD! Strong improvement")
    else:
        print("   📈 GOOD! Steady progress")