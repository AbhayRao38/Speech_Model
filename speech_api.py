from flask import Flask, request, jsonify
from sklearn.preprocessing._label import LabelEncoder
import torch.serialization
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import StandardScaler
import logging
import tempfile
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load speech model (LightGBM-based from your training)
try:
    # Load the best LightGBM model from your training
    speech_model = joblib.load('model_speech/best_lightgbm_model.pkl')
    scaler = joblib.load('model_speech/scaler.pkl')
    selected_indices = np.load('model_speech/selected_indices.npy')  # ✅ Added
    logging.info("Speech model (LightGBM-based), scaler, and selected feature indices loaded successfully")
except Exception as e:
    logging.error(f"Failed to load speech model or associated data: {e}")
    speech_model = None
    scaler = None
    selected_indices = None  # ✅ Added

def extract_speech_features(audio_file_path):
    """
    Extract comprehensive speech features matching your training pipeline
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_file_path, sr=22050, duration=30)
        
        if len(y) == 0:
            raise ValueError("Empty audio file")
        
        features = []
        
        # === CORE FEATURES (matching your training) ===
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.min(mfccs, axis=1),
            np.max(mfccs, axis=1)
        ])
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        features.extend([
            [np.mean(spectral_centroids), np.std(spectral_centroids)],
            [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
            [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
            [np.mean(zero_crossing_rate), np.std(zero_crossing_rate)]
        ])
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend([
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1)
        ])
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append([tempo])
        
        # RMS Energy
        rms_energy = librosa.feature.rms(y=y)[0]
        features.append([np.mean(rms_energy), np.std(rms_energy)])
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.extend([
            np.mean(spectral_contrast, axis=1),
            np.std(spectral_contrast, axis=1)
        ])
        
        # ✅ Flatten all features cleanly
        flattened_features = []
        for feature_group in features:
            if isinstance(feature_group, (list, np.ndarray)):
                flattened_features.extend(np.array(feature_group).flatten())
            else:
                flattened_features.append(feature_group)

        # ✅ Convert to numpy array and clean it
        feature_vector = np.array(flattened_features, dtype=np.float32)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)

        
        return feature_vector
        
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        raise

def create_intelligent_features(X):
    """
    Create intelligent features based on your training pipeline
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    features = []
    
    # Original features (baseline)
    features.append(X)
    
    # === STATISTICAL FEATURES ===
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
    rms_energy = np.sqrt(np.mean(X**2, axis=1, keepdims=True))
    features.append(rms_energy)
    
    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(X), axis=1) != 0, axis=1, keepdims=True)
    features.append(zero_crossings)
    
    # Spectral centroid approximation
    weighted_mean = np.sum(X * np.arange(X.shape[1]), axis=1, keepdims=True) / (np.sum(np.abs(X), axis=1, keepdims=True) + 1e-8)
    features.append(weighted_mean)
    
    # === TEMPORAL FEATURES ===
    if X.shape[1] > 2:
        first_diff = np.diff(X, axis=1)
        features.append(np.mean(first_diff, axis=1, keepdims=True))
        features.append(np.std(first_diff, axis=1, keepdims=True))
        
        if first_diff.shape[1] > 1:
            second_diff = np.diff(first_diff, axis=1)
            features.append(np.mean(second_diff, axis=1, keepdims=True))
            features.append(np.std(second_diff, axis=1, keepdims=True))
    
    # === RATIO FEATURES ===
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
    
    return X_enhanced

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': speech_model is not None,
        'scaler_loaded': scaler is not None,
        'indices_loaded': selected_indices is not None,  # ✅ Added
        'model_type': 'LightGBM-based'
    })

@app.route('/predict/speech', methods=['POST'])
def predict_speech():
    if speech_model is None or scaler is None or selected_indices is None:  # ✅ Added indices check
        return jsonify({
            'success': False,
            'error': 'Speech model, scaler, or feature indices not loaded'
        }), 500
    
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
            
        audio_file = request.files['file']
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            # Extract features
            features = extract_speech_features(temp_file_path)
            
            # Apply intelligent feature engineering (from your training)
            enhanced_features = create_intelligent_features(features.reshape(1, -1))
            
            # ✅ Apply feature selection
            selected_features = enhanced_features[:, selected_indices]
            
            # ✅ Scale features
            scaled_features = scaler.transform(selected_features)
            
            # Get prediction from LightGBM
            probabilities = speech_model.predict_proba(scaled_features)[0]
            predicted_class = np.argmax(probabilities)
            confidence = np.max(probabilities)
            
            # Emotion labels from your training
            emotion_labels = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
            predicted_emotion = emotion_labels[predicted_class] if predicted_class < len(emotion_labels) else 'Unknown'
            
            # Convert to MCI-relevant binary classification
            mci_relevant_emotions = ["Sad", "Angry", "Fearful", "Disgust"]
            mci_probability = confidence if predicted_emotion in mci_relevant_emotions else (1 - confidence)
            
            # Speech changes are strong MCI indicators, so boost if negative emotion detected
            if predicted_emotion in mci_relevant_emotions:
                mci_probability = min(mci_probability * 1.2, 1.0)
            
            # Create binary probabilities [Non-MCI, MCI]
            binary_probs = [1 - mci_probability, mci_probability]
            
            return jsonify({
                'success': True,
                'probabilities': binary_probs,
                'confidence': confidence,
                'predicted_emotion': predicted_emotion,
                'mci_probability': mci_probability,
                'all_probabilities': probabilities.tolist(),
                'emotion_labels': emotion_labels,
                'feature_count': len(scaled_features[0])
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except Exception as e:
        logging.error(f"Error in speech prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=False)
