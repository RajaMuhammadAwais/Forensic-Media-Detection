# Audio Forensics Module

import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, GlobalMaxPooling1D, Dropout
from sklearn.preprocessing import StandardScaler
import os

class AudioDetector:
    def __init__(self, model_type="xvector"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()

    def extract_features(self, audio_path, sr=16000, n_mfcc=20):
        """Extract audio features (MFCCs) for analysis"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Transpose to have time as first dimension and normalize
        mfccs = mfccs.T
        
        # Scale features (important for x-vectors)
        if self.scaler.mean_ is None:
            self.scaler.fit(mfccs)
        mfccs = self.scaler.transform(mfccs)
        
        return mfccs

    def build_model(self, input_shape=(None, 20)):  # (timesteps, n_mfcc)
        """Build x-vector inspired model for speaker verification/deepfake detection"""
        if self.model_type == "xvector":
            input_layer = Input(shape=input_shape)
            
            # Frame-level layers
            x = Conv1D(512, 5, activation="relu", padding="same")(input_layer)
            x = Conv1D(512, 3, activation="relu", padding="same")(x)
            x = Conv1D(512, 3, activation="relu", padding="same")(x)
            
            # Statistics pooling (mean and std deviation)
            mean = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=1))(x)
            std = tf.keras.layers.Lambda(lambda x: tf.keras.backend.std(x, axis=1))(x)
            stats = tf.keras.layers.Concatenate()([mean, std])
            
            # Segment-level layers
            x = Dense(512, activation="relu")(stats)
            x = Dropout(0.3)(x)
            x = Dense(512, activation="relu")(x)
            x = Dropout(0.3)(x)
            
            # Output layer for binary classification (real vs synthetic)
            output = Dense(1, activation="sigmoid")(x)
            
            self.model = Model(inputs=input_layer, outputs=output)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss="binary_crossentropy",
                metrics=["accuracy", "precision", "recall"]
            )
            
        elif self.model_type == "cnn_lstm":
            input_layer = Input(shape=input_shape)
            
            # CNN layers for feature extraction
            x = Conv1D(64, 3, activation="relu", padding="same")(input_layer)
            x = Conv1D(128, 3, activation="relu", padding="same")(x)
            x = Conv1D(256, 3, activation="relu", padding="same")(x)
            
            # LSTM layers for temporal modeling
            x = LSTM(128, return_sequences=True)(x)
            x = LSTM(64)(x)
            
            # Output layer
            output = Dense(1, activation="sigmoid")(x)
            
            self.model = Model(inputs=input_layer, outputs=output)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss="binary_crossentropy",
                metrics=["accuracy", "precision", "recall"]
            )
        else:
            raise ValueError("Unsupported model type for Audio Forensics.")

    def analyze_spectral_anomalies(self, audio_path, sr=16000):
        """Analyze spectral anomalies in audio using Fourier Transform."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Compute Short-Time Fourier Transform (STFT)
        D = librosa.stft(y)
        
        # Convert to decibels for better visualization and analysis
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Analyze frequency distribution (mean and standard deviation across frequency bins)
        freq_mean = np.mean(D_db, axis=1)
        freq_std = np.std(D_db, axis=1)
        
        # Simple anomaly detection based on spectral flatness
        # Spectral flatness measures how noisy or tonal a sound is. 
        # Synthetic speech might have unusually high or low flatness.
        flatness = librosa.feature.spectral_flatness(y=y)
        flatness_mean = np.mean(flatness)
        flatness_std = np.std(flatness)
        
        # Heuristic for anomaly: high variance in frequency or unusual flatness
        anomaly_score = (np.mean(freq_std) + np.abs(flatness_mean - 0.5)) / 2.0 # Normalize to 0-1
        
        return {
            "anomaly_score": float(anomaly_score),
            "spectral_mean_avg": float(np.mean(freq_mean)),
            "spectral_std_avg": float(np.mean(freq_std)),
            "spectral_flatness_mean": float(flatness_mean),
            "spectral_flatness_std": float(flatness_std)
        }

    def detect_voice_synthesis(self, audio_path):
        """Detect synthetic voice characteristics using MFCC variance and spectral analysis."""
        features = self.extract_features(audio_path)
        
        # Analyze MFCC variance: synthetic voices often have lower variance
        mfcc_variance = np.var(features, axis=0)
        avg_mfcc_variance = np.mean(mfcc_variance)
        
        # Combine with spectral flatness from analyze_spectral_anomalies
        spectral_analysis = self.analyze_spectral_anomalies(audio_path)
        
        # Simple heuristic for synthesis detection
        # Lower MFCC variance and unusual spectral flatness could indicate synthesis
        synthesis_confidence = 1.0 - avg_mfcc_variance # Lower variance -> higher confidence
        synthesis_confidence = (synthesis_confidence + spectral_analysis["anomaly_score"]) / 2.0
        
        return {
            "synthesis_confidence": float(synthesis_confidence),
            "avg_mfcc_variance": float(avg_mfcc_variance),
            "spectral_flatness_mean": spectral_analysis["spectral_flatness_mean"]
        }

    def train(self, train_dataset, validation_dataset=None, epochs=10, batch_size=32):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return history

    def predict(self, audio_features):
        """Make prediction on audio features"""
        if self.model is None:
            raise ValueError("Model not built or trained.")
        
        # Ensure input is batched (add batch dimension if missing)
        if len(audio_features.shape) == 2: # (timesteps, n_mfcc)
            audio_features = np.expand_dims(audio_features, axis=0) # Add batch dimension
            
        prediction = self.model.predict(audio_features, verbose=0)
        return prediction

    def save_model(self, path):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(path)

    def load_model(self, path):
        """Load a trained model"""
        if not os.path.exists(path):
            raise ValueError(f"Model file not found: {path}")
        self.model = tf.keras.models.load_model(path)

if __name__ == "__main__":
    print("Testing X-Vector based Audio Detector...")
    detector = AudioDetector(model_type="xvector")
    detector.build_model()
    detector.model.summary()
    
    print("\nTesting CNN-LSTM based Audio Detector...")
    detector_cnn = AudioDetector(model_type="cnn_lstm")
    detector_cnn.build_model()
    detector_cnn.model.summary()

