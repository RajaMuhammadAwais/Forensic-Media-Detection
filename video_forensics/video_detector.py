# Video Forensics Module

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, TimeDistributed, Flatten
from tensorflow.keras.applications import Xception # Using Xception for frame feature extraction
import cv2
import numpy as np
import os

class VideoDetector:
    def __init__(self, model_type='cnn_lstm'):
        self.model_type = model_type
        self.model = None

    def build_model(self, input_shape=(None, 256, 256, 3)):
        """
        Build CNN-LSTM model for video analysis.
        input_shape: (timesteps, height, width, channels). Timesteps can be None for variable length.
        """
        if self.model_type == 'cnn_lstm':
            # Input for sequences of frames
            video_input = Input(shape=input_shape)
            
            # Use a pre-trained CNN (Xception) for spatial feature extraction from each frame
            # We need to wrap Xception in TimeDistributed to apply it to each frame in the sequence
            # Set include_top=False to remove the classification head
            # Use weights="imagenet" for pre-trained weights
            base_cnn = Xception(weights='imagenet', include_top=False, pooling='avg')
            
            # Freeze the layers of the pre-trained CNN to use it as a fixed feature extractor
            for layer in base_cnn.layers:
                layer.trainable = False
            
            # Apply the CNN to each frame in the video sequence
            cnn_features = TimeDistributed(base_cnn)(video_input)
            
            # LSTM layers for temporal analysis
            # return_sequences=True to stack LSTM layers
            lstm_out = LSTM(256, return_sequences=True)(cnn_features)
            lstm_out = tf.keras.layers.Dropout(0.3)(lstm_out)
            lstm_out = LSTM(128)(lstm_out)
            lstm_out = tf.keras.layers.Dropout(0.3)(lstm_out)
            
            # Final classification layer
            output = Dense(1, activation="sigmoid")(lstm_out)

            self.model = Model(inputs=video_input, outputs=output)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss="binary_crossentropy",
                metrics=["accuracy", "precision", "recall"]
            )
        else:
            raise ValueError("Unsupported model type for Video Forensics.")
    def extract_frames(self, video_path, num_frames=30, target_size=(256, 256)):
        """Extract frames from video for analysis and preprocess them."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")
            
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return np.array([])

        # Extract evenly spaced frames
        # Ensure we don't try to extract more frames than available
        if num_frames > total_frames:
            num_frames = total_frames
            
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, target_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                print(f"Warning: Could not read frame at index {idx}")
        
        cap.release()
        
        if not frames:
            return np.array([])
            
        # Normalize pixel values to [0, 1]
        frames = np.array(frames).astype(np.float32) / 255.0
        
        return frames

    def analyze_lip_sync(self, video_path, audio_path):
        """Placeholder for lip-sync analysis using external tools or more advanced models.
        This would typically involve: 
        1. Speech-to-Text on audio.
        2. Facial landmark detection and lip movement analysis on video frames.
        3. Comparing timing and synchronization using DTW or HMMs.
        """
        print(f"Performing simulated lip-sync analysis for video: {video_path} and audio: {audio_path}")
        # Simulate results for now
        sync_score = np.random.uniform(0.5, 1.0) # Higher score means better sync
        mismatch_detected = sync_score < 0.75 # Example threshold
        
        return {
            "sync_score": float(sync_score),
            "mismatch_detected": bool(mismatch_detected),
            "confidence": float(1.0 - sync_score) if mismatch_detected else float(sync_score),
            "mismatch_timestamps": ["00:45", "01:20"] if mismatch_detected else []
        }

    def train(self, train_dataset, validation_dataset=None, epochs=10, batch_size=4):
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

    def predict(self, video_frames):
        """Make prediction on a sequence of video frames"""
        if self.model is None:
            raise ValueError("Model not built or trained.")
        
        # Ensure input is batched (add batch dimension if missing)
        if len(video_frames.shape) == 4: # (num_frames, height, width, channels)
            video_frames = np.expand_dims(video_frames, axis=0) # Add batch dimension
            
        prediction = self.model.predict(video_frames, verbose=0)
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

if __name__ == '__main__':
    print("Testing CNN-LSTM based Video Detector...")
    detector = VideoDetector(model_type="cnn_lstm")
    detector.build_model()
    detector.model.summary()
    
    # Example of extracting frames (requires a dummy video file)
    # For testing, you might need to create a dummy video or use a real one
    # dummy_video_path = "dummy_video.mp4"
    # if os.path.exists(dummy_video_path):
    #     print(f"\nExtracting frames from {dummy_video_path}...")
    #     frames = detector.extract_frames(dummy_video_path)
    #     print(f"Extracted {len(frames)} frames with shape {frames.shape}")
    # else:
    #     print(f"\nSkipping frame extraction test: {dummy_video_path} not found.")
    
    # Example of lip-sync analysis (simulated)
    # print("\nTesting simulated lip-sync analysis...")
    # result = detector.analyze_lip_sync("dummy_video.mp4", "dummy_audio.wav")
    # print(result)


