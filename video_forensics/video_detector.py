# Video Forensics Module

# Lazy-import heavy ML frameworks to avoid unnecessary CI dependencies
import cv2
import numpy as np
import os
import glob

class VideoDetector:
    def __init__(self, model_type='cnn_lstm', use_wavelet_preprocessor=False):
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        if use_wavelet_preprocessor:
            try:
                from .wavelet_preprocessing import WaveletPreprocessor
                self.preprocessor = WaveletPreprocessor()
            except Exception:
                # Fallback: leave preprocessor as None if import fails
                self.preprocessor = None

    def build_model(self, input_shape=(None, 256, 256, 3)):
        """
        Build CNN-LSTM model for video analysis.
        input_shape: (timesteps, height, width, channels). Timesteps can be None for variable length.
        """
        if self.model_type == 'cnn_lstm':
            # Import TensorFlow/Keras only when building the model
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed
            from tensorflow.keras.applications import Xception  # Using Xception for frame feature extraction

            # Input for sequences of frames
            video_input = Input(shape=input_shape)
            
            # Use a pre-trained CNN (Xception) for spatial feature extraction from each frame
            base_cnn = Xception(weights='imagenet', include_top=False, pooling='avg')
            
            # Freeze the layers of the pre-trained CNN to use it as a fixed feature extractor
            for layer in base_cnn.layers:
                layer.trainable = False
            
            # Apply the CNN to each frame in the video sequence
            cnn_features = TimeDistributed(base_cnn)(video_input)
            
            # LSTM layers for temporal analysis
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
        """Extract frames from video for analysis and preprocess them.

        This function attempts several fallbacks to locate the provided video file so
        tests and CI running from different working directories can still find it.
        """
        # Resolve video path with fallbacks to support different test/workdir setups
        candidates = []
        if os.path.exists(video_path):
            resolved_path = video_path
        else:
            candidates.append(video_path)
            # Try path relative to current working directory
            cwd_path = os.path.join(os.getcwd(), video_path)
            candidates.append(cwd_path)

            # Try path relative to the package directory (repo_root/data/<basename>)
            repo_root_data = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            pkg_data_path = os.path.join(repo_root_data, os.path.basename(video_path))
            candidates.append(pkg_data_path)

            # Search by basename anywhere under cwd
            matches = glob.glob(os.path.join(os.getcwd(), '**', os.path.basename(video_path)), recursive=True)
            candidates.extend(matches)

            resolved_path = None
            for c in candidates:
                if os.path.exists(c):
                    resolved_path = c
                    break

            if resolved_path is None:
                tried_str = '\n'.join(candidates)
                raise FileNotFoundError(f"Video file not found: {video_path}. Tried:\n{tried_str}")

        # Open video capture
        cap = cv2.VideoCapture(resolved_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {resolved_path}")

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return np.array([])

        # Extract evenly spaced frames
        if num_frames > total_frames:
            num_frames = total_frames

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
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
        sync_score = np.random.uniform(0.5, 1.0)  # Higher score means better sync
        mismatch_detected = sync_score < 0.75  # Example threshold
        
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
        
        import tensorflow as tf
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
        if len(video_frames.shape) == 4:  # (num_frames, height, width, channels)
            video_frames = np.expand_dims(video_frames, axis=0)  # Add batch dimension
            
        prediction = self.model.predict(video_frames, verbose=0)
        return prediction

    def save_model(self, path):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(path)

    def load_model(self, path):
        """Load a trained model"""
        import tensorflow as tf
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


