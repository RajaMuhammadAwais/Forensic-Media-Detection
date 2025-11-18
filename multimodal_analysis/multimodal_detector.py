# Multimodal Analysis Module

import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from image_forensics.image_detector import ImageDetector
from video_forensics.video_detector import VideoDetector
from audio_forensics.audio_detector import AudioDetector
from multimodal_analysis.audio_visual_verifier import AudioVisualVerifier

class MultimodalDetector:
    def __init__(self, fusion_type="late"):
        self.fusion_type = fusion_type
        self.image_detector = ImageDetector()
        self.video_detector = VideoDetector()
        self.audio_detector = AudioDetector()
        self.av_verifier = AudioVisualVerifier()
        self.fusion_model = None

    def build_fusion_model(self):
        """Build multimodal fusion model"""
        # Import TensorFlow/Keras only when building the fusion model
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout

        if self.fusion_type == "late":
            # Late fusion: combine predictions from individual models
            image_input = Input(shape=(1,), name="image_prediction")
            video_input = Input(shape=(1,), name="video_prediction")
            audio_input = Input(shape=(1,), name="audio_prediction")
            
            # Combine predictions
            combined = Concatenate()([image_input, video_input, audio_input])
            
            # Fusion layers
            x = Dense(64, activation="relu")(combined)
            x = Dropout(0.3)(x)
            x = Dense(32, activation="relu")(x)
            x = Dropout(0.3)(x)
            
            # Final prediction
            output = Dense(1, activation="sigmoid")(x)
            
            self.fusion_model = Model(inputs=[image_input, video_input, audio_input], 
                                    outputs=output)
            self.fusion_model.compile(optimizer="adam", loss="binary_crossentropy", 
                                    metrics=["accuracy"])
            
        elif self.fusion_type == "early":
            # Early fusion: This would require more complex integration at the feature level
            # For now, we focus on late fusion as per the project plan.
            raise NotImplementedError("Early fusion is not yet implemented.")
        else:
            raise ValueError("Unsupported fusion type.")

    def analyze_cross_modal_consistency(self, image_pred, video_pred, audio_pred):
        """Analyze consistency across different modalities"""
        predictions = np.array([image_pred, video_pred, audio_pred])
        
        # Calculate variance across modalities
        consistency_score = 1.0 - np.var(predictions)  # Higher score means more consistent
        
        # Check for outliers (predictions significantly different from the mean)
        mean_pred = np.mean(predictions)
        # A simple heuristic for outlier detection: if a prediction is more than 0.3 away from the mean
        outliers = np.abs(predictions - mean_pred) > 0.3
        
        return {
            "consistency_score": float(consistency_score),
            "mean_prediction": float(mean_pred),
            "outlier_modalities": outliers.tolist(),
            "individual_predictions": {
                "image": float(image_pred),
                "video": float(video_pred),
                "audio": float(audio_pred)
            }
        }

    def ensemble_predict(self, image_pred, video_pred, audio_pred, weights=None):
        """Ensemble prediction with optional weights"""
        if weights is None:
            weights = [1/3, 1/3, 1/3]  # Equal weights
        
        predictions = np.array([image_pred, video_pred, audio_pred])
        weights = np.array(weights)
        
        # Weighted average
        ensemble_pred = np.sum(predictions * weights)
        
        # Confidence based on agreement (lower std dev means higher agreement/confidence)
        agreement = 1.0 - np.std(predictions)
        
        return {
            "ensemble_prediction": float(ensemble_pred),
            "confidence": float(agreement),
            "weights_used": weights.tolist()
        }

    def comprehensive_analysis(self, media_path):
        """Perform comprehensive multimodal analysis on a given media file."""
        results = {}
        
        # Initialize predictions to neutral (0.5) if a modality is not applicable or cannot be processed
        image_pred = 0.5
        video_pred = 0.5
        audio_pred = 0.5

        file_ext = os.path.splitext(media_path)[1].lower()

        # Image analysis
        if file_ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            try:
                # For simplicity, we'll use the ImageDetector's analyze_image method
                # In a real scenario, this would involve loading a trained model
                img_analysis_results = self.image_detector.analyze_image(media_path)
                image_pred = img_analysis_results["overall_assessment"]["deepfake_probability"]
                results["image_analysis"] = img_analysis_results
            except Exception as e:
                print(f"Error during image analysis: {e}")
                results["image_analysis"] = {"error": str(e)}
        else:
            results["image_analysis"] = {"status": "Not applicable for this file type"}

        # Video analysis (if applicable)
        if file_ext in [".mp4", ".avi", ".mov", ".mkv"]:
            try:
                # Extract frames and predict
                video_frames = self.video_detector.extract_frames(media_path)
                if video_frames.size > 0:
                    # Need to load a trained video model here for actual prediction
                    # For now, simulate prediction
                    video_pred = np.random.uniform(0, 1)
                    results["video_analysis"] = {
                        "deepfake_probability": float(video_pred),
                        "lip_sync_mismatch": self.video_detector.analyze_lip_sync(media_path, "dummy_audio.wav"),
                        "audio_visual_consistency": self.av_verifier.verify_consistency(media_path, media_path)
                    }
                else:
                    # Still provide audio-visual consistency structure even if frames were not extracted
                    results["video_analysis"] = {
                        "status": "No frames extracted",
                        "audio_visual_consistency": self.av_verifier.verify_consistency(media_path, media_path)
                    }
            except Exception as e:
                print(f"Error during video analysis: {e}")
                results["video_analysis"] = {"error": str(e)}
        else:
            results["video_analysis"] = {"status": "Not applicable for this file type"}

        # Audio analysis (if applicable)
        if file_ext in [".wav", ".mp3", ".flac", ".mp4", ".avi", ".mov", ".mkv"]:
            try:
                # Extract audio features and predict
                audio_features = self.audio_detector.extract_features(media_path)
                if audio_features.size > 0:
                    # Need to load a trained audio model here for actual prediction
                    # For now, simulate prediction
                    audio_pred = np.random.uniform(0, 1)
                    results["audio_analysis"] = {
                        "deepfake_probability": float(audio_pred),
                        "spectral_anomalies": self.audio_detector.analyze_spectral_anomalies(media_path),
                        "voice_synthesis_detection": self.audio_detector.detect_voice_synthesis(media_path)
                    }
                else:
                    results["audio_analysis"] = {"status": "No audio features extracted"}
            except Exception as e:
                print(f"Error during audio analysis: {e}")
                results["audio_analysis"] = {"error": str(e)}
        else:
            results["audio_analysis"] = {"status": "Not applicable for this file type"}

        # Cross-modal consistency analysis
        consistency = self.analyze_cross_modal_consistency(image_pred, video_pred, audio_pred)
        results["cross_modal_consistency"] = consistency
        
        # Ensemble prediction
        ensemble = self.ensemble_predict(image_pred, video_pred, audio_pred)
        results["ensemble_prediction"] = ensemble
        
        # Overall assessment
        overall_confidence = (consistency["consistency_score"] + ensemble["confidence"]) / 2
        results["overall_assessment"] = {
            "deepfake_probability": ensemble["ensemble_prediction"],
            "confidence": overall_confidence,
            "recommendation": "DEEPFAKE DETECTED" if ensemble["ensemble_prediction"] > 0.7 else 
                           "SUSPICIOUS" if ensemble["ensemble_prediction"] > 0.3 else "LIKELY AUTHENTIC"
        }
        
        return results

    def train_fusion_model(self, dataset, epochs=10, batch_size=32):
        """Train the fusion model"""
        if self.fusion_model is None:
            self.build_fusion_model()
        
        self.fusion_model.fit(dataset, epochs=epochs, batch_size=batch_size)

    def save_models(self, base_path):
        """Save all models"""
        self.image_detector.save_model(f"{base_path}_image.h5")
        self.video_detector.save_model(f"{base_path}_video.h5")
        self.audio_detector.save_model(f"{base_path}_audio.h5")
        if self.fusion_model:
            self.fusion_model.save(f"{base_path}_fusion.h5")

    def load_models(self, base_path):
        """Load all models"""
        self.image_detector.load_model(f"{base_path}_image.h5")
        self.video_detector.load_model(f"{base_path}_video.h5")
        self.audio_detector.load_model(f"{base_path}_audio.h5")
        if self.fusion_model:
            import tensorflow as tf
            self.fusion_model = tf.keras.models.load_model(f"{base_path}_fusion.h5")

if __name__ == "__main__":
    print("Testing Multimodal Detector...")
    detector = MultimodalDetector(fusion_type="late")
    detector.build_fusion_model()
    detector.fusion_model.summary()

    # Example of comprehensive analysis (requires dummy media files)
    # print("\nTesting comprehensive analysis...")
    # dummy_image_path = "dummy_image.jpg"
    # dummy_video_path = "dummy_video.mp4"
    # dummy_audio_path = "dummy_audio.wav"

    # Create dummy files for testing if they don't exist
    # if not os.path.exists(dummy_image_path):
    #     from PIL import Image
    #     Image.new("RGB", (256, 256), color = "red").save(dummy_image_path)
    # if not os.path.exists(dummy_video_path):
    #     # This is more complex, requires a video library like moviepy or ffmpeg
    #     # For now, just create a placeholder file
    #     with open(dummy_video_path, "w") as f: f.write("dummy video content")
    # if not os.path.exists(dummy_audio_path):
    #     # Requires an audio library like pydub or soundfile
    #     with open(dummy_audio_path, "w") as f: f.write("dummy audio content")

    # if os.path.exists(dummy_image_path):
    #     print(f"\nAnalyzing {dummy_image_path}...")
    #     results = detector.comprehensive_analysis(dummy_image_path)
    #     print(json.dumps(results, indent=2))

    # if os.path.exists(dummy_video_path):
    #     print(f"\nAnalyzing {dummy_video_path}...")
    #     results = detector.comprehensive_analysis(dummy_video_path)
    #     print(json.dumps(results, indent=2))

    # if os.path.exists(dummy_audio_path):
    #     print(f"\nAnalyzing {dummy_audio_path}...")
    #     results = detector.comprehensive_analysis(dummy_audio_path)
    #     print(json.dumps(results, indent=2))


