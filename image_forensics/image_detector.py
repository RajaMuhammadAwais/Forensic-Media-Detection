# Image Forensics Module

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose
from tensorflow.keras.applications import Xception
import numpy as np
import cv2
import os

class ImageDetector:
    def __init__(self, model_type='xception'):
        self.model_type = model_type
        self.model = None

    def build_model(self, input_shape=(256, 256, 3)):
        if self.model_type == 'xception':
            # Use Xception as base model for deepfake detection
            base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
            
            # Freeze early layers, fine-tune later layers
            for layer in base_model.layers[:-20]:
                layer.trainable = False
            
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = Dense(1, activation='sigmoid')(x)
            
            self.model = Model(inputs=base_model.input, outputs=x)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
        elif self.model_type == 'autoencoder':
            # Autoencoder for anomaly detection
            input_img = Input(shape=input_shape)
            
            # Encoder
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
            x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

            # Decoder
            x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(encoded)
            x = tf.keras.layers.UpSampling2D((2, 2))(x)
            x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.UpSampling2D((2, 2))(x)
            x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.UpSampling2D((2, 2))(x)
            x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.UpSampling2D((2, 2))(x)
            decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

            self.model = Model(input_img, decoded)
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
        else:
            raise ValueError("Unsupported model type for Image Forensics.")

    def preprocess_image(self, image_path, target_size=(256, 256)):
        """Preprocess image for model input"""
        if isinstance(image_path, str):
            # Load from file path
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Assume it's already a numpy array
            img = image_path
        
        # Resize image
        img = cv2.resize(img, target_size)
        
        # Normalize pixel values
        img = img.astype(np.float32) / 255.0
        
        return img

    def detect_pixel_inconsistencies(self, image):
        """Detect pixel-level inconsistencies using gradient analysis"""
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Apply Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Detect anomalous gradients
        threshold = np.percentile(gradient_magnitude, 95)
        anomalous_pixels = gradient_magnitude > threshold
        
        # Calculate inconsistency score
        inconsistency_score = np.sum(anomalous_pixels) / (image.shape[0] * image.shape[1])
        
        return {
            'inconsistency_score': float(inconsistency_score),
            'anomalous_regions': anomalous_pixels,
            'gradient_magnitude': gradient_magnitude
        }

    def analyze_lighting_consistency(self, image):
        """Analyze lighting consistency across the image"""
        # Convert to LAB color space for better lighting analysis
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Divide image into regions
        h, w = l_channel.shape
        regions = []
        region_size = 64
        
        for i in range(0, h - region_size, region_size):
            for j in range(0, w - region_size, region_size):
                region = l_channel[i:i+region_size, j:j+region_size]
                regions.append(np.mean(region))
        
        # Calculate lighting variance
        lighting_variance = np.var(regions)
        
        # Detect inconsistent lighting (high variance indicates potential manipulation)
        inconsistency_threshold = 400  # Empirically determined
        is_inconsistent = lighting_variance > inconsistency_threshold
        
        return {
            'lighting_variance': float(lighting_variance),
            'is_inconsistent': bool(is_inconsistent),
            'confidence': min(lighting_variance / inconsistency_threshold, 1.0)
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

    def predict(self, image):
        """Make prediction on a single image"""
        if self.model is None:
            raise ValueError("Model not built or trained.")
        
        # Ensure image is in correct format
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image, verbose=0)
        return prediction

    def analyze_image(self, image_path):
        """Comprehensive image analysis"""
        # Preprocess image
        image = self.preprocess_image(image_path)
        
        # Pixel inconsistency analysis
        pixel_analysis = self.detect_pixel_inconsistencies(image)
        
        # Lighting consistency analysis
        lighting_analysis = self.analyze_lighting_consistency(image)
        
        # Model prediction (if available)
        model_prediction = None
        if self.model is not None:
            try:
                pred = self.predict(image)
                model_prediction = float(pred[0][0])
            except Exception as e:
                print(f"Model prediction failed: {e}")
                model_prediction = 0.5  # Neutral prediction
        
        # Combine analyses
        results = {
            'file_path': image_path,
            'model_prediction': model_prediction,
            'pixel_inconsistencies': {
                'score': pixel_analysis['inconsistency_score'],
                'detected': pixel_analysis['inconsistency_score'] > 0.1
            },
            'lighting_analysis': lighting_analysis,
            'overall_assessment': {
                'deepfake_probability': 0.0,
                'confidence': 0.0
            }
        }
        
        # Calculate overall assessment
        scores = []
        if model_prediction is not None:
            scores.append(model_prediction)
        scores.append(pixel_analysis['inconsistency_score'] * 2)  # Scale up
        scores.append(lighting_analysis['confidence'] if lighting_analysis['is_inconsistent'] else 0)
        
        overall_prob = np.mean(scores) if scores else 0.5
        overall_conf = 1.0 - np.std(scores) if len(scores) > 1 else 0.5
        
        results['overall_assessment']['deepfake_probability'] = float(overall_prob)
        results['overall_assessment']['confidence'] = float(overall_conf)
        
        return results

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
    # Example Usage
    print("Testing XceptionNet based Image Detector...")
    detector = ImageDetector(model_type='xception')
    detector.build_model()
    print("XceptionNet model built successfully!")
    detector.model.summary()

    print("\nTesting Autoencoder based Image Detector...")
    autoencoder = ImageDetector(model_type='autoencoder')
    autoencoder.build_model()
    print("Autoencoder model built successfully!")
    autoencoder.model.summary()

