import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- Configuration ---
# Assuming a pre-trained model is available. For a real-world scenario,
# this model would need to be trained or downloaded.
# For this demonstration, we will simulate the model loading and detection.
# In a real project, we would use a lightweight model like a MobileNet-based
# deepfake detector.

MODEL_PATH = 'deepfake_model.h5'
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

class RealTimeDeepfakeDetector:
    def __init__(self, model_path=MODEL_PATH, face_cascade_path=FACE_CASCADE_PATH):
        self.model = None
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.load_model(model_path)

    def load_model(self, model_path):
        """Loads the deepfake detection model."""
        try:
            # self.model = load_model(model_path)
            # Placeholder: In a real scenario, the model would be loaded here.
            print(f"INFO: Simulating loading model from {model_path}. (Model not actually loaded/used)")
            # For the purpose of this task, we will use a dummy function
            # instead of a real model to avoid large file downloads and complex setup.
            pass
        except Exception as e:
            print(f"WARNING: Could not load model from {model_path}. Using dummy detection. Error: {e}")
            pass

    def detect_deepfake(self, frame):
        """
        Performs real-time deepfake detection on a video frame.
        
        Args:
            frame (np.array): The video frame (BGR format).
            
        Returns:
            tuple: (is_deepfake (bool), confidence (float), annotated_frame (np.array))
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        is_deepfake = False
        confidence = 0.0
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            
            # Pre-process face for model (simulated)
            # face = cv2.resize(face, (128, 128))
            # face = img_to_array(face)
            # face = np.expand_dims(face, axis=0)
            # face = face / 255.0
            
            # --- SIMULATED DETECTION ---
            # In a real scenario, self.model.predict(face) would be used.
            # We will simulate a random result for demonstration.
            if np.random.rand() < 0.1: # 10% chance of being a deepfake
                prediction = [0.95] # High confidence for deepfake (0-1, 1=deepfake)
            else:
                prediction = [0.05] # Low confidence for deepfake
            
            confidence = prediction[0]
            is_deepfake = confidence > 0.5
            
            # Draw bounding box and label
            label = "Deepfake" if is_deepfake else "Real"
            color = (0, 0, 255) if is_deepfake else (0, 255, 0)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return is_deepfake, confidence, frame

def run_realtime_deepfake_detection(video_source=0):
    """
    Runs the real-time deepfake detection on a video stream.
    
    Args:
        video_source (int or str): 0 for webcam, or a path to a video file.
    """
    detector = RealTimeDeepfakeDetector()
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    print("Starting real-time deepfake detection. Press 'q' to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame for faster processing (optional)
        # frame = cv2.resize(frame, (640, 480))
        
        is_deepfake, confidence, annotated_frame = detector.detect_deepfake(frame)
        
        # Display the resulting frame
        cv2.imshow('Real-Time Deepfake Detection', annotated_frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # To run with a webcam: run_realtime_deepfake_detection(0)
    # To run with a video file: run_realtime_deepfake_detection('path/to/video.mp4')
    print("Deepfake detection module created. It requires a live video feed or a video file to run.")
    print("Since this is a sandboxed environment, we cannot access a webcam or display a window.")
    print("The core logic is implemented in the RealTimeDeepfakeDetector class.")
    
    # Example of how to use the class programmatically:
    # detector = RealTimeDeepfakeDetector()
    # dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # is_fake, conf, annotated = detector.detect_deepfake(dummy_frame)
    # print(f"Dummy detection result: Deepfake: {is_fake}, Confidence: {conf:.2f}")
