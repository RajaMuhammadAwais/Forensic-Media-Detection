import os
import sys
import tensorflow as tf
from pathlib import Path

# Add the repository root to the path to import modules
sys.path.append(str(Path(__file__).parent))

from image_forensics.image_detector import ImageDetector

# --- Configuration ---
REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data" / "Dataset"
MODEL_PATH = REPO_ROOT / "xception_deepfake_weights.h5"
BATCH_SIZE = 16
EPOCHS = 3 # Use a small number of epochs for a proof-of-concept

# The ImageDetector is configured for 256x256 images
# Set a smaller subset size for resource-constrained environment
SUBSET_SIZE = 5000 # Use 5,000 images for training and 1,000 for validation

IMG_SIZE = (128, 128)

def load_dataset(subset_name):
    """Loads the image dataset using Keras utility."""
    print(f"Loading {subset_name} dataset from {DATA_DIR / subset_name}...")
    return tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / subset_name,
        labels='inferred',
        label_mode='binary',
        image_size=IMG_SIZE,
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        shuffle=True if subset_name == "Train" else False,
        seed=42
    )

def main():
    # 1. Load Data
    # The dataset is large, so we will only take a small number of batches for a quick POC
    STEPS_PER_EPOCH = SUBSET_SIZE // BATCH_SIZE
    VALIDATION_STEPS = (SUBSET_SIZE // 5) // BATCH_SIZE

    try:
        train_ds = load_dataset("Train")
        val_ds = load_dataset("Validation")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset is downloaded and extracted correctly to:")
        print(DATA_DIR)
        return

    # 2. Instantiate and Build Model
    print("\nBuilding XceptionNet Model...")
    detector = ImageDetector(model_type='xception')
    detector.build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # The Xception model in image_detector.py expects normalized input [0, 1]
    # The Keras utility loads images as [0, 255], so we need a normalization layer
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    # Apply normalization to the datasets
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Configure for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 3. Train Model
    print(f"\nStarting training for 3 epochs...")
    try:
        # The train method expects the model to be compiled, which is done in build_model
        detector.train(train_ds.take(STEPS_PER_EPOCH), 
                       validation_dataset=val_ds.take(VALIDATION_STEPS), 
                       epochs=3, 
                       batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # 4. Save Model
    print(f"\nSaving trained model weights to {MODEL_PATH}...")
    try:
        detector.save_model(str(MODEL_PATH))
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")
        return

    print("\nTraining complete.")

if __name__ == "__main__":
    main()
