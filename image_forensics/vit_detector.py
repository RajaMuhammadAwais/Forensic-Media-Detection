import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class ViTDeepfakeDetector(nn.Module):
    """
    Vision Transformer (ViT) based Deepfake Detector.

    This model uses a pre-trained ViT-B/16 backbone and fine-tunes a
    classification head for binary deepfake detection. The architecture
    is chosen based on the research paper's finding that Transformer-based
    models offer superior generalization and accuracy compared to CNNs.
    """
    def __init__(self, num_classes=1, freeze_backbone=True):
        super(ViTDeepfakeDetector, self).__init__()
        # Load pre-trained ViT-B/16 weights
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Freeze the backbone layers
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Replace the classifier head for binary classification
        # The original classifier is self.vit.heads.head
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid() # For binary classification
        )

    def forward(self, x):
        return self.vit(x)

    @staticmethod
    def preprocess_image(image_path):
        """
        Standard preprocessing for ViT-B/16 model.
        """
        # Standard ViT-B/16 input size is 224x224
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        image = Image.open(image_path).convert("RGB")
        return preprocess(image)

    def predict(self, image_path):
        """
        Performs a prediction on a single image file.
        """
        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval() # Set model to evaluation mode

        try:
            # Preprocess and prepare the input tensor
            input_tensor = self.preprocess_image(image_path)
            input_batch = input_tensor.unsqueeze(0).to(device) # Create a mini-batch as expected by the model

            with torch.no_grad():
                output = self(input_batch)
            
            # The output is a tensor with a single value (probability of being fake)
            prediction = output.item()
            
            # Assuming 0 is 'real' and 1 is 'fake'
            result = {
                "is_fake_probability": prediction,
                "prediction": "fake" if prediction > 0.5 else "real"
            }
            return result

        except Exception as e:
            return {"error": f"Prediction failed: {e}"}

# Example usage (for testing purposes, not part of the final library)
if __name__ == '__main__':
    # This part would typically be replaced by a proper testing framework
    # For demonstration, we'll create a dummy image
    dummy_image_path = "dummy_image.png"
    dummy_image = Image.new('RGB', (500, 500), color = 'red')
    dummy_image.save(dummy_image_path)

    # Initialize the model (using dummy weights as it's not trained)
    detector = ViTDeepfakeDetector(freeze_backbone=True)
    
    # Perform a prediction
    prediction_result = detector.predict(dummy_image_path)
    print(f"Prediction for {dummy_image_path}: {prediction_result}")

    # Clean up dummy file
    import os
    os.remove(dummy_image_path)
