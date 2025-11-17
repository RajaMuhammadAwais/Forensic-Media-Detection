import unittest
import os
from PIL import Image
from image_forensics.image_detector import ImageDetector

class TestViTDetector(unittest.TestCase):

    def setUp(self):
        # Create a dummy image for testing
        self.dummy_image_path = "test_image.png"
        dummy_image = Image.new('RGB', (224, 224), color = 'red')
        dummy_image.save(self.dummy_image_path)

    def tearDown(self):
        # Remove the dummy image
        if os.path.exists(self.dummy_image_path):
            os.remove(self.dummy_image_path)

    def test_vit_model_initialization(self):
        """Test if the ViT model can be initialized correctly."""
        try:
            detector = ImageDetector(model_type='vit')
            detector.build_model()
            self.assertIsNotNone(detector.model, "ViT model should be initialized.")
            self.assertTrue(detector.is_pytorch, "is_pytorch flag should be True for ViT model.")
        except Exception as e:
            self.fail(f"ViT model initialization failed with an exception: {e}")

    def test_vit_prediction(self):
        """Test the prediction method of the ViT model."""
        detector = ImageDetector(model_type='vit')
        detector.build_model()
        
        # Prediction should return a dictionary with results
        result = detector.analyze_image(self.dummy_image_path)
        
        self.assertIn('overall_assessment', result, "Result should contain 'overall_assessment'.")
        self.assertIn('deepfake_probability', result['overall_assessment'], "'overall_assessment' should contain 'deepfake_probability'.")
        self.assertIsInstance(result['overall_assessment']['deepfake_probability'], float, "Deepfake probability should be a float.")

if __name__ == '__main__':
    unittest.main()
