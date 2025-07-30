import unittest
import os
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from image_forensics.image_detector import ImageDetector

class TestImageDetector(unittest.TestCase):

    def setUp(self):
        self.detector = ImageDetector(model_type="xception")
        self.detector.build_model()
        # Create a dummy image for testing
        self.dummy_image_path = "fmd_tool/data/dummy_image.png"
        if not os.path.exists("fmd_tool/data"):
            os.makedirs("fmd_tool/data")
        # Create a simple black image using PIL for testing
        from PIL import Image
        img = Image.new("RGB", (256, 256), color = 'black')
        img.save(self.dummy_image_path)

    def tearDown(self):
        # Clean up dummy image
        if os.path.exists(self.dummy_image_path):
            os.remove(self.dummy_image_path)

    def test_build_model(self):
        self.assertIsNotNone(self.detector.model)

    def test_predict(self):
        # Preprocess the image before passing it to predict
        preprocessed_image = self.detector.preprocess_image(self.dummy_image_path)
        prediction = self.detector.predict(preprocessed_image)
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(prediction.shape, (1, 1))
        self.assertGreaterEqual(prediction[0][0], 0.0)
        self.assertLessEqual(prediction[0][0], 1.0)

if __name__ == '__main__':
    unittest.main()

