import unittest
import os
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from multimodal_analysis.multimodal_detector import MultimodalDetector

class TestMultimodalDetector(unittest.TestCase):

    def setUp(self):
        self.detector = MultimodalDetector()

    def test_analyze_cross_modal_consistency(self):
        image_pred = 0.8
        video_pred = 0.7
        audio_pred = 0.9
        consistency = self.detector.analyze_cross_modal_consistency(image_pred, video_pred, audio_pred)
        self.assertIsInstance(consistency, dict)
        self.assertIn("consistency_score", consistency)
        self.assertIn("outlier_modalities", consistency)

    def test_ensemble_predict(self):
        image_pred = 0.8
        video_pred = 0.7
        audio_pred = 0.9
        ensemble = self.detector.ensemble_predict(image_pred, video_pred, audio_pred)
        self.assertIsInstance(ensemble, dict)
        self.assertIn("ensemble_prediction", ensemble)
        self.assertIn("confidence", ensemble)

if __name__ == '__main__':
    unittest.main()

