import unittest
import os
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from video_forensics.video_detector import VideoDetector

class TestVideoDetector(unittest.TestCase):

    def setUp(self):
        self.detector = VideoDetector(model_type="cnn_lstm")
        self.detector.build_model()
        # Create a dummy video file for testing
        self.dummy_video_path = "data/dummy_video.mp4"
        if not os.path.exists("fmd_tool/data"):
            os.makedirs("fmd_tool/data")
        # Create a simple black video using ffmpeg for testing
        os.system(f"ffmpeg -y -f lavfi -i color=c=black:s=1280x720:d=1 -vcodec libx264 -pix_fmt yuv420p -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" {self.dummy_video_path}")

    def tearDown(self):
        # Clean up dummy video
        if os.path.exists(self.dummy_video_path):
            os.remove(self.dummy_video_path)

    def test_build_model(self):
        self.assertIsNotNone(self.detector.model)

    def test_extract_frames(self):
        frames = self.detector.extract_frames(self.dummy_video_path, num_frames=5)
        self.assertEqual(len(frames), 5)
        self.assertEqual(frames[0].shape, (256, 256, 3))

    def test_predict(self):
        # Extract frames before passing them to predict
        video_frames = self.detector.extract_frames(self.dummy_video_path, num_frames=10)
        prediction = self.detector.predict(video_frames)
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(prediction.shape, (1, 1))
        self.assertGreaterEqual(prediction[0][0], 0.0)
        self.assertLessEqual(prediction[0][0], 1.0)

if __name__ == '__main__':
    unittest.main()

