import unittest
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from video_forensics.wavelet_preprocessing import WaveletPreprocessor
from video_forensics.video_detector import VideoDetector

class TestWaveletIntegration(unittest.TestCase):

    def setUp(self):
        # Create a dummy 3-channel frame (e.g., 64x64 RGB)
        self.dummy_frame = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        # Create a dummy video file for the detector to find (even if it can't read it)
        self.dummy_video_path = "dummy_video_for_test.mp4"
        with open(self.dummy_video_path, "w") as f:
            f.write("placeholder")

    def tearDown(self):
        if os.path.exists(self.dummy_video_path):
            os.remove(self.dummy_video_path)

    def test_wavelet_preprocessor_output_shape(self):
        """Test if WaveletPreprocessor produces the expected output shape."""
        preprocessor = WaveletPreprocessor(wavelet='haar', level=1)
        processed_frame = preprocessor.process_frame(self.dummy_frame)
        
        # With level=1, the output size should be half the original size in H and W,
        # and 4 times the original channels (C * 4)
        # The implementation resizes the detail coefficients to match cA, which is H/2 x W/2
        # Output shape: (H/2, W/2, C * 4) -> (32, 32, 12)
        self.assertEqual(processed_frame.shape[0], self.dummy_frame.shape[0] // 2)
        self.assertEqual(processed_frame.shape[1], self.dummy_frame.shape[1] // 2)
        self.assertEqual(processed_frame.shape[2], self.dummy_frame.shape[2] * 4)
        self.assertTrue(np.issubdtype(processed_frame.dtype, np.floating))

    def test_video_detector_with_wavelet(self):
        """Test if VideoDetector can be initialized with the wavelet preprocessor."""
        # The detector will fail to extract frames from the dummy file, but we test initialization
        # and the presence of the preprocessor.
        detector = VideoDetector(use_wavelet_preprocessor=True)
        self.assertIsNotNone(detector.preprocessor)
        self.assertTrue(isinstance(detector.preprocessor, WaveletPreprocessor))

    # Note: A full end-to-end test of extract_frames with wavelet requires a real video file
    # and a working OpenCV setup, which is complex in a sandbox. The above tests cover
    # the core logic and integration points.

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
