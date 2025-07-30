import unittest
import os
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from audio_forensics.audio_detector import AudioDetector

class TestAudioDetector(unittest.TestCase):

    def setUp(self):
        self.detector = AudioDetector(model_type="xvector")
        self.detector.build_model()
        # Create a dummy audio file for testing
        self.dummy_audio_path = "fmd_tool/data/dummy_audio.wav"
        if not os.path.exists("fmd_tool/data"):
            os.makedirs("fmd_tool/data")
        # Create a simple silent WAV file using pydub for testing
        from pydub import AudioSegment
        AudioSegment.silent(duration=1000).export(self.dummy_audio_path, format="wav")
        
        # Fit the scaler with some dummy data to avoid AttributeError
        dummy_mfccs = np.random.rand(10, 20) # 10 timesteps, 20 mfccs
        self.detector.scaler.fit(dummy_mfccs)

    def tearDown(self):
        # Clean up dummy audio
        if os.path.exists(self.dummy_audio_path):
            os.remove(self.dummy_audio_path)

    def test_build_model(self):
        self.assertIsNotNone(self.detector.model)

    def test_analyze_spectral_anomalies(self):
        results = self.detector.analyze_spectral_anomalies(self.dummy_audio_path)
        self.assertIsInstance(results, dict)
        self.assertIn("anomaly_score", results)

    def test_detect_voice_synthesis(self):
        results = self.detector.detect_voice_synthesis(self.dummy_audio_path)
        self.assertIsInstance(results, dict)
        self.assertIn("synthesis_confidence", results)

if __name__ == '__main__':
    unittest.main()

