# Thin module to make WaveletPreprocessor available under video_forensics package path
# to satisfy imports in tests and other modules.

from pathlib import Path
import sys

# Ensure repository root is in path to import the shared wavelet_preprocessing module
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from wavelet_preprocessing import WaveletPreprocessor  # noqa: F401