# Compatibility wrapper to expose AudioVisualVerifier under multimodal_analysis package
# The implementation lives at the repository root: audio_visual_verifier.py

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from audio_visual_verifier import AudioVisualVerifier  # noqa: F401