import argparse
import torch
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# Load model and processor
print("🔄 Loading model...")
model_id = "Hemgg/Deepfake-audio-detection"
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)

# Define label map
label_map = model.config.id2label if hasattr(model.config, "id2label") else {0: "FakeVoice", 1: "HumanVoice"}

def predict(audio_path):
    # Load audio
    speech, sr = sf.read(audio_path)
    # If stereo, take mean to mono
    if len(speech.shape) > 1:
        speech = speech.mean(axis=1)
    # 🔁 Resample if not 16 kHz
    if sr != 16000:
        import librosa
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        sr = 16000
    # Prepare input
    import numpy as np
    inputs = processor(np.array(speech), sampling_rate=sr, return_tensors="pt", padding=True)

    # Predict
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    # Format results
    results = {label_map[i]: float(probs[i]) for i in range(len(probs))}
    predicted = max(results, key=results.get)

    print("\n🎙️ Deepfake Audio Detection")
    print("---------------------------")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print(f"\n✅ Prediction: {predicted}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Audio Detector CLI")
    parser.add_argument("--audio", required=True, help="Path to audio file (.wav, .mp3, etc.)")
    args = parser.parse_args()

    predict(args.audio)
