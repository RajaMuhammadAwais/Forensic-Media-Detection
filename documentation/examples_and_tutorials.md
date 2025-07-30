# Forensic Media Detection (FMD) Tool - Examples and Tutorials

## 1. Getting Started

This document provides practical examples and step-by-step tutorials for using the FMD tool effectively.

## 2. Basic Usage Examples

### 2.1. Analyzing a Single Image

**Scenario**: You have a suspicious image and want to check if it's been manipulated.

```bash
# Basic image analysis
fmd image --check suspicious_photo.jpg

# Using a specific model
fmd image --check suspicious_photo.jpg --model xception

# Saving results to a file
fmd image --check suspicious_photo.jpg --model xception --output analysis_results.json
```

**Expected Output**:
```
Analyzing image: suspicious_photo.jpg
Using model: xception

Image Analysis Results:
- Deepfake Probability: 75.5%
- Pixel Inconsistencies: Detected 64% probability of manipulation distributed.
- Lighting Mismatch: 69% confidence deepfake.
```

### 2.2. Analyzing a Video File

**Scenario**: You want to analyze a video for deepfake content.

```bash
# Basic video analysis
fmd video --check interview_video.mp4

# With output file
fmd video --check interview_video.mp4 --output video_analysis.json
```

**Expected Output**:
```
Analyzing video: interview_video.mp4
Using model: cnn_lstm

Video Analysis Results:
- Deepfake Probability: 24.9%
- Frame-by-Frame Check: 45% probability of manipulation between frames 13-15.
- Lip Sync Mismatch: Detected 38% confidence at timestamp 00:45.
```

### 2.3. Analyzing Audio Content

**Scenario**: You suspect an audio recording contains synthetic speech.

```bash
# Basic audio analysis
fmd audio --check speech_sample.wav

# Using x-vector model
fmd audio --check speech_sample.wav --model xvector --output audio_results.json
```

**Expected Output**:
```
Analyzing audio: speech_sample.wav
Using model: xvector

Audio Analysis Results:
- Deepfake Probability: 67.3%
- Voice Mismatch: Speaker not recognized with 72% confidence.
- Synthetic Speech Artifacts: Detected 65% confidence of voice synthesis.
```

### 2.4. Comprehensive Multimodal Analysis

**Scenario**: You want to perform a complete analysis of a video file that contains both visual and audio content.

```bash
# Comprehensive multimodal analysis
fmd multimodal --check complete_video.mp4

# With detailed output
fmd multimodal --check complete_video.mp4 --output comprehensive_analysis.json
```

**Expected Output**:
```
Analyzing media file: complete_video.mp4
Using model: fusion_model

Multimodal Analysis Results:
- Image: Detected unusual pixel anomalies.
- Video: Lip sync detected at 85% confidence.
- Audio: Voice mismatch detected (76% confidence).
Overall: 78% probability of deepfake detected.
Recommendation: DEEPFAKE DETECTED
```

## 3. Advanced Usage Scenarios

### 3.1. Batch Processing Multiple Files

**Scenario**: You need to analyze multiple files in a directory.

```bash
#!/bin/bash
# Batch processing script

for file in /path/to/media/files/*; do
    if [[ "$file" == *.jpg ]] || [[ "$file" == *.png ]]; then
        echo "Processing image: $file"
        fmd image --check "$file" --output "${file%.jpg}_analysis.json"
    elif [[ "$file" == *.mp4 ]] || [[ "$file" == *.avi ]]; then
        echo "Processing video: $file"
        fmd multimodal --check "$file" --output "${file%.mp4}_analysis.json"
    elif [[ "$file" == *.wav ]] || [[ "$file" == *.mp3 ]]; then
        echo "Processing audio: $file"
        fmd audio --check "$file" --output "${file%.wav}_analysis.json"
    fi
done
```

### 3.2. Comparing Different Models

**Scenario**: You want to compare results from different detection models.

```bash
# Compare image analysis models
echo "=== XceptionNet Analysis ==="
fmd image --check test_image.jpg --model xception --output xception_results.json

echo "=== Autoencoder Analysis ==="
fmd image --check test_image.jpg --model autoencoder --output autoencoder_results.json

# Compare audio analysis models
echo "=== X-Vector Analysis ==="
fmd audio --check test_audio.wav --model xvector --output xvector_results.json

echo "=== CNN-LSTM Analysis ==="
fmd audio --check test_audio.wav --model cnn_lstm --output cnn_lstm_results.json
```

### 3.3. Analyzing Different Media Types

**Scenario**: You have various types of media files to analyze.

```bash
# Image files
fmd image --check portrait.jpg --model xception
fmd image --check landscape.png --model autoencoder

# Video files
fmd video --check news_clip.mp4 --model cnn_lstm
fmd video --check interview.avi --model cnn_lstm

# Audio files
fmd audio --check podcast.wav --model xvector
fmd audio --check speech.mp3 --model cnn_lstm

# Multimodal analysis
fmd multimodal --check full_video.mp4
```

## 4. Understanding Output Results

### 4.1. JSON Output Structure

When using the `--output` option, results are saved in JSON format:

```json
{
  "file": "example_image.jpg",
  "model_used": "xception",
  "detection_type": "deepfake",
  "analysis_results": {
    "deepfake_probability": 0.755,
    "pixel_inconsistencies": {
      "detected": true,
      "probability": 0.64,
      "location": "distributed"
    },
    "lighting_mismatch": {
      "detected": true,
      "confidence": 0.69
    }
  }
}
```

### 4.2. Interpreting Probability Scores

- **0.0 - 0.3**: Likely authentic content
- **0.3 - 0.7**: Suspicious, requires further investigation
- **0.7 - 1.0**: High probability of manipulation/deepfake

### 4.3. Understanding Specific Indicators

**For Images:**
- **Pixel Inconsistencies**: Unusual patterns in pixel gradients
- **Lighting Mismatch**: Inconsistent lighting across different regions

**For Videos:**
- **Frame-by-Frame Anomalies**: Inconsistencies between consecutive frames
- **Lip Sync Mismatch**: Audio-visual synchronization issues

**For Audio:**
- **Voice Mismatch**: Speaker verification failures
- **Synthetic Speech Artifacts**: Indicators of artificial voice generation

## 5. Troubleshooting Common Issues

### 5.1. File Format Issues

**Problem**: "Could not load image/video/audio file"

**Solution**:
```bash
# Check file format and convert if necessary
file suspicious_media.xyz

# Convert image formats
convert suspicious_image.webp suspicious_image.jpg

# Convert video formats
ffmpeg -i suspicious_video.mkv suspicious_video.mp4

# Convert audio formats
ffmpeg -i suspicious_audio.m4a suspicious_audio.wav
```

### 5.2. Memory Issues

**Problem**: Out of memory errors with large files

**Solution**:
```bash
# Resize large images before analysis
convert large_image.jpg -resize 1024x1024 resized_image.jpg
fmd image --check resized_image.jpg

# Extract shorter clips from long videos
ffmpeg -i long_video.mp4 -t 60 -c copy short_clip.mp4
fmd video --check short_clip.mp4
```

### 5.3. Performance Optimization

**Tips for faster analysis**:

1. **Use appropriate models**: XceptionNet is faster than autoencoder for images
2. **Preprocess media**: Resize images and trim videos to reduce processing time
3. **Batch processing**: Process multiple files in sequence rather than parallel

## 6. Integration Examples

### 6.1. Python Script Integration

```python
import subprocess
import json

def analyze_media(file_path, media_type="auto"):
    """Analyze media file and return results"""
    
    # Determine media type if not specified
    if media_type == "auto":
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            media_type = "image"
        elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            media_type = "multimodal"
        elif file_path.lower().endswith(('.wav', '.mp3', '.flac')):
            media_type = "audio"
    
    # Run FMD analysis
    output_file = f"{file_path}_analysis.json"
    cmd = ["fmd", media_type, "--check", file_path, "--output", output_file]
    
    try:
        subprocess.run(cmd, check=True)
        
        # Load and return results
        with open(output_file, 'r') as f:
            results = json.load(f)
        return results
    
    except subprocess.CalledProcessError as e:
        print(f"Analysis failed: {e}")
        return None

# Usage example
results = analyze_media("test_video.mp4")
if results:
    print(f"Deepfake probability: {results['analysis_results']['deepfake_probability']:.1%}")
```

### 6.2. Shell Script Automation

```bash
#!/bin/bash
# Automated media analysis script

MEDIA_DIR="/path/to/media/files"
RESULTS_DIR="/path/to/results"
THRESHOLD=0.7

mkdir -p "$RESULTS_DIR"

for file in "$MEDIA_DIR"/*; do
    filename=$(basename "$file")
    extension="${filename##*.}"
    
    case "$extension" in
        jpg|jpeg|png)
            fmd image --check "$file" --output "$RESULTS_DIR/${filename}_analysis.json"
            ;;
        mp4|avi|mov)
            fmd multimodal --check "$file" --output "$RESULTS_DIR/${filename}_analysis.json"
            ;;
        wav|mp3|flac)
            fmd audio --check "$file" --output "$RESULTS_DIR/${filename}_analysis.json"
            ;;
    esac
    
    # Check if analysis indicates deepfake
    if [ -f "$RESULTS_DIR/${filename}_analysis.json" ]; then
        probability=$(python3 -c "
import json
with open('$RESULTS_DIR/${filename}_analysis.json') as f:
    data = json.load(f)
    print(data['analysis_results']['deepfake_probability'])
")
        
        if (( $(echo "$probability > $THRESHOLD" | bc -l) )); then
            echo "WARNING: $filename shows high probability of manipulation ($probability)"
        fi
    fi
done
```

## 7. Best Practices

### 7.1. File Preparation

1. **Image Quality**: Use high-resolution images (at least 256x256 pixels)
2. **Video Quality**: Ensure good video quality with clear facial features
3. **Audio Quality**: Use clear audio recordings without excessive noise

### 7.2. Analysis Workflow

1. **Start with Multimodal**: For videos, begin with multimodal analysis
2. **Compare Models**: Use multiple models for critical analyses
3. **Save Results**: Always save results for documentation and comparison
4. **Verify Findings**: Cross-reference with other detection tools when possible

### 7.3. Result Interpretation

1. **Consider Context**: High-quality deepfakes may score lower
2. **Look for Patterns**: Multiple indicators increase confidence
3. **Manual Review**: Always perform human verification for critical decisions
4. **Document Process**: Keep detailed records of analysis procedures

