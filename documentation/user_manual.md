# Forensic Media Detection (FMD) Tool - User Manual

## 1. Introduction

The Forensic Media Detection (FMD) Tool is an AI & ML-driven command-line interface (CLI) application designed to detect deepfake artifacts in various media types, including images, videos, and audio files. This tool leverages advanced machine learning models to analyze media for inconsistencies and anomalies that may indicate manipulation.

## 2. Installation

To use the FMD Tool, you need to have Python 3.8+ installed on your system. It is recommended to use a virtual environment to manage dependencies.

```bash
# Clone the repository (if applicable)
# git clone <repository_url>
# cd fmd_tool

# Create a virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install tensorflow torch opencv-python scikit-learn librosa click numpy pandas matplotlib seaborn pydub
```

## 3. Usage

The FMD Tool is accessed via the `fmd` command, which provides subcommands for analyzing different media types. Below are the available commands and their options.

### 3.1. General Options

*   `--version`: Show the version and exit.

### 3.2. Image Analysis (`fmd image`)

This command analyzes image files for deepfake artifacts and manipulation.

**Usage:**

```bash
fmd image --check <path_to_image_file> [OPTIONS]
```

**Options:**

*   `--check <path_to_image_file>`: **(Required)** Path to the image file to analyze.
*   `--model <model_name>`: Model to use for analysis. Supported models: `xception` (default), `autoencoder`.
*   `--detect <detection_type>`: Type of detection to perform. Default: `deepfake`.
*   `--output <output_file>`: Path to save the analysis results in JSON format.

**Example:**

```bash
fmd image --check my_image.jpg --model xception --output results.json
```

### 3.3. Video Analysis (`fmd video`)

This command analyzes video files for deepfake artifacts and manipulation.

**Usage:**

```bash
fmd video --check <path_to_video_file> [OPTIONS]
```

**Options:**

*   `--check <path_to_video_file>`: **(Required)** Path to the video file to analyze.
*   `--model <model_name>`: Model to use for analysis. Supported models: `cnn_lstm` (default).
*   `--detect <detection_type>`: Type of detection to perform. Default: `deepfake`.
*   `--output <output_file>`: Path to save the analysis results in JSON format.

**Example:**

```bash
fmd video --check my_video.mp4 --model cnn_lstm
```

### 3.4. Audio Analysis (`fmd audio`)

This command analyzes audio files for synthetic voice and manipulation.

**Usage:**

```bash
fmd audio --check <path_to_audio_file> [OPTIONS]
```

**Options:**

*   `--check <path_to_audio_file>`: **(Required)** Path to the audio file to analyze.
*   `--model <model_name>`: Model to use for analysis. Supported models: `xvector` (default), `cnn_lstm`.
*   `--detect <detection_type>`: Type of detection to perform. Default: `deepfake`.
*   `--output <output_file>`: Path to save the analysis results in JSON format.

**Example:**

```bash
fmd audio --check my_audio.wav --model xvector
```

### 3.5. Multimodal Analysis (`fmd multimodal`)

This command performs a comprehensive analysis across multiple modalities (image, video, audio) if available in the media file.

**Usage:**

```bash
fmd multimodal --check <path_to_media_file> [OPTIONS]
```

**Options:**

*   `--check <path_to_media_file>`: **(Required)** Path to the media file to analyze.
*   `--model <model_name>`: Model to use for fusion. Supported models: `fusion_model` (default).
*   `--output <output_file>`: Path to save the analysis results in JSON format.

**Example:**

```bash
fmd multimodal --check my_media.mp4
```

### 3.6. Information (`fmd info`)

This command displays information about the FMD tool, including supported analysis types and usage examples.

**Usage:**

```bash
fmd info
```

**Example:**

```bash
fmd info
```

## 4. Output Interpretation

The FMD tool provides detailed analysis results, typically in JSON format when the `--output` option is used. The output includes:

* **Deepfake Probability:** A score indicating the likelihood of the media being a deepfake.
* **Specific Anomalies:** Details about detected inconsistencies in pixels, lighting, frame-by-frame analysis, lip-sync, voice mismatch, and synthetic speech artifacts.
* **Overall Assessment:** A summary recommendation (e.g., "DEEPFAKE DETECTED", "SUSPICIOUS", "LIKELY AUTHENTIC").

## 5. Troubleshooting

* **"File not found" error:** Ensure the path to your media file is correct and the file exists.
* **Dependency issues:** If you encounter errors related to missing libraries, try reinstalling them using `pip install -r requirements.txt` (if a `requirements.txt` is provided) or the individual `pip install` commands listed in the Installation section.
* **Model loading errors:** Ensure you have sufficient memory and the correct TensorFlow/PyTorch versions installed. Some models might require pre-trained weights, which may need to be downloaded separately.

For further assistance, please refer to the project's documentation or contact support.

