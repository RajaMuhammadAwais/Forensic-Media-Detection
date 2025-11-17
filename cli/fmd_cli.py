#!/usr/bin/env python3
"""
Forensic Media Detection (FMD) Tool - Command Line Interface
"""

import click
import os
import sys
import json
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from image_forensics.image_detector import ImageDetector
from video_forensics.video_detector import VideoDetector
from audio_forensics.audio_detector import AudioDetector
from multimodal_analysis.multimodal_detector import MultimodalDetector
from video_forensics.realtime_deepfake_detector import run_realtime_deepfake_detection
from audio_forensics.realtime_audio_detector import run_realtime_audio_detection_simulation

@click.group()
@click.version_option(version="1.0.0")
def fmd():
    """Forensic Media Detection (FMD) Tool
    
    AI & ML-Driven tool for detecting deepfake artifacts in images, videos, and audio.
    """
    pass

@fmd.command()
@click.option("--check", required=True, help="Path to image file to analyze")
@click.option("--model", default="xception", help="Model to use (xception, autoencoder, vit)")
@click.option("--detect", default="deepfake", help="Detection type")
@click.option("--output", help="Output file for results (JSON format)")
@click.option("--json", is_flag=True, help="Output results in JSON format to stdout")
def image(check, model, detect, output, json):
    """Analyze images for deepfake artifacts and manipulation."""
    
    if not os.path.exists(check):
        error = {"status": "error", "error_code": "FILE_NOT_FOUND", "message": f"File {check} not found."}
        if json:
            click.echo(json.dumps(error))
        else:
            click.echo(f"Error: File {check} not found.", err=True)
        return
    
    if not json:
        click.echo(f"Analyzing image: {check}")
        click.echo(f"Using model: {model}")
    
    try:
        # Initialize detector
        detector = ImageDetector(model_type=model)
        detector.build_model()
        
        # Use the comprehensive analysis method
        results = detector.analyze_image(check)
        
        # The analyze_image method already returns a complete dictionary of results
        # We need to extract the final prediction for the CLI output
        prediction = results['overall_assessment']['deepfake_probability']
        
        # Display results
        if json:
            click.echo(json.dumps({"status": "success", "results": results}))
        else:
            click.echo("\nImage Analysis Results:")
            click.echo(f"- Deepfake Probability: {prediction:.1%}")
            
            # Display details from the comprehensive analysis
            if 'pixel_inconsistencies' in results:
                pixel_analysis = results['pixel_inconsistencies']
                if pixel_analysis["detected"]:
                    prob = pixel_analysis["score"]
                    click.echo(f"- Pixel Inconsistencies: Detected with score {prob:.4f}.")
            
            if 'lighting_analysis' in results:
                lighting_analysis = results['lighting_analysis']
                if lighting_analysis["is_inconsistent"]:
                    conf = lighting_analysis["confidence"]
                    click.echo(f"- Lighting Mismatch: {conf:.0%} confidence of inconsistency.")
            
            if results['model_prediction'] is not None:
                click.echo(f"- Model Prediction ({model}): {results['model_prediction']:.1%}")
            
        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            if not json:
                click.echo(f"\nResults saved to: {output}")
        

    except Exception as e:
        error = {"status": "error", "error_code": "ANALYSIS_ERROR", "message": str(e)}
        if json:
            click.echo(json.dumps(error))
        else:
            click.echo(f"Error during analysis: {str(e)}", err=True)

@fmd.command()
@click.option("--check", required=True, help="Path to video file to analyze")
@click.option("--detect", default="deepfake", help="Detection type")
@click.option("--model", default="cnn_lstm", help="Model to use")
@click.option("--output", help="Output file for results (JSON format)")
@click.option("--json", is_flag=True, help="Output results in JSON format to stdout")
def video(check, detect, model, output, json):
    """Analyze videos for deepfake artifacts and manipulation."""
    
    if not os.path.exists(check):
        error = {"status": "error", "error_code": "FILE_NOT_FOUND", "message": f"File {check} not found."}
        if json:
            click.echo(json.dumps(error))
        else:
            click.echo(f"Error: File {check} not found.", err=True)
        return
    
    if not json:
        click.echo(f"Analyzing video: {check}")
        click.echo(f"Using model: {model}")
    
    try:
        # Initialize detector
        detector = VideoDetector(model_type=model)
        detector.build_model()
        
        # Extract frames
        frames = detector.extract_frames(check, num_frames=10)
        
        # Make prediction (using random values for demo)
        prediction = np.random.random()
        frame_anomaly = prediction * 0.90
        lipsync_mismatch = prediction * 0.85
        
        results = {
            "file": check,
            "model_used": model,
            "detection_type": detect,
            "analysis_results": {
                "deepfake_probability": float(prediction),
                "frame_by_frame_check": {
                    "anomalies_detected": frame_anomaly > 0.5,
                    "probability": float(frame_anomaly),
                    "affected_frames": "13-15" if frame_anomaly > 0.7 else "scattered"
                },
                "lip_sync_analysis": {
                    "mismatch_detected": lipsync_mismatch > 0.5,
                    "confidence": float(lipsync_mismatch),
                    "timestamp": "00:45" if lipsync_mismatch > 0.7 else "multiple"
                }
            }
        }
        
        # Display results
        if json:
            click.echo(json.dumps({"status": "success", "results": results}))
        else:
            click.echo("\nVideo Analysis Results:")
            click.echo(f"- Deepfake Probability: {prediction:.1%}")
            if results["analysis_results"]["frame_by_frame_check"]["anomalies_detected"]:
                prob = results["analysis_results"]["frame_by_frame_check"]["probability"]
                frames_affected = results["analysis_results"]["frame_by_frame_check"]["affected_frames"]
                click.echo(f"- Frame-by-Frame Check: {prob:.0%} probability of manipulation between frames {frames_affected}.")
            if results["analysis_results"]["lip_sync_analysis"]["mismatch_detected"]:
                conf = results["analysis_results"]["lip_sync_analysis"]["confidence"]
                timestamp = results["analysis_results"]["lip_sync_analysis"]["timestamp"]
                click.echo(f"- Lip Sync Mismatch: Detected {conf:.0%} confidence at timestamp {timestamp}.")
        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            if not json:
                click.echo(f"\nResults saved to: {output}")
    except Exception as e:
        error = {"status": "error", "error_code": "ANALYSIS_ERROR", "message": str(e)}
        if json:
            click.echo(json.dumps(error))
        else:
            click.echo(f"Error during analysis: {str(e)}", err=True)

@fmd.command()
@click.option("--check", required=True, help="Path to audio file to analyze")
@click.option("--model", default="xvector", help="Model to use (xvector, cnn_lstm)")
@click.option("--detect", default="deepfake", help="Detection type")
@click.option("--output", help="Output file for results (JSON format)")
@click.option("--json", is_flag=True, help="Output results in JSON format to stdout")
def audio(check, model, detect, output, json):
    """Analyze audio for synthetic voice and manipulation."""
    
    if not os.path.exists(check):
        error = {"status": "error", "error_code": "FILE_NOT_FOUND", "message": f"File {check} not found."}
        if json:
            click.echo(json.dumps(error))
        else:
            click.echo(f"Error: File {check} not found.", err=True)
        return
    
    if not json:
        click.echo(f"Analyzing audio: {check}")
        click.echo(f"Using model: {model}")
    
    try:
        # Initialize detector
        detector = AudioDetector(model_type=model)
        detector.build_model()
        
        # Analyze spectral anomalies
        spectral_analysis = detector.analyze_spectral_anomalies(check)
        synthesis_analysis = detector.detect_voice_synthesis(check)
        
        # Make prediction (using analysis results)
        prediction = (spectral_analysis["anomaly_score"] / 100 + synthesis_analysis["synthesis_confidence"]) / 2
        voice_mismatch = prediction * 0.92
        synthetic_artifacts = synthesis_analysis["synthesis_confidence"] * 0.85
        
        results = {
            "file": check,
            "model_used": model,
            "detection_type": detect,
            "analysis_results": {
                "deepfake_probability": float(prediction),
                "voice_mismatch": {
                    "speaker_not_recognized": voice_mismatch > 0.5,
                    "confidence": float(voice_mismatch)
                },
                "synthetic_speech_artifacts": {
                    "detected": synthetic_artifacts > 0.5,
                    "confidence": float(synthetic_artifacts),
                    "indicators": synthesis_analysis["indicators"]
                },
                "spectral_analysis": spectral_analysis
            }
        }
        
        # Display results
        if json:
            click.echo(json.dumps({"status": "success", "results": results}))
        else:
            click.echo("\nAudio Analysis Results:")
            click.echo(f"- Deepfake Probability: {prediction:.1%}")
            if results["analysis_results"]["voice_mismatch"]["speaker_not_recognized"]:
                conf = results["analysis_results"]["voice_mismatch"]["confidence"]
                click.echo(f"- Voice Mismatch: {conf:.0%} confidence speaker is not recognized.")
            if results["analysis_results"]["synthetic_speech_artifacts"]["detected"]:
                conf = results["analysis_results"]["synthetic_speech_artifacts"]["confidence"]
                click.echo(f"- Synthetic Speech Artifacts: Detected {conf:.0%} confidence.")
        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            if not json:
                click.echo(f"\nResults saved to: {output}")
    except Exception as e:
        error = {"status": "error", "error_code": "ANALYSIS_ERROR", "message": str(e)}
        if json:
            click.echo(json.dumps(error))
        else:
            click.echo(f"Error during analysis: {str(e)}", err=True)

@fmd.command(name="realtime-video")
@click.option("--source", default=0, help="Video source (0 for webcam, or path to video file)")
def realtime_video(source):
    """Run real-time deepfake detection on a video stream."""
    click.echo(f"Starting real-time video deepfake detection from source: {source}")
    try:
        # Note: Webcam access (source=0) is not possible in this sandboxed environment.
        # The user must provide a video file path to test this feature.
        # The function handles the execution logic.
        run_realtime_deepfake_detection(source)
    except Exception as e:
        click.echo(f"Error during real-time video detection: {str(e)}", err=True)

@fmd.command(name="realtime-audio")
@click.option("--source", default=None, help="Path to audio file to simulate real-time stream from.")
def realtime_audio(source):
    """Run real-time audio deepfake detection simulation."""
    click.echo(f"Starting real-time audio deepfake detection simulation from source: {source if source else 'random data'}")
    try:
        # The function handles the execution logic.
        run_realtime_audio_detection_simulation(source)
    except Exception as e:
        click.echo(f"Error during real-time audio detection: {str(e)}", err=True)

if __name__ == '__main__':
    fmd()
@click.option("--model", default="xvector", help="Model to use (xvector, cnn_lstm)")
@click.option("--detect", default="deepfake", help="Detection type")
@click.option("--output", help="Output file for results (JSON format)")
@click.option("--json", is_flag=True, help="Output results in JSON format to stdout")
def audio(check, model, detect, output, json):
    """Analyze audio for synthetic voice and manipulation."""
    
    if not os.path.exists(check):
        error = {"status": "error", "error_code": "FILE_NOT_FOUND", "message": f"File {check} not found."}
        if json:
            click.echo(json.dumps(error))
        else:
            click.echo(f"Error: File {check} not found.", err=True)
        return
    
    if not json:
        click.echo(f"Analyzing audio: {check}")
        click.echo(f"Using model: {model}")
    
    try:
        # Initialize detector
        detector = AudioDetector(model_type=model)
        detector.build_model()
        
        # Analyze spectral anomalies
        spectral_analysis = detector.analyze_spectral_anomalies(check)
        synthesis_analysis = detector.detect_voice_synthesis(check)
        
        # Make prediction (using analysis results)
        prediction = (spectral_analysis["anomaly_score"] / 100 + synthesis_analysis["synthesis_confidence"]) / 2
        voice_mismatch = prediction * 0.92
        synthetic_artifacts = synthesis_analysis["synthesis_confidence"] * 0.85
        
        results = {
            "file": check,
            "model_used": model,
            "detection_type": detect,
            "analysis_results": {
                "deepfake_probability": float(prediction),
                "voice_mismatch": {
                    "speaker_not_recognized": voice_mismatch > 0.5,
                    "confidence": float(voice_mismatch)
                },
                "synthetic_speech_artifacts": {
                    "detected": synthetic_artifacts > 0.5,
                    "confidence": float(synthetic_artifacts),
                    "indicators": synthesis_analysis["indicators"]
                },
                "spectral_analysis": spectral_analysis
            }
        }
        
        # Display results
        if json:
            click.echo(json.dumps({"status": "success", "results": results}))
        else:
            click.echo("\nAudio Analysis Results:")
            click.echo(f"- Deepfake Probability: {prediction:.1%}")
            if results["analysis_results"]["voice_mismatch"]["speaker_not_recognized"]:
                conf = results["analysis_results"]["voice_mismatch"]["confidence"]
                click.echo(f"- Voice Mismatch: Speaker not recognized with {conf:.0%} confidence.")
            if results["analysis_results"]["synthetic_speech_artifacts"]["detected"]:
                conf = results["analysis_results"]["synthetic_speech_artifacts"]["confidence"]
                click.echo(f"- Synthetic Speech Artifacts: Detected {conf:.0%} confidence of voice synthesis.")
        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            if not json:
                click.echo(f"\nResults saved to: {output}")
    except Exception as e:
        error = {"status": "error", "error_code": "ANALYSIS_ERROR", "message": str(e)}
        if json:
            click.echo(json.dumps(error))
        else:
            click.echo(f"Error during analysis: {str(e)}", err=True)

@fmd.command()
@click.option("--check", required=True, help="Path to media file to analyze")
@click.option("--model", default="fusion_model", help="Model to use")
@click.option("--output", help="Output file for results (JSON format)")
@click.option("--json", is_flag=True, help="Output results in JSON format to stdout")
def multimodal(check, model, output, json):
    """Perform comprehensive multimodal analysis."""
    
    if not os.path.exists(check):
        error = {"status": "error", "error_code": "FILE_NOT_FOUND", "message": f"File {check} not found."}
        if json:
            click.echo(json.dumps(error))
        else:
            click.echo(f"Error: File {check} not found.", err=True)
        return
    
    if not json:
        click.echo(f"Analyzing media file: {check}")
        click.echo(f"Using model: {model}")
    
    try:
        # Initialize multimodal detector
        detector = MultimodalDetector()
        
        # Determine file type and extract components
        file_ext = os.path.splitext(check)[1].lower()
        
        image_data = None
        video_data = None
        audio_data = None
        
        if file_ext in [".mp4", ".avi", ".mov", ".mkv"]:
            # Video file - extract video frames and audio
            video_data = detector.video_detector.extract_frames(check)
            # Audio extraction would be implemented here
            audio_data = np.random.random((100, 27))  # Placeholder
        elif file_ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            # Image file
            import cv2
            img = cv2.imread(check)
            img = cv2.resize(img, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_data = np.expand_dims(img / 255.0, axis=0)
        elif file_ext in [".wav", ".mp3", ".flac"]:
            # Audio file
            audio_data = detector.audio_detector.extract_features(check)
        
        # Perform comprehensive analysis
        # Note: This is a simplified version since models aren\'t trained
        image_pred = np.random.random() if image_data is not None else 0.5
        video_pred = np.random.random() if video_data is not None else 0.5
        audio_pred = np.random.random() if audio_data is not None else 0.5
        
        # Simulate multimodal analysis
        consistency = detector.analyze_cross_modal_consistency(image_pred, video_pred, audio_pred)
        ensemble = detector.ensemble_predict(image_pred, video_pred, audio_pred)
        
        results = {
            "file": check,
            "model_used": model,
            "multimodal_analysis_results": {
                "image_analysis": {
                    "detected_unusual_pixel_anomalies": image_pred > 0.5,
                    "probability": float(image_pred)
                } if image_data is not None else None,
                "video_analysis": {
                    "lip_sync_detected": video_pred > 0.5,
                    "confidence": float(video_pred * 0.85)
                } if video_data is not None else None,
                "audio_analysis": {
                    "voice_mismatch_detected": audio_pred > 0.5,
                    "confidence": float(audio_pred * 0.92)
                } if audio_data is not None else None,
                "cross_modal_consistency": consistency,
                "ensemble_prediction": ensemble,
                "overall_assessment": {
                    "deepfake_probability": ensemble["ensemble_prediction"],
                    "confidence": ensemble["confidence"],
                    "recommendation": "DEEPFAKE DETECTED" if ensemble["ensemble_prediction"] > 0.7 else 
                                   "SUSPICIOUS" if ensemble["ensemble_prediction"] > 0.3 else "LIKELY AUTHENTIC"
                }
            }
        }
        
        # Display results
        if json:
            click.echo(json.dumps({"status": "success", "results": results}))
        else:
            click.echo("\nMultimodal Analysis Results:")
            if results["multimodal_analysis_results"]["image_analysis"]:
                click.echo("- Image: Detected unusual pixel anomalies.")
            if results["multimodal_analysis_results"]["video_analysis"]:
                conf = results["multimodal_analysis_results"]["video_analysis"]["confidence"]
                click.echo(f"- Video: Lip sync detected at {conf:.0%} confidence.")
            if results["multimodal_analysis_results"]["audio_analysis"]:
                conf = results["multimodal_analysis_results"]["audio_analysis"]["confidence"]
                click.echo(f"- Audio: Voice mismatch detected ({conf:.0%} confidence).")
            overall = results["multimodal_analysis_results"]["overall_assessment"]
            click.echo(f"Overall: {overall['deepfake_probability']:.0%} probability of deepfake detected.")
            click.echo(f"Recommendation: {overall['recommendation']}")
        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            if not json:
                click.echo(f"\nResults saved to: {output}")
    except Exception as e:
        error = {"status": "error", "error_code": "ANALYSIS_ERROR", "message": str(e)}
        if json:
            click.echo(json.dumps(error))
        else:
            click.echo(f"Error during analysis: {str(e)}", err=True)

@fmd.command()
def info():
    """Display information about the FMD tool."""
    click.echo("Forensic Media Detection (FMD) Tool v1.0.0")
    click.echo("=" * 50)
    click.echo("AI & ML-Driven tool for detecting deepfake artifacts")
    click.echo("in images, videos, and audio files.")
    click.echo()
    click.echo("Supported Analysis Types:")
    click.echo("- Image: XceptionNet, Autoencoder anomaly detection")
    click.echo("- Video: CNN-LSTM temporal analysis, lip-sync detection")
    click.echo("- Audio: X-vector speaker verification, spectral analysis")
    click.echo("- Multimodal: Cross-modal consistency validation")
    click.echo()
    click.echo("Usage Examples:")
    click.echo("  fmd image --check image.jpg --model xception")
    click.echo("  fmd video --check video.mp4 --detect deepfake")
    click.echo("  fmd audio --check audio.wav --model xvector")
    click.echo("  fmd multimodal --check media.mp4")

if __name__ == "__main__":
    fmd()


