# Real-time Sign Language Detection

A computer vision system that detects sign language activity in real-time using MediaPipe hand tracking and velocity-based motion analysis.

## Overview

This project implements a real-time sign language detection algorithm that analyzes hand movements to determine when someone is actively signing. The system uses Google's MediaPipe for hand landmark detection and applies velocity analysis with moving average smoothing to classify signing vs. non-signing activity.

## Features

- **Real-time Detection**: Processes live webcam feed for immediate signing detection
- **Bilateral Hand Tracking**: Monitors both left and right hands independently
- **Velocity-based Classification**: Uses movement speed of key hand landmarks
- **Noise Reduction**: Moving average filter smooths velocity measurements
- **Visual Feedback**: Real-time display of detection results and hand landmarks
- **Configurable Parameters**: Adjustable thresholds and smoothing windows

## Algorithm

The detection algorithm works by:

1. **Hand Tracking**: Uses MediaPipe to detect 21 hand landmarks per hand
2. **Key Landmark Selection**: Focuses on 5 critical points:
   - Wrist (landmark 0)
   - Thumb tip (landmark 4)
   - Index finger tip (landmark 8)
   - Ring finger tip (landmark 16)
   - Pinky tip (landmark 20)
3. **Velocity Calculation**: Computes 3D Euclidean distance between consecutive frames
4. **Smoothing**: Applies moving average filter over configurable window size
5. **Classification**: Compares average velocity against threshold for binary classification

## Requirements

### System Requirements
- Python 3.7 or higher
- Webcam or camera device
- macOS, Linux, or Windows

### Dependencies
Install required packages using:
```bash
pip install -r requirements.txt
```

Required packages:
- `opencv-python>=4.8.0` - Computer vision and video processing
- `numpy>=1.21.0` - Numerical computations
- `mediapipe>=0.10.0` - Hand tracking and landmark detection

### MediaPipe Model
The system requires the MediaPipe hand landmark model file:
- File: `hand_landmarker.task`
- Should be placed in the same directory as the script
- Download from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)

## Installation

1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the MediaPipe hand landmark model and place `hand_landmarker.task` in the project directory
4. Run the application:
   ```bash
   python realtime_mediapipe.py
   ```

## Usage

### Basic Usage
```bash
python realtime_mediapipe.py
```

### Command Line Options
```bash
python realtime_mediapipe.py [OPTIONS]
```

**Available Options:**
- `--threshold FLOAT`: Velocity threshold for signing detection (default: 0.01)
- `--window INT`: Moving average window size in frames (default: 5)
- `--camera INT`: Camera device ID (default: 0)

**Examples:**
```bash
# Use higher threshold for less sensitive detection
python realtime_mediapipe.py --threshold 0.02

# Use larger smoothing window for more stable detection
python realtime_mediapipe.py --window 10

# Use different camera (if multiple cameras available)
python realtime_mediapipe.py --camera 1

# Combine multiple options
python realtime_mediapipe.py --threshold 0.015 --window 7 --camera 0
```

### Interactive Controls
While running:
- **'q'**: Quit the application
- **'r'**: Reset velocity history (clears smoothing buffer)

## Configuration

### Threshold Tuning
The velocity threshold determines sensitivity:
- **Lower values (0.005-0.01)**: More sensitive, detects subtle movements
- **Higher values (0.02-0.05)**: Less sensitive, requires more pronounced movements
- **Recommended range**: 0.01-0.03 for most use cases

### Window Size Tuning
The moving average window affects smoothing:
- **Smaller windows (3-5 frames)**: More responsive, less smooth
- **Larger windows (10-15 frames)**: More stable, less responsive
- **Recommended range**: 5-10 frames for balanced performance

## Output Information

The system displays:
- **Detection Status**: "SIGNING" or "NOT SIGNING" with color coding
- **Average Velocity**: Current smoothed velocity measurement
- **Configuration**: Current threshold and window size settings
- **FPS Counter**: Performance monitoring (printed to console)
- **Hand Landmarks**: Visual overlay of detected hand points

## Technical Details

### Algorithm Parameters
- **Hand Detection Confidence**: 0.5 (minimum confidence for hand detection)
- **Hand Presence Confidence**: 0.5 (minimum confidence for hand tracking)
- **Tracking Confidence**: 0.5 (minimum confidence for landmark tracking)
- **Maximum Hands**: 2 (left and right hand)

### Performance
- **Typical FPS**: 15-30 FPS depending on hardware
- **Latency**: Real-time processing with minimal delay
- **Resource Usage**: Moderate CPU usage, GPU acceleration available with MediaPipe

### Coordinate System
- **Input**: Normalized coordinates from MediaPipe (0.0-1.0 range)
- **Velocity Units**: Pixels per frame
- **3D Tracking**: Uses x, y, and z coordinates for depth information

## Limitations

- **Binary Classification**: Only detects presence/absence of signing, not specific signs
- **Movement-based**: Cannot distinguish between signing and other hand movements
- **Lighting Dependent**: Performance varies with lighting conditions
- **Single Person**: Optimized for one person in frame
- **No Gesture Recognition**: Does not identify specific signs or words
