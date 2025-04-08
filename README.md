README
======

Overview
--------
This repository contains a Python script that uses MediaPipe for real-time hand tracking and gesture recognition. The code:

- Captures video from a camera (webcam by default).
- Detects a single hand, identifies the position of various hand landmarks (fingertips, joints, etc.).
- Uses custom-trained classifiers (`KeyPointClassifier` and `PointHistoryClassifier`) to categorize gestures.
- Moves the system mouse pointer based on hand/fingertip position.
- Detects whether the index fingertip points toward the camera or sideways.
- Plays a scary sound and displays a fullscreen GIF if the fingertip enters a specified "restricted" rectangle on the screen.

Features
--------
1. **Hand Tracking**: Utilizes MediaPipe Hands for robust real-time hand detection and landmark extraction.
2. **Gesture Classification**: 
   - A keypoint-based classifier (`KeyPointClassifier`) identifies static hand poses.
   - A point-history-based classifier (`PointHistoryClassifier`) recognizes dynamic gestures from the fingertip’s movement history.
3. **Mouse Control**: When the index finger is pointing forward, the script moves the mouse pointer to match the fingertip location on screen.
4. **Scary GIF Trigger**: If the fingertip enters a specific rectangular area of the screen:
   - An audio file (`girl-scream.mp3`) plays non-blocking.
   - A scary GIF (`scary.gif`) displays fullscreen.
5. **Logging for Retraining**: You can press certain keys to log gesture data (keypoints or point history) to CSV files, allowing easy data collection for training or retraining models.

Dependencies
------------
- Python 3.7+ (Recommended)
- [OpenCV](https://pypi.org/project/opencv-python/) (`opencv-python`)
- [TensorFlow](https://pypi.org/project/tensorflow/)
- [Scikit-learn](https://pypi.org/project/scikit-learn/)
- [Matplotlib](https://pypi.org/project/matplotlib/)
- [Protobuf](https://pypi.org/project/protobuf/)
- [Playsound 1.2.2](https://pypi.org/project/playsound/1.2.2/)
- [ImageIO](https://pypi.org/project/imageio/)
- [pynput](https://pypi.org/project/pynput/)
- [MediaPipe](https://pypi.org/project/mediapipe/)
- [NumPy](https://pypi.org/project/numpy/)
- [tkinter](https://docs.python.org/3/library/tk.html) (Usually pre-installed with Python on most platforms)



Additionally, you need:
- `utils` module containing `CvFpsCalc`
- `model` module containing:
  - `KeyPointClassifier` (model/keypoint_classifier.py)
  - `PointHistoryClassifier` (model/point_history_classifier.py)
- CSV label files for each classifier:
  - `model/keypoint_classifier/keypoint_classifier_label.csv`
  - `model/point_history_classifier/point_history_classifier_label.csv`
- CSV files to store new training samples (if desired):
  - `model/keypoint_classifier/keypoint.csv`
  - `model/point_history_classifier/point_history.csv`

Installation
------------
1. **Clone the repository** (or download the code).
2. **Install the required libraries**:
   ```bash
   pip install opencv-python tensorflow scikit-learn matplotlib protobuf playsound==1.2.2 imageio pynput mediapipe numpy

