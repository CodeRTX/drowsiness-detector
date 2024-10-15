# Drowsiness Detector

A drowsiness detection system using facial landmarks and hand detection to monitor user alertness.

## Features

- Real-time drowsiness detection using facial landmarks.
- Head tilt detection.
- Yawning detection.
- Sound alert when drowsiness is detected.
- Logging of events with timestamps.
- Plotting of drowsiness states over time.

## Requirements

- Python 3.12.x
- OpenCV
- NumPy
- Mediapipe
- Pygame
- Matplotlib
- Winsound (for Windows)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/drowsiness-detector.git
   cd drowsiness-detector
   ```
2. Install the required packages:
  ```sh
   pip install opencv-python numpy mediapipe pygame matplotlib
   ```

## Usage

Run the drowsiness detection script:
   ```sh
   python drowsiness_detection_using_deep_learning.py
   ```
The system will start monitoring your face for signs of drowsiness.
* Press 's' to save a snapshot of the current frame.
* Press 'Esc' to exit the program.

## Files

* `drowsiness_detection_using_deep_learning.py`: Main script for drowsiness detection.
* `ringg.mp3`: Ringtone file for sound alert.
* `flowchart.png`: Flowchart of the workflow.
* `logs/events.csv`: Log file for drowsiness events.
* `logs/enhanced_events_plot.png`: Plot of drowsiness states over time.
* `snapshots/snapshot_20241010-165027.jpg`: Example snapshot of facial features.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the APACHE 2.0 License.
