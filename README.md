# Face and Mouth Movement Detection with Sound Alert

This project uses OpenCV to detect faces and monitor mouth movement, triggering a sound alert when significant movement is detected. The application is useful for scenarios like monitoring whether someone is drinking or eating.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)


## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Fardown1/Face-Consumption-Detector.git
    cd Face-Consumption-Detector
    ```

2. **Install the required dependencies:**
    ```bash
    pip install pygame

    pip install OpenCV

    pip install numpy
    ```

3. **Ensure you have the sound file in the correct path (any sound file would work) :**
    - Place your sound file (e.g., `mixkit-long-pop-2358.wav`) in the same directory as your script or adjust the path in the script accordingly.

## Usage

1. **Run the script:**
    ```bash
    python Face.py
    ```

2. **Press `q` to exit the application.**

## Dependencies

- OpenCV: For face detection and image processing.
- NumPy: For numerical operations.
- Pygame: For playing sound alerts.

Install these dependencies using pip:

```bash
pip install opencv-python numpy pygame
